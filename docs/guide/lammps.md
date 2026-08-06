# LAMMPS

LES models run in LAMMPS through the pair styles in
[`mir-group/pair_nequip_allegro`](https://github.com/mir-group/pair_nequip_allegro).
Everything about *how to use LAMMPS* -- building it, the pair style syntax, Kokkos, MPI,
running with GPUs -- is documented there and in the LAMMPS manual; this page only records
what is specific to LES.

* [`pair_nequip_allegro` README](https://github.com/mir-group/pair_nequip_allegro) --
  installation, patching, CMake options, pair style syntax, FAQ
* [NequIP: LAMMPS pair styles](https://nequip.readthedocs.io/en/latest/integrations/lammps/pair_styles.html) --
  which `nequip-compile` target goes with which pair style
* [LAMMPS: Build with CMake](https://docs.lammps.org/Build_cmake.html) and
  [Kokkos on GPUs](https://docs.lammps.org/Speed_kokkos.html)

## What works

| | LES model in LAMMPS |
|---|---|
| `pair_style nequip` (target `pair_nequip`) | ✅ periodic and non-periodic |
| `pair_style allegro` (target `pair_allegro`) | ⚠️ **non-periodic only** -- the target passes no cell |
| more than one MPI rank | ❌ **single rank only**, see below |
| energies, forces, virial | ✅ |
| `enable_TritonContracter` | ✅ the kernels are inside the compiled artefact |
| `enable_OpenEquivariance`, `enable_CuEquivariance`, `enable_CuEquivarianceContracter` | ❌ see [accelerations](#accelerations-in-lammps) |
| `pair_style mliap unified` (ML-IAP) | ❌ LES unsupported, see [ML-IAP](#ml-iap) |
| reading `LES_q` / `LES_BEC` out of LAMMPS | ❌ see [deployment](deployment.md#latent-charges-are-not-in-a-deployed-model) |

## One MPI rank

**A LES model in LAMMPS must run on a single MPI rank.** This is not an implementation
detail that will be tuned away; it follows from what an Ewald sum is.

The reciprocal-space term needs the structure factor over *every* atom in the cell,

$$S(\mathbf{k}) = \sum_{i=1}^{N} q_i^\text{les}\, e^{i\mathbf{k}\cdot\mathbf{r}_i}$$

which no rank can form from its own subdomain. Forces make it harder still: they are the
autograd derivative of a *global* energy, and as the LES Perspective puts it, this "makes it
nontrivial to distribute force evaluation across multiple GPUs"
([Kim & Cheng 2026](https://doi.org/10.1063/5.0316886), Limitations).

Two consequences:

* `pair_style nequip` is limited to one rank by the pair style itself, so it is already
  consistent with LES.
* `pair_style allegro` supports MPI in general, but you must not use that with LES: each rank
  would evaluate an Ewald sum over its own local + ghost atoms, which double-counts atoms and
  silently gives the wrong long-range energy.

Run with one rank, and use Kokkos to put that rank on a GPU.

## Periodic Allegro models

The `pair_allegro` target's graph carries positions, edge indices and atom types -- **no
cell** -- so a periodic LES model cannot be exported to it and is refused with an error
naming the missing cell (see [deployment](deployment.md#pair_allegro-and-periodic-models)).

The way round it does not need any patching: **export the Allegro model with
`--target pair_nequip` and use `pair_style nequip`.** That target passes the cell and
`edge_cell_shift`, and hands the model only the local atoms -- exactly what an Ewald sum
needs. The pair style does not care which backbone produced the model, only that the input
contract matches, and Allegro is strictly local so it never needed ghost nodes. Nothing is
lost by giving up `pair_style allegro`'s MPI support, because LES is single-rank regardless.

```{note}
This route is what [issue #15](https://github.com/ChengUCB/NequIP-LES/issues/15) reports
using in practice. Our own end-to-end check of it is still pending; the energy/force
agreement for a *NequIP* backbone through `pair_nequip` is verified.
```

## Building

`tests/build_lammps.sh` in this repository builds the two LAMMPS variants with the flags the
upstream sources call for. It is a convenience wrapper, not a replacement for the
documentation above -- read the pair README if anything goes wrong.

```bash
cd tests
module load gcc cuda mpi            # names differ per cluster
./build_lammps.sh check             # preflight only, builds nothing
./build_lammps.sh pair  ~/software  # pair_style nequip / allegro
```

The preflight exists because each of these has cost hours before:

| check | why |
|---|---|
| `CXX11 ABI` | if `torch._C._GLIBCXX_USE_CXX11_ABI` is False, a Kokkos build will not link against those wheels |
| `nvcc` present | the CUDA bundled with pip torch cannot compile LAMMPS |
| CUDA majors match | libtorch and Kokkos must not pull in two CUDA majors; the build is refused otherwise |
| MPI compiler present | `pair_nequip_allegro.cpp` calls `MPI_Comm_split_type()` unconditionally and LAMMPS' bundled STUBS `mpi.h` does not define it, so a real MPI is **required** even though you will run on one rank |
| Kokkos GPU architecture | detected from `nvidia-smi`; on a GPU-less build node pass `KOKKOS_ARCH_OVERRIDE=HOPPER90` |

Compiling needs no GPU -- `nvcc` runs on the CPU -- so a large CPU node is a good place to
build. The *tests* need a GPU.

The flags follow
[`pair_nequip_allegro`'s own CI](https://github.com/mir-group/pair_nequip_allegro/blob/main/.github/workflows/tests.yml),
which is tested against CUDA 12.6; `-DNEQUIP_AOT_COMPILE=ON` is what lets `lmp` load
`.nequip.pt2` (AOTInductor) files at all.

## Running

Compile the model, then point `pair_coeff` at it:

```bash
nequip-compile model.ckpt model.nequip.pt2 \
    --device cuda --mode aotinductor --target pair_nequip
```

```
units           metal
atom_style      atomic
boundary        p p p
newton          off              # pair_style nequip requires off; allegro requires on
read_data       system.data
pair_style      nequip
pair_coeff      * * model.nequip.pt2 H O
```

The names after the model file map LAMMPS atom types 1, 2, … to the model's `type_names`, in
order. Getting that order wrong produces a run that looks healthy and has meaningless
forces, so take the list from the model rather than retyping it.

`newton` differs between the two styles and each errors out if given the wrong one
(`pair_nequip_allegro.cpp:149-150`):

| pair style | required |
|---|---|
| `nequip` | `newton off` |
| `allegro` | `newton on` |

## Accelerations in LAMMPS

Only accelerations whose kernels end up **inside** the compiled artefact work in LAMMPS:

| modifier | export | runs in `lmp` |
|---|---|---|
| `enable_TritonContracter` (Allegro) | ✅ | ✅ |
| `enable_OpenEquivariance` (NequIP) | ✅ | ❌ `Could not find schema for libtorch_tp_jit::jit_conv_forward` |
| `enable_CuEquivariance` (NequIP) | ✅ | ❌ `Could not find schema for cuequivariance::fused_tensor_product` |
| `enable_CuEquivarianceContracter` (Allegro) | ✅ | ❌ `Could not find schema for cuequivariance::uniform_1d` |

OpenEquivariance and cuEquivariance register custom torch operators from **Python**. The C++
`lmp` binary never imports those packages, so the operator schemas are not registered and
loading the artefact fails. The export succeeding proves nothing here -- which is why the
test suite now checks that artefacts actually *run*, not only that they compile.

Triton kernels are generated into the AOTInductor artefact and therefore travel with it.

## ML-IAP

**Not supported for LES.** The
[ML-IAP](https://nequip.readthedocs.io/en/latest/integrations/lammps/mliap.html) wrapper
hands the model `edge_vectors` but neither absolute positions nor the cell, so a LES model
raises `KeyError: 'pos'` -- the Ewald sum has nothing to sum over. Its run-time
`torch.compile` additionally fails on torch 2.13 inside nequip's own cutoff function. Both
are upstream matters in what nequip documents as a beta integration; the test suite records
them as a known gap and runs nothing.
