# LAMMPS

LES models run in LAMMPS through the pair styles in
[`mir-group/pair_nequip_allegro`](https://github.com/mir-group/pair_nequip_allegro). Everything
about *how to use LAMMPS* -- the pair style syntax, Kokkos, MPI, running on GPUs -- is documented
there and in the LAMMPS manual; this page records only what is specific to LES.

* [`pair_nequip_allegro` README](https://github.com/mir-group/pair_nequip_allegro) --
  installation, patching, CMake options, pair style syntax, FAQ
* [NequIP: LAMMPS pair styles](https://nequip.readthedocs.io/en/latest/integrations/lammps/pair_styles.html) --
  which `nequip-compile` target goes with which pair style
* [LAMMPS: Build with CMake](https://docs.lammps.org/Build_cmake.html) and
  [Kokkos on GPUs](https://docs.lammps.org/Speed_kokkos.html)

## The LAMMPS route for LES models

**Compile with `--target pair_nequip` and run `pair_style nequip`, on a single MPI rank.** This
holds for an Allegro backbone too, and needs no patched nequip and no patched pair style.

| | `pair_style nequip` | `pair_style allegro` | why |
|---|---|---|---|
| periodic LES | ✅ | ❌ | `pair_allegro` passes no cell, so the reciprocal-space sum cannot be evaluated; the export is refused |
| non-periodic LES | ✅ | ✅ | |
| energies, forces, virial | ✅ | ✅ | |
| Kokkos | ❌ | ✅ | Kokkos is only implemented for `pair_style allegro` |
| more than one MPI rank | ❌ | ❌ | an Ewald sum cannot be domain-decomposed, [below](#one-mpi-rank) |
| `enable_TritonContracter` | ✅ | ✅ | the kernels are inside the compiled artefact |
| `enable_OpenEquivariance`, `enable_CuEquivariance`, `enable_CuEquivarianceContracter` | ❌ | ❌ | their operators are registered from Python, and upstream scopes them to ASE (OpenEquivariance also to ML-IAP) -- [details](deployment.md#accelerations) |
| ML-IAP | ❌ | ❌ | [below](#ml-iap) |

`pair_nequip` works for either backbone because a target is an input/output contract, not an
architecture: it passes the cell, `edge_cell_shift`, and only the local atoms -- exactly what an
Ewald sum needs. `pair_allegro` instead passes local **+ ghost** atoms and no cell, so even with a
cell added it would count every atom once per periodic image.

Giving up `pair_style allegro` costs no MPI parallelism, since LES is single-rank regardless.
**Losing Kokkos is the real trade**: the model still runs on the GPU through libtorch, but the
LAMMPS-side Kokkos path is
[only implemented for `pair_style allegro`](https://github.com/mir-group/pair_nequip_allegro#kokkos-recommended-best-gpu-performance-and-most-reliable).
Lifting that would need `pair_allegro` to pass the cell *and* the local atom count so the Ewald
sum could be restricted to real atoms.

## One MPI rank

**A LES model in LAMMPS must run on a single MPI rank.** The reciprocal-space term needs the
structure factor over *every* atom in the cell,

$$S(\mathbf{k}) = \sum_{i=1}^{N} q_i^\text{les}\, e^{i\mathbf{k}\cdot\mathbf{r}_i}$$

which no rank can form from its own subdomain. Forces make it harder still: they are the autograd
derivative of a *global* energy, which "makes it nontrivial to distribute force evaluation across
multiple GPUs" ([Kim & Cheng 2026](https://doi.org/10.1063/5.0316886), Limitations).

`pair_style nequip` enforces one rank by itself, so it is already consistent with LES.
`pair_style allegro` supports MPI in general, but with LES each rank would evaluate an Ewald sum
over its own atoms and silently give the wrong long-range energy.

## Running

Compile the model, then use it exactly as the
[pair style documentation](https://github.com/mir-group/pair_nequip_allegro#usage) describes --
nothing about the input script is LES-specific.

```bash
nequip-compile model.ckpt model.nequip.pt2 \
    --device cuda --mode aotinductor --target pair_nequip
```

```
pair_style      nequip
pair_coeff      * * model.nequip.pt2 H O
```

The names after the model file map LAMMPS atom types 1, 2, … to the model's `type_names`, in
order. Getting that order wrong gives a run that looks healthy and has meaningless forces, so
take the list from the model rather than retyping it.

## Building

`tests/build_lammps.sh` builds LAMMPS with the flags the upstream sources call for. It is a
convenience wrapper, not a replacement for the documentation above.

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
| CUDA majors match | libtorch and Kokkos must not pull in two CUDA majors; see [Troubleshooting](troubleshooting.md#which-cuda-build-of-torch-if-you-will-build-lammps) |
| MPI compiler present | `pair_nequip_allegro.cpp` calls `MPI_Comm_split_type()` unconditionally and LAMMPS' bundled STUBS `mpi.h` does not define it, so a real MPI is **required** even though you will run on one rank |
| Kokkos GPU architecture | detected from `nvidia-smi`; on a GPU-less build node pass `KOKKOS_ARCH_OVERRIDE=HOPPER90` |

Compiling needs no GPU -- `nvcc` runs on the CPU -- so a large CPU node is a good place to build.
The *tests* need a GPU. The flags follow
[`pair_nequip_allegro`'s own CI](https://github.com/mir-group/pair_nequip_allegro/blob/main/.github/workflows/tests.yml),
which is tested against CUDA 12.6; `-DNEQUIP_AOT_COMPILE=ON` is what lets `lmp` load
`.nequip.pt2` (AOTInductor) files at all.

## ML-IAP

**Not supported for LES.** The
[ML-IAP](https://nequip.readthedocs.io/en/latest/integrations/lammps/mliap.html) wrapper passes
the model `edge_vectors` but neither absolute positions nor the cell, so the Ewald sum has
nothing to sum over and a LES model raises `KeyError: 'pos'`. It is also the only documented
LAMMPS route for
[OpenEquivariance](https://nequip.readthedocs.io/en/latest/guide/accelerations/openequivariance.html),
so one upstream change would open both.
