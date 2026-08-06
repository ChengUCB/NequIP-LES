# What works: compilation, deployment, accelerations

The NequIP framework's compilation and deployment machinery is documented
[here](https://nequip.readthedocs.io/en/latest/guide/getting-started/workflow.html); this
page records how far a LES model gets along each of those paths, and why it stops where it
does. Everything below assumes the vectorized Ewald (`is_periodic` set) -- with the legacy
module, every row in the compilation table is ❌.

## Capabilities

| | LES model | notes |
|---|---|---|
| eager training (`compile_mode: eager`) | ✅ | works with legacy too |
| train-time compilation (`compile_mode: compile`) | ✅ | needs `is_periodic`; see torch versions below |
| `nequip-package build` | ✅ | |
| `nequip-compile --mode aotinductor --target ase` | ✅ | periodic and non-periodic |
| `nequip-compile --mode aotinductor --target batch` | ✅ | for torch-sim |
| `nequip-compile --mode aotinductor --target pair_nequip` | ✅ | see [LAMMPS](lammps.md) |
| `nequip-compile --mode aotinductor --target pair_allegro` | ⚠️ | non-periodic only -- no cell in the graph |
| `nequip-compile --mode torchscript` | ⚠️ | torch < 2.10 only (nequip refuses above) |
| ASE calculator | ✅ | energies, forces, stress |
| torch-sim | ✅ | `pip install torch-sim-atomistic`, needs python >= 3.11 |
| LAMMPS, single MPI rank | ✅ | |
| LAMMPS, multiple MPI ranks | ❌ | an Ewald sum cannot be domain-decomposed -- [why](lammps.md#one-mpi-rank) |
| LAMMPS ML-IAP | ❌ | LES unsupported upstream -- [why](lammps.md#ml-iap) |
| BEC inference from an eager checkpoint | ✅ | |
| BEC inference from a compile-trained model | ❌ | output keys are fixed at trace time |
| reading `LES_q` / `LES_BEC` from a deployed model | ❌ | the targets export energies and forces only -- see below |
| mixed periodic + isolated dataset | ❌ | legacy module only, and then no compilation |
| CPU / CUDA | ✅ / ✅ | compilation works on both |

## The long-range energy reaches LAMMPS through `per_atom_energy`

The Ewald sum produces one number per structure. Until recently it was added to
`total_energy` only, and `per_atom_energy` carried the short-range part alone. ASE reads
`total_energy` and was therefore correct, but the LAMMPS pair styles have no global energy
channel: `pair_nequip_allegro.cpp` builds LAMMPS' `eng_vdwl` by *summing the per-atom array*,
and the LAMMPS targets export `per_atom_energy`, `forces` and `virial` -- no `total_energy`.

So a LES model in LAMMPS reported the short-range energy while its forces were correct: right
trajectories, wrong thermodynamics, and no error message. It stayed hidden because neither
NVT nor NPT reads the potential energy -- NVT is driven by forces and the kinetic energy, NPT
by the virial. Only NVE energy conservation, or a direct comparison against ASE, exposes it.

`nequip_les` now spreads the long-range energy across the atoms of each structure so that the
per-atom array sums to the total. `total_energy`, `forces` and `stress` are bit-for-bit
unchanged -- the distribution happens *after* the total is formed -- so no trained model
changes its predictions, and an existing checkpoint only has to be **re-exported**, not
retrained. Set `distribute_lr_energy: false` in `les_args` to restore the old behaviour.

The even split is bookkeeping, not a physical decomposition: a per-atom share of an
electrostatic energy is not uniquely defined, and only the sum affects what LAMMPS reports.

## Latent charges are not in a deployed model

`nequip-compile` fixes what a compiled artefact returns, per target:

| target | exported outputs |
|---|---|
| `pair_nequip`, `pair_allegro` | per-atom energy, forces, virial |
| `ase`, `batch` | per-atom energy, total energy, forces, stress |

`LES_q` is not on either list, so a deployed model gives you energies and forces and nothing
else. `LES_BEC` is further out of reach: it is not even computed unless BEC is switched on.

This matters because LAMMPS otherwise looks like it can reach them. `pair_nequip_allegro`
provides a
[`compute`](https://github.com/mir-group/pair_nequip_allegro/tree/main/compute) that pulls an
arbitrary key out of the model's returned dictionary -- `compute q all nequip/atom LES_q 1 0`
reads exactly right. But the key has to be in the dictionary the *compiled* model returns,
and it is not.

So: get latent charges and BECs during **training or testing**, from the checkpoint, using
the callbacks in [Usage](usage.md#predicted-charges-and-becs). Lifting this would mean adding
the LES keys to the target output lists in nequip.

## `pair_allegro` and periodic models

The `pair_allegro` target's graph carries positions, edge indices and atom types -- **no
cell**. A periodic Ewald sum needs the cell for its reciprocal-space part, so a periodic LES
model cannot be exported to that target and is refused with an error naming the missing cell.

This guard exists because without it the cell was silently filled with zeros and the
long-range physics quietly became non-periodic
([issue #15](https://github.com/ChengUCB/NequIP-LES/issues/15)). For the way round it, see
[Periodic Allegro models](lammps.md#periodic-allegro-models).

Non-periodic models export to `pair_allegro` normally. Under `--mode torchscript` the export
also succeeds for periodic models, because TorchScript compiles the source without running
the branch that raises -- the guard is compiled into the artefact and fires when LAMMPS calls
it. `--target ase` always passes a cell and is unaffected.

## Accelerations

LES adds no kernels of its own, so the backbone's accelerations apply unchanged; refer to the
[NequIP](https://nequip.readthedocs.io/en/latest/guide/accelerations/openequivariance.html)
and [Allegro](https://nequip.readthedocs.io/projects/allegro/en/latest/guide/accelerations.html)
pages for what each one does and how to install it. What matters here is where each one can
be *used*:

| modifier | backbone | training | ASE / torch-sim | LAMMPS |
|---|---|---|---|---|
| `enable_OpenEquivariance` | NequIP | ✅ | ✅ | ❌ |
| `enable_CuEquivariance` | NequIP | ❌ | ✅ | ❌ |
| `enable_CuEquivarianceContracter` | Allegro | ✅ | ✅ | ❌ |
| `enable_TritonContracter` | Allegro | ❌ | ✅ | ✅ |

The training ❌ entries are the upstream packages' own limitation -- those two are
inference-only. The LAMMPS column is ours to explain: OpenEquivariance and cuEquivariance
register custom torch operators from Python, and the C++ `lmp` binary never imports those
packages, so loading the artefact fails with `Could not find schema for …`. **Every one of
these artefacts exports successfully**, which is why the tests check that they run rather
than only that they compile. Details in [LAMMPS](lammps.md#accelerations-in-lammps).

`nequip-compile` takes its positional arguments first, because `--modifiers` accepts a list
and would otherwise swallow them:

```bash
nequip-compile model.ckpt out.nequip.pt2 \
    --device cuda --mode aotinductor --target ase \
    --modifiers enable_OpenEquivariance
```

## torch versions

| torch | status |
|---|---|
| < 2.10 | `--mode torchscript` available |
| 2.9.1 | recommended for GPU `compile_mode: compile` |
| 2.12 | GPU train-time compilation broken (a `silu_backward` bug, unrelated to LES; fixed in 2.13) |
| 2.13 | `nequip-compile` fails on CUDA unless TF32 is worked around, below; with that in place the whole GPU suite passes |

### The TF32 workaround

On torch 2.13, `torch.export` reads the legacy aggregate cuDNN TF32 flag, which cannot
represent the "not TF32" state nequip sets -- so every `nequip-compile` on a CUDA device
raises, whatever the model. This is a torch bug, not a nequip or LES one.

[`tests/compile_tf32fix.py`](https://github.com/ChengUCB/NequIP-LES/blob/main/tests/compile_tf32fix.py)
makes that one read answer instead of raising and then calls `nequip-compile`'s entry point.
It takes the same arguments:

```bash
python tests/compile_tf32fix.py model.ckpt out.nequip.pt2 \
    --device cuda --mode aotinductor --target ase
```

Nothing about the model or about nequip's TF32 setting changes.

### Which CUDA build of torch, if you will build LAMMPS

Pick a torch wheel whose CUDA **major** matches the toolkit you compile with, and whose minor
is not newer than the system's -- the pair README's condition. Which wheels exist for a given
CUDA is worth checking before deciding, at
`https://download.pytorch.org/whl/cu126/torch/` and its siblings.
`build_lammps.sh check` refuses a mismatch rather than letting it fail an hour into the build.

## What has actually been tested

Every ✅ and ❌ above is a row in the [test suite](testing.md), except where the text names an
upstream limitation. The most recent runs:

* **GPU, torch 2.13 + CUDA 12.6** -- both backbones × both systems (periodic water, isolated
  dipeptide) × both training paths (eager, compiled), each trained, packaged and exported to
  every applicable target, plus one export per acceleration modifier and two training runs
  with a modifier applied. Clean.
* **Cross-engine agreement** -- the same model through ASE and LAMMPS `pair_nequip` agrees to
  `4e-10 eV/atom` and `1e-06 eV/Å`. Before the per-atom energy fix the same comparison read
  `2e-02 eV/atom` with forces already matching, which is how that bug was found.
* **CPU, torch 2.13** -- 20/20 training configs, 12/12 BEC runs, 48/48 exports, and the les
  Ewald suite unchanged at 16/16.

## Known gap

Exporting a **periodic** model with dynamic shapes *and* gradients of the latent charges
requested is not supported by AOTInductor at present. No deployment target asks for that
combination -- it is a stricter case than any real use -- but it is tracked in the les test
suite as a known gap rather than hidden.
