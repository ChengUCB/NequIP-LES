# What works: compilation, deployment, accelerations

The NequIP framework's compilation and deployment machinery is documented
[here](https://nequip.readthedocs.io/en/latest/guide/getting-started/workflow.html); this page
records how far a LES model gets along each of those paths. Everything below assumes the
vectorized Ewald (`is_periodic` set) -- with the legacy module, every compilation row is ❌.

Checked with nequip 0.19.0, allegro 0.8.3, PyTorch 2.13.0, Python 3.12, on CPU and on CUDA.

## Capabilities

| | LES model | notes |
|---|---|---|
| eager training (`compile_mode: eager`) | ✅ | works with legacy too |
| train-time compilation (`compile_mode: compile`) | ✅ | needs `is_periodic` |
| `nequip-package build` | ✅ | |
| `nequip-compile --mode aotinductor --target ase` | ✅ | periodic and non-periodic |
| `nequip-compile --mode aotinductor --target batch` | ✅ | for torch-sim |
| `nequip-compile --mode aotinductor --target pair_nequip` | ✅ | **the route for LAMMPS, either backbone** -- see [LAMMPS](lammps.md) |
| `nequip-compile --mode aotinductor --target pair_allegro` | ⚠️ | non-periodic only -- no cell in the graph |
| `nequip-compile --mode torchscript` | ✅ | torch < 2.10 only (nequip refuses above) |
| ASE calculator | ✅ | energies, forces, stress |
| torch-sim | ✅ | `pip install torch-sim-atomistic`, needs python >= 3.11 |
| LAMMPS, single MPI rank | ✅ | |
| LAMMPS, multiple MPI ranks | ❌ | an Ewald sum cannot be domain-decomposed -- [why](lammps.md#one-mpi-rank) |
| LAMMPS ML-IAP | ❌ | LES unsupported upstream -- [why](lammps.md#ml-iap) |
| BEC inference from an eager checkpoint | ✅ | |
| BEC inference from a compile-trained model | ❌ | output keys are fixed at trace time |
| CPU / CUDA | ✅ / ✅ | compilation works on both |

## The long-range energy reaches LAMMPS through `per_atom_energy`

The Ewald sum produces one number per structure. Until recently it was added to `total_energy`
only, and `per_atom_energy` carried the short-range part alone. ASE reads `total_energy` and was
therefore correct, but the LAMMPS pair styles have no global energy channel:
`pair_nequip_allegro.cpp` builds LAMMPS' `eng_vdwl` by *summing the per-atom array*, and the
LAMMPS targets export `per_atom_energy`, `forces` and `virial` -- no `total_energy`.

`nequip_les` now spreads the long-range energy across the atoms of each structure so that the
per-atom array sums to the total. `total_energy`, `forces` and `stress` are bit-for-bit
unchanged -- the distribution happens *after* the total is formed -- so no trained model changes
its predictions, and an existing checkpoint only has to be **re-exported**, not retrained. Set
`distribute_lr_energy: false` in `les_args` to restore the old behaviour.

The even split is bookkeeping, not a physical decomposition: a per-atom share of an
electrostatic energy is not uniquely defined, and only the sum affects what LAMMPS reports.

## Latent charges are not in a deployed model

`nequip-compile` fixes the outputs per target:

| target | exported outputs |
|---|---|
| `pair_nequip`, `pair_allegro` | per-atom energy, forces, virial |
| `ase`, `batch` | per-atom energy, total energy, forces, stress |

`LES_q` is on neither list, and `LES_BEC` is not even computed unless BEC is switched on. Get
them during **training or testing** from the checkpoint, with the callbacks in
[Usage](usage.md#predicted-charges-and-becs).

## `pair_allegro` and periodic models

That target's graph carries no cell, so a periodic Ewald sum cannot be evaluated and the export
is refused with an error naming the missing cell. Without the guard the cell was silently filled
with zeros and the long-range physics quietly became non-periodic
([issue #15](https://github.com/ChengUCB/NequIP-LES/issues/15)). Use
[`--target pair_nequip`](lammps.md#the-lammps-route-for-les-models) instead; `--target ase`
always passes a cell and is unaffected.

## Accelerations

LES adds no kernels of its own, so the backbone's accelerations apply unchanged:
[OpenEquivariance](https://nequip.readthedocs.io/en/latest/guide/accelerations/openequivariance.html),
[CuEquivariance (NequIP)](https://nequip.readthedocs.io/en/latest/guide/accelerations/cuequivariance.html),
[CuEquivariance (Allegro)](https://nequip.readthedocs.io/projects/allegro/en/latest/guide/cuequivariance.html),
[Triton (Allegro)](https://nequip.readthedocs.io/projects/allegro/en/latest/guide/triton.html).
Where each can be used, which those pages also state:

| modifier | backbone | training | ASE / torch-sim | LAMMPS pair style |
|---|---|---|---|---|
| `enable_OpenEquivariance` | NequIP | ✅ | ✅ | ❌ (ML-IAP only, which LES cannot use) |
| `enable_CuEquivariance` | NequIP | ❌ | ✅ | ❌ |
| `enable_CuEquivarianceContracter` | Allegro | ✅ | ✅ | ❌ |
| `enable_TritonContracter` | Allegro | ❌ | ✅ | ✅ |

All four export successfully; the ❌ in the LAMMPS column fail at load with
`Could not find schema for …`, since their operators are registered from Python.

`nequip-compile` takes its positional arguments first, because `--modifiers` accepts a list and
would otherwise swallow them:

```bash
nequip-compile model.ckpt out.nequip.pt2 \
    --device cuda --mode aotinductor --target ase \
    --modifiers enable_OpenEquivariance
```

If a compilation check fails on a tolerance, or `nequip-compile` fails on CUDA with torch 2.13,
see [Troubleshooting](troubleshooting.md).
