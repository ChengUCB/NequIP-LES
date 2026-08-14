# MD timing benchmark

How much does LES cost at run time, and how do the deployment paths compare? Everything runs
from this one folder.

## Running it

```bash
cd timing
cp /path/to/*.ckpt models/                  # 6 checkpoints, named below
export LMP=/path/to/lammps-pair/build/lmp   # for the LAMMPS rows
unset LMP_LAUNCHER                          # inside an srun step, do not nest srun

# check the plumbing first, then throw the dry-run rows away
SIZES="1 1 1" REPEATS=1 WARMUP=5 STEPS=10 ./run_timing.sh && rm timing.csv

python prepare.py --policies fixed          # base cell + 12 artefacts   (once)
POLICIES=fixed ./run_timing.sh              # 120 points, the main sweep
python plot.py
```

That is the whole benchmark: 6 checkpoints x 2 engines x 10 sizes. The fixed-accuracy k-grid
question is separate and optional, and adds 32 exports:

```bash
python prepare.py --policies scaled
POLICIES=scaled ./run_timing.sh
```

Both are additive and **resumable** -- anything already exported or already in `timing.csv` is
skipped -- so the sweep can be chunked across jobs and restarted after an OOM.

Checkpoints must be named `<backbone>_<variant>.ckpt`:

```
models/nequip_sr.ckpt    models/nequip_les.ckpt    models/nequip_les_u.ckpt
models/allegro_sr.ckpt   models/allegro_les.ckpt   models/allegro_les_u.ckpt
```

`variant` is `sr` (no LES), `les` (latent charges), `les_u` (charges + dipoles). The driver
reads the backbone from the name to decide which engines apply.

Filters for slices, accepted by `run_timing.sh` and (as `--models` / `--engines`) by `prepare.py`:

```bash
MODELS="nequip_les" ENGINES="lammps" POLICIES="fixed" SIZES="2 2 2;3 3 3" ./run_timing.sh
```

`STEPS`, `WARMUP` and `REPEATS` are overridable the same way, for dry runs only -- the defaults
(100 / 100 / 3) are what a reported sweep should use, and rows from different settings are not
comparable. `timing.csv` records `steps` per row; it does not record warmup or repeat count, so
delete dry-run rows before the real sweep rather than mixing them.

Run one process at a time. Two MD jobs on one GPU invalidate every number in the file.

## What is measured

NVE, 1 fs, identical initial velocities from a fixed seed, on one structure — frame 0 of
`../tests/data/water_train.xyz` relaxed once with the NequIP SR model and stored as `water.xyz`.
Each point is **100 warm-up steps, then 3 × 100 timed steps**; the median is reported and every
repeat is kept in the CSV.

Inside the clock: the MD steps. Outside: model load, the AOTInductor first call, structure
construction, cuBLAS autotuning — all absorbed by the warm-up. `torch.cuda.synchronize()`
brackets the timed segment, without which the wall clock straddles kernels still queued on the
device. For LAMMPS we read its own `Loop time` from a second `run`, which already excludes setup
and the first neighbour build.

TF32 is off everywhere, and the ASE neighbourlist backend is fixed (`matscipy` by default,
`TIMING_NL_BACKEND` to change it) and recorded in the CSV.

## Sizes

192-atom base cell, ten steps, stopping near 10,000 atoms:

| `nx ny nz` | atoms | | `nx ny nz` | atoms |
|---|---|---|---|---|
| 1 1 1 | 192 | | 3 2 2 | 2304 |
| 2 1 1 | 384 | | 3 3 2 | 3456 |
| 2 2 1 | 768 | | 4 3 2 | 4608 |
| 3 2 1 | 1152 | | 4 3 3 | 6912 |
| 2 2 2 | 1536 | | 4 4 3 | 9216 |

A model that runs out of memory records `status=oom` and stops its own ladder; the rest continue.

## Engines

Five per backbone: the uncompiled baseline, ASE with every accelerator kernel that works there,
and LAMMPS plain.

| engine | backbone | how the model is loaded |
|---|---|---|
| `ase-eager` | both | the checkpoint, eager — the uncompiled baseline |
| `ase-aoti` | both | `--target ase`, driven by `ase.md.verlet` |
| `ase-aoti-oeq` | NequIP | + `enable_OpenEquivariance` |
| `ase-aoti-cueq` | both | + `enable_CuEquivariance` / `enable_CuEquivarianceContracter` |
| `ase-aoti-triton` | Allegro | + `enable_TritonContracter` |
| `lammps` | both | `--target pair_nequip`, `pair_style nequip`, one MPI rank |

`ase-eager` needs no export, so it costs nothing to prepare — but it is the slowest row by a
wide margin and the most likely to run out of memory at the top of the ladder. That is itself
the answer to "how much does compiling buy".

The accelerator rows need their package importable — `openequivariance`,
`cuequivariance_torch`, `triton` — at export **and** at run time. A missing one shows up as a
failed export in `prepare.py` and `status=error` for that engine; nothing else is affected.

`--target pair_nequip` is used for **both** backbones: it is the only LAMMPS target that passes a
cell, which a periodic Ewald sum needs.

Everything else is deliberately absent, and `common.py` lists each with its reason. The two that
most often come up:

* **torch-sim** is not installed in the target environment.
* **Accelerator kernels** answer a different question — which kernel backend is fastest, not what
  LES costs — and are unavailable where they would matter. Under a LAMMPS pair style,
  OpenEquivariance and cuEquivariance artefacts export but fail to load with
  `Could not find schema for …`, since their operators are registered from Python.
  OpenEquivariance *does* have a LAMMPS route — ML-IAP — but that one passes the model no
  positions and no cell, so it runs SR models only and can never produce the LES rows those SR
  rows would be compared against. The one modifier that works under a pair style,
  `enable_TritonContracter`, is Allegro-only.

## Reading the numbers

**The overhead number is the trustworthy one.** SR → LES → LES+dipole on the same engine, same
structure, same neighbour-list cost, is apples to apples: the difference is the Ewald sum and
nothing else. That is `les_overhead.png`.

**ASE vs LAMMPS answers a different question.** The ASE per-step time includes a neighbour-list
rebuild on the CPU every step; LAMMPS' does not, because LAMMPS builds its own list internally.
So a LAMMPS row being faster is partly the engine and partly that bookkeeping. Read those as
"what a user experiences in this engine", not as a measurement of the model.

## The two `N_max` policies

`N_max` bounds the integer k-grid. The physical cutoff is set by `dl` (2 Å here, giving
`|k| ≤ π`); `N_max` only has to be large enough to reach it, which needs `N_max ≥ L/dl`. The
12.4 Å training cell needs 6.2, so training at `N_max = 10` was fully converged — but replicating
to 24.8 Å and beyond silently truncates the sphere.

| policy | `N_max` | k-vectors | what it shows |
|---|---|---|---|
| `fixed` | 10, as trained | 9,261 at every size | LES cost is O(N) and the overhead **fraction falls** as N grows. What you get replicating a cell without touching `les_args`. |
| `scaled` | 7 / 13 / 19 / 25 by size | 3,375 → 132,651 | the accuracy the training cell had, at every size. K grows with the cell, so the reciprocal sum is O(N²) and the overhead **grows**. |

Both are in the CSV under `nmax_policy`, and `nmax_policy.png` overlays them. The `scaled` policy
runs on `ase-aoti` and `lammps` only — the k-grid cost does not depend on the engine — and skips SR
entirely, which has no Ewald sum for `N_max` to affect.

Changing `N_max` is safe: the grid it builds is registered `persistent=False`, so it is not in the
state dict and not learned. `prepare.py` writes patched checkpoints into `models/_derived/` and
exports artefacts from them. Expect `scaled` to run out of memory at the top sizes — the
implementation materialises the full grid, so an `[N, K]` tensor at 9,216 atoms and 132,651
k-vectors is about 5 GB and there are several. That limit is itself a result.

## Files

| | |
|---|---|
| `common.py` | the sizes, engines and CSV schema, in one place so nothing can disagree |
| `prepare.py` | relax the base cell, patch `N_max`, export every artefact |
| `time_md.py` | ASE and torch-sim timing for one (model, engine) |
| `time_lammps.py` | LAMMPS timing for one model |
| `run_timing.sh` | the driver, resumable |
| `plot.py` | the three figures |
| `timing.csv` | one row per repeat, `status ∈ ok / oom / error` |

The LAMMPS input-script construction, data-file writing, MPI environment handling and
`Loop time` parsing live in `../tests/engines.py`, shared with the correctness tests.
