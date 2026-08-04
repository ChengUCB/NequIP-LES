# NequIP-LES smoke tests

End-to-end checks that a LES-augmented model **trains, exports and infers Born
effective charges without erroring**, across every combination of backbone, periodicity and
LES code path. These are smoke tests: the models are tiny and train for two
epochs, so they say nothing about accuracy. Numerical correctness of the Ewald
sum itself lives in the [`les`](https://github.com/ChengUCB/les) repository
(`test/test_ewald_vectorized_physics.py`, `test_ewald_vectorized_compile.py`).

## Running

```bash
cd tests
./run_all.sh          # train all 12 configurations
./run_bec.sh          # BEC inference from each trained checkpoint
./run_compile.sh      # export for deployment with nequip-compile
./run_gpu.sh          # CUDA-only: GPU training, TF32, kernel accelerations
```

Each takes an optional substring filter (`./run_all.sh nequip`,
`./run_bec.sh water_compiled`) and honours `KEEP_OUTPUTS=1` to leave the generated
files behind for inspection. All of them use a GPU when CUDA is available and the
CPU otherwise, and exit non-zero if any run fails.

Training and BEC are **separate scripts on purpose**: when they were one
`run: [train, test]` pass, a BEC failure and a training failure looked the same.
`run_bec.sh` trains a model itself if no checkpoint is there yet, so it can also be
run on its own.

## The matrix

| axis | values |
|---|---|
| backbone | `nequip`, `allegro` |
| periodicity | `water` (periodic, `pbc="T T T"`), `dipep` (non-periodic dipeptide) |
| LES path | `compiled`, `eager`, `legacy`, and `sr_compiled` / `sr_eager` (no LES) |

20 configurations, one yaml each in `configs/`. The three LES paths are:

| path | `compile_mode` | `is_periodic` / `N_max` | what it exercises |
|---|---|---|---|
| `compiled` | `compile` | present | the vectorized Ewald under `torch.compile` — the deployment path |
| `eager` | `eager` | present | the same vectorized Ewald, uncompiled |
| `legacy` | `eager` | **absent** | the loop-based Ewald, which decides periodicity per structure and cannot be compiled |

The settings are written out in each yaml rather than passed as command-line
overrides, so every config is a complete example you can copy.

### The `sr_*` rows carry no LES, and that is the point

`{backbone}_{data}_sr_{compiled,eager}` use `nequip.model.NequIPGNNModel` /
`allegro.model.AllegroModel` with the same backbone, schedule and data as their LES
siblings, and no LES at all. They run every time so that a failure is attributable
without a separate investigation. It has already paid for itself twice:

* `derivative for aten::silu_backward is not implemented` under train-time compile
  on CUDA — reproduces without LES,
* non-finite gradients on the first optimizer step for **non-periodic** data under
  train-time compile — reproduces without LES, on both backbones. On torch 2.12 the
  four failing rows are exactly `{nequip,allegro}_dipep_{compiled,sr_compiled}`,
  the LES and non-LES pair failing together.

When only the LES row fails and its `sr_` partner passes, then it is ours.

Each config turns on **all** the long-range terms — dipoles, quadrupoles,
induced charges, induced dipoles, anisotropic polarizability — following the
maintained full-term (`uQiqiu`) configs in
[extended_les_fit](https://github.com/ChengUCB/extended_les_fit). Testing at the
maximum means a failure anywhere shows up here.

## What counts as a pass

`run_all.sh`: `nequip-train` exits 0 for `run: [train]`, **and** the logged metrics
contain no NaN. Exiting 0 is not enough — a NaN loss raises nothing, and the
non-periodic compile failure above trained "successfully" while every gradient was
NaN; only the export caught it. The suite now reads `metrics.csv`.

`run_bec.sh`: the test phase exits 0 **and** `predictions/<tag>_dataset0.xyz`
contains an `LES_BEC` column. Requiring the column means a run that completes but
silently stops producing BECs is a failure, not a pass. The values themselves are
not checked — these are two-epoch models.

BEC inference uses `configs/bec_water.yaml` / `configs/bec_dipep.yaml`, which load
the trained model through `nequip.model.ModelFromCheckpoint` and run `run: [test]`
with `ToggleLESCallback(compute_bec: true)`. Only the checkpoint path and the output
filename are passed on the command line, since both are run-specific artefacts.
They need hydra quoting because lightning's default checkpoint names contain `=`:

```bash
nequip-train -cn bec_water --config-dir configs \
    "++training_module.model.checkpoint_path='ckpt/<tag>/.../epoch=1-step=12.ckpt'"
```

## Data

Small excerpts in `data/`, enough to run in seconds:

| file | frames | atoms | notes |
|---|---|---|---|
| `water_train.xyz` | 10 | 192 | periodic bulk water, split 50/50 into train and val |
| `water_bec.xyz` | 5 | 192 | used as the test set; carries reference BECs |
| `dipep_train.xyz` | 11 | 45 | non-periodic dipeptide, `pbc="F F F"` + 100 A dummy cell (see below), split 50/50 |
| `dipep_test.xyz` | 3 | 45 | test set, same dummy cell |

## Why `batch_size` must be at least 2

`nequip.nn.compile.CompileGraphModel.forward` falls back to the eager model for
any batch with fewer than two frames (or nodes, or edges) — a workaround for
PyTorch's 0/1 shape specialization. With `batch_size: 1` the `compiled` configs
therefore train **without ever compiling**, and pass while a real run crashes:
that is how a periodicity check that could not be traced by nequip's `make_fx`
survived this suite and only failed on a cluster. `run_all.sh` now refuses to run
a `compiled` config whose dataloaders use `batch_size: 1`.

## Two different things called "compilation"

| | what it is | when | checked by |
|---|---|---|---|
| `compile_mode: compile` | compiles the model for the training loop | during training | `run_all.sh` |
| `nequip-compile` | produces the artefact LAMMPS / ASE load (AOTInductor, TorchScript) | after training | `run_compile.sh`, `run_gpu.sh` |

`run_all.sh` does not take the config's word for it. `train_probed.py` wraps
`nequip.utils.fx._nequip_make_fx` — the one function that only runs when a graph is
really traced — and prints `TRACED n_calls=... nodes=...` or `NOT-TRACED`. A
`compiled` config that does not trace is a failure, and so is an `eager` or
`legacy` config that does. Checking the yaml is not enough: see the `batch_size`
section below for how a "compiled" run once trained without ever compiling.

Deployment export is checked for **both** training paths — an eagerly trained
checkpoint must export exactly as well as a compile-trained one. `legacy` is
excluded on purpose: the loop-based Ewald cannot be traced at all, which is why the
vectorized one exists.

## GPU-only checks (`run_gpu.sh`)

Needs CUDA, and each kernel library must be installed; whatever is missing is
skipped rather than failed. The modifier names differ per backbone:

| backbone | modifiers |
|---|---|
| nequip | `enable_OpenEquivariance`, `enable_CuEquivariance` |
| allegro | `enable_TritonContracter`, `enable_CuEquivarianceContracter` |

What it covers, for `{nequip,allegro} x {eager,compiled}`:

* training on CUDA
* export for ASE, for the backbone's LAMMPS pair style, and for **LAMMPS ML-IAP**
* the same export with `--tf32`, since TF32 changes float32 arithmetic and the
  Ewald k-space sum is sensitive to it
* the same export with each available kernel modifier (`nequip-compile --modifiers`)
* **training** with each kernel modifier, which is a different code path from
  exporting an already-trained checkpoint: the modifier nests the model under
  `nequip.model.modify`, which no command-line override can express, so
  `wrap_modifier.py` generates a wrapped copy of the same config

## Non-periodic data needs a finite dummy cell

A non-periodic frame written with a **zero** cell trains fine eagerly but produces
**non-finite gradients on the very first optimizer step** under
`compile_mode: compile`. Training raises nothing, so it looks like a successful run.

The cause is in nequip, not here, and it is worth understanding because it will bite
anyone who trains on molecules:

`ForceStressOutput` computes `stress = virial / volume` (`nequip/nn/grad_output.py`),
and its own comment states the expectation:

```
# NOTE: to support batching periodic and non-periodic structures together,
# the data processing stage is responsible for ensuring that:
# 1. non-periodic systems have a finite dummy cell to prevent infs in the division below
```

With a zero cell the volume is zero and the stress is non-finite. Eagerly that is
harmless — nothing consumes it. But since nequip commit `c70a36e6` ("ensure that
models compiled for training produce full set of eager output dict entries") the
compiled graph returns *every* key the eager model produces, `stress` included, so
the non-finite value enters the joint graph AOTAutograd builds and poisons every
parameter gradient.

Established by experiment:

| varied | result |
|---|---|
| LES on / off (`sr_*`) | both fail, together |
| backbone nequip / allegro | both fail |
| torch 2.12.0 / 2.9.1, CPU / CUDA | all fail |
| two different dipeptide datasets | both fail |
| learning rate 5e-3 / 1e-6 | both fail — not divergence |
| nequip v0.16.0 (before `c70a36e6`) | passes |
| `compile_mode: eager`, any version | passes |
| dropping `stress` from the compiled outputs | passes |
| **a finite dummy cell, 1 / 10 / 100 A, pbc still False** | **passes** |

So `data/dipep_*.xyz` carry a 100 A dummy cell with `pbc="F F F"`. The physics is
unchanged: with `pbc` false the neighborlist ignores the cell (the mean neighbour
count is identical to the zero-cell files), and LES with `is_periodic: false` never
looks at it. Only the stress denominator changes.

If you bring your own non-periodic data, do the same:

```python
from ase.io import read, write
import numpy as np
frames = read("mydata.xyz", index=":")
for a in frames:
    a.set_cell(np.eye(3) * 100.0)   # finite, any size
    a.set_pbc(False)                # still non-periodic
write("mydata.xyz", frames)
```

`nequip.data.transforms.NonPeriodicCellTransform` looks like it should do this from
the config, but in our runs it did not prevent the NaN; setting the cell in the data
did. That discrepancy is not yet explained.

## Train-time compilation needs torch 2.9.x

`compile_mode: compile` on a **GPU** fails on torch 2.12 with

```
RuntimeError: derivative for aten::silu_backward is not implemented
```

and works on **torch 2.9.1**. It is not LES-specific — a plain NequIP model fails
identically — and the CPU is fine either way. The cause is that the
`aten.silu_backward.default` entry of the decomposition table nequip passes to
`make_fx` is not applied on CUDA under 2.12, so the raw op survives into the graph
and AOTAutograd then has to differentiate it. nequip's own comment in
`nequip/utils/fx.py` places this error in the PT 2.9.1 -> 2.10.0 window.

## Note on periodicity and the cell

The vectorized Ewald applies one periodicity to the whole batch, so `is_periodic`
must agree with the data: passing a real cell with `is_periodic: false` silently
drops the reciprocal-space sum. **This is on the user to get right** — `les` no
longer checks it. The check had to read a tensor value, which is exactly what
`make_fx` cannot do, and raising inside nequip's train-time-compile trace left the
tracer in a state where later ops stopped being decomposed (surfacing as
`derivative for aten::silu_backward is not implemented`).

What this package still checks is structural, resolved at trace time and therefore
compile-safe: it raises if `is_periodic: true` but the input graph carries no cell
at all, which is what happens with deployment targets that do not pass one
(LAMMPS `pair_allegro` declares only positions, edge indices and atom types).

For genuinely mixed periodic/non-periodic datasets, omit `is_periodic` and use
the legacy Ewald, which decides per structure.
