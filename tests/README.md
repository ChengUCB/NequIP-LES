# NequIP-LES tests

Smoke tests: a LES-augmented model **trains, infers Born effective charges, and exports
for deployment without erroring**, across every combination of backbone, periodicity and
LES code path. The models are tiny and train for two epochs, so they say nothing about
accuracy. Numerical correctness of the Ewald sum itself lives in the
[`les`](https://github.com/ChengUCB/les) repository.

## Running

```bash
cd tests
./run_all.sh          # train, 20 configs
./run_bec.sh          # BEC inference from each trained checkpoint, 12 configs
./run_compile.sh      # deployment export with nequip-compile
./run_gpu.sh          # CUDA only: GPU training, TF32, kernel accelerations, ML-IAP
```

Run them **one at a time** -- they share the working directory and clean it at the end.
Each takes an optional substring filter (`./run_all.sh nequip`), honours `KEEP_OUTPUTS=1`,
uses a GPU when CUDA is available, and exits non-zero on failure.

## The matrix

| axis | values |
|---|---|
| backbone | `nequip`, `allegro` |
| periodicity | `water` (periodic), `dipep` (non-periodic dipeptide) |
| LES path | `compiled`, `eager`, `legacy`, `sr_compiled`, `sr_eager` |

| LES path | `compile_mode` | `is_periodic` / `N_max` | what it exercises |
|---|---|---|---|
| `compiled` | `compile` | present | vectorized Ewald under `torch.compile` |
| `eager` | `eager` | present | the same vectorized Ewald, uncompiled |
| `legacy` | `eager` | **absent** | loop-based Ewald, which decides periodicity per structure |
| `sr_compiled`, `sr_eager` | either | -- | **no LES at all**, so a failure can be attributed |

20 configs, one yaml each in `configs/`, each a complete copy-able example. All LES configs
enable every long-range term (dipoles, quadrupoles, induced charges, induced dipoles,
anisotropic polarizability), following the maintained full-term (`uQiqiu`) configs in
[extended_les_fit](https://github.com/ChengUCB/extended_les_fit).

The `sr_*` rows use `nequip.model.NequIPGNNModel` / `allegro.model.AllegroModel` with the
same backbone, schedule and data as their LES siblings. They exist so that when something
breaks, one run says whether it is a LES problem or nequip's.

## What counts as a pass

* **`run_all.sh`** -- `nequip-train` exits 0, the logged metrics contain no NaN (a NaN loss
  raises nothing, so exiting 0 is not enough), and compilation really happened:
  `train_probed.py` reports `TRACED` for a `compiled` config and `NOT-TRACED` for an
  `eager` or `legacy` one. Checking the yaml is not enough -- `CompileGraphModel.forward`
  falls back to eager for batches with fewer than two frames, which is why the configs use
  `batch_size: 2`.
* **`run_bec.sh`** -- the test phase exits 0 and `predictions/<tag>_dataset0.xyz` contains
  an `LES_BEC` column. Values are not checked.
* **`run_compile.sh`** -- `nequip-compile` produces a non-empty artefact, for both training
  paths (an eagerly trained checkpoint must export as well as a compile-trained one), for
  ASE and the backbone's LAMMPS pair style. `legacy` is excluded: the loop-based Ewald
  cannot be traced. TorchScript is attempted only on torch < 2.10, where nequip still
  supports it.

One row is expected to be **refused**: a periodic LES model exported for `pair_allegro`,
which declares no cell, so the reciprocal-space sum cannot be evaluated
(ChengUCB/NequIP-LES#15 -- before this was caught, the cell was silently replaced by zeros
and the long-range physics quietly became non-periodic). The refusal is only visible for
`aotinductor`, which traces with example inputs so the branch that raises actually runs.
`torchscript` merely compiles the source, so the export succeeds and the guard fires when
LAMMPS calls the artefact without a cell.

## Data

| file | frames | atoms | notes |
|---|---|---|---|
| `water_train.xyz` | 10 | 192 | periodic bulk water, split 50/50 into train and val |
| `water_bec.xyz` | 5 | 192 | test set for BEC; carries reference BECs |
| `dipep_train.xyz` | 11 | 45 | non-periodic dipeptide, `pbc="F F F"`, split 50/50 |
| `dipep_test.xyz` | 3 | 45 | test set |
| `dipep_*_dummycell.xyz` | 11 / 3 | 45 | same frames with a 100 A cell, used by the `compiled` rows |

The dummy-cell copies exist because nequip computes `stress = virial / cell volume`, which
is non-finite for a zero cell -- harmless eagerly, but under train-time compilation every
gradient comes out NaN. `pbc` stays false, so the neighborlist and LES are unaffected. The
`legacy` rows keep the zero-cell files: the loop-based Ewald decides periodicity from the
cell, and a finite cell would send it down the reciprocal-space branch.

## Helpers

* `train_probed.py` -- `nequip-train` plus a report of whether a graph was really traced.
  Used by `run_all.sh` on every run.
* `wrap_modifier.py` -- generates a config whose model is wrapped in `nequip.model.modify`,
  for the acceleration rows. Only `run_gpu.sh` needs it.
