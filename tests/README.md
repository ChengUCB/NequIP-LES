# NequIP-LES smoke tests

End-to-end checks that a LES-augmented model **trains and infers Born effective
charges without erroring**, across every combination of backbone, periodicity and
LES code path. These are smoke tests: the models are tiny and train for two
epochs, so they say nothing about accuracy. Numerical correctness of the Ewald
sum itself lives in the [`les`](https://github.com/ChengUCB/les) repository
(`test/test_ewald_vectorized_physics.py`, `test_ewald_vectorized_compile.py`).

## Running

```bash
cd tests
./run_all.sh                  # all 12 configurations
./run_all.sh nequip           # only configs whose name contains "nequip"
./run_all.sh water_compiled   # ... or any other substring
KEEP_OUTPUTS=1 ./run_all.sh   # keep predictions/ and outputs/ for inspection
```

It uses a GPU when CUDA is available and the CPU otherwise, and exits non-zero if
any run fails. `predictions/` and `outputs/` are created at the start and removed
at the end (logs of failing runs are kept).

## The matrix

| axis | values |
|---|---|
| backbone | `nequip`, `allegro` |
| periodicity | `water` (periodic, `pbc="T T T"`), `dipep` (non-periodic dipeptide) |
| LES path | `compiled`, `eager`, `legacy` |

12 configurations, one yaml each in `configs/`. The three LES paths are:

| path | `compile_mode` | `is_periodic` / `N_max` | what it exercises |
|---|---|---|---|
| `compiled` | `compile` | present | the vectorized Ewald under `torch.compile` — the deployment path |
| `eager` | `eager` | present | the same vectorized Ewald, uncompiled |
| `legacy` | `eager` | **absent** | the loop-based Ewald, which decides periodicity per structure and cannot be compiled |

The settings are written out in each yaml rather than passed as command-line
overrides, so every config is a complete example you can copy.

Each config turns on **all** the long-range terms — dipoles, quadrupoles,
induced charges, induced dipoles, anisotropic polarizability — following the
maintained full-term (`uQiqiu`) configs in
[extended_les_fit](https://github.com/ChengUCB/extended_les_fit). Testing at the
maximum means a failure anywhere shows up here.

## What counts as a pass

Both of:

1. `nequip-train` exits 0 for `run: [train, test]`, and
2. the test phase writes `LES_BEC` into `predictions/<tag>_dataset0.xyz`.

`ToggleLESCallback` enables BEC only at test time, so one `run: [train, test]`
covers training and BEC inference together. Requiring the BEC column means a run
that trains but silently stops producing BECs is a failure, not a pass.

## Data

Small excerpts in `data/`, enough to run in seconds:

| file | frames | atoms | notes |
|---|---|---|---|
| `water_train.xyz` | 10 | 192 | periodic bulk water |
| `water_bec.xyz` | 5 | 192 | used as the test set; carries reference BECs |
| `dipep_train.xyz` | 2 | 45 | non-periodic dipeptide, `pbc="F F F"` |
| `dipep_test.xyz` | 2 | 45 | test set |

## Note on periodicity and the cell

The vectorized Ewald applies one periodicity to the whole batch, so `is_periodic`
must agree with the data. Passing a real cell with `is_periodic: false` would
silently drop the reciprocal-space sum, and both `les` and this package now raise
instead:

* `les` checks the cell against `is_periodic` eagerly.
* this package raises if `is_periodic: true` but the input graph has no cell at
  all, which is what happens with deployment targets that do not pass one
  (LAMMPS `pair_allegro` declares only positions, edge indices and atom types).

For genuinely mixed periodic/non-periodic datasets, omit `is_periodic` and use
the legacy Ewald, which decides per structure.
