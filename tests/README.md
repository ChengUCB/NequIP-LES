# NequIP-LES tests

Smoke tests: a LES model **trains, infers BECs, exports and runs** across every combination
of backbone, periodicity and LES code path. The models are tiny and train for two epochs, so
they say nothing about accuracy. Correctness of the Ewald sum itself is tested in the
[`les`](https://github.com/ChengUCB/les) repository.

## Running

Run them one at a time -- they share the working directory and clean it at the end. Each
takes an optional substring filter, honours `KEEP_OUTPUTS=1`, and exits non-zero on failure.

```bash
cd tests
pytest test_lr_energy_per_atom.py   # unit test: long-range energy in per_atom_energy
./run_all.sh                        # training, 20 configs
./run_bec.sh                        # BEC inference, 12 configs
./run_compile.sh                    # deployment export
./run_gpu.sh                        # CUDA only: GPU training, packaging, accelerations

KEEP_OUTPUTS=1 ./run_gpu.sh         # then:
python run_inference_matrix.py      # do all those artefacts agree with each other?
```

`check_consistency.py <model.ckpt> <structure.xyz>` does the same comparison for a model of
your own, exporting it fresh. `export LMP=…` brings LAMMPS into either comparison;
`build_lammps.sh` builds it (see the [LAMMPS docs page](../docs/guide/lammps.md)).

## The matrix

| axis | values |
|---|---|
| backbone | `nequip`, `allegro` |
| periodicity | `water` (periodic), `dipep` (non-periodic dipeptide) |
| LES path | `compiled`, `eager`, `legacy`, `sr_compiled`, `sr_eager` |

| LES path | `compile_mode` | `is_periodic` | what it exercises |
|---|---|---|---|
| `compiled` | `compile` | present | vectorized Ewald under `torch.compile` |
| `eager` | `eager` | present | the same vectorized Ewald, uncompiled |
| `legacy` | `eager` | **absent** | loop-based Ewald, periodicity decided per structure |
| `sr_*` | either | -- | **no LES at all**, so a failure can be attributed |

One yaml each in `configs/`, all complete and copy-able. The LES ones enable every long-range
term (`uQiqiu`), following the maintained configs in
[extended_les_fit](https://github.com/ChengUCB/extended_les_fit).

## What counts as a pass

* **`run_all.sh`** -- exits 0, no NaN in the metrics, and compilation really happened:
  `train_probed.py` must report `TRACED` for `compiled` and `NOT-TRACED` otherwise. Reading
  the yaml is not enough, because `CompileGraphModel.forward` falls back to eager below two
  frames -- hence `batch_size: 2`.
* **`run_bec.sh`** -- the test phase exits 0 and the predictions carry an `LES_BEC` column.
  Values are not checked.
* **`run_compile.sh`** -- a non-empty artefact for every applicable target, from both an
  eagerly trained and a compile-trained checkpoint. `legacy` is excluded: the loop-based
  Ewald cannot be traced. TorchScript only on torch < 2.10.
* **`run_inference_matrix.py`** -- every artefact agrees with ASE on energy and forces.
  Accelerated artefacts are held to a looser tolerance since they change the arithmetic.

One row is expected to be **refused**: a periodic LES model exported for `pair_allegro`,
which declares no cell (see the
[deployment page](../docs/guide/deployment.md#pair_allegro-and-periodic-models)).

Two things are deliberately not tested: TF32 exports, which never failed and added nothing,
and LAMMPS ML-IAP, which cannot run LES at all
([why](../docs/guide/lammps.md#ml-iap)).

## Data

| file | frames | atoms | notes |
|---|---|---|---|
| `water_train.xyz` | 10 | 192 | periodic bulk water, split 50/50 |
| `water_bec.xyz` | 5 | 192 | BEC test set, carries reference BECs |
| `dipep_train.xyz` | 11 | 45 | non-periodic dipeptide, `pbc="F F F"` |
| `dipep_test.xyz` | 3 | 45 | test set |
| `dipep_*_dummycell.xyz` | 11 / 3 | 45 | same frames with a 100 Å cell, for the `compiled` rows |

The dummy-cell copies exist because nequip computes `stress = virial / volume`, which is
non-finite for a zero cell: harmless eagerly, but every gradient comes out NaN under
train-time compilation. `pbc` stays false. The `legacy` rows keep the zero-cell files, since
the loop-based Ewald reads the cell to decide periodicity.

## Helpers

* `engines.py` -- runs one structure through ASE, torch-sim or LAMMPS; shared by the two
  comparison scripts.
* `train_probed.py` -- `nequip-train` plus a report of whether a graph was really traced.
* `wrap_modifier.py` -- writes a config wrapped in `nequip.model.modify`, for the
  acceleration rows.
* `compile_tf32fix.py` -- `nequip-compile` with a torch 2.13 TF32 workaround
  ([why](../docs/guide/troubleshooting.md)).
