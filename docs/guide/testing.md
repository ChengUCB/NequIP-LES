# Tests

Everything lives in [`tests/`](https://github.com/ChengUCB/NequIP-LES/tree/main/tests) and
runs from there. Each script trains small models on a handful of frames -- these are
integration tests, not accuracy benchmarks: they answer "does this path work at all",
in a few minutes on a laptop.

```bash
cd tests
pytest test_lr_energy_per_atom.py   # unit test of the per-atom long-range energy
./run_all.sh                        # training
./run_bec.sh                        # BEC inference
./run_compile.sh                    # deployment export
./run_gpu.sh                        # GPU-only: accelerations
python run_inference_matrix.py      # do the artefacts agree with each other?
```

The Ewald sum itself is tested in the [les](https://github.com/ChengUCB/les) repository
(`test/run_all.py`), where the vectorized implementation is compared against the legacy one
multipole by multipole. This suite tests the *integration* -- that NequIP and Allegro models
carrying LES train, compile and export.

## What each script checks

**`run_all.sh` -- training.** 20 configs: both backbones (NequIP, Allegro) × both systems
(periodic water, non-periodic dipeptide) × three paths (`compiled`, `eager`, `legacy`), plus 8
short-range configs with no LES at all. It fails on a NaN anywhere in `metrics.csv`, and it
verifies that compilation *actually happened*: `*_compiled` rows must report `TRACED` and
everything else `NOT-TRACED`. That check exists because a config can silently fall back to
eager -- NequIP short-circuits to eager below two frames -- and then a green run proves
nothing. The no-LES rows are there for attribution: when a row fails, they say whether LES
is involved.

**`run_bec.sh` -- Born effective charges.** 12 runs that load a trained checkpoint with
`ModelFromCheckpoint`, turn BEC on, and write predictions. Checks the extra derivative
survives every multipole combination.

**`run_compile.sh` -- deployment.** Trains 12 models and exports each to every applicable
target and mode, plus `nequip-package`. Two rows are expected to *fail*: periodic LES +
`pair_allegro`, which must be rejected for the reason in
[the capabilities page](deployment.md). A row that should have been rejected and was not is
a failure, same as a broken export.

**`run_gpu.sh` -- GPU only.** The same exports with each acceleration modifier, split into
train-time and inference-time lists because some modifiers are inference-only, plus packaging.
`ROWS=...` filters to a subset, which is useful when only the accelerations changed.

**`run_inference_matrix.py` -- do the artefacts agree?** The scripts above prove each export
*succeeds*. They do not prove the exports agree with each other, and a wrong unit, a wrong type
mapping, or an energy term that never reached the per-atom array all produce a perfectly valid
artefact that returns different numbers. This loads every artefact `run_gpu.sh` left behind
through the interface it was built for and differences them against ASE, per model. It is what
caught the long-range energy missing from LAMMPS' potential energy.

**`check_consistency.py` -- the same check for one model of your own.** Exports a checkpoint
fresh to every target and compares the engines:

```bash
python check_consistency.py path/to/model.ckpt data/water_train.xyz
```

`export LMP=…` brings LAMMPS into the comparison; without it the LAMMPS rows are skipped and
the rest still runs.

**`test_nequip.sh` / `test_allegro.sh` -- one model, every path.** Given a checkpoint, walks
through the deployment steps in order -- package, ASE, LAMMPS pair style, batch,
export-from-package, accelerations, ASE inference, torch-sim -- printing the exact command
before each. Use these when you want to see what a real deployment of *your*
model does, rather than of the test models:

```bash
./test_nequip.sh path/to/model.ckpt [outdir]
```

Steps needing something you do not have installed are skipped with a reason.

## Configs and data

[`tests/configs/`](https://github.com/ChengUCB/NequIP-LES/tree/main/tests/configs) holds 22
complete configs; they double as copy-able examples of every combination.
[`tests/data/`](https://github.com/ChengUCB/NequIP-LES/tree/main/tests/data) holds the frames,
including the `_dummycell` variants explained in
[Non-periodic models need a dummy cell](ewald.md#non-periodic-models-need-a-dummy-cell).

```{note}
Two things are deliberately **not** exercised. TF32 exports were dropped: they never failed and
added nothing over the plain export. LAMMPS ML-IAP is a known gap -- the wrapper passes the
model no positions and no cell, so LES cannot run there at all
([why](lammps.md#ml-iap)) -- and the scripts record it as such rather than attempting it.
```

Two helpers do the work the checks above rely on: `train_probed.py` wraps NequIP's tracing
entry point to report `TRACED` / `NOT-TRACED`, and `wrap_modifier.py` writes a config with
the model wrapped in `nequip.model.modify` so a modifier can be applied at training time.
[`tests/README.md`](https://github.com/ChengUCB/NequIP-LES/blob/main/tests/README.md) has the
per-script details.
