# Updates

Newest first. One line per change that affects how the package is used.

| date | change |
|---|---|
| 2026-08-06 | The long-range energy is now added to `per_atom_energy` as well, so LAMMPS reports the correct potential energy. Existing checkpoints only need re-exporting. New `les_args` option `distribute_lr_energy` (default `true`). |
| 2026-08-06 | Documented that a LES model in LAMMPS must use `--target pair_nequip` and a single MPI rank, whichever backbone. |
| 2026-07-31 | The extended LES with multipoles became `torch.jit.script`-friendly. |
