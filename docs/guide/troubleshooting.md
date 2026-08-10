# Troubleshooting

## `nequip-compile` fails on CUDA with torch 2.13

`torch.export` reads the legacy aggregate cuDNN TF32 flag, which cannot represent the
"not TF32" state nequip sets, so every `nequip-compile` on a CUDA device raises -- whatever the
model. This is a torch bug, not a nequip or LES one.

[`tests/compile_tf32fix.py`](https://github.com/ChengUCB/NequIP-LES/blob/main/tests/compile_tf32fix.py)
makes that one read answer instead of raising, then calls `nequip-compile`'s entry point. It
takes the same arguments:

```bash
python tests/compile_tf32fix.py model.ckpt out.nequip.pt2 \
    --device cuda --mode aotinductor --target ase
```

Nothing about the model or about nequip's TF32 setting changes.

## Which CUDA build of torch, if you will build LAMMPS

Pick a torch wheel whose CUDA **major** matches the toolkit you compile with, and whose minor
is not newer than the system's -- the condition in the
[pair style README](https://github.com/mir-group/pair_nequip_allegro#cuda). Two CUDA majors in
one binary do not work: libtorch brings its own runtime and Kokkos compiles against the
toolkit's.

Check which wheels exist for a given CUDA before deciding, at
`https://download.pytorch.org/whl/cu126/torch/` and its siblings -- not every torch version is
built for every CUDA. `tests/build_lammps.sh check` refuses a mismatch rather than letting it
fail an hour into the build.
