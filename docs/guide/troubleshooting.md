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

## A compilation check fails with a tolerance error

`nequip-compile` compares the compiled model against the eager one with an **absolute**
tolerance, and reports the largest entry of the prediction alongside the error:

```
Compilation check MaxAbsError: 0.0115945 (tol: 0.002) for field `forces` ...
the largest absolute (MaxAbs) entry of the model prediction is 47.1873
```

Divide the two. `0.0116 / 47.19 = 2.5e-4` is float32 round-off with TF32 on -- TF32 keeps about
ten mantissa bits, so a relative error near `1e-3` is what it is supposed to do. A genuine
miscompile shows up orders of magnitude out, not at the fourth digit.

The absolute tolerance bites hardest on undertrained models, whose forces can be tens of
eV/Å while the tolerance stays fixed. Raise it with `NEQUIP_FLOAT32_MODEL_TOL`,
`NEQUIP_FLOAT64_MODEL_TOL`, or `NEQUIP_TF32_MODEL_TOL` -- note that the TF32 variable is the one
in effect whenever TF32 is on, and the float32 one is then ignored.
