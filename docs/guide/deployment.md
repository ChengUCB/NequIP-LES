# What works: compilation, deployment, accelerations

The NequIP framework's compilation and deployment machinery is documented
[here](https://nequip.readthedocs.io/en/latest/guide/getting-started/workflow.html); this
page only records how far a LES model gets along each of those paths, and why it stops where
it does. Everything below assumes the vectorized Ewald (`is_periodic` set) -- with the legacy
module, every row in the compilation table is ❌.

## Capabilities

| | LES model | notes |
|---|---|---|
| eager training (`compile_mode: eager`) | ✅ | works with legacy too |
| train-time compilation (`compile_mode: compile`) | ✅ | needs `is_periodic`; see torch versions below |
| `nequip-package build` | ✅ | |
| `nequip-compile --mode aotinductor --target ase` | ✅ | periodic and non-periodic |
| `nequip-compile --mode aotinductor --target batch` | ✅ | |
| `nequip-compile --mode aotinductor --target pair_nequip` | ✅ | NequIP backbone |
| `nequip-compile --mode aotinductor --target pair_allegro` | ⚠️ | non-periodic only -- see below |
| `nequip-compile --mode torchscript` | ⚠️ | torch < 2.10 only (nequip refuses above) |
| BEC inference from a compile-trained model | ❌ | output keys are fixed at trace time |
| BEC inference from an eager checkpoint | ✅ | |
| reading `LES_q` / `LES_BEC` from a deployed model | ❌ | the targets export energies and forces only -- see below |
| mixed periodic + isolated dataset | ❌ | legacy module only, and then no compilation |
| LAMMPS ML-IAP (`nequip-prepare-lmp-mliap`) | ❓ | not yet verified against a LAMMPS build |
| CPU / CUDA | ✅ / ✅ | compilation works on both |

## `pair_allegro` and periodic models

The `pair_allegro` target's graph carries positions, edge indices and atom types -- **no
cell**. A periodic Ewald sum needs the cell for its reciprocal-space part, so a periodic LES
model cannot be exported to that target and is refused with an error naming the missing cell.

This guard exists because without it the cell was silently filled with zeros and the
long-range physics quietly became non-periodic, with no warning
([issue #15](https://github.com/ChengUCB/NequIP-LES/issues/15)). A loud failure is the
correct behaviour until the target passes a cell.

Non-periodic models export to `pair_allegro` normally. Under `--mode torchscript` the export
also succeeds for periodic models, because TorchScript compiles the source without running
the branch that raises -- the guard is compiled into the artefact and fires when LAMMPS calls
it. `--target ase` always passes a cell and is unaffected.

## Latent charges are not in a deployed model

`nequip-compile` fixes what a compiled artefact returns, per target:

| target | exported outputs |
|---|---|
| `pair_nequip`, `pair_allegro` | per-atom energy, forces, virial |
| `ase`, `batch` | per-atom energy, total energy, forces, stress |

`LES_q` is not on either list, so a deployed model gives you energies and forces and nothing
else. `LES_BEC` is further out of reach: it is not even computed unless BEC is switched on.

This matters because LAMMPS otherwise looks like it can reach them. `pair_nequip_allegro`
provides a [`compute`](https://github.com/mir-group/pair_nequip_allegro/tree/main/compute)
that pulls an arbitrary key out of the model's returned dictionary --
`compute q all nequip/atom LES_q 1 0` reads exactly right. But the key has to be in the
dictionary the *compiled* model returns, and it is not.

So: get latent charges and BECs during **training or testing**, from the checkpoint, using the
callbacks in [Usage](usage.md#predicted-charges-and-becs). They are not available from a
LAMMPS or ASE deployment. Lifting this would mean adding the LES keys to the target output
lists in nequip.

## Accelerations

LES adds no kernels of its own, so the backbone's accelerations apply unchanged; refer to
the [NequIP](https://nequip.readthedocs.io/en/latest/guide/accelerations/openequivariance.html)
and [Allegro](https://nequip.readthedocs.io/projects/allegro/en/latest/guide/accelerations.html)
pages for what each one does and how to install it. What matters here is that they compose
with LES, which the test suite checks:

| modifier | backbone | training | inference |
|---|---|---|---|
| `enable_OpenEquivariance` | NequIP | ✅ | ✅ |
| `enable_CuEquivariance` | NequIP | ❌ | ✅ |
| `enable_CuEquivarianceContracter` | Allegro | ✅ | ✅ |
| `enable_TritonContracter` | Allegro | ❌ | ✅ |

The ❌ entries are the upstream packages' own limitation -- those two are inference-only --
not a LES one. Every ✅ above is exercised on GPU by `run_gpu.sh`: the inference column as an
export with the modifier applied, the training column as a training run with the model wrapped
in `nequip.model.modify`.

`nequip-compile` takes its positional arguments first, because `--modifiers` accepts a list
and would otherwise swallow them:

```bash
nequip-compile model.ckpt out.nequip.pt2 \
    --device cuda --mode aotinductor --target ase \
    --modifiers enable_OpenEquivariance
```

## torch versions

| torch | status |
|---|---|
| < 2.10 | `--mode torchscript` available |
| 2.9.1 | recommended for GPU `compile_mode: compile` |
| 2.12 | GPU train-time compilation broken (a `silu_backward` bug, unrelated to LES; fixed in 2.13) |
| 2.13 | `nequip-compile` fails on CUDA unless TF32 is worked around, below; with that in place the whole GPU suite passes |

### The TF32 workaround

On torch 2.13, `torch.export` reads the legacy aggregate cuDNN TF32 flag, which cannot
represent the "not TF32" state nequip sets -- so every `nequip-compile` on a CUDA device
raises, whatever the model. This is a torch bug, not a nequip or LES one.

[`tests/compile_tf32fix.py`](https://github.com/ChengUCB/NequIP-LES/blob/main/tests/compile_tf32fix.py)
makes that one read answer instead of raising and then calls `nequip-compile`'s entry point.
It takes the same arguments:

```bash
python tests/compile_tf32fix.py model.ckpt out.nequip.pt2 \
    --device cuda --mode aotinductor --target ase
```

Nothing about the model or about nequip's TF32 setting changes.

## What has actually been tested

The tables above are not aspirational -- with the exception of the ❓ row, every entry is a
row in the [test suite](testing.md). The most recent full GPU run:

* **NVIDIA A40, torch 2.13.0+cu130** -- 70 rows, 0 failures. Both backbones × both systems
  (periodic water, isolated dipeptide) × both training paths (eager, compiled), each one
  trained, packaged, exported to its applicable targets, exported again with TF32, and
  exported once per acceleration modifier; plus two training runs with a modifier applied.
* ML-IAP was skipped throughout: no LAMMPS ML-IAP build in that environment. This is the one
  ❓ in the capability table, and the only claim on this page that rests on the documentation
  rather than on a run of our own.

## Known gap

Exporting a **periodic** model with dynamic shapes *and* gradients of the latent charges
requested is not supported by AOTInductor at present. No deployment target asks for that
combination -- it is a stricter case than any real use -- but it is tracked in the les test
suite as a known gap rather than hidden.
