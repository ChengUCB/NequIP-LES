# Ewald implementations: vectorized vs legacy

LES ships two implementations of the same sum. `is_periodic` chooses between them, and the
choice decides whether the model can be compiled.

| | legacy (`is_periodic` absent / `None`) | vectorized (`is_periodic: true` or `false`) |
|---|---|---|
| boundary condition | decided **per structure**, from `det(cell)` | fixed for the whole model by the flag |
| mixed periodic + non-periodic dataset | ✅ | ❌ one condition for everything |
| `torch.compile` (train-time) | ❌ | ✅ |
| `nequip-compile` (deployment) | ❌ | ✅ |
| numerical result | same physics | same physics |

They agree numerically: the vectorized module is checked against the legacy one across
monopole, dipole, quadrupole, induced-charge, induced-dipole and BEC paths, periodic and
non-periodic, in the [les test suite](https://github.com/ChengUCB/les/tree/main/test).

## Why the legacy one cannot be compiled

It loops over the structures in a batch in Python and branches on
`det(cell) < 1e-6` -- a decision made from tensor *values*. A compiler has to know the
graph before it sees the data, so a value-dependent branch stops the trace. The vectorized
module removes both: one batched code path, and the periodicity known at construction time.

```{important}
**Compilation of any kind requires `is_periodic` to be set.** Without it you get the legacy
module, and `compile_mode: compile` and `nequip-compile` both fail. Eager training works
either way.
```

## Which should you use

Set `is_periodic` unless your dataset genuinely mixes periodic and non-periodic structures. It
is faster, it is what deployment needs, and stating the boundary condition explicitly is a
sanity check on your data rather than a cost.

Keep the legacy module only for a mixed dataset, and accept eager-only training. Splitting
such a dataset into two datasets is usually the better answer.

## Non-periodic models need a dummy cell

`is_periodic: false` means LES ignores the cell. NequIP does not: it computes
`stress = virial / volume`, so a zero cell gives `volume = 0` and the loss becomes NaN as
soon as the model is compiled -- silently, with training appearing to proceed.

Give non-periodic structures a large finite cell with `pbc="F F F"`:

```
Lattice="100.0 0.0 0.0 0.0 100.0 0.0 0.0 0.0 100.0" pbc="F F F"
```

The cell is never used physically -- it only keeps the volume finite. nequip's
[`cell_utils`](https://github.com/mir-group/nequip/blob/main/nequip/data/transforms/cell_utils.py)
is worth reading for how the framework handles cells and periodicity. Compare
[`tests/data/dipep_train.xyz`](https://github.com/ChengUCB/NequIP-LES/blob/main/tests/data/dipep_train.xyz)
(zero cell, for eager runs) with
[`dipep_train_dummycell.xyz`](https://github.com/ChengUCB/NequIP-LES/blob/main/tests/data/dipep_train_dummycell.xyz)
(100 Å cell, for compiled runs).

```{warning}
Do not add a dummy cell to a **legacy** model's data. The legacy module reads `det(cell)`
to decide periodicity, so a dummy cell silently switches it to the periodic branch: in our
test system the energy moved from −4.43 to −5.24 eV and the step time from 1 ms to 64 ms.
The dummy cell is for `is_periodic: false` runs only.
```
