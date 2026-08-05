# Using the LES library directly

You do not need NequIP to use LES. The library is a single `torch.nn.Module` that takes
positions, a cell, and per-atom multipoles, and returns the long-range energy. This page is
for adding LES to your own model; if you are training NequIP or Allegro models, use
[NequIP-LES](../guide/usage.md) instead.

## Install

```bash
pip install git+https://github.com/ChengUCB/les.git
```

## The module

```python
import torch
from les import Les

les = Les(les_arguments={"is_periodic": True, "sigma": 1.0, "dl": 2.0})

out = les(
    positions=positions,          # [n_atoms, 3]
    cell=cell,                    # [n_structures, 3, 3]
    latent_charges=q,             # [n_atoms] -- your model's output
    batch=batch,                  # [n_atoms], which structure each atom belongs to
    compute_energy=True,
)

energy = out["E_lr"]              # add this to your short-range energy
```

`les_arguments` is the same dictionary documented in the
[`les_args` reference](../guide/les_args.md) -- it can also be a path to a YAML file holding
it. Forces come from autograd on `positions` as usual; nothing special is needed.

## Inputs

| argument | shape | notes |
|---|---|---|
| `positions` | `[n_atoms, 3]` | required |
| `cell` | `[n_structures, 3, 3]` | required, even for isolated systems -- see below |
| `batch` | `[n_atoms]` | defaults to a single structure |
| `latent_charges` | `[n_atoms]` | your model's per-atom scalar |
| `latent_dipoles` | `[n_atoms, 3]` | optional, must be equivariant |
| `latent_quads` | `[n_atoms, 3, 3]` | optional |
| `latent_kappas` | `[n_atoms]` | optional, charge polarizability |
| `latent_alphas` | `[n_atoms]` or `[n_atoms, 3, 3]` | optional, dipole polarizability |
| `atomic_numbers` | `[n_atoms]` | needed by `use_fixed_atomic_charges` / `use_atomic_alpha` |
| `e_ext` | | optional external field |
| `desc` | `[n_atoms, n_features]` | descriptors, instead of `latent_charges` -- requires `use_atomwise: True`, and LES predicts the charges itself with its own MLP |

Pass either `latent_charges` or `desc`; passing neither raises.

## Outputs

```python
{
    "E_lr":            ...,   # [n_structures] long-range energy
    "latent_charges":  ...,   # [n_atoms], with induced charge added if kappa was used
    "latent_dipoles":  ...,   # with induced dipole added if alpha was used
    "latent_quads":    ...,
    "latent_alphas":   ...,
    "BEC":             ...,   # only if compute_bec=True
}
```

The returned `latent_charges` is what you should write out if you want to inspect the
charges: it includes the induced contribution, while the tensor you passed in does not.

## Flags

* `compute_energy` (default `True`) -- the Ewald sum. Turn it off to get charges only.
* `compute_field` -- also return the electrostatic potential and field per atom.
* `compute_bec` -- Born effective charges. `bec_output_index` restricts them to one
  Cartesian direction, which is three times cheaper when that is all you need.

## Isolated systems

`cell` is required whatever the boundary condition. For an isolated system set
`is_periodic: False` and pass a cell -- LES ignores it, but a zero cell will bite you
elsewhere (NequIP divides by the volume for stress). Use a large finite box; the reasoning is
in [Non-periodic models need a dummy cell](../guide/ewald.md#non-periodic-models-need-a-dummy-cell).

## Choosing the implementation

`is_periodic` selects between the vectorized Ewald (set it to `True`/`False`) and the legacy
one (leave it out). Only the vectorized one can be compiled or exported; the legacy one
supports datasets that mix periodic and isolated structures. See
[Ewald implementations](../guide/ewald.md).

## Hyperparameters

The defaults usually work. The one worth trying differently is `remove_self_interaction`:

> `remove_self_interaction=True` is the default and is the most robust choice.
> `remove_self_interaction=False` can sometimes yield a bit better training accuracy, but is
> less robust when training on finite systems and then extrapolating to periodic systems.

## Other MLIPs with LES

LES is already integrated into several packages, so check whether yours is covered before
wiring it in yourself:

| package | link |
|---|---|
| MACE | [ACEsuit/mace](https://github.com/ACEsuit/mace) (merged upstream) |
| CACE | [BingqingCheng/cace](https://github.com/BingqingCheng/cace) |
| NequIP / Allegro | [ChengUCB/NequIP-LES](https://github.com/ChengUCB/NequIP-LES) -- this package |
| MatGL | [ChengUCB/matgl](https://github.com/ChengUCB/matgl) |

Training scripts and trained models for all of them:
[les_fit](https://github.com/ChengUCB/les_fit) and
[extended_les_fit](https://github.com/ChengUCB/extended_les_fit), including **MACELES-OFF**
trained on SPICE.
