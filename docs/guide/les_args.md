# `les_args` reference

Everything LES-specific lives in one block:

```yaml
model:
  _target_: nequip_les.model.LESModel
  base_model: nequip
  LES:
    les_args:
      is_periodic: true
      N_max: 10
```

Only the keys you set matter; the rest keep the defaults below. Two packages read this
block -- [les](https://github.com/ChengUCB/les) for the Ewald sum itself, and `nequip_les`
for the multipole heads that feed it -- but you write them all in the same place.

## The Ewald sum

| key | default | meaning |
|---|---|---|
| `is_periodic` | `None` | Which Ewald implementation and which boundary condition. `true` = periodic, `false` = isolated, `None` = the legacy implementation, which decides per structure. **Set it explicitly** -- see [Ewald implementations](ewald.md). |
| `sigma` | `1.0` | Width (Å) of the Gaussian each latent charge is smeared over. It sets how far the electrostatics reach before the short-range model takes over; also the Ewald splitting parameter. |
| `dl` | `2.0` | Reciprocal-space resolution (Å). Smaller `dl` = more k-vectors = more accurate and slower. |
| `N_max` | `10` | Cap on k-vectors per direction. Keep `N_max * dl` above your cell's longest side, otherwise the sum is truncated before it converges. Periodic only. |
| `remove_self_interaction` | `True` | Subtract each charge's interaction with its own Gaussian. Leave on: it is a constant offset, not physics. |
| `use_epsilon_r_scaling` | `False` | Learn a dielectric-screening factor that rescales the electrostatic energy. |

## What the network predicts

By default LES predicts one scalar latent charge per atom. Each flag below adds a term.
They compose, and each one costs both parameters and time -- turn on what your physics
needs, not everything.

| key | default | meaning |
|---|---|---|
| `use_dipole` | `False` | Also predict a latent dipole per atom (`1o`). |
| `use_quadrupole` | `False` | Also predict a latent quadrupole per atom (`2e`). Requires `use_dipole`. |
| `use_induced_charge` | `False` | Charges respond to the local field: adds a polarizability `kappa` used to induce extra charge. |
| `use_induced_dipole` | `False` | Dipoles respond to the local field: adds a polarizability `alpha` used to induce extra dipole. |
| `use_anisotropic_polarizability` | `False` | Make `alpha` a tensor rather than a scalar, so the response depends on direction. |
| `alpha_irreps` | `"0e+1o+2e"` | Which irreps make up `alpha`. Must contain `1o` or `2e` for anisotropy to mean anything. |
| `kappa_scale` / `alpha_scale` | `0.1` | Output scale of the two polarizability heads. Small values keep the induced terms a correction at the start of training. |
| `kappa_alpha_positive` | `True` | Constrain the polarizabilities to be positive (physically they are). |
| `l_max` | `1` | Highest rotation order the LES heads read from the backbone features. |

## Charge parameterization

| key | default | meaning |
|---|---|---|
| `use_atomwise` | `False` | Read the charge from an extra MLP on the node features instead of the backbone's own readout. Configured by `n_layers` (`3`), `n_hidden` (`[32, 16]`), `add_linear_nn` (`True`), `output_scaling_factor` (`0.1`). |
| `use_fixed_atomic_charges` | `False` | Give every element one learnable charge, ignoring its environment. A much stiffer model -- useful when the charges should be transferable. `fixed_atomic_charges_scaling_factor` (`0.5`) scales them. |
| `use_atomic_alpha` | `False` | Per-element learnable polarizability. |

## BEC only

These two do **not** affect the Ewald energy or the forces. They are used only when Born
effective charges are computed, and they sit in the same block purely for convenience --
so seeing them next to `sigma` in an example config is not a sign they change training.

| key | default | meaning |
|---|---|---|
| `remove_mean` | `True` | Subtract the mean latent charge of each configuration before differentiating, so the charges sum to zero and the polarization is well defined. |
| `epsilon_factor` | `1.0` | High-frequency dielectric constant ε∞. Enters as `sqrt(epsilon_factor)`, screening the reported BECs. |

## A note on `sigma` and `dl`

The defaults (`sigma: 1.0`, `dl: 2.0`) work for the systems in the papers and are a
reasonable starting point. If you change them, change `sigma` for physics (how far the
long-range part should reach relative to your cutoff) and `dl`/`N_max` for numerics
(convergence of the sum). Increasing accuracy through `dl` shows up directly in runtime,
since the number of k-vectors grows as `1/dl^3`.
