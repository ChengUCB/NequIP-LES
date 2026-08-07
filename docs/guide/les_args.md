# `les_args` reference

Everything LES-specific lives in one block. All of the options are listed here with their
defaults; you only need to write the ones you change.

```yaml
model:
  _target_: nequip_les.model.LESModel
  base_model: nequip                        # or: allegro
  LES:
    les_args:
      # --- what the network predicts ---
      use_dipole: false                     # default: false
      use_quadrupole: false                 # default: false   (requires use_dipole)
      use_induced_charge: false             # default: false
      use_induced_dipole: false             # default: false
      use_anisotropic_polarizability: false # default: false
      alpha_irreps: '0e+1o+2e'              # default: '0e+1o+2e'
      kappa_scale: 0.1                      # default: 0.1
      alpha_scale: 0.1                      # default: 0.1
      kappa_alpha_positive: true            # default: true

      # --- bookkeeping ---
      distribute_lr_energy: true            # default: true

      # --- the Ewald sum ---
      remove_self_interaction: true         # default: true
      is_periodic: true                     # default: null  -> legacy Ewald
      N_max: 10                             # default: 10
      sigma: 1.0                            # default: 1.0
      dl: 2.0                               # default: 2.0
```

Two packages read this block -- [les](https://github.com/ChengUCB/les) for the Ewald sum
itself and `nequip_les` for the multipole heads that feed it -- but you write them all in the
same place.

## What the network predicts

By default LES predicts one scalar latent charge per atom. Each flag below adds a term of the
multipole expansion or of the polarization response. They compose, and each one costs both
parameters and time -- turn on what your physics needs, not everything. Please see
[*Polarizable atomic multipoles for learning long-range electrostatics*](https://arxiv.org/abs/2605.05746)
for more details.

| key | default | meaning |
|---|---|---|
| `use_dipole` | `false` | Also predict a latent dipole per atom (`1o`). |
| `use_quadrupole` | `false` | Also predict a latent traceless quadrupole per atom (`2e`). Requires `use_dipole`. |
| `use_induced_charge` | `false` | Charges respond to the local electrostatic potential: adds a hardness `kappa` from which an induced charge is obtained. |
| `use_induced_dipole` | `false` | Dipoles respond to the local field: adds a polarizability `alpha` from which an induced dipole is obtained. |
| `use_anisotropic_polarizability` | `false` | Make `alpha` a tensor rather than a scalar, so the response depends on direction. |
| `alpha_irreps` | `'0e+1o+2e'` | Which irreps make up `alpha`. Only meaningful when `use_anisotropic_polarizability` is `true`, and it must then contain `1o` or `2e`. |
| `kappa_scale` / `alpha_scale` | `0.1` | Output scale of the two response heads. Small values keep the induced terms a correction at the start of training. |
| `kappa_alpha_positive` | `true` | Constrain the polarizabilities to be positive. Even without the constraint they are usually learned positive in our experience. |

Published model names encode the combination, so you can read one straight off: `-u` dipoles,
`-Q` quadrupoles, `-iq` induced charge, `-iu` induced dipole. A model called
`nequiples-uQiqiu` has all four on. See
[Multipoles and polarization response](https://les.readthedocs.io/en/latest/theory.html#multipoles-and-polarization-response).

## Bookkeeping

| key | default | meaning |
|---|---|---|
| `distribute_lr_energy` | `true` | Spread the long-range energy over the atoms so `per_atom_energy` sums to `total_energy`. Required for LAMMPS, which has no global energy channel; `total_energy`, forces and stress are identical either way. Set `false` to reproduce the pre-fix per-atom values exactly. [Details](deployment.md#the-long-range-energy-reaches-lammps-through-per_atom_energy). |

## The Ewald sum

| key | default | meaning |
|---|---|---|
| `remove_self_interaction` | `true` | Subtract each charge's interaction with its own Gaussian. |
| `is_periodic` | `null` | Which Ewald implementation and which boundary condition. `true` = periodic, `false` = non-periodic, `null` = the legacy implementation, which decides per structure. **Set it explicitly** -- see [Ewald implementations](ewald.md). |
| `N_max` | `10` | Extent of the integer k-grid, `n` in `[-N_max, N_max]` per direction. It must be large enough for the `dl` cutoff sphere to fit: keep `N_max * dl` above your cell's longest side. Periodic only. More k-vectors means more accurate and more expensive. |
| `sigma` | `1.0` | Width (Å) of the Gaussian each latent charge is smeared over. It sets how far the electrostatics reach before the short-range model takes over; also the Ewald splitting parameter. |
| `dl` | `2.0` | Resolution of the reciprocal-space sum, in Å: it sets the cutoff on the k-vector magnitude, `k_max = 2*pi/dl`. The default corresponds to `k_c = pi`. |

```{note}
We have checked that the default `sigma: 1.0` and `dl: 2.0` converge in essentially every case
we have tried, and they are what the published fits use. Changing them is not recommended.
```
