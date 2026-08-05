# How LES works

## The problem

A short-range MLIP writes the energy as a sum over atoms inside a cutoff $r_c$:

$$E = \sum_i \varepsilon_i(\{\mathbf{r}_j : r_{ij} < r_c\})$$

Electrostatics does not fit in that form. A Coulomb interaction decays as $1/r$, which is
slower than the volume of a shell grows, so no finite cutoff converges the sum. Charge
transfer, dielectric screening and polarization response are all outside what such a model
can express -- not because it is not flexible enough, but because the information is not in
its inputs.

## The idea: latent charges

LES splits the energy in two:

$$E = E_\text{SR} + E_\text{LR}$$

$E_\text{SR}$ is the ordinary short-range model. For $E_\text{LR}$, the network predicts one
scalar per atom from its local environment,

$$q_i = f(\text{descriptors of atom } i)$$

and LES computes the electrostatic energy of that set of charges with an Ewald sum.

The important part is what does **not** happen: nothing supervises $q_i$. There is no
reference charge, no population analysis, no partial-charge target. The only training signal
is the total energy and the forces, and the charges are whatever makes those come out right.
This is why they are called *latent* -- they are an internal representation that happens to
behave like charge, not a fit to a definition of charge.

They are not arbitrary, either. Because $E_\text{LR}$ depends on them only through a physical
Coulomb form, the only way they can lower the loss is by describing the actual long-range
part of the interaction. In practice they come out close to physically sensible charges, and
they transfer: a model trained on energies and forces alone predicts dipoles and dielectric
response it never saw.

## The Ewald sum

Each latent charge is smeared into a Gaussian of width $\sigma$. That single choice makes
both halves of the calculation finite.

**Periodic systems** are summed in reciprocal space. With the structure factor

$$S(\mathbf{k}) = \sum_i q_i e^{i\mathbf{k}\cdot\mathbf{r}_i}$$

the energy is

$$E_\text{LR} = \frac{2\pi}{V} \sum_{\mathbf{k} \neq 0} \frac{e^{-\sigma^2 k^2 / 2}}{k^2}\, |S(\mathbf{k})|^2$$

summed over the reciprocal lattice with $|\mathbf{k}| \le 2\pi/\texttt{dl}$. The Gaussian
factor $e^{-\sigma^2 k^2/2}$ cuts the sum off by itself: large $k$ contributes nothing, so a
finite number of terms is exact to any tolerance you like.

**Isolated systems** are summed in real space, where the smeared charges give an error
function instead of a bare $1/r$:

$$E_\text{LR} = \frac{1}{2} \sum_{i \neq j} q_i q_j \frac{\operatorname{erf}\!\left(r_{ij} / \sigma\sqrt{2}\right)}{r_{ij}}$$

At short range $\operatorname{erf}(r/\sigma\sqrt{2})/r \to$ constant rather than diverging --
the Gaussian overlap removes the singularity. This matters for more than numerics: it means
the long-range term is *smooth and weak* where the short-range model is already accurate, so
the two do not fight over the same physics. $\sigma$ is where the handover happens.

`remove_self_interaction` subtracts each charge's interaction with its own Gaussian,
$\sum_i q_i^2 / (\sigma (2\pi)^{3/2})$, which is a spurious constant of the smearing rather
than physics.

## Multipoles

A single scalar per atom is the leading term. The extension adds the next ones -- latent
**dipoles** $\boldsymbol{\mu}_i$ (an equivariant vector, `1o`) and **quadrupoles**
$\mathbf{Q}_i$ (`2e`) -- summed with the same Ewald machinery, since the kernels are just
derivatives of the monopole one.

Polarizabilities go further: instead of fixed multipoles, the atom is given a **response**.
With a polarizability $\kappa_i$ or $\alpha_i$, the local electrostatic field induces extra
charge or extra dipole,

$$q_i \rightarrow q_i + \Delta q_i(\mathbf{E}_i), \qquad \boldsymbol{\mu}_i \rightarrow \boldsymbol{\mu}_i + \Delta\boldsymbol{\mu}_i(\mathbf{E}_i)$$

so the charge distribution is no longer a function of geometry alone -- it reacts to the
electrostatics it is itself generating. This is what lets one model describe dielectric
screening, and it is what the `use_induced_charge` / `use_induced_dipole` flags turn on.
`use_anisotropic_polarizability` makes $\alpha$ a tensor, so the response can depend on
direction.

## Born effective charges

The polarization of a configuration follows from the latent charges,

$$\mathbf{P} = \sum_i q_i \mathbf{r}_i \;(+ \text{dipole terms})$$

and the Born effective charge tensor is its derivative with respect to an atomic position:

$$Z^*_{i,\alpha\beta} = \frac{\partial P_\alpha}{\partial r_{i\beta}}$$

Since $q_i$ is itself a differentiable function of every position, autograd gives this
directly -- no finite differences, no extra training target. `remove_mean` subtracts the mean
latent charge of each configuration first, so the charges sum to zero and $\mathbf{P}$ is
well defined; `epsilon_factor` applies the high-frequency dielectric screening
$\varepsilon_\infty$.

This is the strongest evidence that the latent charges are not a fitting artefact: BECs are a
response property that never appeared in training, and they come out right.

## Cost

The reciprocal-space sum is $O(N K)$ for $N$ atoms and $K$ k-vectors; the real-space one is
$O(N^2)$ over pairs, which is why non-periodic LES is for molecules rather than for large
systems. In both cases the long-range term is a small fraction of the short-range network's
cost at typical settings -- LES is cheap compared to the model it augments.

## Reading

The method and its successive extensions are developed in the papers listed under
[Citation](../citation.md). Start with *Latent Ewald summation for machine learning of
long-range interactions* for the method itself, *Machine learning interatomic potential can
infer electrical response* for the BECs, and *A universal augmentation framework for
long-range electrostatics* for the MLIP-agnostic formulation this package implements.
