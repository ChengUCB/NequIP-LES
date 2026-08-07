# Introduction

## What LES does

A short-range interatomic potential sums energies over a neighbour list, so any physics
that reaches beyond the cutoff -- electrostatics, polarization response, dielectric
screening -- is missing by construction.

LES adds the long-range electrostatics explicitly without requiring reference electrostatic traning labels. The network emits a **latent charge**
per atom, and LES evaluates the electrostatic energy of those charges, which is then added to the short-range energy. The charges are learned indirectly: nothing
supervises them, only the total energy and forces do. Beyond monopoles, LES can also take
latent **dipoles**, **quadrupoles**, and **polarizabilities** (induced charges and induced dipoles), which extend the same idea to higher multipole orders.

Because the latent quantities are a differentiable function of the positions, quantities that follow
from them come for free -- most usefully the **Born effective charges** (BEC), obtained by
differentiating the polarization with respect to atomic positions.

[How LES works](https://les.readthedocs.io/en/latest/theory.html) in the LES documentation goes through this in more detail.

## When you want it

* systems where long-range electrostatics matter: electrolytes, interfaces, charged defects,
  polar and ionic materials, biomolecules;
* properties that depend on the charge response: dielectric constants, IR and Raman spectra,
  BECs, ionic conductivities;
* cases where the interaction that matters reaches beyond the cutoff: non-local charge
  transfer, a dopant that changes adsorption from 10 Å away, an interface with an intrinsic
  field.

LES in general reduces force errors for **every** architecture tested with small computational overhead due to the construction. 
Please check the previous papers in [].

## What this package is

[LES](https://github.com/ChengUCB/les) is a standalone library, plugged into a number of
MLIPs -- [MACE](https://github.com/ACEsuit/mace), [CACE](https://github.com/BingqingCheng/cace),
[MatGL](https://github.com/ChengUCB/matgl), CHGNet, UMA. **NequIP-LES** is the plug-in for the NequIP framework: it
wires LES into [NequIP](https://github.com/mir-group/nequip) and
[Allegro](https://github.com/mir-group/allegro) models, adds the equivariant heads that
predict the multipoles, and makes the whole thing trainable, compilable and deployable
through the ordinary NequIP tooling.

Everything that is not LES-specific -- data, training, loss, metrics, LAMMPS, ASE -- is the
NequIP framework, documented [here](https://nequip.readthedocs.io). This site covers the
LES part and links out for the rest.

## Where to go next

* [Usage](../guide/usage.md) -- turning a NequIP or Allegro config into a LES one
* [`les_args` reference](../guide/les_args.md) -- every option, with defaults
* [Ewald implementations](../guide/ewald.md) -- vectorized vs legacy, and what compilation needs
* [What works](../guide/deployment.md) -- capability tables for compilation, deployment, accelerations
* [How LES works](https://les.readthedocs.io/en/latest/theory.html) -- the method itself, in the LES documentation
* [Citation](../citation.md)
