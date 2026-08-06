# Introduction

## What LES does

A short-range interatomic potential sums energies over a neighbour list, so any physics
that reaches beyond the cutoff -- electrostatics, polarization response, dielectric
screening -- is missing by construction.

LES adds it back without requiring reference charges. The network emits a **latent charge**
per atom, and LES evaluates the electrostatic energy of those charges with an Ewald sum,
which is then added to the short-range energy. The charges are learned indirectly: nothing
supervises them, only the total energy and forces do. Beyond monopoles, LES can also take
latent **dipoles**, **quadrupoles**, and **polarizabilities** (induced charges and induced
dipoles), which extend the same idea to higher multipole orders.

Because the charges are a differentiable function of the positions, quantities that follow
from them come for free -- most usefully the **Born effective charges** (BEC), obtained by
differentiating the polarization with respect to atomic positions.

[How LES works](../les/theory.md) goes through this in more detail.

## When you want it

* systems where long-range electrostatics matter: electrolytes, interfaces, charged defects,
  polar and ionic materials, biomolecules;
* properties that depend on the charge response: dielectric constants, IR and Raman spectra,
  BECs, ionic conductivities;
* cases where a short-range model plateaus in accuracy for reasons the cutoff explains.

How much it buys depends on the backbone's receptive field. Across the published benchmarks,
LES reduces force errors for **every** architecture tested, with the largest gains where the
baseline sees least: single-layer MACE and CACE, and the strictly local Allegro. Multi-layer
NequIP models gain the least, because message passing has already widened the receptive field.
Systems built around non-local charge transfer are the extreme case -- error reductions above
90% on the `Na8/9Cl8+` and `Au2/MgO(001)` sets.

The overhead is small: the reciprocal sum is inexpensive next to the message passing, and MD
timings with and without LES nearly coincide. For a genuinely short-range problem it still buys
nothing -- use plain NequIP or Allegro.

## What this package is

[LES](https://github.com/ChengUCB/les) is a standalone library, plugged into a number of
MLIPs (MACE, CACE, MatGL, ...). **NequIP-LES** is the plug-in for the NequIP framework: it
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
* [How LES works](../les/theory.md) -- the method, and the library on its own
* [Citation](../citation.md)
