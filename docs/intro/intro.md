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

## When you want it

* systems where long-range electrostatics matter: electrolytes, interfaces, charged
  defects, polar solids;
* properties that depend on the charge response: dielectric constants, IR spectra, BECs;
* cases where a short-range model plateaus in accuracy for reasons the cutoff explains.

For a purely short-range problem the extra cost buys nothing -- use plain NequIP or Allegro.

## Papers

* LES: [arXiv:2504.15925](https://arxiv.org/abs/2504.15925)
* Multipole extension (dipoles, quadrupoles, polarizabilities): [arXiv:2605.05746](https://arxiv.org/abs/2605.05746)

See the [repository README](https://github.com/ChengUCB/NequIP-LES) for the citation entries.
