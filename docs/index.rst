NequIP-LES
==========

NequIP-LES explicitly adds a **long-range electrostatic** term to `NequIP <https://github.com/mir-group/nequip>`_
and `Allegro <https://github.com/mir-group/allegro>`_ models, implemented as a NequIP
`extension package <https://nequip.readthedocs.io/en/latest/dev/extension_packages.html>`_.
The interaction is evaluated with `LES <https://github.com/ChengUCB/les>`_
(Latent Ewald Summation): the network predicts latent charges -- and optionally dipoles,
quadrupoles and polarizabilities -- and LES sums their electrostatics.

Nothing about the underlying short-range model changes. Training, testing and deployment
follow the NequIP framework, so this documentation covers what is specific to LES and links to
the `NequIP docs <https://nequip.readthedocs.io>`_ and the
`Allegro docs <https://nequip.readthedocs.io/projects/allegro/en/latest/>`_ for everything else.

Tested with nequip 0.19.0, allegro 0.8.3, PyTorch 2.13.0 and Python 3.12 without problems.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   intro/intro
   guide/guide
   updates
   citation
