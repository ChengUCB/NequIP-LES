NequIP-LES
==========

NequIP-LES adds a **long-range electrostatic** term to `NequIP <https://github.com/mir-group/nequip>`_
and `Allegro <https://github.com/mir-group/allegro>`_ models, implemented as a NequIP
extension package. The interaction is evaluated with `LES <https://github.com/ChengUCB/les>`_
(Latent Ewald Summation): the network predicts latent charges -- and optionally dipoles,
quadrupoles and polarizabilities -- and LES sums their electrostatics with an Ewald sum.

Nothing about the underlying short-range model changes. Training, testing and deployment
follow the NequIP framework, so this documentation covers what is specific to LES and links
to the `NequIP docs <https://nequip.readthedocs.io>`_ for everything else.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   intro/intro
   guide/guide
   les/les
   citation

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
