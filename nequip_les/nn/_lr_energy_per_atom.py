# This file is a part of the `nequip-les` package. Please see LICENSE and README at the root for information on using it.
import torch

from nequip.data import AtomicDataDict
from nequip.nn import GraphModuleMixin

from .. import _keys


class DistributeLREnergyPerAtom(GraphModuleMixin, torch.nn.Module):
    r"""Add the long-range energy into ``per_atom_energy``, spread evenly over the atoms.

    Why this exists
    ---------------
    The Ewald sum produces one number per structure, and it is added to
    ``total_energy`` only -- ``per_atom_energy`` carries the short-range part alone.
    That is invisible in ASE, which reads ``total_energy``, but the LAMMPS pair styles
    have no global energy channel: ``pair_nequip_allegro.cpp`` builds LAMMPS'
    ``eng_vdwl`` by summing the per-atom array, and ``nequip-compile``'s LAMMPS targets
    export ``per_atom_energy``, ``forces`` and ``virial`` -- no ``total_energy``. So a
    LES model in LAMMPS reported the short-range energy while its forces were correct:
    silently wrong thermodynamics with a correct trajectory.

    This module closes that gap by making the per-atom array sum to the total.

    Placement
    ---------
    Appended *after* the module that writes ``total_energy``, so ``total_energy``,
    ``forces`` and ``stress`` are bit-for-bit what they were before -- only
    ``per_atom_energy`` changes, from "inconsistent with the total" to "sums to it".

    On the even split
    -----------------
    A per-atom share of an electrostatic energy is not uniquely defined; the physical
    decomposition would be :math:`\tfrac{1}{2} q_i \phi_i` plus the corresponding
    multipole terms, which needs the per-atom potential that the LES library does not
    currently return. The even split is exact in the sum, which is the only property
    LAMMPS' energy channel depends on, and it costs one broadcast. Should per-atom
    long-range energies ever need to be meaningful on their own, this module is where
    that would change; nothing around it would.
    """

    def __init__(
        self,
        field: str = _keys.LR_ENERGY_KEY,
        out_field: str = AtomicDataDict.PER_ATOM_ENERGY_KEY,
        irreps_in={},
    ):
        super().__init__()
        self.field = field
        self.out_field = out_field
        # the field is modified in place and keeps its irreps, so nothing new is declared
        self._init_irreps(irreps_in=irreps_in)

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        per_atom = data[self.out_field]
        e_lr = data[self.field].reshape(-1)  # (num_frames,)

        # the LAMMPS pair targets pass no batch; same fallback as `LatentEwaldSum`
        batch = data.get(AtomicDataDict.BATCH_KEY)
        if batch is None:
            batch = torch.zeros(
                per_atom.shape[0], dtype=torch.long, device=per_atom.device
            )

        # one-hot matmuls rather than scatter/gather: AOTInductor's CPU backend cannot
        # vectorize the atomic_add a scatter lowers to, which is why the Ewald sum was
        # written this way too
        n_frames = e_lr.shape[0]
        onehot = (
            batch.unsqueeze(0)
            == torch.arange(n_frames, device=batch.device).unsqueeze(1)
        ).to(dtype=per_atom.dtype)  # (num_frames, num_atoms)
        counts = onehot.sum(dim=1).clamp(min=1.0)  # (num_frames,)
        share = torch.matmul(
            onehot.transpose(0, 1), (e_lr.to(per_atom.dtype) / counts).unsqueeze(1)
        )  # (num_atoms, 1)

        data[self.out_field] = per_atom + share.reshape(per_atom.shape)
        return data
