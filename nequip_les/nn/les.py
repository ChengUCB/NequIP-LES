# This file is a part of the `nequip-les` package. Please see LICENSE and README at the root for information on using it.
# This file is added to the nequip package to implement the LES energy model.
import torch
from nequip.data import AtomicDataDict
from nequip.nn._graph_mixin import GraphModuleMixin
from les import Les # https://github.com/ChengUCB/les

from typing import Optional


class LatentEwaldSum(GraphModuleMixin, torch.nn.Module):
    """Latent Ewald Sum module for computing long-range energy contributions."""

    def __init__(
        self,
        field: str,
        out_field: Optional[str] = None,
        irreps_in={},
        les_args: dict = {'use_atomwise': False},
        compute_bec: bool = False,
        bec_output_index: Optional[int] = None,
    ):
        super().__init__()
        self.field = field
        self.out_field = out_field
        self._init_irreps(
            irreps_in=irreps_in,
            irreps_out=(
                {self.out_field: irreps_in[self.field]}
                if self.field in irreps_in
                else {}
            ),
        )
        self.les = Les(les_args)
        self.compute_bec = compute_bec
        self.bec_output_index = bec_output_index


    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:

        q = data[self.field]
        pos = data[AtomicDataDict.POSITIONS_KEY]
        batch = data.get(AtomicDataDict.BATCH_KEY)
        if batch is None:
            batch = torch.zeros(pos.shape[0], dtype=pos.dtype, device=pos.device)

        if AtomicDataDict.CELL_KEY in data:
            cell = data[AtomicDataDict.CELL_KEY].view(-1, 3, 3)
        else:
            cell = torch.zeros((len(torch.unique(batch)), 3, 3), 
                               device=pos.device, dtype=pos.dtype)

        les_result = self.les(
            latent_charges=q, 
            positions=pos,
            batch=batch,
            cell=cell,
            compute_energy=True,
            compute_bec = self.compute_bec,
            bec_output_index=self.bec_output_index,
        )
        e_lr = les_result['E_lr'] # (n_graphs,)
        assert e_lr is not None
        les_energy = e_lr.unsqueeze(-1) # (n_graphs,1)
        if self.compute_bec:
            bec = les_result['BEC']
            assert bec is not None
            data[AtomicDataDict.BEC_KEY] = bec

        data[self.out_field] = les_energy
        return data

class AddEnergy(GraphModuleMixin, torch.nn.Module):
    """Add energy to the total energy of the system."""

    def __init__(
        self,
        field1: str,
        field2: str,
        out_field: Optional[str] = None,
        irreps_in={},
    ):
        super().__init__()
        self.field1 = field1
        self.field2 = field2
        self.out_field = out_field
        self._init_irreps(
            irreps_in=irreps_in,
            irreps_out=(
                {self.out_field: irreps_in[self.field1]}
                if self.field1 in irreps_in
                else {}
            ),
        )
    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        sr_energy = data[self.field1]
        lr_energy = data[self.field2]
        total_energy = sr_energy + lr_energy
        data[self.out_field] = total_energy
        return data