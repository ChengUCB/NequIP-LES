# This file is a part of the `nequip-les` package. Please see LICENSE and README at the root for information on using it.
# This file is added to the nequip package to implement the LES energy model.
import torch
from nequip.data import AtomicDataDict
from nequip.nn._graph_mixin import GraphModuleMixin
from les import Les # https://github.com/ChengUCB/les
from .. import _keys
from typing import Optional
from nequip.nn.model_modifier_utils import model_modifier, replace_submodules
import logging
import torch.nn.functional as F


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
        self.use_dipole = les_args.get("use_dipole", False)
        self.use_induced_charge = les_args.get("use_induced_charge", False)
        self.use_induced_dipole = les_args.get("use_induced_dipole", False)
        self.use_quad = les_args.get("use_quadrupole", False)
        self.use_anisotropic_polarizability = les_args.get("use_anisotropic_polarizability", False)

        self.kappa_alpha_positive = les_args.get("kappa_alpha_positive", True)
        self.kappa_scale = les_args.get("kappa_scale", 0.1)
        self.alpha_scale = les_args.get("alpha_scale", 0.1)

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
            # cell = torch.zeros((len(torch.unique(batch)), 3, 3), # potential issue with torch.compile 
            cell = torch.zeros((AtomicDataDict.num_frames(data), 3, 3),
                               device=pos.device, dtype=pos.dtype)
            
        les_u = data[_keys.LATENT_DIPOLE_KEY] if hasattr(self, 'use_dipole') and self.use_dipole else None
        les_kappa = data[_keys.LATENT_CHEMICAL_SOFTNESS_KEY] if hasattr(self, 'use_induced_charge') and self.use_induced_charge else None
        les_alpha = data[_keys.LATENT_POLARIZABILITY_KEY] if hasattr(self, 'use_induced_dipole') and self.use_induced_dipole else None #[N,1] or [N,3,3]
        les_quad = data.get(_keys.LATENT_QUAD_KEY) if hasattr(self, 'use_quad') and self.use_quad else None

        if hasattr(self, 'kappa_alpha_positive') and self.kappa_alpha_positive:
            if les_kappa is not None:
                les_kappa = F.softplus(les_kappa)
            if les_alpha is not None:
                if les_alpha.dim() == 2:
                    les_alpha = F.softplus(les_alpha) # [N, 1]
                elif les_alpha.dim() == 3:
                    les_alpha = torch.bmm(les_alpha, les_alpha.transpose(-1, -2)) # [N,3,3]  A @ A^T PSD

        if les_kappa is not None and hasattr(self, 'kappa_scale'):
            les_kappa = les_kappa * self.kappa_scale
        if les_alpha is not None and hasattr(self, 'alpha_scale'):
            les_alpha = les_alpha * self.alpha_scale

        les_result = self.les(
            latent_charges=q,
            latent_dipoles=les_u,
            latent_quads=les_quad,
            latent_alphas=les_alpha,
            latent_kappas=les_kappa,
            positions=pos,
            batch=batch,
            cell=cell,
            compute_energy=True,
            compute_bec=self.compute_bec,
            bec_output_index=self.bec_output_index,
        )
        e_lr = les_result['E_lr'] # (n_graphs,)
        assert e_lr is not None
        les_energy = e_lr.unsqueeze(-1) # (n_graphs,1)
        if self.compute_bec:
            bec = les_result['BEC']
            assert bec is not None
            # if bec.dim() > 2 and bec.shape[1] == 2:
                # bec = bec.sum(dim=1, keepdim=False)
            # assert bec.dim() in [2, 3], f'BEC output dimension error: expected 2 or 3, got {bec.dim()}'
            data[_keys.BEC_KEY] = bec

        data[self.out_field] = les_energy
        if les_kappa is not None:
            data[_keys.LATENT_CHEMICAL_SOFTNESS_KEY] = les_kappa
        if les_alpha is not None:
            data[_keys.LATENT_POLARIZABILITY_KEY] = les_alpha
        return data
    
    @model_modifier(persistent=True)
    @classmethod
    def modify_latent_ewald_sum(
        cls,
        model,
        compute_bec: bool = True,
        bec_output_index: Optional[int] = None,
    ):
        """
        Enable Born effective charge inference in the LatentEwaldSum module.
        
        Parameters:
            model (GraphModel): The model to modify.
            compute_bec (bool): Whether to compute the Born effective charge.
            bec_output_index (Optional[int]): Index for the Born effective charge output. 
            (0, 1, or 2 for x, y, z components)
        """

        def factory(old_module: LatentEwaldSum) -> LatentEwaldSum:
            old_module.compute_bec = compute_bec
            old_module.bec_output_index = bec_output_index
            logging.getLogger(__name__).info(f"Setting compute_bec to {compute_bec} in LatentEwaldSum")
            return old_module

        return replace_submodules(model, cls, factory)
    

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