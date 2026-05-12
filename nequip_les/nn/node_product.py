# This file is a part of the `nequip-les` package. Please see LICENSE and README at the root for information on using it.
import torch
from e3nn import o3
from e3nn.io import CartesianTensor
from nequip.data import AtomicDataDict
from nequip.nn import GraphModuleMixin


class NodeOuterProduct(GraphModuleMixin, torch.nn.Module):
    """Maps a node 1o vector field [N, 3] to a symmetric [N, 3, 3] tensor.
    If traceless=True, subtracts trace/3 * I, to produce a traceless tensor (quadrupole).
    If traceless=False, yields a symmetric PSD tensor (anisotropic polarizability).
    """

    def __init__(self, in_field: str, out_field: str, irreps_in: dict, traceless: bool = False):
        super().__init__()
        self.in_field = in_field
        self.out_field = out_field
        self.traceless = traceless
        self._init_irreps(irreps_in=irreps_in, irreps_out={})

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        u = data[self.in_field]  # [N, 3]
        outer = torch.einsum('na,nb->nab', u, u)  # [N, 3, 3] symmetric
        if self.traceless:
            trace = outer.diagonal(dim1=-1, dim2=-2).sum(-1)  # [N]
            eye = torch.eye(3, device=u.device, dtype=u.dtype)
            outer = outer - (trace / 3).view(-1, 1, 1) * eye.unsqueeze(0)
        data[self.out_field] = outer
        return data


class NodeSpherical2eToCartesian(GraphModuleMixin, torch.nn.Module):
    """Maps a 1x2e spherical tensor field [N, 5] to a traceless symmetric Cartesian [N, 3, 3].
    Uses the e3nn CartesianTensor change-of-basis (fixed buffer, no trainable parameters).
    """

    def __init__(self, in_field: str, out_field: str, irreps_in: dict):
        super().__init__()
        self.in_field = in_field
        self.out_field = out_field
        self._init_irreps(irreps_in=irreps_in, irreps_out={})
        # reduced_tensor_products().change_of_basis has shape [irreps.dim, 3, 3] = [6, 3, 3]
        # irreps order: 1x0e (index 0) + 1x2e (indices 1:6)
        cob = CartesianTensor("ij=ji").reduced_tensor_products().change_of_basis  # [6, 3, 3]
        self.register_buffer("change_of_basis_2e", cob[1:])  # [5, 3, 3]

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        x = data[self.in_field]  # [N, 5] spherical 2e components
        data[self.out_field] = torch.einsum('ijk,ni->njk', self.change_of_basis_2e, x)
        return data


class NodeAssembleTensor(GraphModuleMixin, torch.nn.Module):
    """Assembles a final [N, 3, 3] tensor from scalar (0e) and/or higher-order contributions,
    writing the result directly to out_field.
    scalar_field (optional): key holding [N, 1], expanded to scalar * I as the isotropic base.
    contrib_fields: list of keys holding [N, 3, 3] tensors to add on top.
    traceless: if True, removes the trace after assembly (for quadrupole).
    """

    def __init__(
        self,
        out_field: str,
        irreps_in: dict,
        scalar_field: str = None,
        contrib_fields: list = None,
        traceless: bool = False,
    ):
        super().__init__()
        self.out_field = out_field
        self.scalar_field = scalar_field
        self.contrib_fields = contrib_fields or []
        self.traceless = traceless
        self._init_irreps(irreps_in=irreps_in, irreps_out={})

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        pos = data[AtomicDataDict.POSITIONS_KEY]
        eye = torch.eye(3, device=pos.device, dtype=pos.dtype)

        result = None
        if self.scalar_field is not None:
            scalar = data[self.scalar_field]  # [N, 1]
            result = scalar.view(-1, 1, 1) * eye.unsqueeze(0)  # [N, 3, 3]
        for field in self.contrib_fields:
            if field in data:
                contrib = data[field]
                result = contrib if result is None else result + contrib

        if result is not None:
            if self.traceless:
                trace = result.diagonal(dim1=-1, dim2=-2).sum(-1)  # [N]
                result = result - (trace / 3).view(-1, 1, 1) * eye.unsqueeze(0)
            data[self.out_field] = result
        return data
