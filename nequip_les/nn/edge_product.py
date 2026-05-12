# This file is a part of the `nequip-les` package. Please see LICENSE and README at the root for information on using it.
import torch
from math import sqrt as msqrt
from e3nn import o3
from e3nn.io import CartesianTensor
from nequip.data import AtomicDataDict
from nequip.nn import GraphModuleMixin, scatter, AvgNumNeighborsNorm


class EdgeDipoleProduct(GraphModuleMixin, torch.nn.Module):
    """
    Edge module that computes the product of the latent dipole magnitude 
    and the edge direction vector to get a dipole vector for each edge.
    """
    def __init__(self, weight_field: str, attrs_field: str, out_field: str, irreps_in: dict):
        super().__init__()
        self.weight_field = weight_field
        self.attrs_field = attrs_field
        self.out_field = out_field

        self._init_irreps(
            irreps_in=irreps_in,
            irreps_out={self.out_field: o3.Irreps("1o")}
        )

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        weight = data[self.weight_field]  # shape: [n_edges, 1]
        attrs = data[self.attrs_field]    # shape: [n_edges, 4] (1x0e + 1x1o)
        vec_1o = attrs[:, 1:4] # [n_edges, 3]
        data[self.out_field] = weight * vec_1o
        return data


class EdgeOuterProduct(GraphModuleMixin, torch.nn.Module):
    """Maps edge scalar weight [E,1] × outer(edge_unit_vec, edge_unit_vec) → node [N, 3, 3].

    Uses the same scatter + AvgNumNeighborsNorm + 1/sqrt(2) normalization as EdgewiseReduce.
    If traceless=True, subtracts trace/3 * I per edge before scatter (for quadrupole).
    edge_index[0] is the center atom (receiver), matching EdgewiseReduce convention.
    """

    def __init__(
        self,
        weight_field: str,
        attrs_field: str,
        out_field: str,
        irreps_in: dict,
        avg_num_neighbors: float,
        type_names=None,
        traceless: bool = False,
    ):
        super().__init__()
        self.weight_field = weight_field
        self.attrs_field = attrs_field
        self.out_field = out_field
        self.traceless = traceless
        self._init_irreps(irreps_in=irreps_in, irreps_out={})
        self.norm_module = AvgNumNeighborsNorm(
            avg_num_neighbors=avg_num_neighbors, type_names=type_names
        )

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        weight = data[self.weight_field]          # [E, 1]
        attrs = data[self.attrs_field]            # [E, 4+] (0e + 1o + ...)
        edge_index = data[AtomicDataDict.EDGE_INDEX_KEY]
        num_nodes = AtomicDataDict.num_nodes(data)

        vec = attrs[:, 1:4]                                        # [E, 3]
        outer = torch.einsum('ea,eb->eab', vec, vec)               # [E, 3, 3]
        if self.traceless:
            trace = outer.diagonal(dim1=-1, dim2=-2).sum(-1)       # [E]
            eye = torch.eye(3, device=vec.device, dtype=vec.dtype)
            outer = outer - (trace / 3).view(-1, 1, 1) * eye
        weighted = (weight.view(-1, 1, 1) * outer).view(-1, 9)    # [E, 9]

        # scatter to center atoms — same as EdgewiseReduce
        node_flat = scatter(
            weighted, edge_index[0], dim=0,
            dim_size=num_nodes, reduce="sum",
        )                                                           # [N, 9]
        data[AtomicDataDict.NODE_FEATURES_KEY] = node_flat
        data = self.norm_module(data)
        node_flat = data[AtomicDataDict.NODE_FEATURES_KEY] / msqrt(2)

        data[self.out_field] = node_flat.view(num_nodes, 3, 3)
        return data


class EdgeSpherical2eProduct(GraphModuleMixin, torch.nn.Module):
    """Maps edge scalar weight [E,1] × (2e part of edge_attrs → Cartesian [3,3]) → node [N, 3, 3].

    Requires edge_attrs to contain a 1x2e component (l_max >= 2 in the model).
    Uses the same scatter + AvgNumNeighborsNorm + 1/sqrt(2) normalization as EdgewiseReduce.
    """

    def __init__(
        self,
        weight_field: str,
        attrs_field: str,
        out_field: str,
        irreps_in: dict,
        avg_num_neighbors: float,
        type_names=None,
    ):
        super().__init__()
        self.weight_field = weight_field
        self.attrs_field = attrs_field
        self.out_field = out_field
        self._init_irreps(irreps_in=irreps_in, irreps_out={})
        self.norm_module = AvgNumNeighborsNorm(
            avg_num_neighbors=avg_num_neighbors, type_names=type_names
        )
        cob = CartesianTensor("ij=ji").reduced_tensor_products().change_of_basis  # [6, 3, 3]
        self.register_buffer("change_of_basis_2e", cob[1:])  # [5, 3, 3]

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        weight = data[self.weight_field]          # [E, 1]
        attrs = data[self.attrs_field]            # [E, 9+] (0e + 1o + 2e + ...)
        edge_index = data[AtomicDataDict.EDGE_INDEX_KEY]
        num_nodes = AtomicDataDict.num_nodes(data)

        e2e = attrs[:, 4:9]                                        # [E, 5] spherical 2e
        cart = torch.einsum('ijk,ei->ejk', self.change_of_basis_2e, e2e)  # [E, 3, 3]
        weighted = (weight.view(-1, 1, 1) * cart).view(-1, 9)     # [E, 9]

        node_flat = scatter(
            weighted, edge_index[0], dim=0,
            dim_size=num_nodes, reduce="sum",
        )                                                           # [N, 9]
        data[AtomicDataDict.NODE_FEATURES_KEY] = node_flat
        data = self.norm_module(data)
        node_flat = data[AtomicDataDict.NODE_FEATURES_KEY] / msqrt(2)

        data[self.out_field] = node_flat.view(num_nodes, 3, 3)
        return data