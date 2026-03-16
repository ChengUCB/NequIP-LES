# This file is a part of the `nequip-les` package. Please see LICENSE and README at the root for information on using it.
import torch
from e3nn import o3
from nequip.data import AtomicDataDict
from nequip.nn import GraphModuleMixin

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