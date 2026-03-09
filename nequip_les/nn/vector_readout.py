import torch
from nequip.data import AtomicDataDict
from nequip.nn._graph_mixin import GraphModuleMixin

from e3nn import o3
from e3nn.nn import NormActivation
from e3nn.util.jit import compile_mode


@compile_mode("script")
class VectorMultiReadout(GraphModuleMixin, torch.nn.Module):

    def __init__(
        self, 
        irreps_in: None,
        field: str = AtomicDataDict.NODE_FEATURES_KEY,
        out_field: str = AtomicDataDict.FORCE_KEY,
        hidden_dims: list = [16],
        bias: bool = False,
    ):
        super().__init__()
        self.field = field
        self.out_field = out_field
        self._init_irreps(
            irreps_in=irreps_in,
            required_irreps_in=[self.field],
        )
        in_irreps = self.irreps_in[self.field]
        assert len(in_irreps) == 1
        assert in_irreps[0].ir == (1, -1), f"expected 1o, got {in_irreps[0].ir}"  # vectors

        if hidden_dims is None or len(hidden_dims) == 0:
            hidden_dims = []

        self.blocks = torch.nn.ModuleList()
        prev_irreps = in_irreps

        for m in hidden_dims:
            hidden_irreps = o3.Irreps(f"{m}x1o")  # hidden vector irreps

            lin = o3.Linear(prev_irreps, hidden_irreps) # (n_atoms, hidden_dim*3)

            act = NormActivation(
                irreps_in = hidden_irreps,
                scalar_nonlinearity = torch.nn.functional.silu,
                normalize = True,
                bias = bias,
            ) # (n_atoms, hidden_dim*3)
            self.blocks.append(torch.nn.ModuleDict({
                'lin': lin,
                'act': act,
            }))
            prev_irreps = hidden_irreps

        out_irreps = o3.Irreps("1x1o")  # single vector

        self.lin_out = o3.Linear(prev_irreps, out_irreps)

        self.irreps_out.update(self.irreps_in)
        self.irreps_out[self.out_field] = out_irreps

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        x = data[self.field] # (n_atoms, num_feaures*3)
        for block in self.blocks:
            x = block['lin'](x)
            x = block['act'](x)
        x = self.lin_out(x) # (n_atoms, 3)
        data[self.out_field] = x
        return data