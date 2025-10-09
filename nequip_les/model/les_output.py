# This file is a part of the `nequip-les` package. Please see LICENSE and README at the root for information on using it.
import math
from nequip.nn import SequentialGraphNetwork, AtomwiseReduce, ScalarMLP, PerTypeScaleShift
from nequip.data import AtomicDataDict
from allegro.nn import EdgewiseReduce
from ..nn.les import LatentEwaldSum, AddEnergy

from typing import Dict, Optional

def Add_LES_to_NequIP_model(model: SequentialGraphNetwork,
    les_args: Optional[Dict] = None,
    compute_bec: bool = False,
    bec_output_index: Optional[int] = None,
):
    """
    Function to add LES modules to a NequIP model.
    
    Parameters:
        model (SequentialGraphNetwork): The model to which LES will be added.
        les_args (Optional[Dict]): Arguments for the LES module.
        compute_bec (bool): Whether to compute the Born effective charge.
        bec_output_index (Optional[int]): Index for the Born effective charge output.
    
    Returns:
        SequentialGraphNetwork: The modified model with LES modules added.
    """
    # Implementation to add LES to the model
    dict = model._modules
    for name, module in dict.items():
        if (isinstance(module, AtomwiseReduce) 
            and module.out_field == AtomicDataDict.TOTAL_ENERGY_KEY):
            total_energy_readout, total_e_key = module, name
        elif (isinstance(module, PerTypeScaleShift)
            and module.out_field == AtomicDataDict.PER_ATOM_ENERGY_KEY):
            prev_irreps_out = module.irreps_out

    model._modules.pop(total_e_key)

    sr_energy_sum = AtomwiseReduce(
            irreps_in = prev_irreps_out,
            reduce = "sum",
            field = AtomicDataDict.PER_ATOM_ENERGY_KEY,
            out_field = AtomicDataDict.SR_ENERGY_KEY,
        )
    latent_charge_readout = ScalarMLP(
            output_dim = 1,
            bias = False,
            forward_weight_init=True,
            field = AtomicDataDict.NODE_FEATURES_KEY,
            out_field = AtomicDataDict.LATENT_CHARGE_KEY,
            irreps_in = sr_energy_sum.irreps_out,
        )

    lr_energy_sum = LatentEwaldSum(
            irreps_in = latent_charge_readout.irreps_out,
            field = AtomicDataDict.LATENT_CHARGE_KEY,
            out_field = AtomicDataDict.LR_ENERGY_KEY,
            les_args = les_args,
            compute_bec = compute_bec,
            bec_output_index= bec_output_index
        )

    total_energy_sum = AddEnergy(
            irreps_in = lr_energy_sum.irreps_out,
            field1 = AtomicDataDict.SR_ENERGY_KEY,
            field2 = AtomicDataDict.LR_ENERGY_KEY,
            out_field = AtomicDataDict.TOTAL_ENERGY_KEY,
        )

    model.append('sr_energy_sum', sr_energy_sum)
    model.append('latent_charge_readout', latent_charge_readout)
    model.append('lr_energy_sum', lr_energy_sum)
    model.append('total_energy_sum', total_energy_sum)

    return model

def Add_LES_to_Allegro_model(model: SequentialGraphNetwork,
    avg_num_neighbors: float,
    hidden_layers_width: int,
    les_args: Optional[Dict] = None,
    compute_bec: bool = False,
    bec_output_index: Optional[int] = None,
):
    """
    Function to add LES modules to a Allegro model.
    """
    # Implementation to add LES to the model
    dict = model._modules
    for name, module in dict.items():
        if (isinstance(module, AtomwiseReduce) 
            and module.out_field == AtomicDataDict.TOTAL_ENERGY_KEY):
            total_energy_readout, total_e_key = module, name
        elif (isinstance(module, PerTypeScaleShift)
            and module.out_field == AtomicDataDict.PER_ATOM_ENERGY_KEY):
            prev_irreps_out = module.irreps_out

    model._modules.pop(total_e_key)

    sr_energy_sum = AtomwiseReduce(
            irreps_in = prev_irreps_out,
            reduce = "sum",
            field = AtomicDataDict.PER_ATOM_ENERGY_KEY,
            out_field = AtomicDataDict.SR_ENERGY_KEY,
        )
    edge_latent_charge_readout = ScalarMLP(
            output_dim = 1,
            hidden_layers_depth = 1,
            hidden_layers_width = hidden_layers_width,
            # nonlinearity = None,
            bias = False,
            forward_weight_init=True,
            field = AtomicDataDict.EDGE_FEATURES_KEY,
            out_field = AtomicDataDict.EDGE_LATENT_CHARGE_KEY,
            irreps_in = sr_energy_sum.irreps_out,
        )
    
    edge_charge_sum = EdgewiseReduce(
            field = AtomicDataDict.EDGE_LATENT_CHARGE_KEY,
            out_field = AtomicDataDict.LATENT_CHARGE_KEY,
            factor = 1.0 / math.sqrt(2 * avg_num_neighbors),
            # ^ factor of 2 to normalize dE/dr_i which includes both contributions from dE/dr_ij and every other derivative against r_ji
            irreps_in=edge_latent_charge_readout.irreps_out,
        )
    
    lr_energy_sum = LatentEwaldSum(
            irreps_in = edge_charge_sum.irreps_out,
            field = AtomicDataDict.LATENT_CHARGE_KEY,
            out_field = AtomicDataDict.LR_ENERGY_KEY,
            les_args = les_args,
            compute_bec = compute_bec,
            bec_output_index= bec_output_index
        )

    total_energy_sum = AddEnergy(
            irreps_in = lr_energy_sum.irreps_out,
            field1 = AtomicDataDict.SR_ENERGY_KEY,
            field2 = AtomicDataDict.LR_ENERGY_KEY,
            out_field = AtomicDataDict.TOTAL_ENERGY_KEY,
        )

    model.append('sr_energy_sum', sr_energy_sum)
    model.append('edge_latent_charge_readout', edge_latent_charge_readout)
    model.append('edge_charge_sum', edge_charge_sum)
    model.append('lr_energy_sum', lr_energy_sum)
    model.append('total_energy_sum', total_energy_sum)

    return model