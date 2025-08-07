# This file is a part of the `nequip-les` package. Please see LICENSE and README at the root for information on using it.
from nequip.nn import SequentialGraphNetwork, AtomwiseReduce, ScalarMLP
from nequip.data import AtomicDataDict
from . import LatentEwaldSum, AddEnergy

from typing import Dict, Optional

def Add_LES_to_model(model: SequentialGraphNetwork,
    les_args: Optional[Dict] = None,
    compute_bec: bool = False,
    bec_output_index: Optional[int] = None,
):
    """
    Function to add LES modules to a SequentialGraphNetwork model.
    
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
            total_energy_readout = module
            total_e_key = name
            break

    prev_irreps_out = total_energy_readout.irreps_out
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