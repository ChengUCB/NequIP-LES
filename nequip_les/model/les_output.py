# This file is a part of the `nequip-les` package. Please see LICENSE and README at the root for information on using it.
import math
from nequip.nn import (
    SequentialGraphNetwork,
    AtomwiseReduce,
    AtomwiseLinear,
    ScalarMLP,
    PerTypeScaleShift,
)
from nequip.data import AtomicDataDict
from allegro.nn import EdgewiseReduce
from ..nn.les import LatentEwaldSum, AddEnergy
from .. import _keys
from typing import Dict, Optional


def Add_LES_to_NequIP_model(
    model: SequentialGraphNetwork,
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
        if (
            isinstance(module, AtomwiseReduce)
            and module.out_field == AtomicDataDict.TOTAL_ENERGY_KEY
        ):
            _total_energy_readout, total_e_key = module, name
        elif (
            isinstance(module, PerTypeScaleShift)
            and module.out_field == AtomicDataDict.PER_ATOM_ENERGY_KEY
        ):
            prev_irreps_out = module.irreps_out
    
    # options to add additional readouts for dipole if specified in les_args
    if les_args is not None and les_args.get("use_dipole", False):
        # Find last convolutional layer to insert dipole readout before the layer
        last_conv_name = [name for name in dict.keys() if "convnet" in name][-1]
        last_conv_module = dict[last_conv_name]
        has_l1 = any(ir.l == 1 and ir.p == -1 for _, ir in last_conv_module.irreps_in["node_features"])
        if not has_l1:
            raise ValueError(
                f''' Current irreps_in for node features in last conv layer: {last_conv_module.irreps_in["node_features"]}.
                Adding LES dipole readout requires 1o (vector) node features in convolutional layers.'''
            )
        latent_dipole_readout = AtomwiseLinear( #TODO: consider vector MLP for dipole readout instead of single linear layer
            field=AtomicDataDict.NODE_FEATURES_KEY,
            out_field=_keys.LATENT_DIPOLE_KEY,
            irreps_in=last_conv_module.irreps_in,
            irreps_out="1x1o",
        )
        model.insert("latent_dipole_readout", latent_dipole_readout, before=last_conv_name)

    # remove original total energy readout to replace with new one that includes LES energy
    model._modules.pop(total_e_key) 

    sr_energy_sum = AtomwiseReduce(
        irreps_in=prev_irreps_out,
        reduce="sum",
        field=AtomicDataDict.PER_ATOM_ENERGY_KEY,
        out_field=_keys.SR_ENERGY_KEY,
    )
    latent_charge_readout = ScalarMLP(
        output_dim=1,
        bias=False,
        forward_weight_init=True,
        field=AtomicDataDict.NODE_FEATURES_KEY,
        out_field=_keys.LATENT_CHARGE_KEY,
        irreps_in=sr_energy_sum.irreps_out,
    )

    lr_energy_sum = LatentEwaldSum(
        irreps_in=latent_charge_readout.irreps_out,
        field=_keys.LATENT_CHARGE_KEY,
        out_field=_keys.LR_ENERGY_KEY,
        les_args=les_args,
        compute_bec=compute_bec,
        bec_output_index=bec_output_index,
    )

    total_energy_sum = AddEnergy(
        irreps_in=lr_energy_sum.irreps_out,
        field1=_keys.SR_ENERGY_KEY,
        field2=_keys.LR_ENERGY_KEY,
        out_field=AtomicDataDict.TOTAL_ENERGY_KEY,
    )

    # append new modules to the model
    model.append("sr_energy_sum", sr_energy_sum)
    model.append("latent_charge_readout", latent_charge_readout)

    # options to add additional readouts for induced charge and dipole if specified in les_args
    if les_args is not None and les_args.get("use_induced_charge", False):
        latent_kappa_readout = ScalarMLP(
            output_dim=1,
            bias=False,
            forward_weight_init=True,
            field=AtomicDataDict.NODE_FEATURES_KEY,
            out_field=_keys.LATENT_CHEMICAL_SOFTNESS_KEY,
            irreps_in=sr_energy_sum.irreps_out,
        )
        model.append("latent_kappa_readout", latent_kappa_readout)

    if les_args is not None and les_args.get("use_induced_dipole", False):
        latent_alpha_readout = ScalarMLP(
            output_dim=1,
            bias=False,
            forward_weight_init=True,
            field=AtomicDataDict.NODE_FEATURES_KEY,
            out_field=_keys.LATENT_POLARIZABILITY_KEY,
            irreps_in=sr_energy_sum.irreps_out,
        )
        model.append("latent_alpha_readout", latent_alpha_readout)

    # append LES energy modules after readouts
    model.append("lr_energy_sum", lr_energy_sum)
    model.append("total_energy_sum", total_energy_sum)

    return model


def Add_LES_to_Allegro_model(
    model: SequentialGraphNetwork,
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
        if (
            isinstance(module, AtomwiseReduce)
            and module.out_field == AtomicDataDict.TOTAL_ENERGY_KEY
        ):
            _total_energy_readout, total_e_key = module, name
        elif (
            isinstance(module, PerTypeScaleShift)
            and module.out_field == AtomicDataDict.PER_ATOM_ENERGY_KEY
        ):
            prev_irreps_out = module.irreps_out

    model._modules.pop(total_e_key)

    sr_energy_sum = AtomwiseReduce(
        irreps_in=prev_irreps_out,
        reduce="sum",
        field=AtomicDataDict.PER_ATOM_ENERGY_KEY,
        out_field=_keys.SR_ENERGY_KEY,
    )
    edge_latent_charge_readout = ScalarMLP(
        output_dim=1,
        hidden_layers_depth=1,
        hidden_layers_width=hidden_layers_width,
        # nonlinearity = None,
        bias=False,
        forward_weight_init=True,
        field=AtomicDataDict.EDGE_FEATURES_KEY,
        out_field=_keys.EDGE_LATENT_CHARGE_KEY,
        irreps_in=sr_energy_sum.irreps_out,
    )

    edge_charge_sum = EdgewiseReduce(
        field=_keys.EDGE_LATENT_CHARGE_KEY,
        out_field=_keys.LATENT_CHARGE_KEY,
        factor=1.0 / math.sqrt(2 * avg_num_neighbors),
        # ^ factor of 2 to normalize dE/dr_i which includes both contributions from dE/dr_ij and every other derivative against r_ji
        irreps_in=edge_latent_charge_readout.irreps_out,
    )

    lr_energy_sum = LatentEwaldSum(
        irreps_in=edge_charge_sum.irreps_out,
        field=_keys.LATENT_CHARGE_KEY,
        out_field=_keys.LR_ENERGY_KEY,
        les_args=les_args,
        compute_bec=compute_bec,
        bec_output_index=bec_output_index,
    )

    total_energy_sum = AddEnergy(
        irreps_in=lr_energy_sum.irreps_out,
        field1=_keys.SR_ENERGY_KEY,
        field2=_keys.LR_ENERGY_KEY,
        out_field=AtomicDataDict.TOTAL_ENERGY_KEY,
    )

    model.append("sr_energy_sum", sr_energy_sum)
    model.append("edge_latent_charge_readout", edge_latent_charge_readout)
    model.append("edge_charge_sum", edge_charge_sum)
    model.append("lr_energy_sum", lr_energy_sum)
    model.append("total_energy_sum", total_energy_sum)

    return model
