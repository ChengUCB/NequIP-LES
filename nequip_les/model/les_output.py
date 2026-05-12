# This file is a part of the `nequip-les` package. Please see LICENSE and README at the root for information on using it.
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
from ..nn.edge_product import EdgeDipoleProduct, EdgeOuterProduct, EdgeSpherical2eProduct
from ..nn.node_product import NodeOuterProduct, NodeSpherical2eToCartesian, NodeAssembleTensor
from .. import _keys
from typing import Dict, Optional, Union, Sequence


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
    modules = model._modules
    for name, module in modules.items():
        if (
            isinstance(module, AtomwiseReduce)
            and module.out_field == AtomicDataDict.TOTAL_ENERGY_KEY
        ):
            total_e_key = name
        elif (
            isinstance(module, PerTypeScaleShift)
            and module.out_field == AtomicDataDict.PER_ATOM_ENERGY_KEY
        ):
            prev_irreps_out = module.irreps_out

    use_dipole = les_args is not None and les_args.get("use_dipole", False)
    use_quad = les_args is not None and les_args.get("use_quadrupole", False)
    use_aniso_alpha = les_args is not None and les_args.get("use_anisotropic_polarizability", False)
    alpha_irreps = les_args.get("alpha_irreps", "0e+1o+2e") if les_args else "0e"

    # Insert feature-space readouts before the last convolutional layer
    if use_dipole or use_quad or use_aniso_alpha:
        last_conv_name = [name for name in modules.keys() if "convnet" in name][-1]
        last_conv_module = modules[last_conv_name]
        node_features_irreps = last_conv_module.irreps_in["node_features"]
        has_l1 = any(ir.l == 1 and ir.p == -1 for _, ir in node_features_irreps)
        has_l2 = any(ir.l == 2 and ir.p == 1  for _, ir in node_features_irreps)

        if use_dipole:
            if not has_l1:
                raise ValueError(
                    f'LES dipole readout requires 1o node features. Got: {node_features_irreps}'
                )
            latent_dipole_readout = AtomwiseLinear(
                field=AtomicDataDict.NODE_FEATURES_KEY,
                out_field=_keys.LATENT_DIPOLE_KEY,
                irreps_in=last_conv_module.irreps_in,
                irreps_out="1x1o",
            )
            model.insert("latent_dipole_readout", latent_dipole_readout, before=last_conv_name)
            print('*** USE_DIPOLE ***')

        if use_quad:
            if not (has_l1 or has_l2):
                raise ValueError(
                    f'use_quadrupole=True requires 1o or 2e node features, but got: {node_features_irreps}. '
                    f'Set l_max >= 1 and parity=True.'
                )
            quad_contrib_fields = []
            if has_l1:
                quad_1o_readout = AtomwiseLinear(
                    field=AtomicDataDict.NODE_FEATURES_KEY,
                    out_field=_keys._LATENT_QUAD_1O_VEC_KEY,
                    irreps_in=last_conv_module.irreps_in,
                    irreps_out="1x1o",
                )
                quad_1o_outer = NodeOuterProduct(
                    in_field=_keys._LATENT_QUAD_1O_VEC_KEY,
                    out_field=_keys.LATENT_QUAD_1O_KEY,
                    irreps_in=quad_1o_readout.irreps_out,
                    traceless=True,
                )
                model.insert("quad_1o_readout", quad_1o_readout, before=last_conv_name)
                model.insert("quad_1o_outer", quad_1o_outer, before=last_conv_name)
                quad_contrib_fields.append(_keys.LATENT_QUAD_1O_KEY)
                print('*** USE_QUADRUPOLE (1o) ***')
            if has_l2:
                quad_2e_readout = AtomwiseLinear(
                    field=AtomicDataDict.NODE_FEATURES_KEY,
                    out_field=_keys._LATENT_QUAD_2E_SPH_KEY,
                    irreps_in=last_conv_module.irreps_in,
                    irreps_out="1x2e",
                )
                quad_2e_cart = NodeSpherical2eToCartesian(
                    in_field=_keys._LATENT_QUAD_2E_SPH_KEY,
                    out_field=_keys.LATENT_QUAD_2E_KEY,
                    irreps_in=quad_2e_readout.irreps_out,
                )
                model.insert("quad_2e_readout", quad_2e_readout, before=last_conv_name)
                model.insert("quad_2e_cart", quad_2e_cart, before=last_conv_name)
                quad_contrib_fields.append(_keys.LATENT_QUAD_2E_KEY)
                print('*** USE_QUADRUPOLE (2e) ***')

        if use_aniso_alpha:
            if not les_args.get("use_induced_dipole", False):
                raise ValueError("use_anisotropic_polarizability requires use_induced_dipole=True")
            if "1o" not in alpha_irreps and "2e" not in alpha_irreps:
                raise ValueError(f'Set alpha_irreps to include 1o or 2e (e.g. "0e+1o+2e" or "1o+2e" or "1o").')
            if not (has_l1 or has_l2):
                raise ValueError(
                    f'use_anisotropic_polarizability=True requires 1o or 2e node features, but got: {node_features_irreps}. '
                    f'Set l_max >= 1 and parity=True.'
                )
            alpha_contrib_fields = []
            if has_l1 and "1o" in alpha_irreps:
                alpha_1o_readout = AtomwiseLinear(
                    field=AtomicDataDict.NODE_FEATURES_KEY,
                    out_field=_keys._LATENT_ALPHA_1O_VEC_KEY,
                    irreps_in=last_conv_module.irreps_in,
                    irreps_out="1x1o",
                )
                alpha_1o_outer = NodeOuterProduct(
                    in_field=_keys._LATENT_ALPHA_1O_VEC_KEY,
                    out_field=_keys.LATENT_ANISO_ALPHA_1O_KEY,
                    irreps_in=alpha_1o_readout.irreps_out,
                    traceless=False,
                )
                model.insert("alpha_1o_readout", alpha_1o_readout, before=last_conv_name)
                model.insert("alpha_1o_outer", alpha_1o_outer, before=last_conv_name)
                alpha_contrib_fields.append(_keys.LATENT_ANISO_ALPHA_1O_KEY)
                print('*** USE_ANISOTROPIC_POLARIZABILITY (1o) ***')
            if has_l2 and "2e" in alpha_irreps:
                alpha_2e_readout = AtomwiseLinear(
                    field=AtomicDataDict.NODE_FEATURES_KEY,
                    out_field=_keys._LATENT_ALPHA_2E_SPH_KEY,
                    irreps_in=last_conv_module.irreps_in,
                    irreps_out="1x2e",
                )
                alpha_2e_cart = NodeSpherical2eToCartesian(
                    in_field=_keys._LATENT_ALPHA_2E_SPH_KEY,
                    out_field=_keys.LATENT_ANISO_ALPHA_2E_KEY,
                    irreps_in=alpha_2e_readout.irreps_out,
                )
                model.insert("alpha_2e_readout", alpha_2e_readout, before=last_conv_name)
                model.insert("alpha_2e_cart", alpha_2e_cart, before=last_conv_name)
                alpha_contrib_fields.append(_keys.LATENT_ANISO_ALPHA_2E_KEY)
                print('*** USE_ANISOTROPIC_POLARIZABILITY (2e) ***')

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

    # append scalar readouts
    model.append("sr_energy_sum", sr_energy_sum)
    model.append("latent_charge_readout", latent_charge_readout)

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
        print('*** USE_INDUCED_CHARGE ***')

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
        print('*** USE_INDUCED_DIPOLE ***')

    # assemble final tensor fields from contributions computed above
    if use_quad and quad_contrib_fields:
        quad_assemble = NodeAssembleTensor(
            out_field=_keys.LATENT_QUAD_KEY,
            irreps_in=latent_charge_readout.irreps_out,
            scalar_field=None,
            contrib_fields=quad_contrib_fields,
            traceless=True,
        )
        model.append("quad_assemble", quad_assemble)

    if use_aniso_alpha:
        if not alpha_contrib_fields:
            raise ValueError(
                f"use_anisotropic_polarizability=True but no tensor contributions could be built. "
                f"alpha_irreps={alpha_irreps!r}, has_l1={has_l1}, has_l2={has_l2}. "
                f"Check that alpha_irreps matches the available model features."
            )
        alpha_assemble = NodeAssembleTensor(
            out_field=_keys.LATENT_POLARIZABILITY_KEY,
            irreps_in=latent_charge_readout.irreps_out,
            scalar_field=_keys.LATENT_POLARIZABILITY_KEY,
            contrib_fields=alpha_contrib_fields,
            traceless=False,
        )
        model.append("alpha_assemble", alpha_assemble)

    # append LES energy modules after readouts
    model.append("lr_energy_sum", lr_energy_sum)
    model.append("total_energy_sum", total_energy_sum)

    return model


def Add_LES_to_Allegro_model(
    model: SequentialGraphNetwork,
    avg_num_neighbors: float,
    hidden_layers_width: Union[float, Dict[str, float]],
    type_names: Sequence[str],
    les_args: Optional[Dict] = None,
    compute_bec: bool = False,
    bec_output_index: Optional[int] = None,
):
    """
    Function to add LES modules to a Allegro model.
    """
    modules = model._modules
    for name, module in modules.items():
        if (
            isinstance(module, AtomwiseReduce)
            and module.out_field == AtomicDataDict.TOTAL_ENERGY_KEY
        ):
            total_e_key = name
        elif (
            isinstance(module, PerTypeScaleShift)
            and module.out_field == AtomicDataDict.PER_ATOM_ENERGY_KEY
        ):
            prev_irreps_out = module.irreps_out

    use_dipole = les_args is not None and les_args.get("use_dipole", False)
    use_quad = les_args is not None and les_args.get("use_quadrupole", False)
    use_aniso_alpha = les_args is not None and les_args.get("use_anisotropic_polarizability", False)
    alpha_irreps = les_args.get("alpha_irreps", "0e+1o+2e") if les_args else "0e"

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
        avg_num_neighbors=avg_num_neighbors,
        type_names=type_names,
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

    # options to add additional readouts for dipole if specified in les_args
    if use_dipole:
        # weights for dipole readout on edges
        edge_latent_dipole_weight_readout = ScalarMLP(
            output_dim=1,
            hidden_layers_depth=1,
            hidden_layers_width=hidden_layers_width,
            bias=False,
            forward_weight_init=True,
            field=AtomicDataDict.EDGE_FEATURES_KEY,
            out_field=_keys.EDGE_LATENT_DIPOLE_WEIGHT_KEY,
            irreps_in=sr_energy_sum.irreps_out,
        )
        # produce dipole vector for each edge 
        edge_dipole_product = EdgeDipoleProduct(
            weight_field=_keys.EDGE_LATENT_DIPOLE_WEIGHT_KEY,
            attrs_field=AtomicDataDict.EDGE_ATTRS_KEY,
            out_field=_keys.EDGE_LATENT_DIPOLE_KEY,
            irreps_in=edge_latent_dipole_weight_readout.irreps_out,
        )
        edge_dipole_sum = EdgewiseReduce(
            field=_keys.EDGE_LATENT_DIPOLE_KEY,
            out_field=_keys.LATENT_DIPOLE_KEY,
            avg_num_neighbors=avg_num_neighbors,
            type_names=type_names,
            irreps_in=edge_dipole_product.irreps_out,
        )
        model.append("edge_latent_dipole_weight_readout", edge_latent_dipole_weight_readout)
        model.append("edge_dipole_product", edge_dipole_product)
        model.append("edge_dipole_sum", edge_dipole_sum)
        print('*** USE_DIPOLE ***')
    
    # options to add additional readouts for induced charge and dipole if specified in les_args
    if les_args is not None and les_args.get("use_induced_charge", False):
        edge_latent_kappa_readout = ScalarMLP(
            output_dim=1,
            hidden_layers_depth=1,
            hidden_layers_width=hidden_layers_width,
            bias=False,
            forward_weight_init=True,
            field=AtomicDataDict.EDGE_FEATURES_KEY,
            out_field=_keys.EDGE_LATENT_CHEMICAL_SOFTNESS_KEY,
            irreps_in=sr_energy_sum.irreps_out,
        )
        edge_kappa_sum = EdgewiseReduce(
            field=_keys.EDGE_LATENT_CHEMICAL_SOFTNESS_KEY,
            out_field=_keys.LATENT_CHEMICAL_SOFTNESS_KEY,
            avg_num_neighbors=avg_num_neighbors,
            type_names=type_names,
            irreps_in=edge_latent_kappa_readout.irreps_out,
        )
        model.append("edge_latent_kappa_readout", edge_latent_kappa_readout)
        model.append("edge_kappa_sum", edge_kappa_sum)
        print('*** USE_INDUCED_CHARGE ***')

    if les_args is not None and les_args.get("use_induced_dipole", False):
        edge_latent_alpha_readout = ScalarMLP(
            output_dim=1,
            hidden_layers_depth=1,
            hidden_layers_width=hidden_layers_width,
            bias=False,
            forward_weight_init=True,
            field=AtomicDataDict.EDGE_FEATURES_KEY,
            out_field=_keys.EDGE_LATENT_POLARIZABILITY_KEY,
            irreps_in=sr_energy_sum.irreps_out,
        )
        edge_alpha_sum = EdgewiseReduce(
            field=_keys.EDGE_LATENT_POLARIZABILITY_KEY,
            out_field=_keys.LATENT_POLARIZABILITY_KEY,
            avg_num_neighbors=avg_num_neighbors,
            type_names=type_names,
            irreps_in=edge_latent_alpha_readout.irreps_out,
        )
        model.append("edge_latent_alpha_readout", edge_latent_alpha_readout)
        model.append("edge_alpha_sum", edge_alpha_sum)
        print('*** USE_INDUCED_DIPOLE ***')

    if use_quad or use_aniso_alpha:
        # Detect 1o/2e availability from edge_attrs irreps
        edge_attrs_irreps = None
        for _, module in modules.items():
            irreps_out = getattr(module, 'irreps_out', {})
            if AtomicDataDict.EDGE_ATTRS_KEY in irreps_out:
                edge_attrs_irreps = irreps_out[AtomicDataDict.EDGE_ATTRS_KEY]
                break
        if edge_attrs_irreps is not None:
            has_l1 = any(ir.l == 1 and ir.p == -1 for _, ir in edge_attrs_irreps)
            has_l2 = any(ir.l == 2 and ir.p == 1  for _, ir in edge_attrs_irreps)
        else:
            l_max = les_args.get("l_max", 1) if les_args else 1
            has_l1 = l_max >= 1
            has_l2 = l_max >= 2

        if use_quad:
            if not (has_l1 or has_l2):
                raise ValueError(
                    'use_quadrupole=True requires l_max >= 1 in the Allegro model '
                    '(edge_attrs must contain 1o or 2e components).'
                )
            quad_contrib_fields = []
            if has_l1:
                edge_quad_1o_weight = ScalarMLP(
                    output_dim=1,
                    hidden_layers_depth=1,
                    hidden_layers_width=hidden_layers_width,
                    bias=False,
                    forward_weight_init=True,
                    field=AtomicDataDict.EDGE_FEATURES_KEY,
                    out_field=_keys.EDGE_QUAD_1O_WEIGHT_KEY,
                    irreps_in=sr_energy_sum.irreps_out,
                )
                edge_quad_1o_outer = EdgeOuterProduct(
                    weight_field=_keys.EDGE_QUAD_1O_WEIGHT_KEY,
                    attrs_field=AtomicDataDict.EDGE_ATTRS_KEY,
                    out_field=_keys.LATENT_QUAD_1O_KEY,
                    irreps_in=edge_quad_1o_weight.irreps_out,
                    avg_num_neighbors=avg_num_neighbors,
                    type_names=type_names,
                    traceless=True,
                )
                model.append("edge_quad_1o_weight", edge_quad_1o_weight)
                model.append("edge_quad_1o_outer", edge_quad_1o_outer)
                quad_contrib_fields.append(_keys.LATENT_QUAD_1O_KEY)
                print('*** USE_QUADRUPOLE (1o) ***')
            if has_l2:
                edge_quad_2e_weight = ScalarMLP(
                    output_dim=1,
                    hidden_layers_depth=1,
                    hidden_layers_width=hidden_layers_width,
                    bias=False,
                    forward_weight_init=True,
                    field=AtomicDataDict.EDGE_FEATURES_KEY,
                    out_field=_keys.EDGE_QUAD_2E_WEIGHT_KEY,
                    irreps_in=sr_energy_sum.irreps_out,
                )
                edge_quad_2e_outer = EdgeSpherical2eProduct(
                    weight_field=_keys.EDGE_QUAD_2E_WEIGHT_KEY,
                    attrs_field=AtomicDataDict.EDGE_ATTRS_KEY,
                    out_field=_keys.LATENT_QUAD_2E_KEY,
                    irreps_in=edge_quad_2e_weight.irreps_out,
                    avg_num_neighbors=avg_num_neighbors,
                    type_names=type_names,
                )
                model.append("edge_quad_2e_weight", edge_quad_2e_weight)
                model.append("edge_quad_2e_outer", edge_quad_2e_outer)
                quad_contrib_fields.append(_keys.LATENT_QUAD_2E_KEY)
                print('*** USE_QUADRUPOLE (2e) ***')
            quad_assemble = NodeAssembleTensor(
                out_field=_keys.LATENT_QUAD_KEY,
                irreps_in=edge_charge_sum.irreps_out,
                scalar_field=None,
                contrib_fields=quad_contrib_fields,
                traceless=True,
            )
            model.append("quad_assemble", quad_assemble)

        if use_aniso_alpha:
            if not les_args.get("use_induced_dipole", False):
                raise ValueError("use_anisotropic_polarizability requires use_induced_dipole=True")
            if not (has_l1 or has_l2):
                raise ValueError(
                    'use_anisotropic_polarizability=True requires l_max >= 1 in the Allegro model.'
                )
            if "1o" not in alpha_irreps and "2e" not in alpha_irreps:
                raise ValueError(f'Set alpha_irreps to include 1o or 2e (e.g. "0e+1o+2e" or "1o"). Got: {alpha_irreps!r}')
            alpha_contrib_fields = []
            if has_l1 and "1o" in alpha_irreps:
                edge_alpha_1o_weight = ScalarMLP(
                    output_dim=1,
                    hidden_layers_depth=1,
                    hidden_layers_width=hidden_layers_width,
                    bias=False,
                    forward_weight_init=True,
                    field=AtomicDataDict.EDGE_FEATURES_KEY,
                    out_field=_keys.EDGE_ANISO_ALPHA_1O_WEIGHT_KEY,
                    irreps_in=sr_energy_sum.irreps_out,
                )
                edge_alpha_1o_outer = EdgeOuterProduct(
                    weight_field=_keys.EDGE_ANISO_ALPHA_1O_WEIGHT_KEY,
                    attrs_field=AtomicDataDict.EDGE_ATTRS_KEY,
                    out_field=_keys.LATENT_ANISO_ALPHA_1O_KEY,
                    irreps_in=edge_alpha_1o_weight.irreps_out,
                    avg_num_neighbors=avg_num_neighbors,
                    type_names=type_names,
                    traceless=False,
                )
                model.append("edge_alpha_1o_weight", edge_alpha_1o_weight)
                model.append("edge_alpha_1o_outer", edge_alpha_1o_outer)
                alpha_contrib_fields.append(_keys.LATENT_ANISO_ALPHA_1O_KEY)
                print('*** USE_ANISOTROPIC_POLARIZABILITY (1o) ***')
            if has_l2 and "2e" in alpha_irreps:
                edge_alpha_2e_weight = ScalarMLP(
                    output_dim=1,
                    hidden_layers_depth=1,
                    hidden_layers_width=hidden_layers_width,
                    bias=False,
                    forward_weight_init=True,
                    field=AtomicDataDict.EDGE_FEATURES_KEY,
                    out_field=_keys.EDGE_ANISO_ALPHA_2E_WEIGHT_KEY,
                    irreps_in=sr_energy_sum.irreps_out,
                )
                edge_alpha_2e_outer = EdgeSpherical2eProduct(
                    weight_field=_keys.EDGE_ANISO_ALPHA_2E_WEIGHT_KEY,
                    attrs_field=AtomicDataDict.EDGE_ATTRS_KEY,
                    out_field=_keys.LATENT_ANISO_ALPHA_2E_KEY,
                    irreps_in=edge_alpha_2e_weight.irreps_out,
                    avg_num_neighbors=avg_num_neighbors,
                    type_names=type_names,
                )
                model.append("edge_alpha_2e_weight", edge_alpha_2e_weight)
                model.append("edge_alpha_2e_outer", edge_alpha_2e_outer)
                alpha_contrib_fields.append(_keys.LATENT_ANISO_ALPHA_2E_KEY)
                print('*** USE_ANISOTROPIC_POLARIZABILITY (2e) ***')
            if not alpha_contrib_fields:
                raise ValueError(
                    f"use_anisotropic_polarizability=True but no tensor contributions could be built. "
                    f"alpha_irreps={alpha_irreps!r}, has_l1={has_l1}, has_l2={has_l2}. "
                    f"Check that alpha_irreps matches the available model features."
                )
            alpha_assemble = NodeAssembleTensor(
                out_field=_keys.LATENT_POLARIZABILITY_KEY,
                irreps_in=edge_charge_sum.irreps_out,
                scalar_field=_keys.LATENT_POLARIZABILITY_KEY,
                contrib_fields=alpha_contrib_fields,
                traceless=False,
            )
            model.append("alpha_assemble", alpha_assemble)

    model.append("lr_energy_sum", lr_energy_sum)
    model.append("total_energy_sum", total_energy_sum)

    return model
