# This file is a part of the `nequip-les` package. Please see LICENSE and README at the root for information on using it.
from nequip.model import model_builder, NequIPGNNEnergyModel
from nequip.nn import (
    GraphModel, ForceStressOutput, SequentialGraphNetwork
)
from typing import Dict, Optional
from .les_output import Add_LES_to_model



@model_builder
def LESEnergyModel(
    LES: Optional[Dict] = None,
    **kwargs
) -> GraphModel:
    """
    Function to create a LES energy model.
    Parameters:
        les_args (Optional[Dict]): Arguments for the LES module.
        compute_bec (bool): Whether to compute the Born effective charge.
        bec_output_index (Optional[int]): Index for the Born effective charge output.
        **kwargs: Additional keyword arguments for the NequIPGNNEnergyModel.
    Returns:
        GraphModel: The LES energy model.
    """
    if LES is not None:
        les_args = LES.get("les_args", None)
        compute_bec = LES.get("compute_bec", False)
        bec_output_index = LES.get("bec_output_index", None)

    SR_Model = NequIPGNNEnergyModel(**kwargs)
    if not isinstance(SR_Model, SequentialGraphNetwork):
        raise TypeError(f"LEW Wrapper can only be applied to SequentialGraphNetwork, not {type(SR_Model)}")


    model = Add_LES_to_model(
        SR_Model, 
        les_args=les_args, 
        compute_bec=compute_bec, 
        bec_output_index=bec_output_index
    )

    return model

@model_builder
def LESModel(**kwargs) -> GraphModel:
    return ForceStressOutput(func=LESEnergyModel(**kwargs))