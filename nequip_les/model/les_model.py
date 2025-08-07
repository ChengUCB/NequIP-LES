# This file is a part of the `nequip-les` package. Please see LICENSE and README at the root for information on using it.
from nequip.model import model_builder, NequIPGNNEnergyModel
from nequip.nn import (
    GraphModel, ForceStressOutput, SequentialGraphNetwork
)
from typing import Dict, Optional
from ..nn import Add_LES_to_model



@model_builder
def LESEnergyModel(
    use_les: bool = True,
    les_args: Optional[Dict] = None,
    compute_bec: bool = False,
    bec_output_index: Optional[int] = None,
    **kwargs
) -> GraphModel:
    SR_Model = NequIPGNNEnergyModel(**kwargs)
    if use_les:
        if not isinstance(SR_Model, SequentialGraphNetwork):
            raise TypeError(f"LEW Wrapper can only be applied to SequentialGraphNetwork, not {type(SR_Model)}")

        model = Add_LES_to_model(
            SR_Model, 
            les_args=les_args, 
            compute_bec=compute_bec, 
            bec_output_index=bec_output_index
        )
        if bec_output_index is not None:
            print('!'*50, type(bec_output_index))
    else:
        model = SR_Model
    return model

@model_builder
def LESModel(**kwargs) -> GraphModel:
    return ForceStressOutput(func=LESEnergyModel(**kwargs))