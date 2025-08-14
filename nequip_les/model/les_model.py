# This file is a part of the `nequip-les` package. Please see LICENSE and README at the root for information on using it.
from nequip.model import model_builder, NequIPGNNEnergyModel
from nequip.nn import (
    GraphModel, ForceStressOutput, SequentialGraphNetwork
)
from allegro.model import AllegroEnergyModel

from hydra.utils import instantiate
from typing import Dict, Optional
from .les_output import Add_LES_to_NequIP_model, Add_LES_to_Allegro_model




@model_builder
def LESEnergyModel(
    LES: Optional[Dict] = None,
    base_model: Optional[str] = 'nequip',  # 'nequip' or 'allegro'
    **kwargs
) -> GraphModel:
    """
    Function to create a LES energy model.
    Parameters:
        LES (Optional[Dict]): LES configuration including les_args, compute_bec, bec_output_index.
        base_model (Optional[str]): Base model type ("nequip" or "allegro"). Defaults to "nequip".
        **kwargs: Additional keyword arguments for the base model.
    Returns:
        GraphModel: The LES energy model.
    """
    if LES is not None:
        les_args = LES.get("les_args", None)
        compute_bec = LES.get("compute_bec", False)
        bec_output_index = LES.get("bec_output_index", None)

    # Create base model
    if base_model == "nequip":
        SR_Model = NequIPGNNEnergyModel(**kwargs)
        if not isinstance(SR_Model, SequentialGraphNetwork):
            raise TypeError(f"LES Wrapper can only be applied to SequentialGraphNetwork, not {type(SR_Model)}")
        
        model = Add_LES_to_NequIP_model(
            SR_Model,
            les_args=les_args,
            compute_bec=compute_bec,
            bec_output_index=bec_output_index
        )
    elif base_model == "allegro":
        if "avg_num_neighbors" not in kwargs:
            raise ValueError("avg_num_neighbors must be provided for Allegro model")
        SR_Model = AllegroEnergyModel(**kwargs)
        if not isinstance(SR_Model, SequentialGraphNetwork):
            raise TypeError(f"LES Wrapper can only be applied to SequentialGraphNetwork, not {type(SR_Model)}")
        
        model = Add_LES_to_Allegro_model(
            SR_Model,
            hidden_layers_width=kwargs['readout_mlp_hidden_layers_width'],
            avg_num_neighbors=kwargs["avg_num_neighbors"],
            les_args=les_args,
            compute_bec=compute_bec,
            bec_output_index=bec_output_index
        )
    else:
        raise ValueError(f"Unsupported base model: {base_model}")

    return model

@model_builder
def LESModel(**kwargs) -> GraphModel:
    return ForceStressOutput(func=LESEnergyModel(**kwargs))