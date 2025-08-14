# This file is a part of the `nequip-les` package. Please see LICENSE and README at the root for information on using it.
from typing import Final
from nequip.data._key_registry import register_fields
from nequip.data import AtomicDataDict

#key definitions for nequip_les
LATENT_CHARGE_KEY: Final[str] = "LES_q"
LR_ENERGY_KEY: Final[str] = "lr_energy"
SR_ENERGY_KEY: Final[str] = "sr_energy"
BEC_KEY: Final[str] = "LES_BEC"
EDGE_LATENT_CHARGE_KEY: Final[str] = "edge_LES_q"


#key registry for nequip_les
#check nequip.data._key_registry for more information
register_fields(
    node_fields=[
        LATENT_CHARGE_KEY, 
        BEC_KEY
    ],
    graph_fields=[
        LR_ENERGY_KEY, 

        SR_ENERGY_KEY
    ],
    edge_fields=[
        EDGE_LATENT_CHARGE_KEY
    ],
    cartesian_tensor_fields={
        BEC_KEY: "ij" 
    }
)

setattr(AtomicDataDict, 'LATENT_CHARGE_KEY', LATENT_CHARGE_KEY)
setattr(AtomicDataDict, 'LR_ENERGY_KEY', LR_ENERGY_KEY)
setattr(AtomicDataDict, 'SR_ENERGY_KEY', SR_ENERGY_KEY)
setattr(AtomicDataDict, 'BEC_KEY', BEC_KEY)
setattr(AtomicDataDict, 'EDGE_LATENT_CHARGE_KEY', EDGE_LATENT_CHARGE_KEY)