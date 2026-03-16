# This file is a part of the `nequip-les` package. Please see LICENSE and README at the root for information on using it.
from typing import Final
from nequip.data._key_registry import register_fields

# key definitions for nequip_les
LATENT_CHARGE_KEY: Final[str] = "LES_q"
LR_ENERGY_KEY: Final[str] = "lr_energy"
SR_ENERGY_KEY: Final[str] = "sr_energy"
BEC_KEY: Final[str] = "LES_BEC"
EDGE_LATENT_CHARGE_KEY: Final[str] = "edge_LES_q"
# dipole LES
LATENT_DIPOLE_KEY: Final[str] = "LES_u"
LATENT_POLARIZABILITY_KEY: Final[str] = "LES_alpha"
LATENT_CHEMICAL_SOFTNESS_KEY: Final[str] = "LES_kappa"
EDGE_LATENT_DIPOLE_WEIGHT_KEY: Final[str] = "edge_LES_u_weight"
EDGE_LATENT_DIPOLE_KEY: Final[str] = "edge_LES_u"
EDGE_LATENT_POLARIZABILITY_KEY: Final[str] = "edge_LES_alpha"
EDGE_LATENT_CHEMICAL_SOFTNESS_KEY: Final[str] = "edge_LES_kappa"

# key registry for nequip_les
# check nequip.data._key_registry for more information
register_fields(
    node_fields=[LATENT_CHARGE_KEY, BEC_KEY,
                 LATENT_DIPOLE_KEY, LATENT_POLARIZABILITY_KEY, 
                 LATENT_CHEMICAL_SOFTNESS_KEY],
    graph_fields=[LR_ENERGY_KEY, SR_ENERGY_KEY],
    edge_fields=[EDGE_LATENT_CHARGE_KEY, 
                 EDGE_LATENT_DIPOLE_WEIGHT_KEY, EDGE_LATENT_DIPOLE_KEY, 
                 EDGE_LATENT_POLARIZABILITY_KEY, EDGE_LATENT_CHEMICAL_SOFTNESS_KEY],
    cartesian_tensor_fields={BEC_KEY: "ij"},
)
