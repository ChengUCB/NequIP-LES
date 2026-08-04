"""
Write a copy of a training config whose model is wrapped in `nequip.model.modify`,
so a kernel acceleration can be exercised at TRAINING time.

The modifier nests the model rather than replacing it:

    model:
      _target_: nequip.model.modify
      modifiers:
        - modifier: enable_OpenEquivariance
      model:
        _target_: nequip_les.model.LESModel
        ...

which a `++`-style command-line override cannot express -- it would have to move the
whole existing model dict one level down. So generate the file instead, from the same
config the plain training runs use, and keep one source of truth.

    python wrap_modifier.py nequip_water_eager enable_OpenEquivariance out/dir
    -> out/dir/nequip_water_eager+enable_OpenEquivariance.yaml   (prints the config name)
"""

import os
import sys

import yaml


def main(tag: str, modifier: str, out_dir: str) -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "configs", f"{tag}.yaml")) as f:
        cfg = yaml.safe_load(f)

    inner = cfg["training_module"]["model"]
    if inner.get("_target_") == "nequip.model.modify":
        raise SystemExit(f"{tag} is already wrapped in a modifier")
    cfg["training_module"]["model"] = {
        "_target_": "nequip.model.modify",
        "modifiers": [{"modifier": modifier}],
        "model": inner,
    }

    name = f"{tag}+{modifier}"
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{name}.yaml"), "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return name


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    print(main(sys.argv[1], sys.argv[2], sys.argv[3]))
