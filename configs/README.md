To get the tutorial training data,
download the data from: https://github.com/ChengUCB/les_fit/tree/main/data-benchmark

To begin training, run
```bash
nequip-train -cn tutorial_les.yaml
```

To write test xyz file with BEC, run
```bash
nequip-train -cn test_bec_xyz.yaml cutoff_radius=4.5 training_module.model.checkpoint_path={checkpoint.pth}
```

This config file is an example of a NequIP LES model.

The config file is adopted from the NequIP example config file. (https://github.com/mir-group/nequip/blob/main/configs/tutorial.yaml)
The config based tutorial is meant to be used with the user guide: https://nequip.readthedocs.io/en/latest/guide/guide.html