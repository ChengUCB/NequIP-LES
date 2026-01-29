To get the tutorial training data, run:
```bash
python get_tutorial_data.py
```

Data source: https://github.com/ChengUCB/les_fit/tree/main/data-benchmark

To begin training, run
```bash
nequip-train -cn tutorial_les.yaml
```

There are two different ways to write xyz file with BEC inference. 
They give the same result but we leave these for as examples using `callback` and `model modifier`.

1.
To write test xyz file with BEC using `callback`, run
```bash
nequip-train -cn test_bec_xyz_callback.yaml cutoff_radius=4.5 training_module.model.checkpoint_path={best.ckpt}
```

2.
To write test xyz file with BEC using `model modifier`, run
```bash
nequip-train -cn test_bec_xyz_modifier.yaml cutoff_radius=4.5 training_module.model.model.checkpoint_path={best.ckpt}
```

This config file is an example of a NequIP LES model.

The config file is adopted from the NequIP example config file. (https://github.com/mir-group/nequip/blob/main/configs/tutorial.yaml)
The config based tutorial is meant to be used with the user guide: https://nequip.readthedocs.io/en/latest/guide/guide.html

For the Allegro LES example config file, check here [https://github.com/ChengUCB/les_fit/blob/main/MLIPs/Allegro-LES/water/allegroles-r-4.5-nl-3-l-2/lr_r45_nlayer3_lmax2.yaml]


### Compilation
For now, compilation of NequIP-LES needs to use `develop` branch for `LES`.
```
git clone https://github.com/ChengUCB/les.git
cd les
git checkout develop
pip install -e .
```

To begin training with compilation, run 
```bash
nequip-train -cn tutorial_les_compile.yaml
```
For this compilation run, please set 'is_periodic' 
to use the `torch.compile` friendly vectorized Ewald module. 

For more information about NequIP Compilation, please refer to https://nequip.readthedocs.io/en/latest/guide/accelerations/pt2_compilation.html
