# Usage

## Installation

```bash
git clone https://github.com/ChengUCB/les.git
cd les && pip install -e . && cd ..

git clone https://github.com/ChengUCB/NequIP-LES.git
cd NequIP-LES && pip install -e .
```

Editable installs, so the example configs and the test suite come with them. Allegro backbones
additionally need [allegro](https://github.com/mir-group/allegro).

## Turning a NequIP or Allegro config into a LES one

Replace the model target and nest the backbone's own arguments underneath it:

```yaml
training_module:
  _target_: nequip.train.EMALightningModule
  # ... loss, metrics, optimizer as in any NequIP config ...

  model:
    _target_: nequip_les.model.LESModel
    base_model: nequip          # or: allegro

    LES:
      les_args:
        is_periodic: true       # see the Ewald page -- required for compilation
        N_max: 10
        sigma: 1.0
        dl: 2.0

    # torch.compile the model for training. Omit it and you get `eager`, which is the
    # default. Compilation requires `is_periodic` to be set -- see the Ewald page.
    compile_mode: compile

    # everything below is the ordinary backbone config, unchanged
    seed: 123
    model_dtype: float32
    type_names: ${model_type_names}
    r_max: ${cutoff_radius}
    num_layers: 3
    l_max: 1
    num_features: 32
    # ...
```

Everything else -- data, trainer, loss, metrics, `nequip-train`, `nequip-compile` -- is the
NequIP framework as documented [here](https://nequip.readthedocs.io). Switching backbone is
one line: `base_model: nequip` or `base_model: allegro`.

## Running the model

| | how |
|---|---|
| ASE | `nequip-compile … --target ase`, then [`NequIPCalculator.from_compiled_model`](https://nequip.readthedocs.io/en/latest/integrations/ase.html) |
| torch-sim | `--target batch`, then [`NequIPTorchSimCalc`](https://nequip.readthedocs.io/en/latest/integrations/torchsim.html). `pip install torch-sim-atomistic` (python >= 3.11) |
| LAMMPS | `--target pair_nequip`, one MPI rank -- see [LAMMPS](lammps.md) |

## Example configs

| file | what it shows |
|---|---|
| [`configs/tutorial_les.yaml`](https://github.com/ChengUCB/NequIP-LES/blob/main/configs/tutorial_les.yaml) | minimal LES model (latent charges only) |
| [`configs/tutorial_les_extension.yaml`](https://github.com/ChengUCB/NequIP-LES/blob/main/configs/tutorial_les_extension.yaml) | dipoles, quadrupoles, polarizabilities |
| [`configs/tutorial_les_compile.yaml`](https://github.com/ChengUCB/NequIP-LES/blob/main/configs/tutorial_les_compile.yaml) | with train-time compilation |
| [`configs/test_bec_xyz_modifier.yaml`](https://github.com/ChengUCB/NequIP-LES/blob/main/configs/test_bec_xyz_modifier.yaml) | applying a model modifier (`nequip.model.modify`) |
| [`tests/configs/`](https://github.com/ChengUCB/NequIP-LES/tree/main/tests/configs) | 22 configs covering every backbone / periodicity / Ewald path combination, each complete and copy-able |
| [extended_les_fit: NequIP water](https://github.com/ChengUCB/extended_les_fit/blob/main/MLIPs/NequIP-LES/water/nequiples-uQiqiu-r-4.5-nl-3-l-1/water-les_uQiqiu_layer3_lmax1.yaml) | a real production config -- bulk water, all multipole and response terms (`uQiqiu`) |
| [extended_les_fit: Allegro water](https://github.com/ChengUCB/extended_les_fit/tree/main/MLIPs/Allegro-LES/water/allegroles-uQiqiu-r-4.5-nl-3-l-1) | the same with an Allegro backbone |

## Model modifiers

Accelerations are applied by wrapping the model, not by changing it. The LES block stays where
it is:

```yaml
  model:
    _target_: nequip.model.modify
    modifiers:
      - modifier: enable_OpenEquivariance
    model:
      _target_: nequip_les.model.LESModel
      base_model: nequip
      ...
```

Which modifiers exist, and where each can be used, is in
[What works](deployment.md#accelerations).

## Predicted charges and BECs

Latent charges are written to the output dict as `LES_q`. BECs are computed only when asked
for, since they cost an extra derivative:

```yaml
trainer:
  callbacks:
    - _target_: nequip.train.callbacks.TestTimeXYZFileWriter
      out_file: predictions/test          # no extension; the writer adds it
      extra_fields: [LES_q, LES_BEC]
      output_fields_from_original_dataset: [total_energy, forces]
      chemical_symbols: ${model_type_names}
    - _target_: nequip_les.train.callbacks.ToggleLESCallback
      compute_bec: true                   # BEC on for the test phase only
```

With `run: [train, test]` this trains and then writes predictions including BECs. Working BEC
scripts for every architecture live under
[extended_les_fit/MLIPs](https://github.com/ChengUCB/extended_les_fit/tree/main/MLIPs), in each
model's `water` directory. To run BEC
inference on an already-trained model, load it with
`nequip.model.ModelFromCheckpoint` and use `run: [test]` --
[`tests/configs/bec_water.yaml`](https://github.com/ChengUCB/NequIP-LES/blob/main/tests/configs/bec_water.yaml)
is a working example.

```{warning}
BEC inference does **not** currently work with a model trained under
`compile_mode: compile`: the compiled graph's output keys are fixed when it is first traced,
so turning BEC on afterwards produces no `LES_BEC` column. Train with `compile_mode: eager`
if you need BECs, or run the BEC pass from the checkpoint as above.
```
