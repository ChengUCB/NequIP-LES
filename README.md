## NequIP-LES

This package implements the `NequIP-LES` shown in [A universal augmentation framework for long-range electrostatics in machine learning interatomic potentials](https://arxiv.org/abs/2507.14302).

In particular, `NequIP-LES` implements the [LES library](https://github.com/ChengUCB/les) as an **extension package** for the [NequIP framework](https://github.com/mir-group/nequip).

 - [Installation](#installation)
 - [Usage](#usage)
 - [License](#license)
 - [LAMMPS Integration](#lammps-integration)
 - [Citation](#citation)
 - [Contact and questions](#contact-and-questions)

## Installation
`Nequip-LES` requires the `nequip` package. Details on `nequip` and its required PyTorch versions can be found in [the `nequip` docs](https://nequip.readthedocs.io).

`Nequip-LES` can be installed using `pip`
```bash
git clone https://github.com/ChengUCB/NequIP-LES.git
pip install -e . 
```
Installing `Nequip-LES` in this way will also install the `nequip` package from PyPI and `les` package from GitHub.

## Usage

The `Nequip-LES` package provides the Nequip-LES model for use within the [NequIP framework](https://github.com/mir-group/nequip).
[The framework's documentation](https://nequip.readthedocs.io) describes how  to train, test, and use models.

`Nequip-LES` now supports both the **[NequIP](https://github.com/mir-group/nequip)** and **[Allegro](https://github.com/mir-group/allegro)**.

You can use the `Allegro` model by changing `base_model: nequip` to `base_model: allegro` in model details.

A minimal example of a config file for training a Nequip-LES model is provided at [`configs/tutorial_les.yaml`](configs/tutorial_les.yaml).

## License
This project is licensed under the CC BY-NC 4.0 License.

## LAMMPS Integration

LAMMPS Integration has not been tested yet. 

## Citation

If you use this code in your academic work, please cite:

```text
@article{Kim2025universal,
  title = {A Universal Augmentation Framework for Long-Range Electrostatics in Machine Learning Interatomic Potentials},
  author = {Kim, Dongjin and Wang, Xiaoyu and Zhong, Peichen and King, Daniel S. and Inizan, Theo Jaffrelot and Cheng, Bingqing},
  journal={arXiv preprint arXiv:2507.14302},
  year = {2025}
```

And also consider citing:
 1. [Latent Ewald summation for machine learning of long-range interactions](https://www.nature.com/articles/s41524-025-01577-7)
    
 2. [Learning charges and long-range interactions from energies and forces](https://arxiv.org/abs/2412.15455)
    
 3. [Machine learning interatomic potential can infer electrical response](https://arxiv.org/abs/2504.05169)
 
 4. The [original NequIP paper](https://www.nature.com/articles/s41467-022-29939-5)
    
 5. The `e3nn` equivariant neural network package used by NequIP, through its [preprint](https://arxiv.org/abs/2207.09453) and/or [code](https://github.com/e3nn/e3nn)

## Contact and questions

If you find a bug or have a proposal for a feature, please post it in the [Issues](https://github.com/ChengUCB/NequIP-LES/issues)
or reach out to dongjin.kim@berkeley.edu

