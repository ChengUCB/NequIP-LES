## NequIP-LES

This package implements the `NequIP-LES` shown in [A universal augmentation framework for long-range electrostatics in machine learning interatomic potentials](https://pubs.acs.org/doi/10.1021/acs.jctc.5c01400).

In particular, `NequIP-LES` implements the [LES library](https://github.com/ChengUCB/les) as an **extension package** for the [NequIP framework](https://github.com/mir-group/nequip).

 - [Documentation](#Documentation)
 - [Installation](#installation)
 - [Usage](#usage)
 - [License](#license)
 - [LAMMPS Integration](#lammps-integration)
 - [Citation](#citation)
 - [Contact and questions](#contact-and-questions)

## Documentation
Please check the [full documentation](https://nequip-les.readthedocs.io)

## Installation
`Nequip-LES` requires the `nequip` and `allegro` packages. Details on `nequip` and `allegro`and their required PyTorch versions can be found in [the `nequip` docs](https://nequip.readthedocs.io).

`Nequip-LES` can be installed using `pip`
```bash
git clone https://github.com/ChengUCB/NequIP-LES.git
cd NequIP-LES
pip install -e . 
```
Installing `Nequip-LES` in this way will also install the `nequip` package from PyPI and `les` package from GitHub.

## Usage

The `Nequip-LES` package provides the Nequip-LES model for use within the [NequIP framework](https://github.com/mir-group/nequip).
[The framework's documentation](https://nequip.readthedocs.io) describes how  to train, test, and use models.

`Nequip-LES` now supports both the **[NequIP](https://github.com/mir-group/nequip)** and **[Allegro](https://github.com/mir-group/allegro)**.

A minimal example of a config file for training a Nequip-LES model is provided at [`configs/tutorial_les.yaml`](configs/tutorial_les.yaml).
A minimal example of a config file for training a Nequip-LES model with extended features (e.g., dipole, quadrupole, polarizability, etc.) is provided at [`configs/tutorial_les_extension.yaml`](configs/tutorial_les_extension.yaml).

You can use the `Allegro` model by changing `base_model: nequip` to `base_model: allegro` in model details.

## License
This project is licensed under the CC BY-NC 4.0 License.

## LAMMPS Integration

LES is now fully `torch.compile` friendly. 
For detailed usage, please check the [documentation](https://nequip-les.readthedocs.io)
LAMMPS Integration has not been tested yet. 

## Citation

If you use this code in your academic work, please cite:

```text
@article{Kim2025Universalb,
  title = {A Universal Augmentation Framework for Long-Range Electrostatics in Machine Learning Interatomic Potentials},
  author = {Kim, Dongjin and Wang, Xiaoyu and Vargas, Santiago and Zhong, Peichen and King, Daniel S. and Inizan, Theo Jaffrelot and Cheng, Bingqing},
  year = 2025,
  journal = {Journal of Chemical Theory and Computation},
  publisher = {American Chemical Society},
  doi = {10.1021/acs.jctc.5c01400}
}
```

And also consider citing:
 1. [Latent Ewald summation for machine learning of long-range interactions](https://www.nature.com/articles/s41524-025-01577-7)
    
 2. [Machine learning of charges and long-range interactions from energies and forces](https://www.nature.com/articles/s41467-025-63852-x)
    
 3. [Machine learning interatomic potential can infer electrical response](https://www.nature.com/articles/s41524-025-01911-z)

 4. [Long-range electrostatics for machine learning interatomic potentials is easier than we thought](https://pubs.aip.org/jcp/article/164/6/060901/3379367/Long-range-electrostatics-for-machine-learning)

 5. [Polarizable atomic multipoles for learning long-range electrostatics](arXiv preprint arXiv:2605.05746)

 6. The [original NequIP paper](https://www.nature.com/articles/s41467-022-29939-5)

 7. The [Allegro paper](https://www.nature.com/articles/s41467-023-36329-y)
    
 8. The `e3nn` equivariant neural network package used by NequIP, through its [preprint](https://arxiv.org/abs/2207.09453) and/or [code](https://github.com/e3nn/e3nn)

## Contact and questions

If you find a bug or have a proposal for a feature, please post it in the [Issues](https://github.com/ChengUCB/NequIP-LES/issues)
or reach out to dongjin.kim@berkeley.edu

