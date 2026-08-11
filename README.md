## NequIP-LES

This package implements the `NequIP-LES` shown in [A universal augmentation framework for long-range electrostatics in machine learning interatomic potentials](https://pubs.acs.org/doi/10.1021/acs.jctc.5c01400).

In particular, `NequIP-LES` implements the [LES library](https://github.com/ChengUCB/les) as an **extension package** for the [NequIP framework](https://github.com/mir-group/nequip), and supports both **[NequIP](https://github.com/mir-group/nequip)** and **[Allegro](https://github.com/mir-group/allegro)** backbones.

## Documentation

**[nequip-les.readthedocs.io](https://nequip-les.readthedocs.io)** — usage, the `les_args`
reference, what can and cannot be compiled or deployed, LAMMPS, and the test suite.

For now, please use [`develop_dipole` branch](https://github.com/ChengUCB/les/tree/develop_dipole) of `les` to be fully compatible with `torch.compile`.

## Installation

```bash
git clone https://github.com/ChengUCB/les.git
cd les && pip install -e . && cd ..

git clone https://github.com/ChengUCB/NequIP-LES.git
cd NequIP-LES && pip install -e .
```

Allegro backbones additionally need [allegro](https://github.com/mir-group/allegro). Details
and example configs: [Usage](https://nequip-les.readthedocs.io/en/latest/guide/usage.html).

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

Related papers and the other entries to consider:
[Citation](https://nequip-les.readthedocs.io/en/latest/citation.html).

## License

This project is licensed under the CC BY-NC 4.0 License.

## Contact and questions

If you find a bug or have a proposal for a feature, please post it in the [Issues](https://github.com/ChengUCB/NequIP-LES/issues)
or reach out to dongjin.kim@berkeley.edu
