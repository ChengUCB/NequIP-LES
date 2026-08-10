# Citation

If you use NequIP-LES in academic work, please cite the paper this package implements:

```bibtex
@article{Kim2025Universalb,
  title = {A Universal Augmentation Framework for Long-Range Electrostatics in Machine Learning Interatomic Potentials},
  author = {Kim, Dongjin and Wang, Xiaoyu and Vargas, Santiago and Zhong, Peichen and King, Daniel S. and Inizan, Theo Jaffrelot and Cheng, Bingqing},
  year = 2025,
  journal = {Journal of Chemical Theory and Computation},
  publisher = {American Chemical Society},
  doi = {10.1021/acs.jctc.5c01400}
}
```

## LES

The method itself, and the developments this package builds on:

```bibtex
@article{Kim2026Perspective,
  title = {Long-range electrostatics for machine learning interatomic potentials is easier than we thought},
  author = {Kim, Dongjin and Cheng, Bingqing},
  journal = {The Journal of Chemical Physics},
  volume = {164},
  number = {6},
  pages = {060901},
  year = {2026},
  doi = {10.1063/5.0316886}
}

@article{Kim2026Multipoles,
  title = {Polarizable atomic multipoles for learning long-range electrostatics},
  author = {Kim, Dongjin and King, Daniel S. and Park, Yoonjae and Savoj, Roya and Hamel, Sebastien and Wang, Xiaoyu and Cheng, Bingqing},
  journal = {arXiv preprint arXiv:2605.05746},
  year = {2026}
}

@article{cheng2025latent,
  title = {Latent Ewald summation for machine learning of long-range interactions},
  author = {Cheng, Bingqing},
  journal = {npj Computational Materials},
  volume = {11},
  number = {1},
  pages = {80},
  year = {2025},
  publisher = {Nature Publishing Group UK London}
}

@article{King2025Machine,
  title = {Machine Learning of Charges and Long-Range Interactions from Energies and Forces},
  author = {King, Daniel S. and Kim, Dongjin and Zhong, Peichen and Cheng, Bingqing},
  year = 2025,
  journal = {Nature Communications},
  volume = {16},
  number = {1},
  pages = {8763},
  publisher = {Nature Publishing Group}
}

@article{zhong2025machine,
  title = {Machine learning interatomic potential can infer electrical response},
  author = {Zhong, Peichen and Kim, Dongjin and King, Daniel S and Cheng, Bingqing},
  journal = {arXiv preprint arXiv:2504.05169},
  year = {2025}
}
```

* [Latent Ewald summation for machine learning of long-range interactions](https://www.nature.com/articles/s41524-025-01577-7) -- the method
* [Machine learning of charges and long-range interactions from energies and forces](https://www.nature.com/articles/s41467-025-63852-x) -- learning charges from energies and forces alone
* [Machine learning interatomic potential can infer electrical response](https://arxiv.org/abs/2504.05169) -- Born effective charges and dielectric response
* [A universal augmentation framework for long-range electrostatics in MLIPs](https://pubs.acs.org/doi/10.1021/acs.jctc.5c01400) -- the MLIP-agnostic formulation, and this package
* [Long-range electrostatics for MLIPs is easier than we thought](https://doi.org/10.1063/5.0316886) -- the two design principles, and how LES relates to the alternatives
* [Polarizable atomic multipoles for learning long-range electrostatics](https://arxiv.org/abs/2605.05746) -- the multipole and induced-response terms (`-u`, `-Q`, `-iq`, `-iu`) this package implements

## The models you are augmenting

LES adds a term to a NequIP or Allegro model, so please also cite whichever you used, and
`e3nn`:

* [NequIP](https://www.nature.com/articles/s41467-022-29939-5)
* [Allegro](https://www.nature.com/articles/s41467-023-36329-y)
* `e3nn` -- [preprint](https://arxiv.org/abs/2207.09453) and [code](https://github.com/e3nn/e3nn)

## License

Both LES and NequIP-LES are licensed under CC BY-NC 4.0.

## Contact

* NequIP-LES: [issues](https://github.com/ChengUCB/NequIP-LES/issues)
* LES itself: [issues](https://github.com/ChengUCB/les/issues)
* or dongjin.kim@berkeley.edu