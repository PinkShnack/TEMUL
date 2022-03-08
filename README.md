# TEMUL
Functions for image and signal processing, based on the packages Scikit-image, Hyperspy, Atomap, PyPrismatic, ASE, periodictable etc. Written by members of the TEMUL group from the University of Limerick, Ireland (though we welcome all help, hints and contributions!)

## Documentation & Installation

```bash
pip install temul-toolkit
```

For full installation instructions and documentation, go to [temul-toolkit.readthedocs.io](https://temul-toolkit.readthedocs.io/en/latest/).

To use the vast majority of the `temul` functionality,
import it from the api module::

    import temul.api as tml


## Interactive Notebooks in the Browser

Jupyter Notebooks and MyBinder allow us to analyse data in the browser without needing any downloads using the below "launch binder" button.

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/PinkShnack/TEMUL/0.1.7)

If the button does not work, try [this link instead](https://mybinder.org/v2/gh/PinkShnack/TEMUL/0.1.7). You also share this link with others for quick access.

- To run through some code documentation tutorials, go to the "code_tutorials" folder after clicking the above button.

- To analyse data published in the listed scientific papers, go to the "publication_examples" folder after clicking the above button.

 ### Example Publications that have used the TEMUL Toolkit
 
| Publication Details   | Folder Location in TEMUL  |
| :------------------   | :-----------------------  |
| M. Hadjimichael, Y. Li *et al*, [Metal-ferroelectric supercrystals with periodically curved metallic layers](https://www.nature.com/articles/s41563-020-00864-6), Nature Materials 2021        | publication_examples/PTO_supercrystal_hadjimichael              |
| K. Moore *et al* [Highly charged 180 degree head-to-head domain walls in lead titanate](https://www.nature.com/articles/s42005-020-00488-x), Nature Communications Physics 2020          | publication_examples/PTO_Junction_moore                         |
| K. Moore *et al* [Charged Domain Wall and Polar Vortex Topologies in a Room-Temperature Magnetoelectric Multiferroic Thin Film](https://pubs.acs.org/doi/abs/10.1021/acsami.1c17383), ACS Appl. Mater. Interfaces 2022, 14, 4      |                          |



## Cite TEMUL Toolkit/TopoTEM

To cite the TEMUL toolkit in general, use the following DOI:
- [![DOI](https://www.zenodo.org/badge/203785298.svg)](https://www.zenodo.org/badge/latestdoi/203785298)

You can also cite the TopoTEM publication if you only worked with polarisation-type analysis.
- See the [TopopTEM publication here on ArXiv (soon to be released in peer-reviewed journal)](https://arxiv.org/abs/2110.00112).



## Information for developers

### Linting (PEP8)

We use flake8 to enforce coding style:

```bash
pip install -r tests/requirements.txt
flake8 temul
flake8 tests
flake8 docs
```

### Testing

Tests are carried out with `pytest`:

```bash
pip install -r tests/requirements.txt
pytest tests
```

### Documentation (docs)

To build the docs, do the following::

```bash
pip install -r docs/requirements.txt
sphinx-build docs _build  # open "index.html" in the "_build" directory
```

### Incrementing version

`temul` gets its version from the latest git tag. If you think that a
new version should be published, create a tag on the master branch
(if you have the necessary permissions to do so):

```bash
git tag -a 'x.y.z' -m 'new tag'
git push --tags <remote>
```

GitHub Actions will then automatically build source package and wheels and
publish them on PyPI.
