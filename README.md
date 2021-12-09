# NeuroModels
[![PyPi version](https://img.shields.io/pypi/v/neuromodels.svg)](https://pypi.python.org/pypi/neuromodels)
[![python compatibility](https://img.shields.io/pypi/pyversions/neuromodels.svg)](https://pypi.python.org/pypi/neuromodels)
[![GitHub license](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/nicolossus/neuromodels/blob/master/LICENSE)
<!--[![Documentation Status](https://readthedocs.org/projects/neuromodels/badge/?version=latest)](https://neuromodels.readthedocs.io/en/latest/?badge=latest)
[![Tests](https://github.com/nicolossus/neuromodels/workflows/Tests/badge.svg?branch=main)](https://github.com/nicolossus/neuromodels/actions)-->


`NeuroModels` is a Python toolbox for simulating neuroscientific models, post-simulation analysis and feature extraction.

## Overview
`NeuroModels` is a software that aims to provide a framework for neuroscientific simulator models and methods for extracting features from the simulated recording. The simulator models are implemented to be flexible, particularly in the sense of parameterizing the models.

`NeuroModels` presently includes the following neural simulators:

* Hodgkin-Huxley Model
* Brunel Network Model

`NeuroModels` is a part of the author's [Master thesis](https://github.com/nicolossus/Master-thesis).

## Installation instructions

### Install with pip
`NeuroModels` can be installed directly from [PyPI](https://pypi.org/project/neuromodels/):

```
pip install neuromodels
```

## Requirements
* `Python` >= 3.8
* The standard scientific libraries; `NumPy`, `Matplotlib`, `SciPy`

## Documentation
Documentation can be found at [neuromodels.readthedocs.io](https://neuromodels.readthedocs.io/).

## Getting started
Check out the [Examples gallery](https://neuromodels.readthedocs.io/en/latest/auto_examples/index.html) in the documentation.

## Automated build and test
The repository uses continuous integration (CI) workflows to build and test the project directly with GitHub Actions. Tests are provided in the [`tests`](tests) folder. Run tests locally with `pytest`:

```
python -m pytest tests -v
```
