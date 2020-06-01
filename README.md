﻿# DeepChem
[![Build Status](https://travis-ci.org/deepchem/deepchem.svg?branch=master)](https://travis-ci.org/deepchem/deepchem)
[![Coverage Status](https://coveralls.io/repos/github/deepchem/deepchem/badge.svg?branch=master)](https://coveralls.io/github/deepchem/deepchem?branch=master)
[![Anaconda-Server Badge](https://anaconda.org/deepchem/deepchem/badges/version.svg)](https://anaconda.org/deepchem/deepchem)
[![PyPI version](https://badge.fury.io/py/deepchem.svg)](https://badge.fury.io/py/deepchem)


DeepChem aims to provide a high quality open-source toolchain
that democratizes the use of deep-learning in drug discovery,
materials science, quantum chemistry, and biology.

### Table of contents:

* [Requirements](#requirements)
* [Installation](#installation)
    * [Easy Install with Conda](#easy-install-with-conda)
    * [Docker](#using-a-docker-image)
    * [Conda Environment](#installing-from-source-in-a-conda-environment)
    * [Windows](#installing-in-windows)
* [FAQ and Troubleshooting](#faq-and-troubleshooting)
* [Getting Started](#getting-started)
* [Contributing to DeepChem](/CONTRIBUTING.md)
    * [Code Style Guidelines](/CONTRIBUTING.md#code-style-guidelines)
    * [Documentation Style Guidelines](/CONTRIBUTING.md#documentation-style-guidelines)
    * [Gitter](#gitter)
* [DeepChem Publications](#deepchem-publications)
* [Examples](/examples)
* [About Us](#about-us)
* [Citing DeepChem](#citing-deepchem)

## Requirements
* [pandas](http://pandas.pydata.org/)
* [joblib](https://pypi.python.org/pypi/joblib)
* [sklearn](https://github.com/scikit-learn/scikit-learn.git)
* [numpy](https://store.continuum.io/cshop/anaconda/)
* [tensorflow](https://www.tensorflow.org/)

### Soft Requirements
DeepChem has a number of "soft" requirements. These are packages which are needed for various submodules of DeepChem but not for the package as a whole.

* [RDKit](http://www.rdkit.org/docs/Install.html)
* [six](https://pypi.python.org/pypi/six)
* [MDTraj](http://mdtraj.org/)
* [PDBFixer](https://github.com/pandegroup/pdbfixer)
* [Pillow](https://pillow.readthedocs.io/en/stable/)

### Super easy install via pip3

```bash
pip3 install joblib pandas sklearn tensorflow pillow deepchem
```

### Easy Install via Conda

```bash
conda install -c deepchem -c rdkit -c conda-forge -c omnia deepchem=2.3.0
```
If you want GPU support:
```bash
conda install -c deepchem -c rdkit -c conda-forge -c omnia deepchem-gpu=2.3.0
```

**Note:** The above commands install the latest stable version of `deepchem` and _do not install from source_. If you need to install from source make sure you follow the steps [here](#using-a-conda-environment).

### Using a Docker Image
Using a docker image requires an NVIDIA GPU.  If you do not have a GPU please follow the directions for [using a conda environment](#installing-from-source-in-a-conda-environment)
In order to get GPU support you will have to use the [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) plugin.
``` bash
# This will the download the latest stable deepchem docker image into your images
docker pull deepchemio/deepchem

# This will create a container out of our latest image with GPU support
nvidia-docker run -i -t deepchemio/deepchem

# You are now in a docker container whose python has deepchem installed
# For example you can run our tox21 benchmark
cd deepchem/examples
python benchmark.py -d tox21

# Or you can start playing with it in the command line
pip install jupyter
ipython
import deepchem as dc
```

### Installing from source in a conda environment
You can install deepchem in a new conda environment using the conda commands in scripts/install_deepchem_conda.sh
Installing via this script will ensure that you are **installing from the source**.

```bash
git clone https://github.com/deepchem/deepchem.git      # Clone deepchem source code from GitHub
cd deepchem
```
If you don't want GPU support:
```
bash scripts/install_deepchem_conda.sh deepchem         # If you don't want GPU support
```
If you want GPU support:
```
gpu=1 bash scripts/install_deepchem_conda.sh deepchem         # If you want GPU support
```
Note : `gpu=0 bash scripts/install_deepchem_conda.sh deepchem` will also install CPU supported `deepchem`.
```
source activate deepchem
python setup.py install                                # Manual install
nosetests -a '!slow' -v deepchem --nologcapture        # Run tests
```
This creates a new conda environment `deepchem` and installs in it the dependencies that
are needed. To access it, use the `conda activate deepchem` command (if your conda version >= 4.4) and use `source activate deepchem` command (if your conda version < 4.4).

Check [this link](https://conda.io/docs/using/envs.html) for more information about
the benefits and usage of conda environments. **Warning**: Segmentation faults can [still happen](https://github.com/deepchem/deepchem/pull/379#issuecomment-277013514)
via this installation procedure.

### Installing in Windows

Currently you have to install from source in windows. The following scripts requires `conda>4.6`.

If you want gpu support, use the following command in powershell:
```ps1
.\scripts\install_deepchem_conda.ps1 -gpu 1 deepchem
```
Or you can use the following command to install deepchem without gpu support.
```ps1
.\scripts\install_deepchem_conda.ps1 -gpu 0 deepchem
```

Before activating deepchem envrionment, make sure conda-powershell has been initialized.
Check if there is a `(base)` before `PS` in powershell. If not, use `conda init powershell`
to activate it, then:
```
conda activate deepchem
python setup.py install
nosetests -a '!slow' -v deepchem --nologcapture
```
## FAQ and Troubleshooting

1. DeepChem currently supports Python 3.5 through 3.7, and is supported on 64 bit Linux and Mac OSX. Note that DeepChem is not currently maintained for older versions of Python or with other operating systems.
2. Question: I'm seeing some failures in my test suite having to do with MKL
   ```Intel MKL FATAL ERROR: Cannot load libmkl_avx.so or libmkl_def.so.```

   Answer: This is a general issue with the newest version of `scikit-learn` enabling MKL by default. This doesn't play well with many linux systems. See [BVLC/caffe#3884](https://github.com/BVLC/caffe/issues/3884) for discussions. The following seems to fix the issue
   ```bash
   conda install nomkl numpy scipy scikit-learn numexpr
   conda remove mkl mkl-service
   ```
3.  Note that when using Ubuntu 16.04 server or similar environments, you may need to ensure libxrender is provided via e.g.:
   ```bash
   sudo apt-get install -y libxrender-dev
   ```

## Getting Started
The DeepChem project maintains an extensive colelction of [tutorials](https://github.com/deepchem/deepchem/tree/master/examples/tutorials). All tutorials are designed to be run on Google colab (or locally if you prefer). Tutorials are arranged in a suggested learning sequence which will take you from beginner to proficient at molecular machine learning and computational biology more broadly.

After working through the tutorials, you can also go through other [examples](https://github.com/deepchem/deepchem/tree/master/examples). To apply `deepchem` to a new problem, try starting from one of the existing examples or tutorials and modifying it step by step to work with your new use-case. If you have questions or comments you can raise them on our [gitter](https://gitter.im/deepchem/Lobby).

### Gitter
Join us on gitter at [https://gitter.im/deepchem/Lobby](https://gitter.im/deepchem/Lobby). Probably the easiest place to ask simple questions or float requests for new features.

## About Us
DeepChem is managed by a team of open source contributors. Anyone is free to join and contribute!

## Citing DeepChem

If you have used DeepChem in the course of your research, we ask that you cite the "Deep Learning for the Life Sciences" book by the DeepChem core team.

To cite this book, please use this bibtex entry:

```
@book{Ramsundar-et-al-2019,
    title={Deep Learning for the Life Sciences},
    author={Bharath Ramsundar and Peter Eastman and Patrick Walters and Vijay Pande and Karl Leswing and Zhenqin Wu},
    publisher={O'Reilly Media},
    note={\url{https://www.amazon.com/Deep-Learning-Life-Sciences-Microscopy/dp/1492039837}},
    year={2019}
}
```

## Version
2.4.0rc
