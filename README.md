# granite-common


Python library that provides enhanced prompt creation and output parsing for IBM 
Granite models.


## Installation

To install from the main development branch, type:

```
pip install git+https://github.com/ibm-granite/granite-common.git
```

## Developer setup

For compatibility with different underlying operating system versions, we recommend using `conda` to create a consistent base Python environment for development and testing.

Detailed instructions:

1. Install [MiniForge](https://github.com/conda-forge/miniforge) or another package that provides the `conda` command-line utility.
1. Set the environment variable `CONDA_HOME` to point to the root of your `conda` install. If you installed MiniForge in your home directory, this value should be `${HOME}/miniforge3`.
1. Check out a copy of this repository.
1. Run the script [`scripts/env.sh`](scripts/env.sh) from the root of your local copy of the repository. The script will create a Conda environment in `./env` and will install the source code of your local copy as an editable Pip package. The script will also install and enable pre-commit hooks with [pre-commit](https://pre-commit.com/).
1. Before running commands such as `python` or `jupyter` from the command line, activate the Conda environment by typing `conda activate ./env` from the root of your local copy of this repository.
1. If you are using Visual Studio Code or a similar IDE, configure your IDE to use the environment at `./env`

## Running tests

After following the instructions in the previous section, you should be able to run tests on your local machine by typing:
```
pytest tests
```
from the root of your local copy of this repository, using the conda environment 
described in the previous section.

The build automation for this project uses the [`tox`](https://tox.wiki/en) environment manager. Sometimes you will need to run tests from inside a `tox` managed environment to replicate issues from the continuous integration environment. To run tests with `tox`, activate the Python environment `./env` created earlier, then choose from among the following:

* Run regression tests: `tox -e unit`
* Run `pylint` checks: `tox -e lint`
* Run `ruff` formatter: `tox -e format`
* Run a full continuous integration suite: `tox`
