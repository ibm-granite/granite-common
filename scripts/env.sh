#! /bin/bash

################################################################################
# Create conda environment for local development.
#
# See usage() below for the current set of arguments this script accepts.
################################################################################
# Argument processing

# Default values for parameters passed on command line
# Use environment variables if present.
# (-z predicate means "unset or empty string")
if [ -z "$PYTHON_VERSION" ]; then
    PYTHON_VERSION="3.11"
fi
ENV_PATH="./env"


usage() {
    echo "Usage: ./scripts/env.sh [-h] [--env_path <path>] "
    echo "                     [--use_active_env]"
    echo "                     [--python_version <version>]"
    echo ""
    echo "You can also use the following environment variables:"
    echo "      PYTHON_VERSION: Version of Python to install"
    echo "(command-line arguments override environment variables values)"
}

die() {
    echo $1
    usage
    exit 1
}

# Read command line arguments
# See Bash docs at http://mywiki.wooledge.org/BashFAQ/035
while :; do
    case $1 in
        -h|--help)
            usage
            exit 0
            ;;
        --env_path)
            if [ "$2" ]; then ENV_PATH=$2; shift
            else die "ERROR: --env_path requires an environment name"
            fi
            ;;
        --use_active_env)
            unset ENV_PATH; shift
            ;;
        --python_version)
            if [ "$2" ]; then PYTHON_VERSION=$2; shift
            else die "ERROR: --python_version requires a python version"
            fi
            ;;
        ?*)
            die "Unknown option '$1'"
            ;;
        *) # No more options
            break
    esac
    shift # Move on to next argument
done

if [ -n "${ENV_PATH}" ]; then
    echo "Creating environment at ${ENV_PATH} with Python '${PYTHON_VERSION}'."
else
    echo "Using active environment with Python '${PYTHON_VERSION}'."
fi


############################
# HACK ALERT *** HACK ALERT 
# The friendly folks at Anaconda thought it would be a good idea to make the
# "conda" command a shell function. 
# See https://github.com/conda/conda/issues/7126
# The following workaround will probably be fragile.
if [ -z "$CONDA_HOME" ]
then 
    echo "Error: CONDA_HOME environment variable not set."
    exit
fi
if [ -e "${CONDA_HOME}/etc/profile.d/conda.sh" ]
then
    # shellcheck disable=SC1090
    . "${CONDA_HOME}/etc/profile.d/conda.sh"
else
    echo "Error: CONDA_HOME (${CONDA_HOME}) does not appear to be set up."
    exit
fi
# END HACK
############################

################################################################################
# Create the environment 

if [ -e "${ENV_PATH}" ]; then

    # Remove the detrius of any previous runs of this script
    conda env remove --prefix ${ENV_PATH}
fi

conda create -y --prefix ${ENV_PATH} python=${PYTHON_VERSION} pip

################################################################################
# All the installation steps that follow must be done from within the new
# environment.
conda activate ${ENV_PATH}

################################################################################
# Install packages with pip

# If Anaconda's version of Pip is backlevel, Pip will complain bitterly until you 
# update it. Start out with a preemptive update to prevent these complaints.
pip install --upgrade pip

# Install our local copy of the source tree in editable mode.
# Also install development-only packages.
pip install --editable ".[all]" 

# Also install pre-commit hooks in the local working copy
pip install pre-commit
pre-commit install

conda deactivate

echo "Anaconda environment '${ENV_PATH}' successfully created."
echo "To use, type 'conda activate ${ENV_PATH}'."

