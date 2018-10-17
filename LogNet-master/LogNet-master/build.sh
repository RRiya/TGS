#!/bin/bash
echo "Hello LogNet!"
echo "Building on: " $NODE_NAME
export PATH="/ldata/sribkain/anaconda3/bin:$PATH"

# Check if python3 exists on PATH
if ! [ -x "$(command -v python3)" ]; then
    echo "ERROR: python3 is not available on PATH"
    exit 1
fi

# Store source directory
source_dir=$(pwd)
echo "Source directory: " $source_dir

env_name="test_env_"$GIT_COMMIT"_"$GIT_LOCAL_BRANCH

# Create and activate new environment
echo "Creating new virtual environment in: " $env_name
mkdir ../$env_name
cd ../$env_name
echo "Currently in: " $(pwd)
python3 -m venv test_env
. test_env/bin/activate
echo "New virtual environment activated!"
cd $source_dir

# Install the library
pip install -r requirements.txt
pip install -e . --verbose

# Before anything check the static type
mypy --config-file mypy.ini lognet

# Run tests here
python -m pytest tests/test_*.py

# Deactivate
deactivate test_env
