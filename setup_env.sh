#!/usr/bin/env bash
# Create the Intelliswitch virtual environment and register a Jupyter kernel.
# Run from the project root on your Mac:   bash setup_env.sh
set -e
cd "$(dirname "$0")"

# 1) create the venv (needs Python 3.10+)
python3 -m venv .venv
source .venv/bin/activate

# 2) install the package (editable) + notebook tooling
#    setuptools/wheel upgrade is needed for PEP 660 editable installs
python -m pip install --upgrade pip setuptools wheel
pip install -e ".[dev]"

# 3) register the venv as a Jupyter kernel so notebooks can select it
python -m ipykernel install --user --name intelliswitch \
    --display-name "Python (intelliswitch)"

echo
echo "Done. Activate with:  source .venv/bin/activate"
echo "Then:  jupyter lab   (and pick the 'Python (intelliswitch)' kernel)"
