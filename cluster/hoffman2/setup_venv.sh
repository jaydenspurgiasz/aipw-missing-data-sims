#!/bin/bash
# Build a Hoffman2 (Slurm) Python venv with this repo's deps.
# Run on a *compute* node (interactive srun / h2jupynb session), not a login node.
#
# Usage:
#   bash cluster/hoffman2/setup_venv.sh
#   bash cluster/hoffman2/setup_venv.sh /path/to/missing_data_2

set -euo pipefail

REPO_ROOT="${1:-$(cd "$(dirname "$0")/../.." && pwd)}"
VENV_DIR="${H2_VENV:-$HOME/.virtualenvs/missing_data_2}"
PYTHON_MODULE="${H2_PYTHON_MODULE:-python/3.12.13}"

echo "Repo:   $REPO_ROOT"
echo "Venv:   $VENV_DIR"
echo "Module: $PYTHON_MODULE"

module load "$PYTHON_MODULE"
python3 -m venv "$VENV_DIR"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip
python -m pip install -r "$REPO_ROOT/requirements.txt"
python -m ipykernel install --user --name missing_data_2 --display-name "Python (missing_data_2)"

echo
echo "Done. For h2jupynb from your laptop:"
echo "  python3 h2jupynb -u \$USER -v 3.12.13 -k '\$HOME/.virtualenvs/missing_data_2' -t 8 -m 16"
echo
echo "Or activate on a compute node:"
echo "  module load $PYTHON_MODULE"
echo "  source $VENV_DIR/bin/activate"
