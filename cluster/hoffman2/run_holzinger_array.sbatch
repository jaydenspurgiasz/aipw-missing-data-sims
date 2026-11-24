#!/bin/bash
#SBATCH --job-name=holzinger_arr
#SBATCH --output=/u/scratch/j/jspurgia/research/missing_data_2/logs/holzinger_arr_%A_%a.out
#SBATCH --error=/u/scratch/j/jspurgia/research/missing_data_2/logs/holzinger_arr_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --array=0-3
#
# One Slurm array task per misspec scenario.
# Submit from repo root on Hoffman2:
#   mkdir -p $SCRATCH/research/missing_data_2/logs
#   sbatch --export=ALL,N_LIST=250,500,1000,REPS=1000 cluster/hoffman2/run_holzinger_array.sbatch

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$HOME/research/missing_data_2}"
cd "$REPO_ROOT"

VENV_DIR="${H2_VENV:-$HOME/.virtualenvs/missing_data_2}"
PYTHON_MODULE="${H2_PYTHON_MODULE:-python/3.12.13}"

module purge
module load "$PYTHON_MODULE"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# Ensure scratch log/run dirs exist (HOME runs/ should already symlink here).
mkdir -p "$SCRATCH/research/missing_data_2/logs" "$SCRATCH/research/missing_data_2/runs"
if [[ ! -e "$REPO_ROOT/runs" ]]; then
  ln -s "$SCRATCH/research/missing_data_2/runs" "$REPO_ROOT/runs"
fi

SCENARIOS=(correct outcome_mis propensity_mis both_mis)
SCENARIO="${SCENARIOS[$SLURM_ARRAY_TASK_ID]}"

N_LIST="${N_LIST:-250,500,1000}"
REPS="${REPS:-1000}"
COV_XZ_FRAC="${COV_XZ_FRAC:-0.9}"
SEED="${SEED:-20261123}"
RUN_ID_PREFIX="${RUN_ID_PREFIX:-holzinger}"

echo "Host: $(hostname)  user=$(whoami)  task=$SLURM_ARRAY_TASK_ID  scenario=$SCENARIO"
echo "Start: $(date -Is)"
echo "Repo: $REPO_ROOT"

# User R library (lavaan) is not needed for the Python sim itself, but keep
# R_LIBS_USER set in case any helper shells out to R later.
export R_LIBS_USER="${R_LIBS_USER:-$HOME/R/library}"

python cluster/hoffman2/run_holzinger.py \
  --n "$N_LIST" \
  --reps "$REPS" \
  --scenario "$SCENARIO" \
  --seed "$SEED" \
  --cov-xz-frac "$COV_XZ_FRAC" \
  --run-id-prefix "$RUN_ID_PREFIX" \
  --repo-root "$REPO_ROOT"

echo "End: $(date -Is)"
