#!/bin/bash
#SBATCH --job-name=holzinger_aipw
#SBATCH --output=/u/scratch/j/jspurgia/research/missing_data_2/logs/holzinger_%x_%j.out
#SBATCH --error=/u/scratch/j/jspurgia/research/missing_data_2/logs/holzinger_%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#
# Single (non-array) job. Prefer run_holzinger_array.sbatch for the real study.
#   mkdir -p $SCRATCH/research/missing_data_2/logs
#   sbatch --export=ALL,N_LIST=250,REPS=2,SCENARIO=correct,ALL_SCENARIOS=0 \
#     cluster/hoffman2/run_holzinger.sbatch

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$HOME/research/missing_data_2}"
cd "$REPO_ROOT"

VENV_DIR="${H2_VENV:-$HOME/.virtualenvs/missing_data_2}"
PYTHON_MODULE="${H2_PYTHON_MODULE:-python/3.12.13}"

module purge
module load "$PYTHON_MODULE"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

mkdir -p "$SCRATCH/research/missing_data_2/logs" "$SCRATCH/research/missing_data_2/runs"
if [[ ! -e "$REPO_ROOT/runs" ]]; then
  ln -s "$SCRATCH/research/missing_data_2/runs" "$REPO_ROOT/runs"
fi

N_LIST="${N_LIST:-250,500,1000}"
REPS="${REPS:-1000}"
SCENARIO="${SCENARIO:-}"
ALL_SCENARIOS="${ALL_SCENARIOS:-1}"
COV_XZ_FRAC="${COV_XZ_FRAC:-0.9}"
SEED="${SEED:-20261123}"
RUN_ID_PREFIX="${RUN_ID_PREFIX:-holzinger}"
export R_LIBS_USER="${R_LIBS_USER:-$HOME/R/library}"

ARGS=(
  --n "$N_LIST"
  --reps "$REPS"
  --seed "$SEED"
  --cov-xz-frac "$COV_XZ_FRAC"
  --run-id-prefix "$RUN_ID_PREFIX"
  --repo-root "$REPO_ROOT"
)

if [[ -n "$SCENARIO" ]]; then
  ARGS+=(--scenario "$SCENARIO")
elif [[ "$ALL_SCENARIOS" == "1" ]]; then
  ARGS+=(--all-scenarios)
else
  ARGS+=(--scenario correct)
fi

echo "Host: $(hostname)  user=$(whoami)"
echo "Start: $(date -Is)"
echo "Cmd: python cluster/hoffman2/run_holzinger.py ${ARGS[*]}"

python cluster/hoffman2/run_holzinger.py "${ARGS[@]}"

echo "End: $(date -Is)"
