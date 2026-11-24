#!/bin/bash
# First-time Hoffman2 setup for user jspurgia (or override H2_USER).
# Run on a *compute* node after the repo has been copied to $HOME.
#
#   ssh jspurgia@hoffman2.oarc.ucla.edu
#   srun --pty --time=2:00:00 --mem=8G bash
#   bash $HOME/research/missing_data_2/cluster/hoffman2/bootstrap_once.sh
#
# What this does:
#   1. Layout under $HOME/research/missing_data_2 (code) + $SCRATCH/... (runs/logs)
#   2. Python venv + deps
#   3. R user library + lavaan
#   4. Export Holzinger FA params into params/plasmode_holzinger/

set -euo pipefail

H2_USER="${H2_USER:-jspurgia}"
REPO_ROOT="${REPO_ROOT:-$HOME/research/missing_data_2}"
SCRATCH_ROOT="${SCRATCH_ROOT:-$SCRATCH/research/missing_data_2}"
VENV_DIR="${H2_VENV:-$HOME/.virtualenvs/missing_data_2}"
PYTHON_MODULE="${H2_PYTHON_MODULE:-python/3.12.13}"
# If module spider R shows a versioned module, set it here (e.g. R/4.4.1).
R_MODULE="${H2_R_MODULE:-R}"
R_LIBS_USER="${R_LIBS_USER:-$HOME/R/library}"

echo "==== Hoffman2 bootstrap ===="
echo "user:     $H2_USER (whoami=$(whoami))"
echo "repo:     $REPO_ROOT"
echo "scratch:  $SCRATCH_ROOT"
echo "venv:     $VENV_DIR"
echo "R libs:   $R_LIBS_USER"
echo

if [[ ! -d "$REPO_ROOT/scripts" ]]; then
  echo "ERROR: repo not found at $REPO_ROOT"
  echo "Copy the project first, e.g. from your laptop:"
  echo "  scp -r missing_data_2 ${H2_USER}@hoffman2.oarc.ucla.edu:~/research/"
  exit 1
fi

mkdir -p "$SCRATCH_ROOT/runs" "$SCRATCH_ROOT/logs" "$REPO_ROOT/logs"
mkdir -p "$R_LIBS_USER"

# Code stays in $HOME (backed up, 60GB). Large sim artifacts live on $SCRATCH
# (2TB, not backed up; idle files may be deleted after ~14 days).
if [[ -e "$REPO_ROOT/runs" && ! -L "$REPO_ROOT/runs" ]]; then
  echo "Moving existing runs/ directory aside -> runs.bak_before_scratch"
  mv "$REPO_ROOT/runs" "$REPO_ROOT/runs.bak_before_scratch"
fi
if [[ ! -L "$REPO_ROOT/runs" ]]; then
  ln -s "$SCRATCH_ROOT/runs" "$REPO_ROOT/runs"
  echo "Linked $REPO_ROOT/runs -> $SCRATCH_ROOT/runs"
fi
if [[ -e "$REPO_ROOT/logs" && ! -L "$REPO_ROOT/logs" ]]; then
  # Keep a local logs dir for tiny files; also link scratch logs as logs_scratch
  true
fi
ln -sfn "$SCRATCH_ROOT/logs" "$REPO_ROOT/logs_scratch"
echo "Linked $REPO_ROOT/logs_scratch -> $SCRATCH_ROOT/logs"

# -----------------------------------------------------------------------------
# Python
# -----------------------------------------------------------------------------
echo
echo "==== Python venv ===="
module purge
module load "$PYTHON_MODULE"
python3 -m venv "$VENV_DIR"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip
python -m pip install -r "$REPO_ROOT/requirements.txt"
python -m ipykernel install --user --name missing_data_2 --display-name "Python (missing_data_2)"

# -----------------------------------------------------------------------------
# R + lavaan (user library under $HOME)
# -----------------------------------------------------------------------------
echo
echo "==== R + lavaan ===="
module load "$R_MODULE"
export R_LIBS_USER
mkdir -p "$R_LIBS_USER"

Rscript -e "dir.create(Sys.getenv('R_LIBS_USER'), recursive=TRUE, showWarnings=FALSE); \
  if (!requireNamespace('lavaan', quietly=TRUE)) { \
    install.packages('lavaan', lib=Sys.getenv('R_LIBS_USER'), repos='https://cloud.r-project.org', dependencies=TRUE); \
  }; \
  library(lavaan, lib.loc=Sys.getenv('R_LIBS_USER')); \
  cat('lavaan OK', as.character(packageVersion('lavaan')), '\n')"

# -----------------------------------------------------------------------------
# Plasmode FA params
# -----------------------------------------------------------------------------
echo
echo "==== Holzinger FA export ===="
export R_LIBS_USER
Rscript "$REPO_ROOT/scripts/R/plasmode_holzinger_fa.R" "$REPO_ROOT/params/plasmode_holzinger"

echo
echo "==== Bootstrap complete ===="
echo "Next (from laptop):"
echo "  python h2jupynb -u ${H2_USER} -v 3.12.13 -k '\$HOME/.virtualenvs/missing_data_2' -t 8 -m 16"
echo
echo "Or submit the real Slurm array (from repo on Hoffman):"
echo "  cd $REPO_ROOT && mkdir -p logs_scratch"
echo "  sbatch --export=ALL,N_LIST=250,500,1000,REPS=1000 cluster/hoffman2/run_holzinger_array.sbatch"
