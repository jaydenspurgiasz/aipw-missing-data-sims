# Hoffman2 setup for `jspurgia`

Production login: **`hoffman2.oarc.ucla.edu`** (Slurm).  
Docs: [Storage](https://www.hoffman2.idre.ucla.edu/Using-H2/Storage/Storage.html), [Computing/Slurm](https://www.hoffman2.idre.ucla.edu/Using-H2/Computing/Computing-Slurm.html), [Jupyter / h2jupynb](https://www.hoffman2.idre.ucla.edu/Using-H2/Connecting/Connecting.html).

## Storage (docs-backed)

| Place | Quota | Backed up? | Use for |
|---|---|---|---|
| `$HOME` (`/u/home/j/jspurgia`) | 60 GB | Yes | **code**, venv, R user libs, FA CSVs |
| `$SCRATCH` (`/u/scratch/j/jspurgia`) | 2 TB | **No** (idle ~14 days may be deleted) | **`runs/`**, Slurm logs, big I/O |

Putting the **repo in `$HOME` is fine**. Do **not** dump large `runs/` only into `$HOME` for `R=1000` grids — bootstrap symlinks `runs/` → `$SCRATCH/.../runs`.


## 0. Copy the project from your laptop (nothing exists on Hoffman yet)

From PowerShell on your PC (repo parent folder):

```powershell
ssh jspurgia@hoffman2.oarc.ucla.edu "mkdir -p ~/research"
scp -r missing_data_2 jspurgia@hoffman2.oarc.ucla.edu:~/research/
```

Or from inside the repo:

```powershell
scp -r . jspurgia@hoffman2.oarc.ucla.edu:~/research/missing_data_2
```

Include `params/plasmode_holzinger/` if you already exported FA locally (bootstrap can also re-export with R).


## 1. One-time bootstrap on a compute node

```bash
ssh jspurgia@hoffman2.oarc.ucla.edu
srun --pty --time=2:00:00 --mem=8G bash
bash $HOME/research/missing_data_2/cluster/hoffman2/bootstrap_once.sh
```

That script:

1. Links `$HOME/.../runs` → `$SCRATCH/.../runs`
2. Builds Python venv `$HOME/.virtualenvs/missing_data_2`
3. Installs **lavaan** into `$HOME/R/library`
4. Runs `plasmode_holzinger_fa.R` → `params/plasmode_holzinger/`

If `module load R` fails, run `module spider R` / `h2 modules R` and re-run with:

```bash
H2_R_MODULE=R/4.4.1 bash cluster/hoffman2/bootstrap_once.sh
```

(use the version spider shows).


## 2. Interactive notebook (smoke / debugging)

On your **laptop** (not inside SSH):

```powershell
curl.exe -o h2jupynb https://raw.githubusercontent.com/rdauria/jupyter-notebook/main/h2jupynb_slurm
python h2jupynb -u jspurgia -v 3.12.13 -k "$HOME/.virtualenvs/missing_data_2" -t 8 -m 16
```

Open `notebooks/HolzingerSwineford_hoffman.ipynb`, kernel **Python (missing_data_2)**.  
Keep `reps` small in the notebook; use Slurm for the real study.


## 3. Real run = Slurm array (one job per misspec scenario)

On Hoffman, from the repo:

```bash
cd $HOME/research/missing_data_2
mkdir -p logs_scratch

sbatch --export=ALL,N_LIST=250,500,1000,REPS=1000 \
  cluster/hoffman2/run_holzinger_array.sbatch
```

Array tasks: `0=correct`, `1=outcome_mis`, `2=propensity_mis`, `3=both_mis`.  
Each task writes `runs/holzinger_<scenario>_n{250,500,1000}/` (on scratch via the symlink).

Check:

```bash
squeue -u jspurgia
sacct -j JOBID --format=JobID,State,Elapsed,MaxRSS,ExitCode
myquota
```


## 4. DGP reminder

`cov_XZ = 0.9 * max_equal_cov_xz` (~0.195). Joint Σ stays PD; legacy 0.64 is not used.
