# Missing-data simulation repo

Python-first simulation code for MAR missingness and AIPW estimation.
Write code like a careful senior engineer: clear, boring, easy to search,
easy to change. Prefer readable structure over cleverness.

Codex and Cursor both load this file from the repo root. Nested
`AGENTS.md` / `AGENTS.override.md` files can add directory-local rules;
closer files win on conflicts.

StatLab (MCP / validated experiment tooling) lives at:
`C:/Users/spurg/Documents/Research/statlab-ai`.


## Layout

```text
dgp/         # data generation only (joint X,Z law + missingness)
scripts/     # estimators, simulation runners, utilities
scripts/R/   # legacy / optional R helpers
notebooks/   # interactive exploration
runs/        # indexed experiment artifacts for new work
legacy/      # old runs, parked helpers, failed-experiment debris
docs/        # papers / memos
```

Do not put new experiment dumps in `legacy/`. Do not revive failed
experiment scripts into `scripts/` without an explicit request.


## Runs

Each experiment gets one indexed folder:

```text
runs/<run_id>/
  params.json
  data/                 # CSV draws (language-agnostic)
    rep_0000/           # optional per-replication subfolder
  output/               # bias, var, CI, charts, logs
```

- DGP writes `data/` (+ `params.json`).
- Estimation / simulation writes `output/`.
- Prefer CSV interchange so R or other tools can consume the same files.
- Old studies live under `legacy/runs/`.


## Master-branch scope

- Primary path: AIPW covariance with linear first-order nuisances
  (`aipw_covariance` / method ``aipw`` = augmented §2.1). Simple Z-only
  §3 is kept as ``aipw_covariance_simple`` / ``aipw_simple`` for optional
  re-comparison (augmented had better variance in n=500/1000 sims).
- Sandwich SEs: ``aipw_m_estimation`` (analytic Jacobian for simple; μ/ν
  derivatives w.r.t. β,Σ use finite differences when augmented). Methods
  write ``covariance_se.csv`` and ``covariance_vcov.csv`` per replication.
- CFA: Semopy equal-block CFA on the AIPW covariance (factor vars fixed at
  1, all loadings free including the first, uniquenesses free, factor
  correlations free). Default 2-factor; plasmode Holzinger uses 3 via
  ``RunConfig.cfa_n_factors``. Writes ``cfa_parameters.csv`` plus Heywood /
  PSD / status flags in ``result_meta.json``. Custom ``factor_analyzer``
  CFA is parked under ``legacy/``.
- Plasmode: ``scripts/R/plasmode_holzinger_fa.R`` exports FA CSVs under
  ``params/plasmode_holzinger/``; ``scripts/plasmode.py`` builds joint Σ
  with ``cov_XZ = 0.9 * max_equal_cov_xz`` by default. Notebooks:
  ``HolzingerSwineford_sim.ipynb``, ``HolzingerSwineford_hoffman.ipynb``.
- Hoffman2 (Slurm, ``hoffman2.oarc.ucla.edu``): ``cluster/hoffman2/``
  (venv setup, ``h2jupynb`` notes, ``sbatch`` runners).
- DGP owns data generation and the missingness mechanism.
- Misspecification (wrong Z, wrong propensity design, etc.) is a separate
  step between DGP and estimation — not inside `dgp/`.
- Factor-model helpers that build covariance matrices are optional later
  tooling, not part of the core DGP API.
- Keep the active tree lean. Park dead code under `legacy/` instead of
  leaving it beside production scripts.


## Pipeline

Ordered stages for a replication:

1. `dgp.generate` → pandas draw (+ optional `data/rep_XXXX/` CSVs)
2. `misspec.build_estimation_draw` → attaches ``Z_outcome`` / ``Z_propensity``
   independently (`true` or `squared`); never overwrites DGP ``Z`` / ``pi``.
   Named presets: ``correct``, ``outcome_mis``, ``propensity_mis``, ``both_mis``
   (via ``RunConfig.scenario``).
3. `methods.run_method` → uses ``Z_outcome`` for AIPW design and either
   fitted ``Z_propensity`` or oracle ``pi`` (`use_true_pi`); fits Semopy
   CFA to the estimated covariance
4. `simulation.run_simulation` → loops reps, owns `runs/<id>/`.
   Default ids are ``sim_1``, ``sim_2``, … (never overwrite).
   ``RunConfig.n`` may be a list; each ``n`` gets its own folder.

Do not fold misspecification into DGP or bury estimators inside the runner.
Do not pass a single ambiguous ``Z`` into methods after this cleanup.


## DGP API (current)

`dgp.generate(n, Sigma, mean=..., n_z=1, ...)` takes the **joint**
covariance (and mean) of `(X, Z)`. The last `n_z` coordinates are Z.

Returns a dict of pandas objects (`X_true`, `X_obs`, `R`, `Z`, `pi`,
`Sigma`, `mean`, `params`). If `run_dir` is set, also writes CSVs under
that run's `data/` folder. Draw-level `params.json` lives beside those
CSVs; the simulation runner owns the run-level `params.json`.

Do not reintroduce a stateful mutate-in-place DGP class unless asked.


## Coding standards

Write as if a senior engineer with years of non-AI experience wrote it.

### Structure
- One clear job per function / section.
- Keep functions short. If a function needs a scroll break, split it.
- Prefer explicit inputs and return values over hidden object state.
- Pass matrices, paths, seeds, and knobs as arguments — do not hardcode
  process-critical values inside algorithm bodies.
- Separate stages: generate data → optional misspecification → estimate →
  summarize / save.

### Comments
- Use searchable section headers for separable blocks, for example:

  ```python
  # =============================================================================
  # Missingness mechanism (unit nonresponse, logistic in Z)
  # =============================================================================
  ```

- Comment what a section does, not every line.
- Do not add narrating comments like "import numpy" or "return the result".

### Style
- Readable line lengths. Avoid dense one-liners and clever packing.
- Plain names (`Sigma`, `run_dir`, `propensity_coef`) over abbreviations
  unless the abbreviation is standard in the paper/codebase.
- Prefer pandas for tabular draws and results tables.
- Match surrounding file style when editing; do not drive-by reformat
  unrelated code.
- No AI-flavored noise: emoji in code, decorative banners beyond section
  headers, filler docstrings, or speculative abstractions.

### Files
- Prefer updating an existing module over adding a new file.
- Do not create markdown docs unless asked.
- Do not commit secrets, large run artifacts, or scratch notebooks by
  default.


## Research / StatLab workflow

When doing research experiments with StatLab available:

1. Call `retrieve_research_context`.
2. State estimand, assumptions, hypothesis, control, repetitions, and seed.
3. Validate before running.
4. Use StatLab tools for numerical claims; do not invent metrics, SEs,
   derivative checks, or citations.
5. Record and inspect the returned experiment ID before reporting.
6. Use critique / adversarial tools when assessing robustness.
7. Treat failed runs as recorded failures; require human review before
   publishable conclusions.

If StatLab MCP is unavailable, say so and continue with local code only.
Do not fabricate StatLab results.


## Working agreements

- Ask before deleting tracked research outputs or rewriting public APIs
  beyond the current step.
- Refactors in this repo are step-by-step (DGP, then misspecification /
  estimation, then run orchestration). Do not boil the ocean.
- After substantive edits, run a small smoke import / generate check when
  practical.
- Keep `master` focused on the clean AIPW path described above.


## Summaries

After a run finishes, ``metrics.summarize_run`` writes under ``output/``:

- ``summary_<method>_covariance_by_entry.csv`` — bias, emp var, M-est
  ``pred_var``, ``var_ratio = pred_var / emp_var``, 95% coverage
- ``summary_<method>_cfa_by_param.csv`` — CFA param means / bias (when present)
- ``summary_overall.csv`` / ``SUMMARY.json`` — rel bias, var trace, MSE,
  mean/median var ratio, coverage, PSD / Heywood rates

Poster-style charts are still tabled.


## Later (tabled)

- Poster-style plots for summaries.
- Seed-fixed unit tests for small `n` / small `reps` simulations that check
  AIPW (and later other methods) against expected numeric fixtures. Keep
  them fast enough for routine runs; do not build that harness until the
  AIPW path and run I/O are stable.
