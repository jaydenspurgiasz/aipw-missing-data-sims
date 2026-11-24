# Simulation runs

Each run lives under one ID so data and results stay indexed together.
Default ids are `sim_1`, `sim_2`, … (existing folders are never overwritten).
`RunConfig.n` may be a list; each sample size gets its own folder
(`sim_K_n250`, `sim_K_n500`, …).

```text
runs/<run_id>/
  params.json              # run-level config (owned by simulation)
  Sigma.csv / mean.csv     # joint law used for the run
  data/
    rep_0000/              # DGP CSVs + draw params.json
  output/
    rep_0000/
      aipw/                # covariance.csv, covariance_se.csv,
                           # covariance_vcov.csv, cfa_parameters.csv,
                           # pi_hat.csv, result_meta.json (incl. CFA flags)
    RUN_COMPLETE.json
    SUMMARY.json
    summary_overall.csv
    summary_<method>_covariance_by_entry.csv   # includes var_ratio
    summary_<method>_cfa_by_param.csv          # when CFA present
```


Pipeline: `generate` → `build_estimation_draw(outcome_z, propensity_z)` → methods → save → summarize.

`outcome_z` / `propensity_z` are chosen independently (`true` or `squared`).
Or set `RunConfig.scenario` to a preset: `correct`, `outcome_mis`,
`propensity_mis`, `both_mis`.

`use_true_pi=True` uses DGP `pi` and skips the propensity fit.

Older experiment dumps live in `legacy/runs/`.
