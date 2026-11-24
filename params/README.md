# Plasmode / FA parameter exports

R writes language-agnostic CSVs here; Python reads them to build `Σ`.

```text
params/plasmode_holzinger/
  lambda.csv
  phi.csv
  omega.csv
  sigma_x.csv
  uniqueness.csv
  meta.json
```

Generate with:

```bash
Rscript scripts/R/plasmode_holzinger_fa.R
```
