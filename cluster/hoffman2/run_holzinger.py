#!/usr/bin/env python3
"""CLI entrypoint for Holzinger plasmode sims on Hoffman2 (or locally).

Examples
--------
# One scenario, one n (good for Slurm array tasks):
python cluster/hoffman2/run_holzinger.py --n 250 --reps 1000 --scenario correct

# Full board grid in one process (long):
python cluster/hoffman2/run_holzinger.py --n 250,500,1000 --reps 1000 --all-scenarios
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_SCRIPTS = _REPO / "scripts"
for path in (_SCRIPTS, _REPO):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from metrics import summarize_run  # noqa: E402
from misspec import available_scenarios  # noqa: E402
from plasmode import load_plasmode_joint  # noqa: E402
from simulation import RunConfig, run_simulation  # noqa: E402


def _parse_n(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _parse_scenarios(raw: str | None, all_scenarios: bool) -> list[str]:
    if all_scenarios:
        return list(available_scenarios())
    if raw is None:
        return ["correct"]
    return [s.strip() for s in raw.split(",") if s.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", default="250", help="Comma-separated sample sizes")
    parser.add_argument("--reps", type=int, default=1000)
    parser.add_argument("--scenario", default=None, help="One scenario, or comma list")
    parser.add_argument(
        "--all-scenarios",
        action="store_true",
        help="Run correct / outcome_mis / propensity_mis / both_mis",
    )
    parser.add_argument("--seed", type=int, default=20261123)
    parser.add_argument(
        "--params-dir",
        default=None,
        help="Directory with lambda.csv / phi.csv / omega.csv (default: params/plasmode_holzinger)",
    )
    parser.add_argument(
        "--cov-xz-frac",
        type=float,
        default=0.9,
        help="cov_XZ = frac * max_equal_cov_xz (default 0.9)",
    )
    parser.add_argument("--save-data", action="store_true")
    parser.add_argument(
        "--run-id-prefix",
        default="holzinger",
        help="Prefix for runs/<prefix>_<scenario>_nN folders",
    )
    parser.add_argument(
        "--repo-root",
        default=str(_REPO),
        help="Repo root (used only for logging)",
    )
    args = parser.parse_args(argv)

    n_values = _parse_n(args.n)
    scenarios = _parse_scenarios(args.scenario, args.all_scenarios)

    plas = load_plasmode_joint(
        args.params_dir,
        cov_XZ=None,
        cov_xz_frac_of_max=float(args.cov_xz_frac),
    )

    print(
        json.dumps(
            {
                "repo_root": args.repo_root,
                "p": plas["p"],
                "n_factors": plas["n_factors"],
                "cov_XZ": plas["cov_XZ"],
                "cov_XZ_max": plas["cov_XZ_max"],
                "n": n_values,
                "reps": args.reps,
                "scenarios": scenarios,
            },
            indent=2,
        ),
        flush=True,
    )

    run_dirs: list[Path] = []
    for scenario in scenarios:
        cfg = RunConfig(
            n=n_values if len(n_values) > 1 else n_values[0],
            reps=int(args.reps),
            Sigma=plas["Sigma"],
            mean=plas["mean"],
            n_z=1,
            seed=int(args.seed),
            scenario=scenario,
            methods=("aipw",),
            propensity_intercept=-0.6,
            propensity_coef=1.0,
            use_true_pi=False,
            save_data=bool(args.save_data),
            cfa_n_factors=plas["cfa_n_factors"],
            run_id=f"{args.run_id_prefix}_{scenario}",
        )
        out = run_simulation(cfg)
        if not isinstance(out, list):
            out = [out]
        run_dirs.extend(out)
        for d in out:
            print(f"FINISHED {scenario} -> {d}", flush=True)

    # Refresh summaries and write a small index next to the first run's parent.
    rows = []
    for run_dir in run_dirs:
        summary = summarize_run(run_dir, write=True)
        overall = summary["overall_table"].copy()
        params = json.loads((run_dir / "params.json").read_text())
        overall.insert(0, "run_id", run_dir.name)
        overall.insert(1, "n", int(params["n"]))
        overall.insert(2, "scenario", params.get("scenario"))
        rows.append(overall)
        print(f"SUMMARY {run_dir / 'output' / 'SUMMARY.json'}", flush=True)

    if rows:
        import pandas as pd

        index = pd.concat(rows, ignore_index=True)
        index_path = _REPO / "runs" / f"{args.run_id_prefix}_index.csv"
        index.to_csv(index_path, index=False)
        print(f"INDEX {index_path}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
