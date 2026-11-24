"""Simulation summary metrics from a finished ``runs/<id>/`` folder.

Covariance summaries use ``vech(Σ)`` (lower triangle, including diagonal).
When sandwich SEs are present: 95% coverage and variance ratio
``mean(se²) / empirical variance`` (M-estimation vs Monte Carlo).

CFA summaries use Semopy parameter estimates and Heywood / PSD flags from
``result_meta.json`` when available.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import f as f_dist

from utils import is_invertible

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))


# =============================================================================
# Paths / truth
# =============================================================================

def _rep_dirs(output_dir: Path) -> list[Path]:
    return sorted(
        d for d in output_dir.iterdir()
        if d.is_dir() and d.name.startswith("rep_")
    )


def _load_truth_sigma(run_dir: Path) -> tuple[np.ndarray, int]:
    params = json.loads((run_dir / "params.json").read_text())
    n_z = int(params.get("n_z", 1))
    Sigma = pd.read_csv(run_dir / "Sigma.csv").to_numpy(dtype=float)
    p = Sigma.shape[0] - n_z
    if p < 1:
        raise ValueError("Sigma has no X block after removing n_z")
    return Sigma[:p, :p], n_z


def _pair_labels(p: int, names: list[str] | None = None) -> list[str]:
    if names is None:
        names = [f"x{j + 1}" for j in range(p)]
    ii, jj = np.tril_indices(p)
    return [f"{names[j]}__{names[k]}" for j, k in zip(ii, jj)]


def _truth_vech(truth: np.ndarray) -> pd.Series:
    p = truth.shape[0]
    names = [f"x{j + 1}" for j in range(p)]
    ii, jj = np.tril_indices(p)
    labels = _pair_labels(p, names)
    return pd.Series(truth[ii, jj], index=labels, name="truth")


# =============================================================================
# Collect per-replication stacks
# =============================================================================

def _stack_covariance(method_dirs: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Return (est_by_rep, se_by_rep). se_by_rep is None if SEs are missing."""
    est_rows = []
    se_rows = []
    have_se = True

    for method_dir in method_dirs:
        cov_path = method_dir / "covariance.csv"
        if not cov_path.exists():
            continue
        cov = pd.read_csv(cov_path, index_col=0)
        names = list(cov.columns)
        p = len(names)
        ii, jj = np.tril_indices(p)
        labels = [f"{names[j]}__{names[k]}" for j, k in zip(ii, jj)]
        vech_path = method_dir / "covariance_vech.csv"
        if vech_path.exists():
            vech = pd.read_csv(vech_path, index_col=0).iloc[:, 0]
            vech = vech.reindex(labels)
        else:
            vech = pd.Series(cov.to_numpy()[ii, jj], index=labels)
        est_rows.append(vech)

        se_path = method_dir / "covariance_se.csv"
        if have_se and se_path.exists():
            se = pd.read_csv(se_path, index_col=0).iloc[:, 0].reindex(labels)
            se_rows.append(se)
        else:
            have_se = False

    if not est_rows:
        raise FileNotFoundError("No covariance.csv files found for method")

    est = pd.DataFrame(est_rows).reset_index(drop=True)
    se = pd.DataFrame(se_rows).reset_index(drop=True) if have_se and se_rows else None
    return est, se


def _stack_cfa(method_dirs: list[Path]) -> tuple[pd.DataFrame | None, pd.DataFrame]:
    """Return (cfa_params_by_rep or None, status flags by rep)."""
    param_rows = []
    status_rows = []

    for method_dir in method_dirs:
        meta_path = method_dir / "result_meta.json"
        meta = {}
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())

        status_rows.append(
            {
                "cfa_converged": meta.get("cfa_converged"),
                "cfa_is_psd": meta.get("cfa_is_psd"),
                "cfa_is_heywood": meta.get("cfa_is_heywood"),
                "cfa_status": meta.get("cfa_status"),
            }
        )

        cfa_path = method_dir / "cfa_parameters.csv"
        if cfa_path.exists():
            params = pd.read_csv(cfa_path, index_col=0).iloc[:, 0]
            param_rows.append(params)

    status = pd.DataFrame(status_rows)
    if not param_rows:
        return None, status
    return pd.DataFrame(param_rows).reset_index(drop=True), status


def _psd_from_covariances(method_dirs: list[Path]) -> np.ndarray:
    flags = []
    for method_dir in method_dirs:
        cov = pd.read_csv(method_dir / "covariance.csv", index_col=0).to_numpy(dtype=float)
        vals = np.linalg.eigvalsh(cov)
        flags.append(bool(np.all(vals > 1e-10)))
    return np.asarray(flags, dtype=bool)


# =============================================================================
# Covariance metrics
# =============================================================================

def _covariance_entry_table(
    est: pd.DataFrame,
    truth: pd.Series,
    se: pd.DataFrame | None,
) -> pd.DataFrame:
    truth = truth.reindex(est.columns)
    mean_est = est.mean(axis=0)
    bias = mean_est - truth
    emp_var = est.var(axis=0, ddof=1)

    rows = pd.DataFrame(
        {
            "pair": est.columns,
            "truth": truth.to_numpy(),
            "mean_est": mean_est.to_numpy(),
            "bias": bias.to_numpy(),
            "emp_var": emp_var.to_numpy(),
        }
    )

    if se is not None and len(se) == len(est):
        pred_var = (se.to_numpy(dtype=float) ** 2).mean(axis=0)
        rows["pred_var"] = pred_var
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = pred_var / emp_var.to_numpy(dtype=float)
        rows["var_ratio"] = ratio
        z = 1.96
        covered = (
            (est.to_numpy(dtype=float) - z * se.to_numpy(dtype=float) <= truth.to_numpy())
            & (est.to_numpy(dtype=float) + z * se.to_numpy(dtype=float) >= truth.to_numpy())
        )
        rows["coverage95"] = covered.mean(axis=0)
        rows["mean_se"] = se.mean(axis=0).to_numpy()
    else:
        rows["pred_var"] = np.nan
        rows["var_ratio"] = np.nan
        rows["coverage95"] = np.nan
        rows["mean_se"] = np.nan

    return rows


def _covariance_overall(entry: pd.DataFrame, n_reps: int) -> dict[str, Any]:
    truth = entry["truth"].to_numpy(dtype=float)
    bias = entry["bias"].to_numpy(dtype=float)
    emp_var = entry["emp_var"].to_numpy(dtype=float)

    bias_l2 = float(np.sqrt(np.sum(bias ** 2)))
    truth_l2 = float(np.sqrt(np.sum(truth ** 2)))
    rel_bias = bias_l2 / truth_l2 if truth_l2 > 0 else np.nan
    var_trace = float(np.sum(emp_var))
    mse = var_trace + bias_l2 ** 2

    out = {
        "n_reps": int(n_reps),
        "n_pairs": int(len(entry)),
        "bias_l2": bias_l2,
        "rel_bias": rel_bias,
        "var_trace": var_trace,
        "mse": mse,
        "mean_abs_bias": float(np.mean(np.abs(bias))),
        "mean_emp_var": float(np.mean(emp_var)),
    }

    if entry["var_ratio"].notna().any():
        out["mean_pred_var"] = float(np.nanmean(entry["pred_var"]))
        out["mean_var_ratio"] = float(np.nanmean(entry["var_ratio"]))
        out["median_var_ratio"] = float(np.nanmedian(entry["var_ratio"]))
        out["mean_coverage95"] = float(np.nanmean(entry["coverage95"]))
        out["min_coverage95"] = float(np.nanmin(entry["coverage95"]))
        out["max_coverage95"] = float(np.nanmax(entry["coverage95"]))
    else:
        out["mean_pred_var"] = None
        out["mean_var_ratio"] = None
        out["median_var_ratio"] = None
        out["mean_coverage95"] = None
        out["min_coverage95"] = None
        out["max_coverage95"] = None

    return out


# =============================================================================
# CFA metrics
# =============================================================================

def _cfa_truth_parameters(
    truth_sigma: np.ndarray,
    n_obs: int,
    n_factors: int | None = None,
) -> pd.Series | None:
    from cfa import fit_cfa

    fit = fit_cfa(truth_sigma, n_obs=n_obs, n_factors=n_factors)
    if fit["parameters"] is None:
        return None
    return fit["parameters"]


def _cfa_param_table(
    params: pd.DataFrame,
    truth: pd.Series | None,
    status: pd.DataFrame,
) -> pd.DataFrame:
    # Summarize among reps that produced finite CFA params.
    ok = params.notna().all(axis=1).to_numpy()
    if status["cfa_is_heywood"].notna().any():
        # Prefer non-Heywood successful fits when flags exist.
        hey = status["cfa_is_heywood"].fillna(False).to_numpy(dtype=bool)
        use = ok & ~hey
        if use.sum() < 2:
            use = ok
    else:
        use = ok

    used = params.loc[use]
    mean_est = used.mean(axis=0)
    emp_var = used.var(axis=0, ddof=1)

    rows = pd.DataFrame(
        {
            "parameter": used.columns,
            "mean_est": mean_est.to_numpy(),
            "emp_var": emp_var.to_numpy(),
            "n_used": int(use.sum()),
        }
    )
    if truth is not None:
        truth = truth.reindex(used.columns)
        rows["truth"] = truth.to_numpy()
        rows["bias"] = (mean_est - truth).to_numpy()
    else:
        rows["truth"] = np.nan
        rows["bias"] = np.nan
    return rows


def _cfa_rates(status: pd.DataFrame, psd_fallback: np.ndarray | None) -> dict[str, Any]:
    n = len(status)
    rates: dict[str, Any] = {"n_reps": n}

    if status["cfa_is_psd"].notna().any():
        rates["psd_rate"] = float(status["cfa_is_psd"].astype(float).mean())
    elif psd_fallback is not None and len(psd_fallback) == n:
        rates["psd_rate"] = float(np.mean(psd_fallback))
    else:
        rates["psd_rate"] = None

    if status["cfa_is_heywood"].notna().any():
        rates["heywood_rate"] = float(status["cfa_is_heywood"].astype(float).mean())
    else:
        rates["heywood_rate"] = None

    if status["cfa_converged"].notna().any():
        rates["cfa_converged_rate"] = float(status["cfa_converged"].astype(float).mean())
    else:
        rates["cfa_converged_rate"] = None

    if status["cfa_status"].notna().any():
        counts = status["cfa_status"].value_counts(dropna=False).to_dict()
        rates["cfa_status_counts"] = {str(k): int(v) for k, v in counts.items()}
    else:
        rates["cfa_status_counts"] = None

    return rates


# =============================================================================
# Public: summarize one run
# =============================================================================

def summarize_run(run_dir: str | Path, *, write: bool = True) -> dict[str, Any]:
    """Build summary tables for every method under ``run_dir/output``."""
    run_dir = Path(run_dir)
    output_dir = run_dir / "output"
    if not output_dir.exists():
        raise FileNotFoundError(f"Missing output dir: {output_dir}")

    params = json.loads((run_dir / "params.json").read_text())
    truth_sigma, _ = _load_truth_sigma(run_dir)
    truth_vech = _truth_vech(truth_sigma)
    n_obs = int(params.get("n", 0)) or None

    methods = list(params.get("methods") or [])
    if not methods:
        # Infer from first rep folder.
        reps = _rep_dirs(output_dir)
        if not reps:
            raise FileNotFoundError(f"No rep_* folders under {output_dir}")
        methods = sorted(d.name for d in reps[0].iterdir() if d.is_dir())

    cfa_n_factors = params.get("cfa_n_factors")
    if cfa_n_factors is not None:
        cfa_n_factors = int(cfa_n_factors)

    cfa_truth = None
    if n_obs is not None:
        try:
            cfa_truth = _cfa_truth_parameters(
                truth_sigma, n_obs=n_obs, n_factors=cfa_n_factors
            )
        except Exception:
            cfa_truth = None

    overall_rows = []
    summary: dict[str, Any] = {
        "run_dir": str(run_dir),
        "n": params.get("n"),
        "reps": params.get("reps"),
        "scenario": params.get("scenario"),
        "outcome_z": params.get("outcome_z"),
        "propensity_z": params.get("propensity_z"),
        "methods": {},
    }

    for method in methods:
        method_dirs = []
        for rep_dir in _rep_dirs(output_dir):
            method_dir = rep_dir / method
            if method_dir.exists():
                method_dirs.append(method_dir)
        if not method_dirs:
            continue

        est, se = _stack_covariance(method_dirs)
        entry = _covariance_entry_table(est, truth_vech, se)
        cov_overall = _covariance_overall(entry, n_reps=len(method_dirs))

        cfa_params, cfa_status = _stack_cfa(method_dirs)
        psd_fallback = _psd_from_covariances(method_dirs)
        rates = _cfa_rates(cfa_status, psd_fallback)

        method_summary = {
            "covariance": cov_overall,
            "cfa_rates": rates,
        }

        if write:
            entry.to_csv(
                output_dir / f"summary_{method}_covariance_by_entry.csv",
                index=False,
            )
            if cfa_params is not None:
                cfa_table = _cfa_param_table(cfa_params, cfa_truth, cfa_status)
                cfa_table.to_csv(
                    output_dir / f"summary_{method}_cfa_by_param.csv",
                    index=False,
                )
                method_summary["cfa_n_used"] = int(cfa_table["n_used"].iloc[0])

        overall_rows.append(
            {
                "method": method,
                **{f"cov_{k}": v for k, v in cov_overall.items()},
                "psd_rate": rates["psd_rate"],
                "heywood_rate": rates["heywood_rate"],
                "cfa_converged_rate": rates["cfa_converged_rate"],
            }
        )
        summary["methods"][method] = method_summary

    overall_df = pd.DataFrame(overall_rows)
    if write:
        overall_df.to_csv(output_dir / "summary_overall.csv", index=False)
        (output_dir / "SUMMARY.json").write_text(json.dumps(summary, indent=2, default=str))

    summary["overall_table"] = overall_df
    return summary


# =============================================================================
# Legacy multivariate table (old simulation class)
# =============================================================================

def result_table(est, theta0, method_names):
    _, k, d = est.shape
    rows = []

    for i in range(k):
        est_i = est[:, i, :]
        est_i = est_i[~np.isnan(est_i).any(axis=1)]
        r = est_i.shape[0]

        if r < 2:
            rows.append(
                dict(
                    Method=method_names[i],
                    Var=np.nan,
                    Bias=np.nan,
                    **{"Rel. Bias": np.nan},
                    MSE=np.nan,
                    F=np.nan,
                    pval=np.nan,
                )
            )
            continue

        m = est_i.mean(axis=0)
        A = np.cov(est_i.T)
        b = m - theta0

        var = np.trace(A)
        bias = np.sqrt(np.sum(b ** 2))
        rel_bias = bias / np.sqrt(np.sum(theta0 ** 2))
        mse = var + bias ** 2

        if is_invertible(A) and r > d:
            df1 = r - d
            df2 = d * (r - 1)
            F_stat = (df1 / df2) * r * (b @ np.linalg.solve(A, b))
            pval = f_dist.sf(F_stat, dfn=df1, dfd=df2)
        else:
            F_stat = np.nan
            pval = np.nan

        rows.append(
            dict(
                Method=method_names[i],
                Var=var,
                Bias=bias,
                **{"Rel. Bias": rel_bias},
                MSE=mse,
                F=F_stat,
                pval=pval,
            )
        )

    return pd.DataFrame(rows).set_index("Method")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python metrics.py <run_dir>")
    out = summarize_run(sys.argv[1])
    print(out["overall_table"].to_string(index=False))
