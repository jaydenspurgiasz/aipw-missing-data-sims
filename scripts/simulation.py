"""Simulation runner: generate → estimation view → methods → save.

Thin orchestrator only. DGP, misspecification, and estimators live elsewhere.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

_SCRIPTS = Path(__file__).resolve().parent
_ROOT = _SCRIPTS.parent
for _path in (_SCRIPTS, _ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from dgp import generate
from methods import available_methods, run_method
from metrics import summarize_run
from misspec import (
    available_scenarios,
    available_z_transforms,
    build_estimation_draw,
    resolve_scenario,
)


# =============================================================================
# Run configuration
# =============================================================================

@dataclass
class RunConfig:
    # Single n or several sample sizes (each n gets its own run folder).
    n: int | Sequence[int]
    reps: int
    Sigma: Any
    mean: Any = None
    n_z: int = 1
    seed: int | None = None
    run_id: str | None = None
    run_dir: str | Path | None = None
    # Optional preset: "correct", "outcome_mis", "propensity_mis", "both_mis".
    # If set, overrides outcome_z / propensity_z.
    scenario: str | None = None
    # Nuisance covariate choices: "true" or "squared" (independent).
    outcome_z: str = "true"
    propensity_z: str = "true"
    methods: Sequence[str] = field(default_factory=lambda: ("aipw",))
    propensity_intercept: float = -0.6
    propensity_coef: float = 1.0
    save_data: bool = True
    # If True, AIPW uses DGP pi and ignores Z_propensity.
    use_true_pi: bool = False
    # Semopy CFA factor count (None => infer; plasmode Holzinger uses 3).
    cfa_n_factors: int | None = None


def _apply_scenario(config: RunConfig) -> RunConfig:
    """Return a copy with outcome_z / propensity_z filled from scenario."""
    if config.scenario is None:
        return config
    resolved = resolve_scenario(config.scenario)
    return RunConfig(
        **{
            **asdict(config),
            "outcome_z": resolved["outcome_z"],
            "propensity_z": resolved["propensity_z"],
        }
    )


# =============================================================================
# Paths / params
# =============================================================================

def _as_n_list(n: int | Sequence[int]) -> list[int]:
    if isinstance(n, (str, bytes)):
        raise TypeError("n must be an int or a sequence of ints")
    if isinstance(n, Sequence):
        values = [int(v) for v in n]
    else:
        values = [int(n)]
    if not values:
        raise ValueError("n must contain at least one sample size")
    if any(v < 1 for v in values):
        raise ValueError(f"all n values must be >= 1, got {values}")
    return values


def _next_sim_id(runs_dir: Path, prefix: str = "sim") -> str:
    """Return the next unused ``sim_K`` id under ``runs/``."""
    runs_dir.mkdir(parents=True, exist_ok=True)
    max_i = 0
    pattern = re.compile(rf"^{re.escape(prefix)}_(\d+)(?:_|$)")
    for path in runs_dir.iterdir():
        if not path.is_dir():
            continue
        match = pattern.match(path.name)
        if match:
            max_i = max(max_i, int(match.group(1)))
    return f"{prefix}_{max_i + 1}"


def _unique_run_dir(runs_dir: Path, run_id: str) -> Path:
    """Avoid overwrites: ``run_id``, then ``run_id_2``, ``run_id_3``, ..."""
    runs_dir.mkdir(parents=True, exist_ok=True)
    candidate = runs_dir / run_id
    if not candidate.exists():
        return candidate
    suffix = 2
    while (runs_dir / f"{run_id}_{suffix}").exists():
        suffix += 1
    return runs_dir / f"{run_id}_{suffix}"


def _resolve_run_dir(config: RunConfig, *, n_tag: int | None = None) -> Path:
    """Allocate a non-colliding run directory under ``runs/``.

    Default ids are ``sim_1``, ``sim_2``, ... When ``n_tag`` is set (multi-n
    grids), append ``_n{n_tag}`` to the base id.
    """
    if config.run_dir is not None:
        run_dir = Path(config.run_dir)
        if n_tag is not None:
            run_dir = run_dir.parent / f"{run_dir.name}_n{n_tag}"
        return _unique_run_dir(run_dir.parent, run_dir.name)

    runs_dir = _ROOT / "runs"
    if config.run_id is None:
        base = _next_sim_id(runs_dir)
    else:
        base = str(config.run_id)
    if n_tag is not None:
        base = f"{base}_n{n_tag}"
    return _unique_run_dir(runs_dir, base)


def _params_dict(config: RunConfig, run_dir: Path) -> dict[str, Any]:
    Sigma = np.asarray(config.Sigma, dtype=float)
    mean = None if config.mean is None else np.asarray(config.mean, dtype=float)
    return {
        "n": int(config.n),
        "reps": int(config.reps),
        "seed": config.seed,
        "n_z": int(config.n_z),
        "scenario": config.scenario,
        "outcome_z": config.outcome_z,
        "propensity_z": config.propensity_z,
        "methods": list(config.methods),
        "propensity_intercept": float(config.propensity_intercept),
        "propensity_coef": (
            float(config.propensity_coef)
            if np.ndim(config.propensity_coef) == 0
            else np.asarray(config.propensity_coef, dtype=float).tolist()
        ),
        "use_true_pi": bool(config.use_true_pi),
        "save_data": bool(config.save_data),
        "cfa_n_factors": config.cfa_n_factors,
        "run_id": run_dir.name,
        "run_dir": str(run_dir),
        "Sigma_shape": list(Sigma.shape),
        "mean": None if mean is None else mean.tolist(),
    }


def _write_params(run_dir: Path, params: dict[str, Any]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "output").mkdir(parents=True, exist_ok=True)
    (run_dir / "params.json").write_text(json.dumps(params, indent=2, default=str))


# =============================================================================
# Per-replication save
# =============================================================================

def _save_method_result(
    output_dir: Path,
    method_name: str,
    result: dict[str, Any],
) -> None:
    method_dir = output_dir / method_name
    method_dir.mkdir(parents=True, exist_ok=True)

    result["covariance"].to_csv(method_dir / "covariance.csv", index=True)

    if "covariance_vech" in result:
        result["covariance_vech"].to_frame().to_csv(
            method_dir / "covariance_vech.csv", index=True
        )
    if "covariance_se" in result:
        result["covariance_se"].to_frame().to_csv(
            method_dir / "covariance_se.csv", index=True
        )
    if "covariance_vcov" in result:
        result["covariance_vcov"].to_csv(
            method_dir / "covariance_vcov.csv", index=True
        )

    if "pi_hat" in result:
        result["pi_hat"].to_frame().to_csv(method_dir / "pi_hat.csv", index=False)

    cfa_params = result.get("cfa_parameters")
    if cfa_params is not None:
        cfa_params.to_frame().to_csv(method_dir / "cfa_parameters.csv", index=True)

    cfa_status = result.get("cfa_status") or {}
    meta = {
        "method": result.get("method", method_name),
        "variant": result.get("variant"),
        "use_true_pi": result.get("use_true_pi"),
        "outcome_z": result.get("outcome_z"),
        "propensity_z": result.get("propensity_z"),
        "score_mean_max_abs": (result.get("diagnostics") or {}).get(
            "score_mean_max_abs"
        ),
        "cfa_converged": cfa_status.get("converged"),
        "cfa_is_psd": cfa_status.get("is_psd"),
        "cfa_is_heywood": cfa_status.get("is_heywood"),
        "cfa_status": cfa_status.get("status"),
    }
    (method_dir / "result_meta.json").write_text(json.dumps(meta, indent=2))


# =============================================================================
# One replication
# =============================================================================

def _rep_seed(base_seed: int | None, rep: int) -> int | None:
    if base_seed is None:
        return None
    return int(base_seed) + int(rep)


def run_replication(
    config: RunConfig,
    rep: int,
    run_dir: Path,
) -> dict[str, Any]:
    """generate → estimation view → methods for one replication."""
    draw = generate(
        n=config.n,
        Sigma=config.Sigma,
        mean=config.mean,
        n_z=config.n_z,
        propensity_intercept=config.propensity_intercept,
        propensity_coef=config.propensity_coef,
        seed=_rep_seed(config.seed, rep),
        run_dir=run_dir if config.save_data else None,
        rep=rep if config.save_data else None,
    )

    est_draw = build_estimation_draw(
        draw,
        outcome_z=config.outcome_z,
        propensity_z=config.propensity_z,
    )
    if config.cfa_n_factors is not None:
        est_draw["params"]["cfa_n_factors"] = int(config.cfa_n_factors)

    output_rep = run_dir / "output" / f"rep_{rep:04d}"
    output_rep.mkdir(parents=True, exist_ok=True)

    results = {}
    for name in config.methods:
        result = run_method(name, est_draw, use_true_pi=config.use_true_pi)
        results[name] = result
        _save_method_result(output_rep, name, result)

    return results


# =============================================================================
# Full simulation
# =============================================================================

def _validate_config(config: RunConfig) -> None:
    if config.scenario is not None and config.scenario not in available_scenarios():
        raise ValueError(f"Unknown scenario '{config.scenario}'")

    allowed_z = available_z_transforms()
    if config.outcome_z not in allowed_z:
        raise ValueError(f"Unknown outcome_z '{config.outcome_z}'")
    if config.propensity_z not in allowed_z:
        raise ValueError(f"Unknown propensity_z '{config.propensity_z}'")
    for name in config.methods:
        if name not in available_methods():
            raise ValueError(f"Unknown method '{name}'")


def _run_simulation_one(config: RunConfig, *, n_tag: int | None = None) -> Path:
    """Run one sample size; ``config.n`` must already be a single int."""
    _validate_config(config)

    run_dir = _resolve_run_dir(config, n_tag=n_tag)
    params = _params_dict(config, run_dir)
    _write_params(run_dir, params)

    Sigma = pd.DataFrame(np.asarray(config.Sigma, dtype=float))
    Sigma.to_csv(run_dir / "Sigma.csv", index=False)
    if config.mean is not None:
        pd.Series(np.asarray(config.mean, dtype=float), name="mean").to_frame().to_csv(
            run_dir / "mean.csv", index=False
        )

    for rep in range(int(config.reps)):
        run_replication(config, rep, run_dir)

    metrics_summary = summarize_run(run_dir, write=True)

    summary = {
        "run_dir": str(run_dir),
        "run_id": run_dir.name,
        "n": int(config.n),
        "reps_completed": int(config.reps),
        "methods": list(config.methods),
        "scenario": config.scenario,
        "outcome_z": config.outcome_z,
        "propensity_z": config.propensity_z,
        "use_true_pi": config.use_true_pi,
        "summary_written": True,
        "summary_methods": list(metrics_summary.get("methods", {})),
    }
    (run_dir / "output" / "RUN_COMPLETE.json").write_text(
        json.dumps(summary, indent=2)
    )
    return run_dir


def run_simulation(config: RunConfig) -> Path | list[Path]:
    """Run all replications and write artifacts under ``runs/<run_id>/``.

    ``config.n`` may be one int or a sequence (e.g. ``[250, 500, 1000]``).
    Each sample size gets its own non-colliding run folder (``sim_1``,
    ``sim_2``, ... by default). Returns a ``Path`` for one ``n``, or a
    ``list[Path]`` when several sample sizes are requested.
    """
    config = _apply_scenario(config)
    n_values = _as_n_list(config.n)

    if len(n_values) == 1:
        return _run_simulation_one(replace(config, n=n_values[0]))

    # Share one study index across the n-grid: sim_K_n250, sim_K_n500, ...
    if config.run_dir is None and config.run_id is None:
        config = replace(config, run_id=_next_sim_id(_ROOT / "runs"))

    run_dirs: list[Path] = []
    for n in n_values:
        run_dirs.append(_run_simulation_one(replace(config, n=n), n_tag=n))
    return run_dirs


def config_to_dict(config: RunConfig) -> dict[str, Any]:
    """Serialize config without dumping huge arrays twice in callers."""
    data = asdict(config)
    data["n"] = _as_n_list(config.n)
    data["Sigma"] = np.asarray(config.Sigma, dtype=float).tolist()
    if config.mean is not None:
        data["mean"] = np.asarray(config.mean, dtype=float).tolist()
    data["methods"] = list(config.methods)
    return data
