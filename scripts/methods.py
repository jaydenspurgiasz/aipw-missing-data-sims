"""Estimation methods used by the simulation runner.

Each method takes an estimation draw with explicit nuisance fields:

- ``X_obs``
- ``Z_outcome``      design for the outcome / AIPW prediction model
- ``Z_propensity``   design for the fitted propensity (unless use_true_pi)
- ``pi``             true DGP propensity (oracle option)

Point estimates and sandwich SEs come from ``aipw_m_estimation``.
``aipw`` is the main augmented estimator; ``aipw_simple`` is optional.
CFA (Semopy two-factor) is fit to the AIPW covariance after estimation.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd

from aipw_m_estimation import aipw_m_estimation
from cfa import fit_cfa


# =============================================================================
# Shared draw helpers
# =============================================================================

def _x_matrix(draw: dict[str, Any]) -> np.ndarray:
    return draw["X_obs"].to_numpy(dtype=float)


def _z_matrix(draw: dict[str, Any], key: str) -> np.ndarray:
    if key not in draw:
        raise KeyError(
            f"Estimation draw is missing '{key}'. "
            "Build the draw with misspec.build_estimation_draw first."
        )
    Z = draw[key]
    if isinstance(Z, pd.Series):
        return Z.to_numpy(dtype=float).reshape(-1, 1)
    return Z.to_numpy(dtype=float)


# =============================================================================
# AIPW + sandwich
# =============================================================================

def _run_aipw(
    draw: dict[str, Any],
    *,
    augmented: bool,
    method_name: str,
    use_true_pi: bool,
) -> dict[str, Any]:
    X = _x_matrix(draw)
    Z_outcome = _z_matrix(draw, "Z_outcome")
    Z_propensity = _z_matrix(draw, "Z_propensity")

    pi = None
    if use_true_pi:
        pi = draw["pi"].to_numpy(dtype=float)

    fit = aipw_m_estimation(
        X,
        Z_outcome,
        None if use_true_pi else Z_propensity,
        augmented=augmented,
        pi=pi,
    )

    names = list(draw["X_obs"].columns)
    pairs = fit["pairs"]
    pair_labels = [f"{names[j]}__{names[k]}" for j, k in pairs]
    n_obs = int(draw["X_obs"].shape[0])

    cfa_n_factors = draw.get("params", {}).get("cfa_n_factors")
    if cfa_n_factors is not None:
        cfa_n_factors = int(cfa_n_factors)
    cfa = fit_cfa(
        fit["covariance"],
        n_obs,
        n_factors=cfa_n_factors,
        names=names,
    )

    return {
        "method": method_name,
        "covariance": pd.DataFrame(fit["covariance"], index=names, columns=names),
        "covariance_vech": pd.Series(
            fit["covariance_vech"], index=pair_labels, name="vech"
        ),
        "covariance_se": pd.Series(
            fit["covariance_se"], index=pair_labels, name="se"
        ),
        "covariance_vcov": pd.DataFrame(
            fit["covariance_vcov"], index=pair_labels, columns=pair_labels
        ),
        "cfa": cfa,
        "cfa_parameters": cfa.get("parameters"),
        "cfa_status": {
            "converged": cfa["converged"],
            "is_psd": cfa["is_psd"],
            "is_heywood": cfa["is_heywood"],
            "status": cfa["status"],
        },
        "pi_hat": pd.Series(fit["pi"], name="pi_hat"),
        "use_true_pi": bool(use_true_pi),
        "outcome_z": draw.get("params", {}).get("outcome_z"),
        "propensity_z": draw.get("params", {}).get("propensity_z"),
        "variant": fit["variant"],
        "diagnostics": {
            "beta": fit["beta"],
            "alpha": fit["alpha"],
            "residual_covariance": fit["residual_covariance"],
            "score_mean_max_abs": float(np.max(np.abs(fit["score_mean"]))),
            "fit_propensity": fit["fit_propensity"],
        },
    }


def aipw_main(draw: dict[str, Any], **kwargs) -> dict[str, Any]:
    """Main AIPW: augmented §2.1 with M-estimation sandwich SEs."""
    return _run_aipw(
        draw,
        augmented=True,
        method_name="aipw",
        use_true_pi=bool(kwargs.get("use_true_pi", False)),
    )


def aipw_simple(draw: dict[str, Any], **kwargs) -> dict[str, Any]:
    """Z-only §3 AIPW with analytic M-estimation sandwich SEs."""
    return _run_aipw(
        draw,
        augmented=False,
        method_name="aipw_simple",
        use_true_pi=bool(kwargs.get("use_true_pi", False)),
    )


# =============================================================================
# Registry
# =============================================================================

MethodFn = Callable[..., dict[str, Any]]

METHODS: dict[str, MethodFn] = {
    "aipw": aipw_main,
    "aipw_simple": aipw_simple,
}


def run_method(name: str, draw: dict[str, Any], **kwargs) -> dict[str, Any]:
    if name not in METHODS:
        known = ", ".join(sorted(METHODS))
        raise ValueError(f"Unknown method '{name}'. Expected one of: {known}")
    return METHODS[name](draw, **kwargs)


def available_methods() -> list[str]:
    return sorted(METHODS)
