"""CFA via Semopy (equal-sized factor blocks).

Factor variances fixed at 1, all loadings free (including the first),
diagonal uniquenesses, free factor correlations. First loading of each
factor is oriented positive after estimation.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from semopy import Model


# =============================================================================
# Model description
# =============================================================================

def _variable_names(p: int, names) -> list[str]:
    if names is None:
        return [f"x{j + 1}" for j in range(p)]
    names = list(names)
    if len(names) != p:
        raise ValueError(f"names must have length {p}, got {len(names)}")
    return names


def _default_n_factors(p: int) -> int:
    """Legacy default: two equal blocks when possible."""
    if p >= 4 and p % 2 == 0:
        return 2
    raise ValueError(f"Cannot infer n_factors for p={p}; pass n_factors explicitly")


def factor_model_description(names: list[str], n_factors: int) -> str:
    """Semopy model string for a CFA with equal-sized blocks."""
    p = len(names)
    if n_factors < 1:
        raise ValueError(f"n_factors must be >= 1, got {n_factors}")
    if p < 2 * n_factors or p % n_factors != 0:
        raise ValueError(
            f"CFA expects p divisible by n_factors with >=2 indicators each; "
            f"got p={p}, n_factors={n_factors}"
        )

    n_per = p // n_factors
    lines: list[str] = []
    for f in range(n_factors):
        block = names[f * n_per:(f + 1) * n_per]
        lines.append(f"f{f + 1} =~ " + " + ".join(block))
    for f in range(n_factors):
        lines.append(f"f{f + 1} ~~ 1*f{f + 1}")
    for a in range(n_factors):
        for b in range(a + 1, n_factors):
            lines.append(f"f{a + 1} ~~ f{b + 1}")
    return "\n".join(lines) + "\n"


def two_factor_description(names: list[str]) -> str:
    return factor_model_description(names, n_factors=2)


# =============================================================================
# Diagnostics
# =============================================================================

HEYWOOD_UNIQUENESS = 1e-4
INADMISSIBLE_CORR = 0.995


def is_positive_definite(matrix: np.ndarray, tol: float = 1e-10) -> bool:
    vals = np.linalg.eigvalsh(np.asarray(matrix, dtype=float))
    return bool(np.all(vals > tol))


def _orient_loadings(loadings_blocks: list[np.ndarray], phi: np.ndarray):
    """Flip factor signs so the first loading on each factor is positive."""
    blocks = [np.asarray(b, dtype=float).copy() for b in loadings_blocks]
    phi = np.asarray(phi, dtype=float).copy()
    k = len(blocks)

    for f in range(k):
        if blocks[f][0] < 0:
            blocks[f] *= -1.0
            phi[f, :] *= -1.0
            phi[:, f] *= -1.0
            phi[f, f] = 1.0
    return blocks, phi


def _phi_offdiag(phi: np.ndarray) -> np.ndarray:
    k = phi.shape[0]
    return np.array([phi[i, j] for i in range(k) for j in range(i + 1, k)], dtype=float)


# =============================================================================
# Fit
# =============================================================================

def _free_first_loadings(model: Model, start: float = 0.8) -> None:
    """Semopy fixes the first loading by default; free it (factor vars are 1)."""
    for key, param in model.parameters.items():
        if key.startswith("_b") and not param.active:
            param.active = True
            param.start = start


def _extract_parameters(
    inspect_df: pd.DataFrame,
    names: list[str],
    n_factors: int,
) -> dict[str, Any]:
    n_per = len(names) // n_factors
    loadings = np.zeros(len(names))
    uniquenesses = np.zeros(len(names))
    phi = np.eye(n_factors)

    factor_names = [f"f{f + 1}" for f in range(n_factors)]
    blocks = {
        f"f{f + 1}": names[f * n_per:(f + 1) * n_per]
        for f in range(n_factors)
    }

    for _, row in inspect_df.iterrows():
        op = row["op"]
        lval, rval = row["lval"], row["rval"]
        est = float(row["Estimate"])

        if op == "~" and rval in blocks and lval in blocks[rval]:
            loadings[names.index(lval)] = est
        elif op == "~~" and lval == rval and lval in names:
            uniquenesses[names.index(lval)] = est
        elif op == "~~" and lval in factor_names and rval in factor_names:
            i = factor_names.index(lval)
            j = factor_names.index(rval)
            phi[i, j] = est
            phi[j, i] = est

    loading_blocks = [
        loadings[f * n_per:(f + 1) * n_per] for f in range(n_factors)
    ]
    loading_blocks, phi = _orient_loadings(loading_blocks, phi)
    loadings = np.concatenate(loading_blocks)
    phi_vec = _phi_offdiag(phi)

    return {
        "loadings": loadings,
        "uniquenesses": uniquenesses,
        "factor_correlation": float(phi_vec[0]) if phi_vec.size == 1 else phi_vec,
        "factor_corr_matrix": phi,
        "theta": np.concatenate([loadings, uniquenesses, phi_vec]),
    }


def fit_cfa(
    covariance,
    n_obs: int,
    *,
    n_factors: int | None = None,
    names=None,
    heywood_threshold: float = HEYWOOD_UNIQUENESS,
) -> dict[str, Any]:
    """Fit an equal-block CFA to a covariance matrix with Semopy."""
    covariance = np.asarray(covariance, dtype=float)
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError("covariance must be square")

    p = covariance.shape[0]
    var_names = _variable_names(p, names)
    if n_factors is None:
        n_factors = _default_n_factors(p)
    n_factors = int(n_factors)

    cov_df = pd.DataFrame(covariance, index=var_names, columns=var_names)

    empty = {
        "converged": False,
        "is_psd": False,
        "is_heywood": False,
        "status": "non_spd",
        "n_factors": n_factors,
        "loadings": None,
        "uniquenesses": None,
        "factor_correlation": None,
        "factor_corr_matrix": None,
        "theta": None,
        "parameters": None,
        "inspect": None,
    }

    psd = is_positive_definite(covariance)
    if not psd:
        return empty

    try:
        model = Model(factor_model_description(var_names, n_factors))
        _free_first_loadings(model, start=0.8)
        result = model.fit(cov=cov_df, obj="MLW", n_samples=int(n_obs))
        success = bool(getattr(result, "success", True))
        inspect_df = model.inspect()
        extracted = _extract_parameters(inspect_df, var_names, n_factors)
    except Exception:
        out = dict(empty)
        out["is_psd"] = True
        out["status"] = "error"
        return out

    uniq = extracted["uniquenesses"]
    phi_mat = extracted["factor_corr_matrix"]
    phi_vec = _phi_offdiag(phi_mat)
    heywood = bool(np.any(uniq <= heywood_threshold))
    inadmissible = heywood or bool(np.any(np.abs(phi_vec) >= INADMISSIBLE_CORR))

    if not success or not np.all(np.isfinite(extracted["theta"])):
        status = "nonconvergence"
        converged = False
    elif inadmissible:
        status = "heywood" if heywood else "inadmissible"
        converged = True
    else:
        status = "success"
        converged = True

    corr_names = [
        f"factor_corr_f{i + 1}_f{j + 1}"
        for i in range(n_factors)
        for j in range(i + 1, n_factors)
    ]
    param_index = (
        [f"loading_{name}" for name in var_names]
        + [f"uniqueness_{name}" for name in var_names]
        + corr_names
    )
    # Keep legacy name for the single two-factor correlation.
    if n_factors == 2 and len(corr_names) == 1:
        param_index[-1] = "factor_correlation"

    parameters = pd.Series(extracted["theta"], index=param_index, name="estimate")

    return {
        "converged": converged,
        "is_psd": True,
        "is_heywood": heywood,
        "status": status,
        "n_factors": n_factors,
        "loadings": extracted["loadings"],
        "uniquenesses": uniq,
        "factor_correlation": extracted["factor_correlation"],
        "factor_corr_matrix": phi_mat,
        "theta": extracted["theta"],
        "parameters": parameters,
        "inspect": inspect_df,
    }


def fit_two_factor_cfa(
    covariance,
    n_obs: int,
    *,
    names=None,
    heywood_threshold: float = HEYWOOD_UNIQUENESS,
) -> dict[str, Any]:
    """Fit a 2-factor CFA (wrapper around ``fit_cfa``)."""
    return fit_cfa(
        covariance,
        n_obs,
        n_factors=2,
        names=names,
        heywood_threshold=heywood_threshold,
    )


# =============================================================================
# Thin wrappers (legacy / delta-method callers)
# =============================================================================

def cfa_two_factor(covariance, n_obs, names=None):
    """Return ``(theta, converged)`` for callers that expect the old API."""
    fit = fit_two_factor_cfa(covariance, n_obs, names=names)
    if fit["theta"] is None:
        return None, False
    ok = bool(fit["converged"] and fit["status"] in ("success", "heywood", "inadmissible"))
    return fit["theta"], ok


def cfa_pass(params, converged) -> bool:
    return bool(converged and params is not None and np.all(np.isfinite(params)))
