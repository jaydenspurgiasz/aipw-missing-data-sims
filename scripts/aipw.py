"""AIPW covariance estimators under MAR.

Main estimator (``aipw_covariance``) is memo §2.1: linear nuisances in Z,
with predictions that also condition on other observed outcomes.

``aipw_covariance_simple`` keeps memo §3 (Z-only predictions) for comparison.
In n=500/1000, 100-rep sims, the augmented estimator had better variance
(and MSE) with similar bias; we may re-test both later.

Numpy in, numpy out. Propensity weights assume item-wise independence:
π_jk = π_j π_k (diagonal second-moment terms use π_j).
"""

from __future__ import annotations

import numpy as np

from covariance import pairwise_cov


# =============================================================================
# Input shaping
# =============================================================================

def _as_2d(name: str, values) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 1-D or 2-D, got shape {arr.shape}")
    return arr


def _response_and_filled(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    R = (~np.isnan(X)).astype(float)
    X_filled = np.where(R == 1.0, X, 0.0)
    return R, X_filled


def _propensity_matrix(pi, n: int, p: int) -> np.ndarray:
    pi = np.asarray(pi, dtype=float)
    if pi.ndim == 1:
        if pi.shape[0] != n:
            raise ValueError(f"pi must have length n={n}, got {pi.shape[0]}")
        pi = np.broadcast_to(pi[:, None], (n, p)).copy()
    elif pi.shape != (n, p):
        raise ValueError(f"pi must have shape (n,) or (n, p)={(n, p)}, got {pi.shape}")
    if np.any(pi <= 0.0) or np.any(pi > 1.0):
        raise ValueError("pi entries must lie in (0, 1].")
    return pi


def _design_with_intercept(Z: np.ndarray) -> np.ndarray:
    Z = _as_2d("Z", Z)
    return np.column_stack([np.ones(Z.shape[0]), Z])


# =============================================================================
# Nuisance models: linear E[X|Z] and residual covariance Σ
# =============================================================================

def _fit_linear_outcome_models(X: np.ndarray, design: np.ndarray, R: np.ndarray):
    """OLS of each column of X on design, using observed rows only."""
    n, p = X.shape
    d = design.shape[1]
    beta = np.zeros((d, p))

    for j in range(p):
        observed = R[:, j] == 1.0
        if observed.sum() < d:
            continue
        coef, *_ = np.linalg.lstsq(design[observed], X[observed, j], rcond=None)
        beta[:, j] = coef

    fitted_mean = design @ beta
    return beta, fitted_mean


def _residual_covariance(X: np.ndarray, fitted_mean: np.ndarray) -> np.ndarray:
    """Pairwise residual covariance; missing X entries stay missing."""
    return pairwise_cov(X - fitted_mean)


# =============================================================================
# Simple AIPW predictions (§3): condition on Z only
# =============================================================================

def _simple_predictions(fitted_mean: np.ndarray, Sigma: np.ndarray):
    """Return per-row mean predictions and second-moment predictions."""
    # mean_pred: (n, p)
    # second_pred: (p, p, n) with second_pred[j, k, i] = Σ_jk + m_j m_k
    n, p = fitted_mean.shape
    second_pred = np.empty((p, p, n))
    for i in range(n):
        m = fitted_mean[i]
        second_pred[:, :, i] = Sigma + np.outer(m, m)
    return fitted_mean, second_pred


# =============================================================================
# Augmented AIPW predictions (§2.1): condition on other observed X + Z
# =============================================================================

def _solve_o(Sigma_oo: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Solve Σ_oo x = rhs; fall back to least squares if needed."""
    try:
        return np.linalg.solve(Sigma_oo, rhs)
    except np.linalg.LinAlgError:
        sol, *_ = np.linalg.lstsq(Sigma_oo, rhs, rcond=None)
        return sol


def _augmented_mean_one(
    j: int,
    x_row: np.ndarray,
    observed: np.ndarray,
    mean_row: np.ndarray,
    Sigma: np.ndarray,
) -> float:
    """E[Xj | X_{o\\j}, Z] under the linear residual model."""
    o_mask = observed.copy()
    o_mask[j] = False
    o_idx = np.flatnonzero(o_mask)

    if o_idx.size == 0:
        return float(mean_row[j])

    resid_o = x_row[o_idx] - mean_row[o_idx]
    adj = Sigma[j, o_idx] @ _solve_o(Sigma[np.ix_(o_idx, o_idx)], resid_o)
    return float(mean_row[j] + adj)


def _augmented_second_one(
    j: int,
    k: int,
    x_row: np.ndarray,
    observed: np.ndarray,
    mean_row: np.ndarray,
    Sigma: np.ndarray,
) -> float:
    """E[Xj Xk | X_{o\\{j,k}}, Z] under the linear residual model."""
    o_mask = observed.copy()
    o_mask[j] = False
    o_mask[k] = False
    o_idx = np.flatnonzero(o_mask)

    if o_idx.size == 0:
        return float(Sigma[j, k] + mean_row[j] * mean_row[k])

    resid_o = x_row[o_idx] - mean_row[o_idx]
    Sigma_oo_inv_r = _solve_o(Sigma[np.ix_(o_idx, o_idx)], resid_o)

    m_j = mean_row[j] + Sigma[j, o_idx] @ Sigma_oo_inv_r
    m_k = mean_row[k] + Sigma[k, o_idx] @ Sigma_oo_inv_r
    cov_jk = Sigma[j, k] - Sigma[j, o_idx] @ _solve_o(
        Sigma[np.ix_(o_idx, o_idx)], Sigma[o_idx, k]
    )
    return float(cov_jk + m_j * m_k)


def _augmented_predictions(
    X: np.ndarray,
    R: np.ndarray,
    fitted_mean: np.ndarray,
    Sigma: np.ndarray,
):
    n, p = X.shape
    mean_pred = np.empty((n, p))
    second_pred = np.empty((p, p, n))
    observed = R == 1.0

    for i in range(n):
        x_row = X[i]
        obs_row = observed[i]
        mean_row = fitted_mean[i]

        for j in range(p):
            mean_pred[i, j] = _augmented_mean_one(j, x_row, obs_row, mean_row, Sigma)
            for k in range(j + 1):
                value = _augmented_second_one(
                    j, k, x_row, obs_row, mean_row, Sigma
                )
                second_pred[j, k, i] = value
                second_pred[k, j, i] = value

    return mean_pred, second_pred


# =============================================================================
# AIPW aggregation
# =============================================================================

def _aipw_moments(
    X_filled: np.ndarray,
    R: np.ndarray,
    pi: np.ndarray,
    mean_pred: np.ndarray,
    second_pred: np.ndarray,
):
    """Build AIPW contributions for first and second moments."""
    n, p = X_filled.shape

    # Independence: off-diagonal weight 1/(π_j π_k); diagonal second moment 1/π_j.
    w1 = R / pi
    w2 = (R[:, None, :] * R[:, :, None]) / (pi[:, None, :] * pi[:, :, None])
    for j in range(p):
        w2[:, j, j] = w1[:, j]

    # Rearrange weights to (p, p, n) to match second_pred layout.
    w2_ppn = np.transpose(w2, (1, 2, 0))

    cross = np.einsum("ij,ik->jki", X_filled, X_filled)
    mean_contrib = mean_pred + w1 * (X_filled - mean_pred)
    second_contrib = second_pred + w2_ppn * (cross - second_pred)

    mu = mean_contrib.mean(axis=0)
    second_mean = second_contrib.mean(axis=2)
    covariance = second_mean - np.outer(mu, mu)

    return covariance, mean_contrib, second_contrib


# =============================================================================
# Public API
# =============================================================================

def _estimate(
    X,
    Z,
    pi,
    *,
    augmented: bool,
    return_diagnostics: bool,
):
    X = _as_2d("X", X)
    Z = _as_2d("Z", Z)
    if X.shape[0] != Z.shape[0]:
        raise ValueError("X and Z must have the same number of rows")

    n, p = X.shape
    R, X_filled = _response_and_filled(X)
    pi = _propensity_matrix(pi, n, p)
    design = _design_with_intercept(Z)

    beta, fitted_mean = _fit_linear_outcome_models(X, design, R)
    Sigma = _residual_covariance(X, fitted_mean)

    if augmented:
        mean_pred, second_pred = _augmented_predictions(X, R, fitted_mean, Sigma)
        label = "augmented"
    else:
        mean_pred, second_pred = _simple_predictions(fitted_mean, Sigma)
        label = "simple"

    covariance, mean_contrib, second_contrib = _aipw_moments(
        X_filled, R, pi, mean_pred, second_pred
    )

    if not return_diagnostics:
        return covariance

    return covariance, {
        "variant": label,
        "beta": beta,
        "fitted_mean_z": fitted_mean,
        "residual_covariance": Sigma,
        "mean_prediction": mean_pred,
        "second_moment_prediction": second_pred,
        "first_moment_contribution": mean_contrib,
        "second_moment_contribution": second_contrib,
    }


def aipw_covariance(X, Z, pi, *, return_diagnostics: bool = False):
    """Main AIPW covariance estimator (memo §2.1, augmented).

    Uses linear E[X|Z] and residual Σ, with predictions that condition on
    other observed outcomes. Comparison sims showed better variance than
    ``aipw_covariance_simple``; both may be re-tested later.
    """
    return _estimate(
        X, Z, pi, augmented=True, return_diagnostics=return_diagnostics
    )


def aipw_covariance_simple(X, Z, pi, *, return_diagnostics: bool = False):
    """Simple Z-only AIPW (memo §3). Kept for future head-to-head checks.

    Predictions use only m(Z) and mjk(Z) = Σ_jk + m_j(Z) m_k(Z). Prefer
    ``aipw_covariance`` for routine work unless comparing variants.
    """
    return _estimate(
        X, Z, pi, augmented=False, return_diagnostics=return_diagnostics
    )


# Older name: now points at the main (augmented) estimator.
def aipw_z(X, Z, pi, return_diagnostics=False):
    return aipw_covariance(X, Z, pi, return_diagnostics=return_diagnostics)
