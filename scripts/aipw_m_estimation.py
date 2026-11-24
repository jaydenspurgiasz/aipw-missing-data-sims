"""M-estimation sandwich variance for AIPW covariance.

Point estimate and analytic Jacobian follow the simple (Z-only) AIPW
estimating equations when ``augmented=False`` (memo §3 / variance reference).

When ``augmented=True`` (main estimator), μ/ν scores use §2.1 predictions.
Nuisance blocks stay analytic; derivatives of μ/ν scores w.r.t. β and Σ use
finite differences with other parameters held fixed.

Moment coordinates are ``(μ, vech(E[XXᵀ]))``. Reported ``covariance_vcov``
is the sandwich covariance of ``vech(Σ)`` with ``Σ = E[XXᵀ] - μμᵀ``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import expit

from aipw import (
    _aipw_moments,
    _augmented_predictions,
    _design_with_intercept,
    _fit_linear_outcome_models,
    _simple_predictions,
)


# =============================================================================
# Small helpers
# =============================================================================

def _as_2d(name: str, values) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 1-D or 2-D, got shape {arr.shape}")
    return arr


def _pairs(p: int) -> list[tuple[int, int]]:
    return list(zip(*np.tril_indices(p)))


def _vech_from_matrix(matrix: np.ndarray, pairs) -> np.ndarray:
    return np.array([matrix[j, k] for j, k in pairs], dtype=float)


def _matrix_from_vech(vech: np.ndarray, pairs, p: int) -> np.ndarray:
    out = np.zeros((p, p))
    for value, (j, k) in zip(vech, pairs):
        out[j, k] = out[k, j] = value
    return out


def _sandwich(bread: np.ndarray, meat: np.ndarray, n: int) -> np.ndarray:
    try:
        inv_bread = np.linalg.solve(bread, np.eye(bread.shape[0]))
    except np.linalg.LinAlgError as exc:
        raise ValueError("Bread matrix is singular; sandwich unavailable.") from exc
    return inv_bread @ meat @ inv_bread.T / n


# =============================================================================
# Parameter layout
# =============================================================================

@dataclass(frozen=True)
class ParameterSlices:
    beta: slice
    alpha: slice | None
    rho: slice
    sigma: slice
    mu: slice
    nu: slice


def _parameter_slices(p: int, q_outcome: int, q_propensity: int, *, fit_propensity: bool):
    start = 0
    beta = slice(start, start + q_outcome * p)
    start = beta.stop

    if fit_propensity:
        alpha = slice(start, start + q_propensity)
        start = alpha.stop
    else:
        alpha = None

    rho = slice(start, start + p * p)
    start = rho.stop

    n_pairs = p * (p + 1) // 2
    sigma = slice(start, start + n_pairs)
    start = sigma.stop

    mu = slice(start, start + p)
    start = mu.stop
    nu = slice(start, start + n_pairs)
    return ParameterSlices(beta=beta, alpha=alpha, rho=rho, sigma=sigma, mu=mu, nu=nu)


# =============================================================================
# Nuisance fits
# =============================================================================

def _fit_shared_propensity(response: np.ndarray, design: np.ndarray,
                           tolerance: float = 1e-10, max_iter: int = 100):
    """Shared binomial-logit for the p item responses on a row."""
    alpha = np.zeros(design.shape[1])
    response_count = response.sum(axis=1)
    p = response.shape[1]

    for _ in range(max_iter):
        pi = expit(design @ alpha)
        score = design.T @ (response_count - p * pi)
        weight = p * pi * (1.0 - pi)
        information = design.T @ (weight[:, None] * design)
        try:
            step = np.linalg.solve(information, score)
        except np.linalg.LinAlgError as exc:
            raise ValueError("Propensity design is singular or separated.") from exc
        alpha_next = alpha + step
        if np.max(np.abs(step)) <= tolerance * (1.0 + np.max(np.abs(alpha_next))):
            return alpha_next
        alpha = alpha_next

    raise RuntimeError("Propensity model did not converge.")


def _fit_pairwise_residual_moments(residual: np.ndarray, response: np.ndarray, pairs):
    """Match pairwise_cov: pairwise residual means and unbiased covariances."""
    _, p = residual.shape
    rho = np.zeros((p, p))
    sigma = np.zeros((p, p))

    for j in range(p):
        for k in range(p):
            paired = (response[:, j] * response[:, k]).astype(bool)
            if paired.sum() == 0:
                raise ValueError(f"No jointly observed residuals for pair ({j}, {k}).")
            rho[j, k] = residual[paired, j].mean()

    for j, k in pairs:
        paired = (response[:, j] * response[:, k]).astype(bool)
        count = int(paired.sum())
        if count <= 1:
            raise ValueError(f"Too few jointly observed residuals for pair ({j}, {k}).")
        centered_j = residual[paired, j] - rho[j, k]
        centered_k = residual[paired, k] - rho[k, j]
        sigma[j, k] = sigma[k, j] = centered_j @ centered_k / (count - 1)

    return rho, sigma


def _residual_count_factor(response: np.ndarray, j: int, k: int) -> float:
    count = int(np.sum(response[:, j] * response[:, k]))
    if count <= 1:
        raise ValueError(f"Too few jointly observed residuals for pair ({j}, {k}).")
    return (count - 1) / count


# =============================================================================
# Predictions and AIPW contributions
# =============================================================================

def _predictions(X, R, fitted_mean, Sigma, *, augmented: bool):
    if augmented:
        return _augmented_predictions(X, R, fitted_mean, Sigma)
    return _simple_predictions(fitted_mean, Sigma)


def _covariance_delta(mu: np.ndarray, p: int, pairs) -> np.ndarray:
    """Jacobian of vech(Σ) w.r.t. (μ, vech(E[XXᵀ]))."""
    n_pairs = len(pairs)
    delta = np.zeros((n_pairs, p + n_pairs))
    for pair_index, (j, k) in enumerate(pairs):
        delta[pair_index, p + pair_index] = 1.0
        delta[pair_index, j] -= mu[k]
        delta[pair_index, k] -= mu[j]
    return delta


# =============================================================================
# Scores and bread
# =============================================================================

def _nuisance_scores_and_bread(
    *,
    observed_x,
    response,
    outcome_design,
    propensity_design,
    beta,
    alpha,
    pi,
    fitted_mean,
    residual,
    rho,
    sigma,
    pairs,
    slices: ParameterSlices,
    fit_propensity: bool,
):
    """β / α / ρ / σ scores and their analytic Jacobian blocks."""
    n, p = observed_x.shape
    q_outcome = outcome_design.shape[1]
    dimension = slices.nu.stop
    scores = np.zeros((n, dimension))
    bread = np.zeros((dimension, dimension))

    for i in range(n):
        z_out = outcome_design[i]
        r = response[i]
        e = residual[i]

        # ---- outcome regressions ----
        for j in range(p):
            block = slice(
                slices.beta.start + j * q_outcome,
                slices.beta.start + (j + 1) * q_outcome,
            )
            scores[i, block] = r[j] * z_out * e[j]
            bread[block, block] -= r[j] * np.outer(z_out, z_out)

        # ---- shared propensity ----
        if fit_propensity:
            z_prop = propensity_design[i]
            propensity = pi[i]
            scores[i, slices.alpha] = z_prop * (r.sum() - p * propensity)
            bread[slices.alpha, slices.alpha] -= (
                p * propensity * (1.0 - propensity) * np.outer(z_prop, z_prop)
            )

        # ---- pairwise residual means ----
        for j in range(p):
            for k in range(p):
                rho_index = slices.rho.start + j * p + k
                paired = r[j] * r[k]
                scores[i, rho_index] = paired * (e[j] - rho[j, k])
                beta_block = slice(
                    slices.beta.start + j * q_outcome,
                    slices.beta.start + (j + 1) * q_outcome,
                )
                bread[rho_index, beta_block] -= paired * z_out
                bread[rho_index, rho_index] -= paired

        # ---- residual covariance ----
        for pair_index, (j, k) in enumerate(pairs):
            sigma_index = slices.sigma.start + pair_index
            paired = r[j] * r[k]
            rho_jk = slices.rho.start + j * p + k
            rho_kj = slices.rho.start + k * p + j
            centered_j = e[j] - rho[j, k]
            centered_k = e[k] - rho[k, j]
            count_factor = _residual_count_factor(response, j, k)
            scores[i, sigma_index] = paired * (
                centered_j * centered_k - count_factor * sigma[j, k]
            )

            beta_j = slice(
                slices.beta.start + j * q_outcome,
                slices.beta.start + (j + 1) * q_outcome,
            )
            beta_k = slice(
                slices.beta.start + k * q_outcome,
                slices.beta.start + (k + 1) * q_outcome,
            )
            if j == k:
                bread[sigma_index, beta_j] -= 2.0 * paired * centered_j * z_out
                bread[sigma_index, rho_jk] -= 2.0 * paired * centered_j
            else:
                bread[sigma_index, beta_j] -= paired * centered_k * z_out
                bread[sigma_index, beta_k] -= paired * centered_j * z_out
                bread[sigma_index, rho_jk] -= paired * centered_k
                bread[sigma_index, rho_kj] -= paired * centered_j
            bread[sigma_index, sigma_index] -= paired * count_factor

    return scores, bread


def _moment_scores(
    mean_contrib: np.ndarray,
    second_contrib: np.ndarray,
    mu: np.ndarray,
    nu: np.ndarray,
    pairs,
    slices: ParameterSlices,
):
    # mean_contrib: (n, p); second_contrib: (p, p, n)
    n = mean_contrib.shape[0]
    scores = np.zeros((n, slices.nu.stop))
    scores[:, slices.mu] = mean_contrib - mu
    for pair_index, (j, k) in enumerate(pairs):
        scores[:, slices.nu.start + pair_index] = second_contrib[j, k, :] - nu[pair_index]
    return scores


def _simple_moment_bread(
    *,
    observed_x,
    response,
    outcome_design,
    propensity_design,
    fitted_mean,
    sigma,
    pi,
    pairs,
    slices: ParameterSlices,
    fit_propensity: bool,
):
    """Analytic ∂ψ_{μ,ν}/∂θ for simple Z-only predictions."""
    n, p = observed_x.shape
    q_outcome = outcome_design.shape[1]
    bread = np.zeros((slices.nu.stop, slices.nu.stop))

    for i in range(n):
        # ∂ψ_μ/∂μ = ∂ψ_ν/∂ν = -I for each observation.
        bread[slices.mu, slices.mu] -= np.eye(p)
        bread[slices.nu, slices.nu] -= np.eye(len(pairs))

        z_out = outcome_design[i]
        r = response[i]
        x = observed_x[i]
        m = fitted_mean[i]
        propensity = pi[i]

        for j in range(p):
            beta_block = slice(
                slices.beta.start + j * q_outcome,
                slices.beta.start + (j + 1) * q_outcome,
            )
            bread[slices.mu.start + j, beta_block] += (1.0 - r[j] / propensity) * z_out
            if fit_propensity:
                z_prop = propensity_design[i]
                bread[slices.mu.start + j, slices.alpha] -= (
                    r[j] / propensity * (x[j] - m[j]) * (1.0 - propensity) * z_prop
                )

        for pair_index, (j, k) in enumerate(pairs):
            row = slices.nu.start + pair_index
            paired = r[j] * r[k]
            exponent = 1 if j == k else 2
            weight = paired / propensity ** exponent
            multiplier = 1.0 - weight
            beta_j = slice(
                slices.beta.start + j * q_outcome,
                slices.beta.start + (j + 1) * q_outcome,
            )
            beta_k = slice(
                slices.beta.start + k * q_outcome,
                slices.beta.start + (k + 1) * q_outcome,
            )
            if j == k:
                bread[row, beta_j] += multiplier * 2.0 * m[j] * z_out
            else:
                bread[row, beta_j] += multiplier * m[k] * z_out
                bread[row, beta_k] += multiplier * m[j] * z_out

            conditional_second = m[j] * m[k] + sigma[j, k]
            bread[row, slices.sigma.start + pair_index] += multiplier
            if fit_propensity:
                z_prop = propensity_design[i]
                bread[row, slices.alpha] -= (
                    exponent * weight * (x[j] * x[k] - conditional_second)
                    * (1.0 - propensity) * z_prop
                )

    return bread


def _augmented_moment_bread_fd(
    *,
    X,
    R,
    X_filled,
    outcome_design,
    propensity_design,
    beta,
    fitted_mean,
    sigma,
    pi,
    pairs,
    slices: ParameterSlices,
    fit_propensity: bool,
    step: float = 1e-6,
):
    """Finite-difference ∂ψ_{μ,ν}/∂(β, Σ), plus analytic α and -I blocks."""
    n, p = X.shape
    q_outcome = outcome_design.shape[1]
    bread = np.zeros((slices.nu.stop, slices.nu.stop))
    # Same as the simple analytic path: -I once per observation.
    bread[slices.mu, slices.mu] -= n * np.eye(p)
    bread[slices.nu, slices.nu] -= n * np.eye(len(pairs))

    # Analytic propensity derivatives using current augmented predictions.
    mean_pred, second_pred = _augmented_predictions(X, R, fitted_mean, sigma)
    _, mean_contrib, second_contrib = _aipw_moments(
        X_filled, R, np.broadcast_to(pi[:, None], (n, p)).copy(), mean_pred, second_pred
    )

    if fit_propensity:
        for i in range(n):
            z_prop = propensity_design[i]
            propensity = pi[i]
            r = R[i]
            x = X_filled[i]
            m = mean_pred[i]
            for j in range(p):
                bread[slices.mu.start + j, slices.alpha] -= (
                    r[j] / propensity * (x[j] - m[j]) * (1.0 - propensity) * z_prop
                )
            for pair_index, (j, k) in enumerate(pairs):
                row = slices.nu.start + pair_index
                paired = r[j] * r[k]
                exponent = 1 if j == k else 2
                weight = paired / propensity ** exponent
                q_jk = second_pred[j, k, i]
                bread[row, slices.alpha] -= (
                    exponent * weight * (x[j] * x[k] - q_jk)
                    * (1.0 - propensity) * z_prop
                )

    def contrib_from_beta_sigma(beta_mat, sigma_mat):
        fitted = outcome_design @ beta_mat
        pred_mean, pred_second = _augmented_predictions(X, R, fitted, sigma_mat)
        pi_mat = np.broadcast_to(pi[:, None], (n, p)).copy()
        _, m_contrib, s_contrib = _aipw_moments(
            X_filled, R, pi_mat, pred_mean, pred_second
        )
        return m_contrib, s_contrib

    base_mean, base_second = mean_contrib, second_contrib

    # ∂/∂β  (accumulate sums over i; caller divides by n)
    for j in range(p):
        for q in range(q_outcome):
            beta_pert = beta.copy()
            beta_pert[q, j] += step
            pert_mean, pert_second = contrib_from_beta_sigma(beta_pert, sigma)
            col = slices.beta.start + j * q_outcome + q
            d_mu = (pert_mean - base_mean) / step
            bread[slices.mu, col] += d_mu.sum(axis=0)
            for pair_index, (a, b) in enumerate(pairs):
                d_nu = (pert_second[a, b, :] - base_second[a, b, :]).sum() / step
                bread[slices.nu.start + pair_index, col] += d_nu

    # ∂/∂Σ (vech)
    for pair_index, (a, b) in enumerate(pairs):
        sigma_pert = sigma.copy()
        sigma_pert[a, b] += step
        sigma_pert[b, a] = sigma_pert[a, b]
        pert_mean, pert_second = contrib_from_beta_sigma(beta, sigma_pert)
        col = slices.sigma.start + pair_index
        d_mu = (pert_mean - base_mean) / step
        bread[slices.mu, col] += d_mu.sum(axis=0)
        for other_index, (c, d) in enumerate(pairs):
            d_nu = (pert_second[c, d, :] - base_second[c, d, :]).sum() / step
            bread[slices.nu.start + other_index, col] += d_nu

    return bread


# =============================================================================
# Public API
# =============================================================================

def aipw_m_estimation(
    X,
    Z_outcome,
    Z_propensity=None,
    *,
    augmented: bool = True,
    pi: np.ndarray | None = None,
    fd_step: float = 1e-6,
):
    """AIPW covariance point estimate plus sandwich variance for vech(Σ).

    Parameters
    ----------
    X:
        Outcomes (n, p) with ``np.nan`` missing.
    Z_outcome:
        Covariates for the linear outcome model.
    Z_propensity:
        Covariates for the shared binomial propensity model. Defaults to
        ``Z_outcome``. Ignored when ``pi`` is supplied.
    augmented:
        ``True`` = main §2.1 estimator; ``False`` = simple §3.
    pi:
        If given, propensity is treated as fixed (no α block in the sandwich).
    """
    X = _as_2d("X", X)
    n, p = X.shape
    outcome_design = _design_with_intercept(_as_2d("Z_outcome", Z_outcome))
    if outcome_design.shape[0] != n:
        raise ValueError("Z_outcome must have n rows")

    response = (~np.isnan(X)).astype(float)
    observed_x = np.where(response == 1.0, X, 0.0)
    pairs = _pairs(p)
    fit_propensity = pi is None

    if fit_propensity:
        if Z_propensity is None:
            Z_propensity = Z_outcome
        propensity_design = _design_with_intercept(
            _as_2d("Z_propensity", Z_propensity)
        )
        if propensity_design.shape[0] != n:
            raise ValueError("Z_propensity must have n rows")
        alpha = _fit_shared_propensity(response, propensity_design)
        pi_hat = expit(propensity_design @ alpha)
    else:
        propensity_design = None
        alpha = None
        pi_hat = np.asarray(pi, dtype=float).reshape(-1)
        if pi_hat.shape != (n,):
            raise ValueError(f"pi must have shape (n,)={(n,)}, got {pi_hat.shape}")
        if np.any(pi_hat <= 0.0) or np.any(pi_hat >= 1.0):
            raise ValueError("pi entries must lie in (0, 1).")

    slices = _parameter_slices(
        p,
        outcome_design.shape[1],
        0 if not fit_propensity else propensity_design.shape[1],
        fit_propensity=fit_propensity,
    )

    beta, fitted_mean = _fit_linear_outcome_models(X, outcome_design, response)
    residual = np.where(response == 1.0, X - fitted_mean, 0.0)
    rho, sigma = _fit_pairwise_residual_moments(residual, response, pairs)

    mean_pred, second_pred = _predictions(
        X, response, fitted_mean, sigma, augmented=augmented
    )
    pi_mat = np.broadcast_to(pi_hat[:, None], (n, p)).copy()
    covariance, mean_contrib, second_contrib = _aipw_moments(
        observed_x, response, pi_mat, mean_pred, second_pred
    )
    mu = mean_contrib.mean(axis=0)
    nu = np.array([second_contrib[j, k, :].mean() for j, k in pairs])

    scores_nuisance, bread_nuisance = _nuisance_scores_and_bread(
        observed_x=observed_x,
        response=response,
        outcome_design=outcome_design,
        propensity_design=propensity_design,
        beta=beta,
        alpha=alpha,
        pi=pi_hat,
        fitted_mean=fitted_mean,
        residual=np.where(response == 1.0, residual, 0.0),
        rho=rho,
        sigma=sigma,
        pairs=pairs,
        slices=slices,
        fit_propensity=fit_propensity,
    )

    scores = scores_nuisance.copy()
    scores += _moment_scores(mean_contrib, second_contrib, mu, nu, pairs, slices)

    if augmented:
        bread_moments = _augmented_moment_bread_fd(
            X=X,
            R=response,
            X_filled=observed_x,
            outcome_design=outcome_design,
            propensity_design=propensity_design,
            beta=beta,
            fitted_mean=fitted_mean,
            sigma=sigma,
            pi=pi_hat,
            pairs=pairs,
            slices=slices,
            fit_propensity=fit_propensity,
            step=fd_step,
        )
    else:
        bread_moments = _simple_moment_bread(
            observed_x=observed_x,
            response=response,
            outcome_design=outcome_design,
            propensity_design=propensity_design,
            fitted_mean=fitted_mean,
            sigma=sigma,
            pi=pi_hat,
            pairs=pairs,
            slices=slices,
            fit_propensity=fit_propensity,
        )

    bread = (bread_nuisance + bread_moments) / n
    score_mean = scores.mean(axis=0)
    if np.max(np.abs(score_mean)) > 1e-6:
        raise RuntimeError(
            f"Stacked score did not average to zero (max |mean|={np.max(np.abs(score_mean)):.3e})."
        )

    meat = scores.T @ scores / n
    parameter_vcov = _sandwich(bread, meat, n)
    moment_vcov = parameter_vcov[slices.mu.start:slices.nu.stop, slices.mu.start:slices.nu.stop]
    delta = _covariance_delta(mu, p, pairs)
    covariance_vcov = delta @ moment_vcov @ delta.T
    covariance_se = np.sqrt(np.clip(np.diag(covariance_vcov), 0.0, None))

    return {
        "variant": "augmented" if augmented else "simple",
        "beta": beta,
        "alpha": alpha,
        "pi": pi_hat,
        "residual_means": rho,
        "residual_covariance": sigma,
        "mean": mu,
        "raw_second_moment_vech": nu,
        "covariance": covariance,
        "covariance_vech": _vech_from_matrix(covariance, pairs),
        "covariance_vcov": covariance_vcov,
        "covariance_se": covariance_se,
        "pairs": pairs,
        "scores": scores,
        "score_mean": score_mean,
        "bread": bread,
        "meat": meat,
        "parameter_vcov": parameter_vcov,
        "moment_vcov": moment_vcov,
        "parameter_slices": slices,
        "fit_propensity": fit_propensity,
    }
