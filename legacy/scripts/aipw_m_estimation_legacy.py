"""Analytic M-estimation variance for the repository's AIPW covariance estimator.

The function in this module mirrors ``aipw.aipw_z`` but also returns the
per-observation estimating functions, their analytic Jacobian, and the
sandwich covariance matrix.  It uses the simulation's shared response model:
one binomial-logit propensity fitted to the p response indicators per row.
"""

from dataclasses import dataclass

import numpy as np
from scipy.special import expit


@dataclass(frozen=True)
class _Slices:
    beta: slice
    alpha: slice
    rho: slice
    sigma: slice
    mu: slice
    nu: slice


def aipw_m_estimation_variance(X, Z_outcome, Z_propensity=None, beta=None,
                                alpha=None, tolerance=1e-10, max_iter=100):
    """Estimate AIPW moments and their analytic sandwich covariance.

    Parameters
    ----------
    X : array-like, shape (n, p)
        Outcome matrix; missing entries must be ``np.nan``.
    Z_outcome : array-like, shape (n, q) or (n,)
        Covariates for the conditional-mean model.  An intercept is added.
    Z_propensity : array-like, optional
        Covariates for the shared binomial-logit response model.  Defaults to
        ``Z_outcome``.  Supplying this separately supports the simulation's
        propensity-misspecification scenarios.
    beta, alpha : array-like, optional
        Previously fitted nuisance estimates.  If omitted, they are fitted
        from ``X`` using the same model definitions as the simulation.

    Returns
    -------
    dict
        Point estimates, nuisance estimates, per-observation scores, analytic
        bread and meat matrices, and sandwich covariance matrices.  Moment
        coordinates are ``[mu, vech(raw second moment)]``.

    Notes
    -----
    The residual covariance is estimated with the same pairwise, unbiased
    formula used by :func:`aipw.aipw_z`.  Its pair-specific residual means and
    covariance entries are included in the stacked estimating equation, so
    their uncertainty is not silently treated as fixed.
    """
    X = _as_2d_float(X, "X")
    outcome_design = _add_intercept(Z_outcome, X.shape[0], "Z_outcome")
    if Z_propensity is None:
        Z_propensity = Z_outcome
    propensity_design = _add_intercept(
        Z_propensity, X.shape[0], "Z_propensity"
    )

    n, p = X.shape
    response = (~np.isnan(X)).astype(float)
    observed_x = np.where(response == 1, X, 0.0)
    pairs = list(zip(*np.tril_indices(p)))
    slices = _parameter_slices(p, outcome_design.shape[1], propensity_design.shape[1])

    if beta is None:
        beta = _fit_outcome_models(observed_x, response, outcome_design)
    else:
        beta = _validate_beta(beta, outcome_design.shape[1], p)

    if alpha is None:
        alpha = _fit_shared_propensity(
            response, propensity_design, tolerance, max_iter
        )
    else:
        alpha = _validate_alpha(alpha, propensity_design.shape[1])

    pi = expit(propensity_design @ alpha)
    if np.any(pi <= 0) or np.any(pi >= 1):
        raise ValueError("Propensity estimates must be strictly between zero and one.")

    fitted_mean = outcome_design @ beta
    residual = observed_x - fitted_mean
    rho, sigma = _fit_pairwise_residual_moments(residual, response, pairs)
    first_contribution, second_contribution = _aipw_contributions(
        observed_x, response, fitted_mean, pi, sigma, pairs
    )
    mu = first_contribution.mean(axis=0)
    nu = np.array([second_contribution[:, j, k].mean() for j, k in pairs])

    scores, bread = _scores_and_bread(
        observed_x=observed_x,
        response=response,
        outcome_design=outcome_design,
        propensity_design=propensity_design,
        beta=beta,
        alpha=alpha,
        pi=pi,
        fitted_mean=fitted_mean,
        residual=residual,
        rho=rho,
        sigma=sigma,
        mu=mu,
        nu=nu,
        pairs=pairs,
        slices=slices,
    )

    score_mean = scores.mean(axis=0)
    if np.max(np.abs(score_mean)) > 1e-7:
        raise RuntimeError("Stacked score did not converge to zero.")

    meat = scores.T @ scores / n
    bread_condition_number = np.linalg.cond(bread)
    if not np.isfinite(bread_condition_number):
        raise ValueError("Analytic bread is ill-conditioned; sandwich variance is unavailable.")
    parameter_covariance = _sandwich(bread, meat, n)
    moment_covariance = parameter_covariance[slices.mu.start:slices.nu.stop,
                                              slices.mu.start:slices.nu.stop]
    covariance_estimate = _covariance_from_moments(mu, nu, pairs, p)
    covariance_delta = _covariance_delta(mu, p, pairs)
    covariance_covariance = covariance_delta @ moment_covariance @ covariance_delta.T

    return {
        "beta": beta,
        "alpha": alpha,
        "pi": pi,
        "pair_residual_means": rho,
        "residual_covariance": sigma,
        "mean": mu,
        "raw_second_moment": nu,
        "covariance": covariance_estimate,
        "scores": scores,
        "score_mean": score_mean,
        "bread": bread,
        "bread_condition_number": bread_condition_number,
        "meat": meat,
        "parameter_covariance": parameter_covariance,
        "moment_covariance": moment_covariance,
        "covariance_covariance": covariance_covariance,
        "parameter_slices": slices,
        "pairs": pairs,
    }


def _scores_and_bread(*, observed_x, response, outcome_design,
                      propensity_design, beta, alpha, pi, fitted_mean,
                      residual, rho, sigma, mu, nu, pairs, slices):
    """Evaluate the stacked scores and average analytic Jacobian."""
    n, p = observed_x.shape
    q_outcome = outcome_design.shape[1]
    q_propensity = propensity_design.shape[1]
    dimension = slices.nu.stop
    scores = np.zeros((n, dimension))
    bread = np.zeros((dimension, dimension))

    for i in range(n):
        z_outcome = outcome_design[i]
        z_propensity = propensity_design[i]
        r = response[i]
        x = observed_x[i]
        m = fitted_mean[i]
        e = residual[i]
        propensity = pi[i]

        beta_score = slices.beta.start
        for j in range(p):
            beta_block = slice(beta_score + j * q_outcome,
                               beta_score + (j + 1) * q_outcome)
            scores[i, beta_block] = r[j] * z_outcome * e[j]
            bread[beta_block, beta_block] -= r[j] * np.outer(z_outcome, z_outcome)

        scores[i, slices.alpha] = z_propensity * (r.sum() - p * propensity)
        bread[slices.alpha, slices.alpha] -= (
            p * propensity * (1 - propensity) * np.outer(z_propensity, z_propensity)
        )

        rho_start = slices.rho.start
        for j in range(p):
            for k in range(p):
                rho_index = rho_start + j * p + k
                paired = r[j] * r[k]
                scores[i, rho_index] = paired * (e[j] - rho[j, k])
                beta_block = slice(beta_score + j * q_outcome,
                                   beta_score + (j + 1) * q_outcome)
                bread[rho_index, beta_block] -= paired * z_outcome
                bread[rho_index, rho_index] -= paired

        for pair_index, (j, k) in enumerate(pairs):
            sigma_index = slices.sigma.start + pair_index
            paired = r[j] * r[k]
            rho_jk = rho_start + j * p + k
            rho_kj = rho_start + k * p + j
            centered_j = e[j] - rho[j, k]
            centered_k = e[k] - rho[k, j]
            count_factor = _residual_count_factor(response, j, k)
            scores[i, sigma_index] = paired * (
                centered_j * centered_k - count_factor * sigma[j, k]
            )

            beta_j = slice(beta_score + j * q_outcome,
                           beta_score + (j + 1) * q_outcome)
            beta_k = slice(beta_score + k * q_outcome,
                           beta_score + (k + 1) * q_outcome)
            if j == k:
                bread[sigma_index, beta_j] -= 2 * paired * centered_j * z_outcome
                bread[sigma_index, rho_jk] -= 2 * paired * centered_j
            else:
                bread[sigma_index, beta_j] -= paired * centered_k * z_outcome
                bread[sigma_index, beta_k] -= paired * centered_j * z_outcome
                bread[sigma_index, rho_jk] -= paired * centered_k
                bread[sigma_index, rho_kj] -= paired * centered_j
            bread[sigma_index, sigma_index] -= paired * count_factor

        first, second = _single_aipw_contribution(x, r, m, propensity, sigma)
        scores[i, slices.mu] = first - mu
        scores[i, slices.nu] = np.array([second[j, k] for j, k in pairs]) - nu
        bread[slices.mu, slices.mu] -= np.eye(p)
        bread[slices.nu, slices.nu] -= np.eye(len(pairs))

        for j in range(p):
            beta_block = slice(beta_score + j * q_outcome,
                               beta_score + (j + 1) * q_outcome)
            bread[slices.mu.start + j, beta_block] += (1 - r[j] / propensity) * z_outcome
            bread[slices.mu.start + j, slices.alpha] -= (
                r[j] / propensity * (x[j] - m[j]) * (1 - propensity) * z_propensity
            )

        for pair_index, (j, k) in enumerate(pairs):
            row = slices.nu.start + pair_index
            paired = r[j] * r[k]
            exponent = 1 if j == k else 2
            weight = paired / propensity ** exponent
            multiplier = 1 - weight
            beta_j = slice(beta_score + j * q_outcome,
                           beta_score + (j + 1) * q_outcome)
            beta_k = slice(beta_score + k * q_outcome,
                           beta_score + (k + 1) * q_outcome)
            if j == k:
                bread[row, beta_j] += multiplier * 2 * m[j] * z_outcome
            else:
                bread[row, beta_j] += multiplier * m[k] * z_outcome
                bread[row, beta_k] += multiplier * m[j] * z_outcome
            conditional_second = m[j] * m[k] + sigma[j, k]
            bread[row, slices.alpha] -= (
                exponent * weight * (x[j] * x[k] - conditional_second)
                * (1 - propensity) * z_propensity
            )
            bread[row, slices.sigma.start + pair_index] += multiplier

    return scores, bread / n


def _fit_outcome_models(observed_x, response, design):
    """Fit observed-case least-squares regressions for every outcome."""
    n, p = observed_x.shape
    q = design.shape[1]
    beta = np.empty((q, p))
    for j in range(p):
        observed = response[:, j].astype(bool)
        if observed.sum() <= q:
            raise ValueError(f"Outcome {j} has too few observed values for its regression.")
        coefficients, _, rank, _ = np.linalg.lstsq(design[observed], observed_x[observed, j], rcond=None)
        if rank < q:
            raise ValueError(f"Outcome design is rank deficient for outcome {j}.")
        beta[:, j] = coefficients
    return beta


def _fit_shared_propensity(response, design, tolerance, max_iter):
    """Fit the shared binomial-logit response model by Newton updates."""
    n, p = response.shape
    alpha = np.zeros(design.shape[1])
    response_count = response.sum(axis=1)
    for _ in range(max_iter):
        pi = expit(design @ alpha)
        score = design.T @ (response_count - p * pi)
        information = design.T @ ((p * pi * (1 - pi))[:, None] * design)
        try:
            step = np.linalg.solve(information, score)
        except np.linalg.LinAlgError as error:
            raise ValueError("Propensity design is rank deficient or separated.") from error
        alpha_next = alpha + step
        if np.max(np.abs(step)) <= tolerance * (1 + np.max(np.abs(alpha_next))):
            return alpha_next
        alpha = alpha_next
    raise RuntimeError("Shared propensity model did not converge.")


def _fit_pairwise_residual_moments(residual, response, pairs):
    """Match ``pairwise_cov`` including its pairwise centering and N - 1 divisor."""
    _, p = residual.shape
    rho = np.zeros((p, p))
    sigma = np.zeros((p, p))
    for j in range(p):
        for k in range(p):
            paired = (response[:, j] * response[:, k]).astype(bool)
            count = paired.sum()
            if count == 0:
                raise ValueError(f"No jointly observed residuals for pair ({j}, {k}).")
            rho[j, k] = residual[paired, j].mean()
    for j, k in pairs:
        paired = (response[:, j] * response[:, k]).astype(bool)
        count = paired.sum()
        if count <= 1:
            raise ValueError(f"Too few jointly observed residuals for pair ({j}, {k}).")
        centered_j = residual[paired, j] - rho[j, k]
        centered_k = residual[paired, k] - rho[k, j]
        sigma[j, k] = sigma[k, j] = centered_j @ centered_k / (count - 1)
    return rho, sigma


def _aipw_contributions(observed_x, response, fitted_mean, pi, sigma, pairs):
    n, p = observed_x.shape
    first = np.empty((n, p))
    second = np.empty((n, p, p))
    for i in range(n):
        first[i], second[i] = _single_aipw_contribution(
            observed_x[i], response[i], fitted_mean[i], pi[i], sigma
        )
    return first, second


def _single_aipw_contribution(x, r, m, propensity, sigma):
    first = m + r / propensity * (x - m)
    conditional_second = np.outer(m, m) + sigma
    paired = np.outer(r, r)
    weight = paired / propensity ** 2
    diagonal = np.diag_indices_from(weight)
    weight[diagonal] = r / propensity
    second = conditional_second + weight * (np.outer(x, x) - conditional_second)
    return first, second


def _parameter_slices(p, q_outcome, q_propensity):
    start = 0
    beta = slice(start, start + q_outcome * p)
    start = beta.stop
    alpha = slice(start, start + q_propensity)
    start = alpha.stop
    rho = slice(start, start + p * p)
    start = rho.stop
    n_pairs = p * (p + 1) // 2
    sigma = slice(start, start + n_pairs)
    start = sigma.stop
    mu = slice(start, start + p)
    start = mu.stop
    nu = slice(start, start + n_pairs)
    return _Slices(beta=beta, alpha=alpha, rho=rho, sigma=sigma, mu=mu, nu=nu)


def _residual_count_factor(response, j, k):
    count = int(np.sum(response[:, j] * response[:, k]))
    if count <= 1:
        raise ValueError(f"Too few jointly observed residuals for pair ({j}, {k}).")
    return (count - 1) / count


def _sandwich(bread, meat, n):
    identity = np.eye(bread.shape[0])
    try:
        inverse_bread = np.linalg.solve(bread, identity)
    except np.linalg.LinAlgError as error:
        raise ValueError("Analytic bread is singular; sandwich variance is unavailable.") from error
    return inverse_bread @ meat @ inverse_bread.T / n


def _covariance_from_moments(mu, nu, pairs, p):
    raw_second = np.zeros((p, p))
    for value, (j, k) in zip(nu, pairs):
        raw_second[j, k] = raw_second[k, j] = value
    return raw_second - np.outer(mu, mu)


def _covariance_delta(mu, p, pairs):
    n_pairs = len(pairs)
    delta = np.zeros((n_pairs, p + n_pairs))
    for pair_index, (j, k) in enumerate(pairs):
        delta[pair_index, p + pair_index] = 1
        delta[pair_index, j] -= mu[k]
        delta[pair_index, k] -= mu[j]
    return delta


def _add_intercept(values, n, name):
    values = _as_2d_float(values, name)
    if values.shape[0] != n:
        raise ValueError(f"{name} must have the same number of rows as X.")
    return np.column_stack([np.ones(n), values])


def _as_2d_float(values, name):
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2:
        raise ValueError(f"{name} must be a one- or two-dimensional numeric array.")
    return values


def _validate_beta(beta, q, p):
    beta = np.asarray(beta, dtype=float)
    if beta.shape != (q, p):
        raise ValueError(f"beta must have shape ({q}, {p}).")
    return beta


def _validate_alpha(alpha, q):
    alpha = np.asarray(alpha, dtype=float)
    if alpha.shape != (q,):
        raise ValueError(f"alpha must have shape ({q},).")
    return alpha
