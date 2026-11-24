import numpy as np
from covariance import pairwise_cov


# One EM step; groups rows by missing-data pattern for efficiency
def _em_step(X, mu, Sigma):
    n, p = X.shape
    patterns, inverse = np.unique(~np.isnan(X), axis=0, return_inverse=True)

    T1 = np.zeros(p)
    T2 = np.zeros((p, p))

    for pat_idx, pattern in enumerate(patterns):
        group = np.where(inverse == pat_idx)[0]
        n_g   = len(group)
        o     = np.where(pattern)[0]
        m     = np.where(~pattern)[0]

        if len(m) == 0:
            xi   = X[group]
            T1  += xi.sum(axis=0)
            T2  += xi.T @ xi

        elif len(o) == 0:
            T1  += n_g * mu
            T2  += n_g * (Sigma + np.outer(mu, mu))

        else:
            xo   = X[np.ix_(group, o)]
            mu_o, mu_m = mu[o], mu[m]
            Soo  = Sigma[np.ix_(o, o)]
            Smo  = Sigma[np.ix_(m, o)]
            Smm  = Sigma[np.ix_(m, m)]

            try:
                Smo_Soo_inv = np.linalg.solve(Soo, Smo.T).T
            except np.linalg.LinAlgError:
                Smo_Soo_inv = Smo @ np.linalg.pinv(Soo)

            Sigma_m_o    = Smm - Smo_Soo_inv @ Smo.T
            mu_m_given_o = mu_m + (xo - mu_o) @ Smo_Soo_inv.T

            T1[o] += xo.sum(axis=0)
            T1[m] += mu_m_given_o.sum(axis=0)

            T2[np.ix_(o, o)] += xo.T @ xo
            T2[np.ix_(o, m)] += xo.T @ mu_m_given_o
            T2[np.ix_(m, o)] += mu_m_given_o.T @ xo
            T2[np.ix_(m, m)] += mu_m_given_o.T @ mu_m_given_o + n_g * Sigma_m_o

    mu_new    = T1 / n
    Sigma_new = T2 / n - np.outer(mu_new, mu_new)
    return mu_new, Sigma_new


def _conditional_mean(X, mu, Sigma):
    """Return E[X | observed entries] under the fitted normal model."""
    completed = np.where(np.isnan(X), mu, X).copy()
    patterns, inverse = np.unique(~np.isnan(X), axis=0, return_inverse=True)

    for pat_idx, pattern in enumerate(patterns):
        group = np.where(inverse == pat_idx)[0]
        observed = np.where(pattern)[0]
        missing = np.where(~pattern)[0]
        if len(observed) == 0 or len(missing) == 0:
            continue

        Soo = Sigma[np.ix_(observed, observed)]
        Smo = Sigma[np.ix_(missing, observed)]
        try:
            weights = np.linalg.solve(Soo, Smo.T).T
        except np.linalg.LinAlgError:
            weights = Smo @ np.linalg.pinv(Soo)
        completed[np.ix_(group, missing)] = (
            mu[missing] + (X[np.ix_(group, observed)] - mu[observed]) @ weights.T
        )
    return completed


# Full Information Maximum Likelihood covariance estimator via EM
def fiml(X, max_iter=500, tol=1e-8, return_diagnostics=False):
    n, p  = X.shape
    mu    = np.nanmean(X, axis=0)
    mu    = np.where(np.isnan(mu), 0.0, mu)
    Sigma = pairwise_cov(X)

    for j in range(p):
        if Sigma[j, j] <= 0:
            Sigma[j, j] = np.nanvar(X[:, j]) if not np.all(np.isnan(X[:, j])) else 1.0

    iterations = 0
    for iterations in range(1, max_iter + 1):
        mu_new, Sigma_new = _em_step(X, mu, Sigma)
        if np.max(np.abs(Sigma_new - Sigma)) < tol:
            break
        mu, Sigma = mu_new, Sigma_new

    if not return_diagnostics:
        return Sigma_new

    return Sigma_new, {
        "mean": mu_new,
        "imputed_mean": _conditional_mean(X, mu_new, Sigma_new),
        "iterations": np.array(iterations),
    }
