import numpy as np
from covariance import pairwise_cov
from utils import outer_array

def aipw_z(X, Z, pi, return_diagnostics=False):
    n, p = X.shape
    R = (~np.isnan(X)).astype(float)
    XR = np.where(~np.isnan(X), X, 0.0)
    Zd = np.column_stack([np.ones(n), Z])
    d = Zd.shape[1]

    outer_Zd = np.einsum('ij,ik->ijk', Zd, Zd)
    A = np.einsum('ijk,il->ljk', outer_Zd, R)
    b = (Zd.T @ XR).T
    try:
        beta = np.linalg.solve(A, b[..., np.newaxis]).squeeze(-1).T   # (d, p)
    except np.linalg.LinAlgError:
        beta = np.zeros((d, p))
        for j in range(p):
            obs = R[:, j].astype(bool)
            if obs.sum() > d:
                beta[:, j], *_ = np.linalg.lstsq(Zd[obs], X[obs, j], rcond=None)

    M1 = Zd @ beta
    E = X - M1
    sig_e = pairwise_cov(E)

    M2 = outer_array(M1) + sig_e[:, :, np.newaxis]

    W1 = 1.0 / pi
    W2 = 1.0 / outer_array(pi)
    W2[np.arange(p), np.arange(p), :] = W1.T

    A1 = M1 + (W1 * R) * (XR - M1)
    A2 = M2 + (W2 * outer_array(R)) * (outer_array(XR) - M2)

    covariance = A2.mean(axis=2) - np.outer(A1.mean(axis=0), A1.mean(axis=0))
    if not return_diagnostics:
        return covariance

    return covariance, {
        "alpha_beta": beta,
        "conditional_mean": M1,
        "residual_covariance": sig_e,
        "first_moment_contribution": A1,
        "second_moment_contribution": A2,
    }
