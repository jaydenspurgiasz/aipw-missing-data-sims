import numpy as np


def pairwise_cov(X):
    R = (~np.isnan(X)).astype(float)
    X0 = np.where(np.isnan(X), 0.0, X)
    XR = X0 * R

    N = R.T @ R
    sum_x = XR.T @ R
    sum_xy = XR.T @ XR

    with np.errstate(invalid='ignore', divide='ignore'):
        A = (sum_xy - sum_x * sum_x.T / N) / (N - 1)
    A[N <= 1] = 0.0
    return A

def complete_case_cov(X):
    mask = ~np.isnan(X).any(axis=1)
    Xc = X[mask]
    p = X.shape[1]
    return np.cov(Xc.T) if Xc.shape[0] > 1 else np.full((p, p), np.nan)
