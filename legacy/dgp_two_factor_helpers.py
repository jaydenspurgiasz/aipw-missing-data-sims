"""Parked two-factor matrix builders (not part of the active DGP API).

Later: a separate script can build the joint (X, Z) covariance from a
factor model and pass that matrix into ``dgp.generate``.
"""

from __future__ import annotations

import numpy as np


def two_factor_matrices(
    p: int = 6,
    n_factors: int = 2,
    loading: float = 0.8,
    phi_offdiag: float = 0.3,
):
    if p % n_factors != 0:
        raise ValueError(f"p={p} must be divisible by n_factors={n_factors}")

    n_per = p // n_factors
    Lambda = np.kron(np.eye(n_factors), np.full((n_per, 1), loading))
    Phi = np.eye(n_factors)
    Phi[(0, 1), (1, 0)] = phi_offdiag

    factor_var = np.diag(Lambda @ Phi @ Lambda.T)
    Omega = np.diag(1.0 - factor_var)
    Sigma_X = Lambda @ Phi @ Lambda.T + Omega
    return {
        "Lambda": Lambda,
        "Phi": Phi,
        "Omega": Omega,
        "Sigma_X": Sigma_X,
    }


def joint_sigma_from_two_factor(
    p: int = 6,
    loading: float = 0.8,
    phi_offdiag: float = 0.3,
    cov_XZ: float = 0.64,
    var_Z: float = 1.0,
):
    """Build the joint (X, Z) covariance matrix used by the old simulations."""
    Sigma_X = two_factor_matrices(
        p=p, loading=loading, phi_offdiag=phi_offdiag
    )["Sigma_X"]
    Sigma = np.zeros((p + 1, p + 1))
    Sigma[:p, :p] = Sigma_X
    Sigma[:p, p] = cov_XZ
    Sigma[p, :p] = cov_XZ
    Sigma[p, p] = var_Z
    return Sigma
