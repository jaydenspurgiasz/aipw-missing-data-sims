"""Plasmode joint covariance from exported lavaan FA parameters.

Reads ``params/plasmode_holzinger/`` (CSV + meta.json written by
``scripts/R/plasmode_holzinger_fa.R``), builds

    Sigma_X = Lambda @ Phi @ Lambda.T + Omega

and the joint ``(X, Z)`` covariance used by ``dgp.generate``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_SCRIPTS = Path(__file__).resolve().parent
_ROOT = _SCRIPTS.parent
DEFAULT_PARAMS_DIR = _ROOT / "params" / "plasmode_holzinger"


# =============================================================================
# Load FA params
# =============================================================================

def load_fa_params(params_dir: str | Path | None = None) -> dict[str, Any]:
    """Load Lambda, Phi, Omega (+ meta) from the R export directory."""
    params_dir = Path(params_dir) if params_dir is not None else DEFAULT_PARAMS_DIR
    if not params_dir.exists():
        raise FileNotFoundError(
            f"Missing plasmode params at {params_dir}. "
            "Run: Rscript scripts/R/plasmode_holzinger_fa.R"
        )

    Lambda = pd.read_csv(params_dir / "lambda.csv", index_col=0).to_numpy(dtype=float)
    Phi = pd.read_csv(params_dir / "phi.csv", index_col=0).to_numpy(dtype=float)
    Omega = pd.read_csv(params_dir / "omega.csv", index_col=0).to_numpy(dtype=float)

    meta_path = params_dir / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    variables = list(meta.get("variables") or [f"x{j + 1}" for j in range(Lambda.shape[0])])
    factors = list(meta.get("factors") or [f"f{j + 1}" for j in range(Lambda.shape[1])])

    return {
        "Lambda": Lambda,
        "Phi": Phi,
        "Omega": Omega,
        "variables": variables,
        "factors": factors,
        "n_factors": int(Lambda.shape[1]),
        "p": int(Lambda.shape[0]),
        "meta": meta,
        "params_dir": params_dir,
    }


def sigma_x_from_fa(Lambda, Phi, Omega) -> np.ndarray:
    """Step 3: Sigma_X = Lambda Phi Lambda^T + Omega."""
    Lambda = np.asarray(Lambda, dtype=float)
    Phi = np.asarray(Phi, dtype=float)
    Omega = np.asarray(Omega, dtype=float)
    return Lambda @ Phi @ Lambda.T + Omega


def max_equal_cov_xz(Sigma_X, var_Z: float = 1.0) -> float:
    """Largest |c| such that Cov(X_j,Z)=c for all j keeps joint Σ PD."""
    Sigma_X = np.asarray(Sigma_X, dtype=float)
    ones = np.ones(Sigma_X.shape[0])
    quad = float(ones @ np.linalg.solve(Sigma_X, ones))
    if quad <= 0:
        raise ValueError("Sigma_X must be positive definite")
    return float(np.sqrt(float(var_Z) / quad))


# =============================================================================
# Joint (X, Z) law
# =============================================================================

def joint_sigma_from_fa(
    Lambda,
    Phi,
    Omega,
    *,
    cov_XZ: float = 0.64,
    var_Z: float = 1.0,
) -> np.ndarray:
    """Build joint covariance of (X, Z) with scalar cov(X_j, Z)=cov_XZ.

    Legacy two-factor sims used ``cov_XZ=0.64`` at p=6. For Holzinger p=9
    that value is typically too large; use ``max_equal_cov_xz`` and pick a
    smaller c (notebook default: 0.9 * max).
    """
    Sigma_X = sigma_x_from_fa(Lambda, Phi, Omega)
    p = Sigma_X.shape[0]
    limit = max_equal_cov_xz(Sigma_X, var_Z=var_Z)
    if abs(cov_XZ) >= limit:
        raise ValueError(
            f"cov_XZ={cov_XZ} makes joint Sigma non-PD for this Sigma_X "
            f"(need |cov_XZ| < {limit:.6g})."
        )

    Sigma = np.zeros((p + 1, p + 1), dtype=float)
    Sigma[:p, :p] = Sigma_X
    Sigma[:p, p] = cov_XZ
    Sigma[p, :p] = cov_XZ
    Sigma[p, p] = var_Z
    return Sigma


def load_plasmode_joint(
    params_dir: str | Path | None = None,
    *,
    cov_XZ: float | None = None,
    var_Z: float = 1.0,
    mean_value: float = 2.0,
    cov_xz_frac_of_max: float = 0.9,
) -> dict[str, Any]:
    """Convenience: load FA params and return joint Sigma / mean / metadata.

    If ``cov_XZ`` is None, uses ``cov_xz_frac_of_max * max_equal_cov_xz``
    so the joint law stays PD (Holzinger p=9 cannot use legacy 0.64).
    """
    fa = load_fa_params(params_dir)
    Sigma_X = sigma_x_from_fa(fa["Lambda"], fa["Phi"], fa["Omega"])
    limit = max_equal_cov_xz(Sigma_X, var_Z=var_Z)
    if cov_XZ is None:
        cov_XZ = float(cov_xz_frac_of_max) * limit
    Sigma = joint_sigma_from_fa(
        fa["Lambda"], fa["Phi"], fa["Omega"], cov_XZ=cov_XZ, var_Z=var_Z
    )
    mean = np.full(fa["p"] + 1, float(mean_value))
    return {
        **fa,
        "Sigma_X": Sigma_X,
        "Sigma": Sigma,
        "mean": mean,
        "cov_XZ": float(cov_XZ),
        "cov_XZ_max": float(limit),
        "var_Z": float(var_Z),
        "cfa_n_factors": fa["n_factors"],
    }
