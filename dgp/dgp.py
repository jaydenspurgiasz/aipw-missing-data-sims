"""Data generation for missing-data simulations.

Input is the joint mean / covariance of (X, Z). Sample complete data, apply
a MAR missingness mechanism, return pandas objects, and optionally write CSVs
under ``runs/<run_id>/data/`` for language-agnostic use.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from scipy.special import expit


# =============================================================================
# Input validation / column labels
# =============================================================================

def _as_1d(name: str, values, length: int) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size != length:
        raise ValueError(f"{name} must have length {length}, got {arr.size}")
    return arr


def _as_square(name: str, values) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"{name} must be a square matrix, got shape {arr.shape}")
    return arr


def _variable_names(p: int, n_z: int, x_columns, z_columns) -> tuple[list[str], list[str]]:
    if x_columns is None:
        x_names = [f"x{j + 1}" for j in range(p)]
    else:
        x_names = list(x_columns)
        if len(x_names) != p:
            raise ValueError(f"x_columns must have length {p}, got {len(x_names)}")

    if z_columns is None:
        z_names = ["Z"] if n_z == 1 else [f"Z{j + 1}" for j in range(n_z)]
    else:
        z_names = list(z_columns)
        if len(z_names) != n_z:
            raise ValueError(f"z_columns must have length {n_z}, got {len(z_names)}")

    return x_names, z_names


# =============================================================================
# Complete-data sampling from joint (X, Z) law
# =============================================================================

def _sample_complete(n: int, mean: np.ndarray, Sigma: np.ndarray, rng: np.random.Generator):
    return rng.multivariate_normal(mean, Sigma, size=n)


# =============================================================================
# Missingness mechanism (unit nonresponse, logistic in Z)
# =============================================================================

def _propensity_from_Z(Z: np.ndarray, intercept: float, coef) -> np.ndarray:
    """logit(pi) = intercept + Z @ coef. Z is (n,) or (n, n_z)."""
    Z = np.asarray(Z, dtype=float)
    coef = np.asarray(coef, dtype=float).reshape(-1)
    if Z.ndim == 1:
        if coef.size != 1:
            raise ValueError("propensity_coef must be length 1 when Z is univariate")
        linear = intercept + coef[0] * Z
    else:
        if coef.size != Z.shape[1]:
            raise ValueError(
                f"propensity_coef must have length {Z.shape[1]}, got {coef.size}"
            )
        linear = intercept + Z @ coef
    return expit(linear)


def _sample_response(n: int, p: int, pi: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    # Same pi for every item (unit nonresponse).
    return rng.binomial(1, np.broadcast_to(pi[:, None], (n, p))).astype(float)


# =============================================================================
# Pack pandas outputs
# =============================================================================

def _pack_draw(
    *,
    X_true: np.ndarray,
    R: np.ndarray,
    Z: np.ndarray,
    pi: np.ndarray,
    Sigma: np.ndarray,
    mean: np.ndarray,
    x_names: list[str],
    z_names: list[str],
    params: dict[str, Any],
) -> dict[str, Any]:
    X_true_df = pd.DataFrame(X_true, columns=x_names)
    R_df = pd.DataFrame(R, columns=x_names)
    X_obs_df = X_true_df.where(R_df.astype(bool))

    if Z.ndim == 1:
        Z_out: Any = pd.Series(Z, name=z_names[0])
    else:
        Z_out = pd.DataFrame(Z, columns=z_names)

    all_names = x_names + z_names
    return {
        "X_true": X_true_df,
        "X_obs": X_obs_df,
        "R": R_df,
        "Z": Z_out,
        "pi": pd.Series(pi, name="pi"),
        "Sigma": pd.DataFrame(Sigma, index=all_names, columns=all_names),
        "mean": pd.Series(mean, index=all_names, name="mean"),
        "params": params,
    }


# =============================================================================
# CSV / JSON persistence (language-agnostic run data)
# =============================================================================

def save_draw(
    draw: Mapping[str, Any],
    run_dir: str | Path,
    *,
    rep: int | None = None,
) -> Path:
    """Write one draw under ``<run_dir>/data[/rep_XXXX]/`` plus ``params.json``."""
    run_dir = Path(run_dir)
    data_dir = run_dir / "data"
    if rep is not None:
        data_dir = data_dir / f"rep_{int(rep):04d}"
    data_dir.mkdir(parents=True, exist_ok=True)

    draw["X_true"].to_csv(data_dir / "X_true.csv", index=False)
    draw["X_obs"].to_csv(data_dir / "X_obs.csv", index=False)
    draw["R"].to_csv(data_dir / "R.csv", index=False)

    Z = draw["Z"]
    if isinstance(Z, pd.Series):
        Z.to_frame().to_csv(data_dir / "Z.csv", index=False)
    else:
        Z.to_csv(data_dir / "Z.csv", index=False)

    draw["pi"].to_frame().to_csv(data_dir / "pi.csv", index=False)
    draw["Sigma"].to_csv(data_dir / "Sigma.csv", index=True)
    draw["mean"].to_frame().to_csv(data_dir / "mean.csv", index=True)

    # Draw-level params stay beside the CSVs. Run-level params.json is owned
    # by the simulation runner.
    params = dict(draw.get("params", {}))
    if rep is not None:
        params["rep"] = int(rep)
    (data_dir / "params.json").write_text(json.dumps(params, indent=2, default=str))
    return data_dir


def load_draw(run_dir: str | Path, *, rep: int | None = None) -> dict[str, Any]:
    """Read a previously saved draw back into pandas objects."""
    run_dir = Path(run_dir)
    data_dir = run_dir / "data"
    if rep is not None:
        data_dir = data_dir / f"rep_{int(rep):04d}"

    params = {}
    for params_path in (data_dir / "params.json", run_dir / "params.json"):
        if params_path.exists():
            params = json.loads(params_path.read_text())
            break

    Z_df = pd.read_csv(data_dir / "Z.csv")
    Z_out: Any = Z_df.iloc[:, 0] if Z_df.shape[1] == 1 else Z_df

    return {
        "X_true": pd.read_csv(data_dir / "X_true.csv"),
        "X_obs": pd.read_csv(data_dir / "X_obs.csv"),
        "R": pd.read_csv(data_dir / "R.csv"),
        "Z": Z_out,
        "pi": pd.read_csv(data_dir / "pi.csv")["pi"],
        "Sigma": pd.read_csv(data_dir / "Sigma.csv", index_col=0),
        "mean": pd.read_csv(data_dir / "mean.csv", index_col=0).iloc[:, 0],
        "params": params,
    }


# =============================================================================
# Public generate entrypoint
# =============================================================================

def generate(
    n: int,
    Sigma,
    *,
    mean=None,
    n_z: int = 1,
    propensity_intercept: float = -0.6,
    propensity_coef=1.0,
    seed: int | None = None,
    x_columns=None,
    z_columns=None,
    run_dir: str | Path | None = None,
    rep: int | None = None,
    extra_params: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Sample complete (X, Z) from a joint law, apply MAR, return a pandas draw.

    Parameters
    ----------
    Sigma:
        Joint covariance of ``(X, Z)``, shape ``(p + n_z, p + n_z)``.
        The last ``n_z`` coordinates are the auxiliary variables Z.
    mean:
        Joint mean of ``(X, Z)``, length ``p + n_z``. Defaults to zeros.
    n_z:
        Number of trailing Z columns in ``Sigma`` / ``mean`` (default 1).
    propensity_intercept, propensity_coef:
        ``logit(pi) = intercept + Z @ coef``. Same ``pi`` for every X item.
    run_dir:
        If set, also write CSVs under ``<run_dir>/data`` and ``params.json``.

    Misspecification is not handled here — apply that between DGP and estimation.
    """
    Sigma = _as_square("Sigma", Sigma)
    dim = Sigma.shape[0]
    if n_z < 1 or n_z >= dim:
        raise ValueError(f"n_z must be in 1..{dim - 1}, got {n_z}")

    p = dim - n_z
    x_names, z_names = _variable_names(p, n_z, x_columns, z_columns)

    if mean is None:
        mean = np.zeros(dim)
    mean = _as_1d("mean", mean, dim)

    coef = np.asarray(propensity_coef, dtype=float).reshape(-1)
    if coef.size == 1 and n_z > 1:
        coef = np.repeat(coef, n_z)
    if coef.size != n_z:
        raise ValueError(f"propensity_coef must have length {n_z}, got {coef.size}")

    rng = np.random.default_rng(seed)
    U = _sample_complete(n, mean, Sigma, rng)

    X_true = U[:, :p]
    Z = U[:, p] if n_z == 1 else U[:, p:]
    pi = _propensity_from_Z(Z, propensity_intercept, coef)
    R = _sample_response(n, p, pi, rng)

    params = {
        "n": int(n),
        "p": int(p),
        "n_z": int(n_z),
        "seed": seed,
        "propensity_intercept": float(propensity_intercept),
        "propensity_coef": coef.tolist(),
        "x_columns": x_names,
        "z_columns": z_names,
        "missingness": "MAR unit nonresponse, logit(pi) = a + Z @ b",
    }
    if extra_params:
        params.update(dict(extra_params))

    draw = _pack_draw(
        X_true=X_true,
        R=R,
        Z=Z,
        pi=pi,
        Sigma=Sigma,
        mean=mean,
        x_names=x_names,
        z_names=z_names,
        params=params,
    )

    if run_dir is not None:
        save_draw(draw, run_dir, rep=rep)

    return draw
