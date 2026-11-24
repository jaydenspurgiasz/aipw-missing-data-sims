"""Build the estimation view between DGP and methods.

DGP always keeps the true ``Z`` and ``pi``. This module attaches the
covariates each nuisance model should actually use:

- ``Z_outcome``     → linear outcome / AIPW prediction design
- ``Z_propensity``  → propensity model design (ignored if use_true_pi)

Transforms are chosen independently so outcome and propensity misspec
do not get tangled into a single overwritten ``Z``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


Z_CHOICES = ("true", "squared")


# =============================================================================
# Z transforms
# =============================================================================

def _z_values(Z: pd.Series | pd.DataFrame) -> np.ndarray:
    if isinstance(Z, pd.Series):
        return Z.to_numpy(dtype=float)
    if Z.shape[1] != 1:
        raise ValueError("squared Z transform currently supports a single Z column")
    return Z.iloc[:, 0].to_numpy(dtype=float)


def _as_z_frame(values: np.ndarray, template: pd.Series | pd.DataFrame, name: str):
    if isinstance(template, pd.Series):
        return pd.Series(values, name=name)
    return pd.DataFrame({name: values})


def _transform_z(Z: pd.Series | pd.DataFrame, how: str, name: str):
    if how not in Z_CHOICES:
        known = ", ".join(Z_CHOICES)
        raise ValueError(f"Unknown Z transform '{how}'. Expected one of: {known}")

    if how == "true":
        if isinstance(Z, pd.Series):
            out = Z.copy()
            out.name = name
            return out
        out = Z.copy()
        out.columns = [name] if out.shape[1] == 1 else list(out.columns)
        return out

    # how == "squared"
    return _as_z_frame(_z_values(Z) ** 2, Z, name)


# =============================================================================
# Estimation-view builder
# =============================================================================

def build_estimation_draw(
    draw: dict[str, Any],
    *,
    outcome_z: str = "true",
    propensity_z: str = "true",
) -> dict[str, Any]:
    """Return a copy of ``draw`` with explicit nuisance covariate fields.

    Never overwrites DGP ``Z`` / ``pi``. Methods should read ``Z_outcome``
    and ``Z_propensity`` instead of guessing from a single ``Z``.
    """
    out = dict(draw)
    out["params"] = dict(draw.get("params", {}))

    out["Z_true"] = draw["Z"]
    out["Z_outcome"] = _transform_z(draw["Z"], outcome_z, name="Z_outcome")
    out["Z_propensity"] = _transform_z(draw["Z"], propensity_z, name="Z_propensity")

    out["params"]["outcome_z"] = outcome_z
    out["params"]["propensity_z"] = propensity_z
    return out


def available_z_transforms() -> list[str]:
    return list(Z_CHOICES)


# =============================================================================
# Named scenarios (convenience presets over the two Z choices)
# =============================================================================

# outcome_z / propensity_z pairs used in misspecification studies.
SCENARIOS = {
    "correct": {"outcome_z": "true", "propensity_z": "true"},
    "outcome_mis": {"outcome_z": "squared", "propensity_z": "true"},
    "propensity_mis": {"outcome_z": "true", "propensity_z": "squared"},
    "both_mis": {"outcome_z": "squared", "propensity_z": "squared"},
}


def resolve_scenario(name: str) -> dict[str, str]:
    if name not in SCENARIOS:
        known = ", ".join(sorted(SCENARIOS))
        raise ValueError(f"Unknown scenario '{name}'. Expected one of: {known}")
    return dict(SCENARIOS[name])


def available_scenarios() -> list[str]:
    return sorted(SCENARIOS)
