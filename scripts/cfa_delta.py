"""Delta-method uncertainty propagation from an AIPW covariance to CFA parameters."""

import numpy as np

from cfa import cfa_two_factor


def cfa_delta_variance(covariance, covariance_covariance, n_obs,
                       step_scales=(1e-4, 1e-5, 1e-6),
                       n_direction_checks=3):
    """Fit two-factor CFA and propagate covariance uncertainty by central differences.

    Parameters
    ----------
    covariance : array-like, shape (p, p)
        Raw AIPW covariance estimate. It must be positive definite; this
        function deliberately does not apply the repository's PSD projection.
    covariance_covariance : array-like, shape (q, q)
        Sandwich covariance of ``vech(covariance)``, ordered with
        ``np.tril_indices(p)``. Here, ``q = p * (p + 1) / 2``.
    n_obs : int
        Sample size supplied to the existing CFA fitting routine.
    step_scales : sequence of float
        Relative central-difference steps used to assess derivative stability.
    n_direction_checks : int
        Number of random directional derivative checks. The default is intended
        for a full one-sample calculation; simulations may set this to zero
        after a separate pilot has validated the selected step size.

    Returns
    -------
    dict
        CFA parameter estimates, the selected sensitivity matrix, all tested
        sensitivities, stability diagnostics, and delta-method covariance and
        standard errors for the CFA parameters.
    """
    covariance = np.asarray(covariance, dtype=float)
    covariance_covariance = np.asarray(covariance_covariance, dtype=float)
    _validate_inputs(
        covariance, covariance_covariance, n_obs, step_scales, n_direction_checks
    )

    parameters = _fit_and_validate(covariance, n_obs, covariance.shape[0])
    parameters = _canonicalize_parameters(parameters, covariance.shape[0])

    pairs = list(zip(*np.tril_indices(covariance.shape[0])))
    parameter_names = _parameter_names(covariance.shape[0])
    sensitivities = {}
    for step_scale in step_scales:
        sensitivities[step_scale] = _central_difference_sensitivity(
            covariance, n_obs, parameters, pairs, step_scale
        )

    selected_scale = step_scales[len(step_scales) // 2]
    sensitivity = sensitivities[selected_scale]
    stability = _stability_diagnostics(sensitivities, selected_scale)
    direction_check = []
    if n_direction_checks:
        direction_check = _directional_derivative_check(
            covariance, n_obs, parameters, sensitivity, pairs, selected_scale,
            n_directions=n_direction_checks,
        )
    factor_covariance = sensitivity @ covariance_covariance @ sensitivity.T
    factor_covariance = (factor_covariance + factor_covariance.T) / 2
    factor_variances = np.diag(factor_covariance)
    if np.any(factor_variances < -1e-10):
        raise RuntimeError("Delta-method covariance has materially negative diagonal entries.")
    if np.linalg.eigvalsh(factor_covariance).min() < -1e-9:
        raise RuntimeError(
            "Delta-method covariance is materially indefinite; inspect the "
            "finite-difference stability diagnostics."
        )

    return {
        "parameters": parameters,
        "parameter_names": parameter_names,
        "pairs": pairs,
        "sensitivity": sensitivity,
        "sensitivity_by_step": sensitivities,
        "selected_step_scale": selected_scale,
        "stability": stability,
        "direction_check": direction_check,
        "minimum_input_eigenvalue": float(np.linalg.eigvalsh(covariance).min()),
        "factor_covariance": factor_covariance,
        "factor_standard_errors": np.sqrt(np.maximum(factor_variances, 0)),
    }


def aipw_and_cfa_m_estimation_variance(aipw_result, n_obs,
                                        step_scales=(1e-4, 1e-5, 1e-6),
                                        n_direction_checks=3):
    """Return joint sandwich covariance for all AIPW and CFA parameters.

    The AIPW estimating-equation vector has 98 coordinates in the current
    six-outcome simulation. CFA parameters are derived from the AIPW covariance
    estimate, so their cross-covariance with every AIPW coordinate is retained.
    """
    required = {
        "beta", "alpha", "pair_residual_means", "residual_covariance",
        "mean", "raw_second_moment", "covariance", "parameter_covariance",
        "covariance_covariance", "parameter_slices", "pairs",
    }
    missing = required.difference(aipw_result)
    if missing:
        raise ValueError(f"aipw_result is missing keys: {sorted(missing)}")

    aipw_covariance = np.asarray(aipw_result["parameter_covariance"], dtype=float)
    p = aipw_result["covariance"].shape[0]
    slices = aipw_result["parameter_slices"]
    pairs = list(aipw_result["pairs"])
    if aipw_covariance.shape[0] != aipw_covariance.shape[1]:
        raise ValueError("AIPW parameter covariance must be square.")

    cfa_result = cfa_delta_variance(
        aipw_result["covariance"],
        aipw_result["covariance_covariance"],
        n_obs=n_obs,
        step_scales=step_scales,
        n_direction_checks=n_direction_checks,
    )
    covariance_map = _aipw_covariance_map(
        aipw_result["mean"], p, pairs, slices, aipw_covariance.shape[0]
    )
    cfa_map = cfa_result["sensitivity"] @ covariance_map
    cross_covariance = aipw_covariance @ cfa_map.T
    joint_covariance = np.block([
        [aipw_covariance, cross_covariance],
        [cross_covariance.T, cfa_result["factor_covariance"]],
    ])
    joint_covariance = (joint_covariance + joint_covariance.T) / 2

    aipw_values, aipw_names, aipw_groups = _aipw_values_and_names(aipw_result)
    parameter_values = np.concatenate([aipw_values, cfa_result["parameters"]])
    parameter_names = aipw_names + cfa_result["parameter_names"]
    parameter_groups = aipw_groups + ["cfa"] * len(cfa_result["parameter_names"])

    return {
        "parameter_values": parameter_values,
        "parameter_names": parameter_names,
        "parameter_groups": parameter_groups,
        "joint_covariance": joint_covariance,
        "aipw_parameter_count": aipw_covariance.shape[0],
        "cfa_parameter_count": cfa_result["parameters"].size,
        "cfa_result": cfa_result,
        "cfa_map_from_aipw": cfa_map,
    }


def _aipw_covariance_map(mean, p, pairs, slices, parameter_count):
    """Derivative of vech(AIPW covariance) with respect to the full AIPW stack."""
    n_pairs = len(pairs)
    output = np.zeros((n_pairs, parameter_count))
    for pair_index, (j, k) in enumerate(pairs):
        output[pair_index, slices.nu.start + pair_index] = 1
        output[pair_index, slices.mu.start + j] -= mean[k]
        output[pair_index, slices.mu.start + k] -= mean[j]
    return output


def _aipw_values_and_names(aipw_result):
    """Flatten the stacked AIPW coordinates in the score-vector ordering."""
    beta = np.asarray(aipw_result["beta"])
    alpha = np.asarray(aipw_result["alpha"])
    rho = np.asarray(aipw_result["pair_residual_means"])
    sigma = np.asarray(aipw_result["residual_covariance"])
    mean = np.asarray(aipw_result["mean"])
    nu = np.asarray(aipw_result["raw_second_moment"])
    pairs = list(aipw_result["pairs"])
    p = beta.shape[1]

    values = np.concatenate([
        beta.T.reshape(-1), alpha, rho.reshape(-1),
        np.array([sigma[j, k] for j, k in pairs]), mean, nu,
    ])
    names = []
    groups = []
    for outcome in range(p):
        for design_index in range(beta.shape[0]):
            names.append(f"beta_x{outcome + 1}_z{design_index}")
            groups.append("beta")
    for design_index in range(alpha.size):
        names.append(f"alpha_z{design_index}")
        groups.append("alpha")
    for row in range(p):
        for col in range(p):
            names.append(f"rho_x{row + 1}_x{col + 1}")
            groups.append("rho")
    for row, col in pairs:
        names.append(f"sigma_x{row + 1}_x{col + 1}")
        groups.append("sigma")
    for outcome in range(p):
        names.append(f"mu_x{outcome + 1}")
        groups.append("mu")
    for row, col in pairs:
        names.append(f"nu_x{row + 1}_x{col + 1}")
        groups.append("nu")
    return values, names, groups


def _central_difference_sensitivity(covariance, n_obs, reference, pairs, step_scale):
    """Differentiate the CFA fitting map with symmetric covariance perturbations."""
    p = covariance.shape[0]
    sensitivity = np.empty((reference.size, len(pairs)))

    for column, (row, col) in enumerate(pairs):
        perturbation_size = step_scale * np.sqrt(covariance[row, row] * covariance[col, col])
        plus = covariance.copy()
        minus = covariance.copy()
        plus[row, col] += perturbation_size
        minus[row, col] -= perturbation_size
        if row != col:
            plus[col, row] += perturbation_size
            minus[col, row] -= perturbation_size

        _require_positive_definite(plus, "Positive perturbation")
        _require_positive_definite(minus, "Negative perturbation")
        plus_parameters = _fit_and_align(plus, n_obs, reference, p)
        minus_parameters = _fit_and_align(minus, n_obs, reference, p)
        sensitivity[:, column] = (plus_parameters - minus_parameters) / (
            2 * perturbation_size
        )

    return sensitivity


def _fit_and_align(covariance, n_obs, reference, p):
    """Refit CFA and remove equivalent factor-sign changes before differencing."""
    parameters = _fit_and_validate(covariance, n_obs, p)

    n_per_factor = p // 2
    aligned = parameters.copy()
    sign_one = _matching_sign(reference[:n_per_factor], aligned[:n_per_factor])
    sign_two = _matching_sign(reference[n_per_factor:p], aligned[n_per_factor:p])
    aligned[:n_per_factor] *= sign_one
    aligned[n_per_factor:p] *= sign_two
    aligned[-1] *= sign_one * sign_two
    return aligned


def _matching_sign(reference_block, candidate_block):
    return 1.0 if reference_block @ candidate_block >= 0 else -1.0


def _canonicalize_parameters(parameters, p):
    """Use positive first loading in each simple-structure block as the orientation."""
    n_per_factor = p // 2
    canonical = parameters.copy()
    sign_one = 1.0 if canonical[0] >= 0 else -1.0
    sign_two = 1.0 if canonical[n_per_factor] >= 0 else -1.0
    canonical[:n_per_factor] *= sign_one
    canonical[n_per_factor:p] *= sign_two
    canonical[-1] *= sign_one * sign_two
    return canonical


def _fit_and_validate(covariance, n_obs, p):
    parameters, converged = cfa_two_factor(covariance, n_obs)
    if not converged:
        raise RuntimeError("CFA returned non-finite parameters.")
    n_per_factor = p // 2
    loadings = parameters[:p]
    uniquenesses = parameters[p:2 * p]
    correlation = parameters[-1]
    if np.linalg.norm(loadings[:n_per_factor]) < 1e-8:
        raise RuntimeError("First CFA loading block is too close to zero to orient.")
    if np.linalg.norm(loadings[n_per_factor:]) < 1e-8:
        raise RuntimeError("Second CFA loading block is too close to zero to orient.")
    if np.any(uniquenesses <= 1e-8):
        raise RuntimeError("CFA returned a boundary or Heywood residual variance.")
    if abs(correlation) >= 0.999:
        raise RuntimeError("CFA factor correlation is too close to a boundary.")
    return parameters


def _stability_diagnostics(sensitivities, selected_scale):
    """Compare each tested finite-difference Jacobian to the selected one."""
    reference = sensitivities[selected_scale]
    diagnostics = {}
    for step_scale, sensitivity in sensitivities.items():
        difference = sensitivity - reference
        diagnostics[step_scale] = {
            "max_absolute_difference": float(np.max(np.abs(difference))),
            "relative_frobenius_difference": float(
                np.linalg.norm(difference, ord="fro")
                / max(np.linalg.norm(reference, ord="fro"), np.finfo(float).eps)
            ),
            "max_column_relative_difference": float(
                max(
                    np.linalg.norm(difference[:, column])
                    / max(np.linalg.norm(reference[:, column]), np.finfo(float).eps)
                    for column in range(reference.shape[1])
                )
            ),
        }
    return diagnostics


def _directional_derivative_check(covariance, n_obs, reference, sensitivity,
                                  pairs, step_scale, n_directions=3):
    """Cross-check the Jacobian against random symmetric covariance directions."""
    rng = np.random.default_rng(20261005)
    p = covariance.shape[0]
    coordinate_scales = np.array([
        np.sqrt(covariance[row, row] * covariance[col, col])
        for row, col in pairs
    ])
    checks = []
    for _ in range(n_directions):
        direction = rng.normal(size=len(pairs))
        direction /= np.linalg.norm(direction)
        direction *= coordinate_scales
        plus = _add_vech_direction(covariance, pairs, step_scale * direction)
        minus = _add_vech_direction(covariance, pairs, -step_scale * direction)
        _require_positive_definite(plus, "Directional positive perturbation")
        _require_positive_definite(minus, "Directional negative perturbation")
        numerical = (
            _fit_and_align(plus, n_obs, reference, p)
            - _fit_and_align(minus, n_obs, reference, p)
        ) / (2 * step_scale)
        analytic = sensitivity @ direction
        checks.append({
            "max_absolute_difference": float(np.max(np.abs(numerical - analytic))),
            "relative_l2_difference": float(
                np.linalg.norm(numerical - analytic)
                / max(np.linalg.norm(analytic), np.finfo(float).eps)
            ),
        })
    return checks


def _add_vech_direction(covariance, pairs, direction):
    output = covariance.copy()
    for value, (row, col) in zip(direction, pairs):
        output[row, col] += value
        if row != col:
            output[col, row] += value
    return output


def _parameter_names(p):
    n_per_factor = p // 2
    return (
        [f"loading_{index + 1}_factor_1" for index in range(n_per_factor)]
        + [f"loading_{index + 1}_factor_2" for index in range(n_per_factor, p)]
        + [f"residual_variance_{index + 1}" for index in range(p)]
        + ["factor_correlation"]
    )


def _validate_inputs(covariance, covariance_covariance, n_obs, step_scales,
                     n_direction_checks):
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError("covariance must be a square matrix.")
    p = covariance.shape[0]
    if p % 2 != 0:
        raise ValueError("The current two-factor CFA requires an even number of variables.")
    if not np.allclose(covariance, covariance.T, atol=1e-10):
        raise ValueError("covariance must be symmetric.")
    _require_positive_definite(covariance, "Raw AIPW covariance")

    q = p * (p + 1) // 2
    if covariance_covariance.shape != (q, q):
        raise ValueError(f"covariance_covariance must have shape ({q}, {q}).")
    if not np.allclose(covariance_covariance, covariance_covariance.T, atol=1e-10):
        raise ValueError("covariance_covariance must be symmetric.")
    if int(n_obs) != n_obs or n_obs <= p:
        raise ValueError("n_obs must be an integer greater than the number of variables.")
    if not step_scales or any(step <= 0 for step in step_scales):
        raise ValueError("step_scales must contain positive values.")
    if int(n_direction_checks) != n_direction_checks or n_direction_checks < 0:
        raise ValueError("n_direction_checks must be a non-negative integer.")


def _require_positive_definite(matrix, label):
    smallest_eigenvalue = np.linalg.eigvalsh(matrix).min()
    if smallest_eigenvalue <= 0:
        raise ValueError(
            f"{label} is not positive definite (minimum eigenvalue: {smallest_eigenvalue})."
        )
