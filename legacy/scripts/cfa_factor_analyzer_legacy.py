import warnings
import numpy as np
from factor_analyzer.confirmatory_factor_analyzer import (
    ConfirmatoryFactorAnalyzer, ModelSpecification,
)


def _make_spec(n_vars):
    n_per = n_vars // 2
    loadings = np.zeros((n_vars, 2))
    loadings[:n_per, 0] = 1
    loadings[n_per:, 1] = 1
    return ModelSpecification(loadings=loadings, n_factors=2, n_variables=n_vars)

def cfa_two_factor(A, n_obs):
    n_vars = A.shape[0]
    n_per = n_vars // 2
    spec = _make_spec(n_vars)
    cfa = ConfirmatoryFactorAnalyzer(spec, is_cov_matrix=True, n_obs=n_obs, disp=False, max_iter=2000)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cfa.fit(A)
    except Exception:
        return None, False

    params = np.concatenate([
        cfa.loadings_[:n_per, 0],
        cfa.loadings_[n_per:, 1],
        cfa.error_vars_.ravel(),
        [cfa.factor_varcovs_[0, 1]],
    ])
    converged = np.all(np.isfinite(params))
    return params, converged

def cfa_pass(params, converged):
    return converged and params is not None
