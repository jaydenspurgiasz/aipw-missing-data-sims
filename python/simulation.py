import numpy as np
import pandas as pd
import statsmodels.api as sm
from tqdm.auto import tqdm

from dgp import DGPTwoFac
from covariance import pairwise_cov, complete_case_cov
from aipw import aipw_z
from fiml import fiml
from utils import psd_project, is_invertible
from cfa import cfa_two_factor, cfa_pass
from metrics import result_table
import re
from matplotlib import pyplot as plt
from matplotlib.patches import Patch


class Simulation:
    """
    n: Sample size
    r: Number of runs
    seed: Random seed
    **dgp_kw: p, loading, phi_offdiag, cov_z_x, mu,
            pi_offset, pi2_scale, pi2_offset
    """
    '''
    METHODS = [
        'Full', 'Cases', 'Pairs',
        'AIPW_pi_hat', 'AIPW_pi', 'AIPW_pi_mis', 'AIPW_z2',
        'FIML', 'FIML_Z2', 'FIML_Z',
    ]
    '''
    
    METHODS = [
        "Full", "Pair Del.", "List Del.",
        "AIPW (Both correct)", "AIPW ($$\\mathbb{P}$$ mis.)", "AIPW ($$\\mathbb{E}$$ mis.)",
        "FIML (Z normal)", "FIML (Z omitted)", "FIML (Z non-normal)"
    ]

    def __init__(self, n, r, seed=None, **dgp_kw):
        self.n = n
        self.r = r
        self.seed = seed
        self.dgp = DGPTwoFac(n, **dgp_kw)

        self._est = None
        self._est_fa = None
        self._theta_cov = None
        self._theta_fa = None

    def run(self):
        if self.seed is not None:
            np.random.seed(self.seed)
            self.dgp._rng = np.random.default_rng(self.seed)

        dgp = self.dgp
        dgp.gen()
        p = dgp.p
        tril_rows, tril_cols = np.tril_indices(p)
        self._tril = (tril_rows, tril_cols)
        n_cov = len(tril_rows)
        n_fa = 2 * p + 1
        k = len(self.METHODS)

        self._theta_cov = dgp.sig_X[tril_rows, tril_cols].copy()
        self._theta_fa = np.concatenate([
            dgp.Lambda[:p//2, 0],
            dgp.Lambda[p//2:, 1],
            np.diag(dgp.Omega),
            [dgp.Phi[0, 1]],
        ])
        self._tril = (tril_rows, tril_cols)

        est = np.full((self.r, k, n_cov), np.nan)
        est_fa = np.full((self.r, k, n_fa),  np.nan)

        def broadcast(v):
            return np.repeat(v[:, np.newaxis], p, axis=1)

        for i in tqdm(range(self.r)):
            dgp.gen()

            obs_count = (~np.isnan(dgp.X_o)).sum(axis=1)
            miss_count = p - obs_count
            obs_count2 = (~np.isnan(dgp.X_o2)).sum(axis=1)
            miss_count2 = p - obs_count2

            pi_hat = self.get_pi_hat(obs_count,  miss_count,  dgp.Z)
            pi_mis = self.get_pi_hat(obs_count,  miss_count,  dgp.Z2)
            #pi_hat2 = self.get_pi_hat(obs_count2, miss_count2, dgp.Z2)

            pi_hat_m = broadcast(pi_hat)
            pi_mis_m = broadcast(pi_mis)
            #pi_hat2_m = broadcast(pi_hat2)
            pi_true_m = broadcast(dgp.pi)

            est_cov = [
                np.cov(dgp.X_true.T),
                pairwise_cov(dgp.X_o),
                complete_case_cov(dgp.X_o),
                aipw_z(dgp.X_o, dgp.Z, pi_hat_m),
                aipw_z(dgp.X_o, dgp.Z, pi_mis_m),
                aipw_z(dgp.X_o, dgp.Z2, pi_hat_m),
                fiml(np.column_stack([dgp.X_o,  dgp.Z ]))[:p, :p],
                fiml(dgp.X_o),
                fiml(np.column_stack([dgp.X_o, dgp.Z2]))[:p, :p]
            ]

            for j, A in enumerate(est_cov):
                est[i, j, :] = A[tril_rows, tril_cols]
                try:
                    A_psd = psd_project(A)
                    params, converged = cfa_two_factor(A_psd, self.n)
                    if cfa_pass(params, converged):
                        est_fa[i, j, :] = params
                except:
                    pass

        self._est = est
        self._est_fa = np.abs(est_fa)

    # Helper
    def _ran(self):
        if self._est is None:
            self.run()

    def get_pi_hat(self, obs, mis, X):
        X_with_constant = sm.add_constant(X)
        model = sm.GLM(np.column_stack([obs, mis]), X_with_constant, family=sm.families.Binomial())
        
        return model.fit(disp=0).predict()

    # Results
    def covariance_results(self) -> pd.DataFrame:
        # Cov metrics

        self._ran()
        return result_table(self._est, self._theta_cov, self.METHODS)

    def factor_analysis_results(self) -> pd.DataFrame:
        # FA metrics

        self._ran()
        return result_table(self._est_fa, self._theta_fa, self.METHODS)

    def results_chart(self):
        self._ran()
        df = self.covariance_results()

        methods  = df.index.tolist()
        rel_bias = df['Rel. Bias'].tolist()

        # Strip $$\mathbb{X}$$ → X so matplotlib's mathtext parser doesn't choke
        labels = [re.sub(r'\$+\\mathbb\{(\w+)\}\$+', r'\1', m) for m in methods]

        colors = []
        for method in methods:
            if method.startswith('AIPW'):
                colors.append('#1f77b4')
            elif method.startswith('FIML'):
                colors.append('#2ca02c')
            else:
                colors.append('#ff7f0e')

        plt.figure(figsize=(10, 6))
        bars = plt.bar(labels, rel_bias, color=colors, edgecolor='black', linewidth=0.5)

        plt.ylabel('Relative Bias')
        plt.title('Relative Bias by Method', fontsize=18)
        plt.xticks(rotation=45, ha='right')

        legend_elements = [
            Patch(facecolor='#ff7f0e', edgecolor='black', label='Other Methods'),
            Patch(facecolor='#1f77b4', edgecolor='black', label='AIPW Methods'),
            Patch(facecolor='#2ca02c', edgecolor='black', label='FIML Methods'),
        ]
        plt.legend(handles=legend_elements)

        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.002,
                     round(yval, 4), ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        plt.show()




    @property
    def valid_fa_counts(self) -> pd.Series:
        # Times FA converged i.e. no error

        self._ran()
        return pd.Series(
            (~np.isnan(self._est_fa[:, :, 0])).sum(axis=0),
            index=self.METHODS,
            name='valid_fits',
        )
