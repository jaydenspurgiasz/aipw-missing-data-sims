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
import json
from pathlib import Path
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

    def __init__(self, n, r, seed=None, save_data=False,
                 output_dir="simulation_output", **dgp_kw):
        self.n = n
        self.r = r
        self.seed = seed
        self.save_data = save_data
        self.output_dir = Path(output_dir)
        self.dgp = DGPTwoFac(n, **dgp_kw)

        self._est = None
        self._est_fa = None
        self._theta_cov = None
        self._theta_fa = None

    def run(self):
        if self.save_data:
            (self.output_dir / "replications").mkdir(parents=True, exist_ok=True)
            metadata = {
                "n": self.n,
                "r": self.r,
                "seed": self.seed,
                "methods": self.METHODS,
                "dgp": {k: v for k, v in self.dgp.__dict__.items()
                        if not k.startswith("_") and np.isscalar(v)},
            }
            (self.output_dir / "metadata.json").write_text(
                json.dumps(metadata, indent=2, default=str)
            )

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

            if self.save_data:
                aipw_both, aipw_both_diag = aipw_z(
                    dgp.X_o, dgp.Z, pi_hat_m, return_diagnostics=True
                )
                aipw_mis_p, aipw_mis_p_diag = aipw_z(
                    dgp.X_o, dgp.Z, pi_mis_m, return_diagnostics=True
                )
                aipw_mis_e, aipw_mis_e_diag = aipw_z(
                    dgp.X_o, dgp.Z2, pi_hat_m, return_diagnostics=True
                )
                fiml_z, fiml_z_diag = fiml(
                    np.column_stack([dgp.X_o, dgp.Z]), return_diagnostics=True
                )
                fiml_omit, fiml_omit_diag = fiml(
                    dgp.X_o, return_diagnostics=True
                )
                fiml_z2, fiml_z2_diag = fiml(
                    np.column_stack([dgp.X_o, dgp.Z2]), return_diagnostics=True
                )
            else:
                aipw_both = aipw_z(dgp.X_o, dgp.Z, pi_hat_m)
                aipw_mis_p = aipw_z(dgp.X_o, dgp.Z, pi_mis_m)
                aipw_mis_e = aipw_z(dgp.X_o, dgp.Z2, pi_hat_m)
                fiml_z = fiml(np.column_stack([dgp.X_o, dgp.Z]))[:p, :p]
                fiml_omit = fiml(dgp.X_o)
                fiml_z2 = fiml(np.column_stack([dgp.X_o, dgp.Z2]))[:p, :p]

            est_cov = [
                np.cov(dgp.X_true.T),
                pairwise_cov(dgp.X_o),
                complete_case_cov(dgp.X_o),
                aipw_both,
                aipw_mis_p,
                aipw_mis_e,
                fiml_z[:p, :p] if fiml_z.shape[0] != p else fiml_z,
                fiml_omit,
                fiml_z2[:p, :p] if fiml_z2.shape[0] != p else fiml_z2,
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

            if self.save_data:
                self._save_replication(
                    i, dgp, pi_hat, pi_mis, est_cov, est_fa[i],
                    {
                        "aipw_both": aipw_both_diag,
                        "aipw_mis_p": aipw_mis_p_diag,
                        "aipw_mis_e": aipw_mis_e_diag,
                        "fiml_z": fiml_z_diag,
                        "fiml_omit": fiml_omit_diag,
                        "fiml_z2": fiml_z2_diag,
                    },
                )

        self._est = est
        self._est_fa = np.abs(est_fa)
        if self.save_data:
            self.covariance_results().to_csv(self.output_dir / "covariance_results.csv")
            self.factor_analysis_results().to_csv(self.output_dir / "factor_analysis_results.csv")

    def _save_replication(self, i, dgp, pi_hat, pi_mis, est_cov, fa_estimates, diagnostics):
        """Save one replication as readable row data plus matrix diagnostics."""
        p = dgp.p
        data = {
            "Z": dgp.Z.ravel(), "Z2": dgp.Z2.ravel(),
            "pi": dgp.pi, "pi2": dgp.pi2,
            "pi_hat": pi_hat, "pi_mis": pi_mis,
        }
        for label, matrix in {
            "X_true": dgp.X_true, "X_o": dgp.X_o, "X_o2": dgp.X_o2,
            "R": dgp.R, "R2": dgp.R2,
        }.items():
            for j in range(p):
                data[f"{label}_{j + 1}"] = matrix[:, j]
        pd.DataFrame(data).to_csv(
            self.output_dir / "replications" / f"replication_{i + 1:04d}_data.csv",
            index=False,
        )

        arrays = {
            "covariance_estimates": np.stack(est_cov),
            "factor_estimates": fa_estimates,
            "true_covariance": dgp.sig_X,
        }
        for name, values in diagnostics.items():
            for key, value in values.items():
                arrays[f"{name}_{key}"] = value
        np.savez_compressed(
            self.output_dir / "replications" / f"replication_{i + 1:04d}_derived.npz",
            **arrays,
        )

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
