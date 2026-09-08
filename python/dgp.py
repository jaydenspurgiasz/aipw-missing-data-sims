import numpy as np
from scipy.special import expit


class DGPTwoFac:
    def __init__(self, n, p=6, loading=0.8, phi_offdiag=0.3, cov_z_x=0.64,
                 mu=2.0, pi_offset=1.4, pi2_scale=2.0, pi2_offset=2.0):
        self.n = n
        self.p = p
        self.loading = loading
        self.phi_offdiag = phi_offdiag
        self.cov_z_x = cov_z_x
        self.mu = mu
        self.pi_offset = pi_offset
        self.pi2_scale = pi2_scale
        self.pi2_offset = pi2_offset

        self.Lambda = self.Phi = self.Omega = self.sig_X = None
        self.X_true = self.X_o = self.X_o2 = None
        self.Z = self.Z2 = None
        self.pi = self.pi2 = None
        self.R = self.R2 = None

        self._rng = np.random.default_rng()

    def set_seed(self, seed=None):
        if seed is None:
            seed = int(self._rng.integers(0, 999999))
        self._rng = np.random.default_rng(abs(seed))
        return seed

    def gen(self, seed=None):
        self.set_seed(seed)
        rng = self._rng
        n, p = self.n, self.p
        n_factors = 2
        n_per = p // n_factors

        self.Lambda = np.kron(np.eye(n_factors), np.full((n_per, 1), self.loading))
        self.Phi = np.array([[1.0, self.phi_offdiag], [self.phi_offdiag, 1.0]])

        factor_var = np.diag(self.Lambda @ self.Phi @ self.Lambda.T)
        self.Omega = np.diag(1.0 - factor_var)
        self.sig_X = self.Lambda @ self.Phi @ self.Lambda.T + self.Omega

        sig_U = np.zeros((p + 1, p + 1))
        sig_U[:p, :p] = self.sig_X
        sig_U[:p, p] = sig_U[p, :p] = self.cov_z_x
        sig_U[p, p] = 1.0

        U = rng.multivariate_normal(np.full(p + 1, self.mu), sig_U, n)
        self.X_true = U[:, :p]
        self.Z = U[:, p:p + 1]

        self.pi = expit(self.Z.ravel() - self.mu + self.pi_offset)

        Z_sq_scaled = (self.Z.ravel() ** 2)
        Z_sq_scaled = (Z_sq_scaled - Z_sq_scaled.mean()) / Z_sq_scaled.std()
        self.pi2 = expit(self.pi2_scale * Z_sq_scaled + self.pi2_offset)
        self.Z2 = self.Z ** 2

        self.R = rng.binomial(1, np.broadcast_to(self.pi[:, None], (n, p))).astype(float)
        self.R2 = rng.binomial(1, np.broadcast_to(self.pi2[:, None], (n, p))).astype(float)

        self.X_o = np.where(self.R.astype(bool), self.X_true, np.nan)
        self.X_o2 = np.where(self.R2.astype(bool), self.X_true, np.nan)
