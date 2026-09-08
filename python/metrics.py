import numpy as np
import pandas as pd
from scipy.stats import f as f_dist
from utils import is_invertible


def result_table(est, theta0, method_names):
    _, k, d = est.shape
    rows = []

    for i in range(k):
        est_i = est[:, i, :]
        est_i = est_i[~np.isnan(est_i).any(axis=1)]
        r = est_i.shape[0]

        if r < 2:
            rows.append(dict(Method=method_names[i], Var=np.nan, Bias=np.nan, **{"Rel. Bias": np.nan}, MSE=np.nan, F=np.nan, pval=np.nan))
            continue

        m = est_i.mean(axis=0)
        A = np.cov(est_i.T)
        b = m - theta0

        var = np.trace(A)
        bias = np.sqrt(np.sum(b ** 2))
        rel_bias = bias / np.sqrt(np.sum(theta0 ** 2))
        mse = var + bias ** 2

        if is_invertible(A) and r > d:
            df1 = r - d
            df2 = d * (r - 1)
            F_stat = (df1 / df2) * r * (b @ np.linalg.solve(A, b))
            pval = f_dist.sf(F_stat, dfn=df1, dfd=df2)
        else:
            F_stat = np.nan
            pval = np.nan

        rows.append(dict(Method=method_names[i], Var=var, Bias=bias, **{"Rel. Bias": rel_bias}, MSE=mse, F=F_stat, pval=pval))

    return pd.DataFrame(rows).set_index("Method")
