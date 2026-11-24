import numpy as np


def outer_array(X):
    return np.einsum("ij,ik->jki", X, X)

# Project A >= 0
def psd_project(A, min_eig=1e-4):
    vals, vecs = np.linalg.eigh(A)
    vals = np.maximum(vals, min_eig)
    return vecs @ np.diag(vals) @ vecs.T

def is_invertible(A):
    try:
        np.linalg.solve(A, np.ones(A.shape[0]))  # one RHS is sufficient to detect singularity
        return True
    except np.linalg.LinAlgError:
        return False
