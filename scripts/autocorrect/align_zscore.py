import numpy as np

def compute_ref_stats(X):
    """
    X: (n_samples, n_features) from unshifted TRAIN data
    """
    mu = X.mean(axis=0)
    sigma = X.std(axis=0) + 1e-8
    return mu, sigma


def zscore_align(X, mu_ref, sigma_ref):
    """
    Align features to reference distribution
    """
    return (X - mu_ref) / sigma_ref
