# scripts/autocorrect/align_coral.py
from __future__ import annotations
import numpy as np


def _cov(X: np.ndarray) -> np.ndarray:
    """
    Covariance with small regularization for stability.
    X: (n_samples, n_features)
    """
    X = np.asarray(X, dtype=np.float64)
    if X.shape[0] < 2:
        # not enough rows; return identity
        return np.eye(X.shape[1], dtype=np.float64)

    C = np.cov(X, rowvar=False, bias=False)
    # regularize (important!)
    reg = 1e-3 * np.trace(C) / max(1, C.shape[0])
    return C + reg * np.eye(C.shape[0], dtype=np.float64)


def _sqrtm_psd(C: np.ndarray) -> np.ndarray:
    """
    Symmetric PSD matrix square root via eigen decomposition.
    """
    w, V = np.linalg.eigh(C)
    w = np.maximum(w, 1e-12)
    return (V * np.sqrt(w)) @ V.T


def _invsqrtm_psd(C: np.ndarray) -> np.ndarray:
    """
    Symmetric PSD matrix inverse square root via eigen decomposition.
    """
    w, V = np.linalg.eigh(C)
    w = np.maximum(w, 1e-12)
    return (V * (1.0 / np.sqrt(w))) @ V.T


def coral_align(X: np.ndarray, X_ref: np.ndarray) -> np.ndarray:
    """
    CORAL alignment: match covariance of X (target/shifted) to X_ref (source/unshifted).

    Steps:
      1) center X and X_ref
      2) whiten X with invsqrt(cov(X))
      3) recolor with sqrt(cov(X_ref))
      4) add back ref mean (optional, but usually helps)
    """
    X = np.asarray(X, dtype=np.float64)
    X_ref = np.asarray(X_ref, dtype=np.float64)

    mu_t = X.mean(axis=0)
    mu_s = X_ref.mean(axis=0)

    Xt = X - mu_t
    Xs = X_ref - mu_s

    Ct = _cov(Xt)
    Cs = _cov(Xs)

    Xt_white = Xt @ _invsqrtm_psd(Ct)
    Xt_coral = Xt_white @ _sqrtm_psd(Cs)

    return Xt_coral + mu_s
