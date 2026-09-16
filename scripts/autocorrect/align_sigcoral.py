# scripts/autocorrect/align_sigcoral.py
from __future__ import annotations
import numpy as np

def _sqrtm_psd(C: np.ndarray) -> np.ndarray:
    w, V = np.linalg.eigh(C)
    w = np.maximum(w, 1e-12)
    return (V * np.sqrt(w)) @ V.T

def _invsqrtm_psd(C: np.ndarray) -> np.ndarray:
    w, V = np.linalg.eigh(C)
    w = np.maximum(w, 1e-12)
    return (V * (1.0 / np.sqrt(w))) @ V.T

def cov2(S: np.ndarray) -> np.ndarray:
    # S: (n,2)
    S0 = S - S.mean(axis=0, keepdims=True)
    C = (S0.T @ S0) / max(1, (len(S0) - 1))
    # tiny ridge for numerical stability
    C = C + 1e-6 * np.eye(2)
    return C

def sigcoral_align(S: np.ndarray, C_ref: np.ndarray) -> np.ndarray:
    """
    Signal-level CORAL (2x2):
      whiten TARGET covariance, recolor to REF covariance
    """
    mu = S.mean(axis=0, keepdims=True)
    X = S - mu
    C_t = cov2(X)
    Xw = X @ _invsqrtm_psd(C_t)
    Xc = Xw @ _sqrtm_psd(C_ref)
    return Xc + mu

def sigwhiten_align(S: np.ndarray) -> np.ndarray:
    """
    Just whiten target to unit covariance (no recolor).
    """
    mu = S.mean(axis=0, keepdims=True)
    X = S - mu
    C_t = cov2(X)
    Xw = X @ _invsqrtm_psd(C_t)
    return Xw + mu
