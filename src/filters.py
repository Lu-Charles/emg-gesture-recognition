from __future__ import annotations
import numpy as np
from scipy.signal import butter, filtfilt, iirnotch

def bandpass(x: np.ndarray, fs: float, lo: float, hi: float, order: int = 4) -> np.ndarray:
    lo_n = lo / (fs / 2.0)
    hi_n = hi / (fs / 2.0)
    b, a = butter(order, [lo_n, hi_n], btype="bandpass")
    return filtfilt(b, a, x)

def notch(x: np.ndarray, fs: float, f0: float = 60.0, q: float = 30.0) -> np.ndarray:
    w0 = f0 / (fs / 2.0)
    b, a = iirnotch(w0, q)
    return filtfilt(b, a, x)

def preprocess_emg(x: np.ndarray, fs: float, lo: float, hi: float, notch_hz: float, notch_q: float) -> np.ndarray:
    y = x.astype(np.float64)
    y = y - np.median(y)  # robust DC removal
    y = notch(y, fs=fs, f0=notch_hz, q=notch_q)
    y = bandpass(y, fs=fs, lo=lo, hi=hi, order=4)
    return y