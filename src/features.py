from __future__ import annotations
import numpy as np

def _safe_eps(x: float = 1e-12) -> float:
    return x

def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x * x) + _safe_eps()))

def mav(x: np.ndarray) -> float:
    return float(np.mean(np.abs(x)))

def waveform_length(x: np.ndarray) -> float:
    return float(np.sum(np.abs(np.diff(x))))

def zero_crossings(x: np.ndarray, thresh: float) -> float:
    s = np.sign(x)
    ds = s[1:] * s[:-1]
    zc = np.sum((ds < 0) & (np.abs(x[1:]-x[:-1]) > thresh))
    return float(zc)

def slope_sign_changes(x: np.ndarray, thresh: float) -> float:
    dx1 = np.diff(x)
    dx2 = np.diff(x, n=2)
    # SSC: sign change in slope, with amplitude thresholding
    ssc = np.sum(((dx1[1:] * dx1[:-1]) < 0) & (np.abs(dx2) > thresh))
    return float(ssc)

def spectrum_feats(x: np.ndarray, fs: float) -> dict:
    # FFT-based power spectrum
    n = len(x)
    w = np.hanning(n)
    X = np.fft.rfft(x * w)
    P = (np.abs(X) ** 2)
    freqs = np.fft.rfftfreq(n, d=1.0/fs)

    p_sum = np.sum(P) + 1e-12
    mnf = float(np.sum(freqs * P) / p_sum)

    cdf = np.cumsum(P) / p_sum
    mdf = float(freqs[np.searchsorted(cdf, 0.5)])

    # bandpower ratio (25–80 vs 80–250 by default-ish)
    def bandpow(f_lo, f_hi):
        mask = (freqs >= f_lo) & (freqs < f_hi)
        return float(np.sum(P[mask]) + 1e-12)

    low = bandpow(25, 80)
    high = bandpow(80, 250)
    ratio_lh = float(low / high)

    return {"mnf": mnf, "mdf": mdf, "bp_ratio_lh": ratio_lh}

def extract_window_features(xw: np.ndarray, fs: float) -> dict:
    thr = 0.01 * (np.std(xw) + 1e-12)
    feats = {
        "rms": rms(xw),
        "mav": mav(xw),
        "wl": waveform_length(xw),
        "zc": zero_crossings(xw, thr),
        "ssc": slope_sign_changes(xw, thr),
    }
    feats.update(spectrum_feats(xw, fs))
    feats["hfd"] = higuchi_fd(xw)   # <<< ADD THIS
    return feats


def higuchi_fd(x: np.ndarray, kmax: int = 8) -> float:
    N = len(x)
    L = []
    for k in range(1, kmax + 1):
        Lk = []
        for m in range(k):
            idx = np.arange(m, N, k)
            if len(idx) < 2:
                continue
            dist = np.sum(np.abs(np.diff(x[idx])))
            norm = (N - 1) / ((len(idx) - 1) * k)
            Lk.append(dist * norm)
        if len(Lk) > 0:
            L.append(np.mean(Lk))
    if len(L) < 2:
        return 0.0
    logL = np.log(L)
    logk = np.log(1.0 / np.arange(1, len(L) + 1))
    return float(np.polyfit(logk, logL, 1)[0])
