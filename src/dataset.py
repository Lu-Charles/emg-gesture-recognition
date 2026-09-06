from __future__ import annotations
import numpy as np
import pandas as pd

def sliding_windows(x: np.ndarray, fs: int, win_sec: float, hop_sec: float):
    win = int(round(win_sec * fs))
    hop = int(round(hop_sec * fs))
    if win <= 0 or hop <= 0:
        raise ValueError("Bad win/hop.")
    for start in range(0, len(x) - win + 1, hop):
        yield start, x[start:start+win]

def build_feature_table(x: np.ndarray, fs: int, win_sec: float, hop_sec: float, extractor):
    rows = []
    for start, xw in sliding_windows(x, fs, win_sec, hop_sec):
        feats = extractor(xw, fs)
        feats["start_idx"] = start
        rows.append(feats)
    return pd.DataFrame(rows)