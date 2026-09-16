# scripts/evaluate.py
from __future__ import annotations

import argparse
from pathlib import Path
from collections import Counter

import joblib
import numpy as np
import pandas as pd

from src.features import extract_window_features
from src.config import StreamConfig


def resolve_trial_path(data_dir: Path, file_field: str) -> Path:
    p = Path(str(file_field))

    if p.is_absolute():
        if p.exists():
            return p
        raise FileNotFoundError(f"Absolute path does not exist: {p}")

    cand = data_dir / p
    if cand.exists():
        return cand

    raise FileNotFoundError(
        f"Could not resolve file '{file_field}' under data_dir='{data_dir}'.\n"
        f"Tried: {cand}"
    )


def sliding_windows(x, fs, win_sec, hop_sec):
    win = int(win_sec * fs)
    hop = int(hop_sec * fs)
    for i in range(0, len(x) - win + 1, hop):
        yield x[i : i + win]


def majority_vote(preds):
    return Counter(preds).most_common(1)[0][0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/processed", help="processed CSV directory")
    ap.add_argument("--label_map", required=True)
    ap.add_argument("--model", required=True)

    ap.add_argument(
        "--channels",
        nargs="+",
        default=["ch0_f", "ch1_f"],
        help="One or more processed channels, e.g. --channels ch0_f ch1_f",
    )

    # NEW: drop amplitude-heavy features to match no-amp training
    ap.add_argument(
        "--drop_amp",
        action="store_true",
        help="Drop __rms/__mav/__wl features (must match no-amp training).",
    )

    args = ap.parse_args()

    cfg = StreamConfig()
    data_dir = Path(args.data_dir)
    labels = pd.read_csv(args.label_map)
    model = joblib.load(args.model)

    channels = args.channels
    allowed = {"ch0_f", "ch1_f"}
    for ch in channels:
        if ch not in allowed:
            raise ValueError(f"Unknown channel '{ch}'. Allowed: {sorted(allowed)}")

    y_true = []
    y_pred = []

    for _, row in labels.iterrows():
        fpath = resolve_trial_path(data_dir, row["file"])
        true_label = row["label"]

        print(f"[SANITY] label_csv={row['file']} -> resolved_path={fpath}")

        if not fpath.exists():
            raise FileNotFoundError(
                f"Missing processed file: {fpath}\n"
                f"label_map file field was: {row['file']}\n"
                f"data_dir was: {data_dir}"
            )

        df = pd.read_csv(fpath)

        chan_feat_frames = []
        for ch in channels:
            if ch not in df.columns:
                raise ValueError(
                    f"{fpath} missing column '{ch}'. Available columns: {list(df.columns)}"
                )

            x = df[ch].to_numpy(dtype=np.float64)
            feats = [
                extract_window_features(xw, cfg.fs_hz)
                for xw in sliding_windows(x, cfg.fs_hz, cfg.win_sec, cfg.hop_sec)
            ]
            ft = pd.DataFrame(feats).add_prefix(f"{ch}__")
            chan_feat_frames.append(ft)

        X_df = pd.concat(chan_feat_frames, axis=1)

        if args.drop_amp:
            drop_suffixes = ("__rms", "__mav", "__wl")
            drop_cols = [c for c in X_df.columns if c.endswith(drop_suffixes)]
            X_df = X_df.drop(columns=drop_cols)

        X = X_df.to_numpy(dtype=np.float64)

        preds = model.predict(X)
        trial_pred = majority_vote(preds)

        print(f"{fpath.name}: true={true_label}, pred={trial_pred}")

        y_true.append(true_label)
        y_pred.append(trial_pred)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    acc = (y_true == y_pred).mean()
    print(f"Trial-level accuracy: {acc:.3f}")

    classes = sorted(set(y_true))
    cm = pd.crosstab(
        pd.Series(y_true, name="true"),
        pd.Series(y_pred, name="pred"),
        dropna=False,
    ).reindex(index=classes, columns=classes, fill_value=0)

    print("Confusion matrix:")
    print(cm)
    print("Classes:", classes)
    print("Channels:", channels)
    print("drop_amp:", args.drop_amp)


if __name__ == "__main__":
    main()
