# scripts/train.py
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import StreamConfig
from src.dataset import build_feature_table
from src.features import extract_window_features
from src.model import make_rf, save_model
from src.utils import ensure_dir


def resolve_trial_path(data_dir: Path, file_field: str) -> Path:
    p = Path(str(file_field))

    if p.is_absolute() and p.exists():
        return p

    cand = data_dir / p
    if cand.exists():
        return cand

    # fallback: just the basename inside data_dir
    return data_dir / p.name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/processed")
    ap.add_argument("--out_model", default="data/models/rf_gesture.joblib")
    ap.add_argument("--label_map", default="data/logs/labels.csv")

    ap.add_argument(
        "--channels",
        nargs="+",
        default=["ch0_f", "ch1_f"],
        help="One or more processed channels, e.g. --channels ch0_f ch1_f",
    )
    args = ap.parse_args()

    cfg = StreamConfig()
    data_dir = Path(args.data_dir)
    labels_path = Path(args.label_map)

    channels = args.channels
    allowed = {"ch0_f", "ch1_f"}
    for ch in channels:
        if ch not in allowed:
            raise ValueError(f"Unknown channel '{ch}'. Allowed: {sorted(allowed)}")

    labels = pd.read_csv(labels_path)

    feats_all = []
    y_all: list[str] = []

    for _, row in labels.iterrows():
        fpath = resolve_trial_path(data_dir, row["file"])
        label = row["label"]

        if not fpath.exists():
            raise FileNotFoundError(
                f"Missing processed file: {fpath}\n"
                f"label_map file field was: {row['file']}\n"
                f"data_dir was: {data_dir}"
            )

        df = pd.read_csv(fpath)

        chan_tables = []
        for ch in channels:
            if ch not in df.columns:
                raise ValueError(
                    f"{fpath} missing column '{ch}'. Available columns: {list(df.columns)}"
                )

            x = df[ch].to_numpy(dtype=np.float64)

            ft = build_feature_table(
                x, cfg.fs_hz, cfg.win_sec, cfg.hop_sec, extract_window_features
            ).drop(columns=["start_idx"], errors="ignore")

            ft = ft.add_prefix(f"{ch}__")
            chan_tables.append(ft)

        ft_concat = pd.concat(chan_tables, axis=1)
        feats_all.append(ft_concat)
        y_all.extend([label] * len(ft_concat))

    X_df = pd.concat(feats_all, ignore_index=True)
    feature_names = list(X_df.columns)
    X = X_df.to_numpy(dtype=np.float64)
    y = np.array(y_all)

    model = make_rf()
    model.fit(X, y)

    rf = model.named_steps["rf"]
    print("Feature importances (top 25):")
    order = np.argsort(rf.feature_importances_)[::-1][:25]
    for i in order:
        print(f"{feature_names[i]}: {rf.feature_importances_[i]:.6f}")

    out_model = Path(args.out_model)
    ensure_dir(out_model.parent)
    save_model(model, str(out_model))
    print(f"Saved model: {out_model} | X={X.shape} | channels={channels}")


if __name__ == "__main__":
    main()
