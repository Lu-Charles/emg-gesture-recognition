# scripts/viz_pca.py
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.config import StreamConfig
from src.dataset import build_feature_table
from src.features import extract_window_features


def resolve_trial_path(data_dir: Path, file_field: str) -> Path:
    p = Path(str(file_field))

    if p.is_absolute():
        return p

    return data_dir / p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="processed data root")
    ap.add_argument("--label_map", required=True, help="labels CSV")
    ap.add_argument("--channel", choices=["ch0_f", "ch1_f"], default="ch0_f")
    ap.add_argument("--out", default="outputs/pca_features.png")
    args = ap.parse_args()

    cfg = StreamConfig()
    data_dir = Path(args.data_dir)
    labels = pd.read_csv(args.label_map)

    X_all = []
    y_all = []

    for _, row in labels.iterrows():
        fpath = resolve_trial_path(data_dir, row["file"])

        if not fpath.exists():
            raise FileNotFoundError(f"Missing file: {fpath}")

        label = row["label"]

        df = pd.read_csv(fpath)

        if args.channel not in df.columns:
            raise ValueError(
                f"{fpath} missing column '{args.channel}'. "
                f"Available: {list(df.columns)}"
            )

        x = df[args.channel].to_numpy(dtype=np.float64)

        feats = build_feature_table(
            x,
            cfg.fs_hz,
            cfg.win_sec,
            cfg.hop_sec,
            extract_window_features,
        ).drop(columns=["start_idx"], errors="ignore")

        X_all.append(feats)
        y_all.extend([label] * len(feats))

    X = pd.concat(X_all, ignore_index=True).to_numpy()
    y = np.array(y_all)

    # ---- Standardize + PCA ----
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    pca = PCA(n_components=2)
    Z = pca.fit_transform(Xs)

    # ---- Plot ----
    plt.figure(figsize=(7, 6))

    for lab in np.unique(y):
        mask = (y == lab)
        plt.scatter(
            Z[mask, 0],
            Z[mask, 1],
            label=str(lab),
            alpha=0.55,
            s=25
        )

    channel_name = "Flexor channel" if args.channel == "ch0_f" else "Extensor channel"

    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title(f"EMG feature-space projection — {channel_name}")
    plt.legend(title="Gesture")
    plt.grid(alpha=0.25)

    plt.gca().set_aspect("equal", adjustable="box")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, dpi=200)

    print(f"[viz] wrote {out}")


if __name__ == "__main__":
    main()