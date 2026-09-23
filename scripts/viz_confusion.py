# scripts/viz_confusion.py
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import confusion_matrix

from src.config import StreamConfig
from src.features import extract_window_features


def sliding_windows(x, fs, win_sec, hop_sec):
    win = int(win_sec * fs)
    hop = int(hop_sec * fs)
    for i in range(0, len(x) - win + 1, hop):
        yield x[i:i + win]


def majority_vote(preds):
    vals, counts = np.unique(preds, return_counts=True)
    return vals[np.argmax(counts)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--label_map", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--channel", choices=["ch0_f", "ch1_f", "dual"], default="ch0_f")
    ap.add_argument("--out", default="outputs/confusion_matrix.png")
    args = ap.parse_args()

    cfg = StreamConfig()
    data_dir = Path(args.data_dir)
    labels = pd.read_csv(args.label_map)
    model = joblib.load(args.model)

    y_true, y_pred = [], []

    for _, row in labels.iterrows():
        fpath = data_dir / row["file"]
        df = pd.read_csv(fpath)

        if args.channel == "dual":
            x0 = df["ch0_f"].to_numpy()
            x1 = df["ch1_f"].to_numpy()
            feats = []
            for w0, w1 in zip(
                sliding_windows(x0, cfg.fs_hz, cfg.win_sec, cfg.hop_sec),
                sliding_windows(x1, cfg.fs_hz, cfg.win_sec, cfg.hop_sec),
            ):
                f0 = extract_window_features(w0, cfg.fs_hz)
                f1 = extract_window_features(w1, cfg.fs_hz)
                feats.append({**f0, **{f"ch1_{k}": v for k, v in f1.items()}})
            X = pd.DataFrame(feats).to_numpy()
        else:
            x = df[args.channel].to_numpy()
            feats = [
                extract_window_features(w, cfg.fs_hz)
                for w in sliding_windows(x, cfg.fs_hz, cfg.win_sec, cfg.hop_sec)
            ]
            X = pd.DataFrame(feats).to_numpy()

        preds = model.predict(X)
        trial_pred = majority_vote(preds)

        y_true.append(row["label"])
        y_pred.append(trial_pred)

    classes = sorted(np.unique(y_true))
    cm = confusion_matrix(y_true, y_pred, labels=classes)

    # ---- Plot ----
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(cm, cmap="Blues")

    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)

    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Gesture classification confusion matrix")

    # numbers + percentages
    for i in range(len(classes)):
        for j in range(len(classes)):
            count = cm[i, j]
            pct = count / cm[i].sum() * 100 if cm[i].sum() else 0
            ax.text(
                j, i, f"{count}\n{pct:.1f}%",
                ha="center", va="center",
                color="white" if count > cm.max() * 0.5 else "black",
                fontsize=11,
            )

    plt.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=300)
    print(f"[viz] wrote {out}")


if __name__ == "__main__":
    main()
