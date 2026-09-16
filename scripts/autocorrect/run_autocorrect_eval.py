# scripts/autocorrect/run_autocorrect_eval.py
from __future__ import annotations

import argparse
from pathlib import Path
from collections import Counter

import joblib
import numpy as np
import pandas as pd

from src.config import StreamConfig
from src.dataset import build_feature_table
from src.features import extract_window_features

from scripts.autocorrect.align_sigcoral import cov2, sigcoral_align, sigwhiten_align


def extract_features_from_signals(x0: np.ndarray, x1: np.ndarray, cfg: StreamConfig) -> pd.DataFrame:
    """
    Build dual-channel feature table with stable column ordering:
      ch0_* then ch1_*
    """
    f0 = build_feature_table(
        x0, cfg.fs_hz, cfg.win_sec, cfg.hop_sec, extract_window_features
    ).drop(columns=["start_idx"], errors="ignore").add_prefix("ch0_")

    f1 = build_feature_table(
        x1, cfg.fs_hz, cfg.win_sec, cfg.hop_sec, extract_window_features
    ).drop(columns=["start_idx"], errors="ignore").add_prefix("ch1_")

    return pd.concat([f0, f1], axis=1)


def resolve_path(data_dir: Path, file_field: str) -> Path:
    p = Path(str(file_field))
    cand = data_dir / p
    if cand.exists():
        return cand
    return data_dir / p.name


def majority_vote(preds: np.ndarray) -> str:
    return pd.Series(preds).mode().iloc[0]


def confusion_tables(y_true: np.ndarray, y_pred: np.ndarray):
    classes = sorted(set(y_true) | set(y_pred))
    cm = pd.crosstab(
        pd.Series(y_true, name="true"),
        pd.Series(y_pred, name="pred"),
        dropna=False,
    ).reindex(index=classes, columns=classes, fill_value=0)
    cm_norm = cm.div(cm.sum(axis=1).replace(0, 1), axis=0)
    return cm, cm_norm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--train_labels", required=True)
    ap.add_argument("--test_labels", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--align", choices=["none", "sigwhiten", "sigcoral"], default="none")
    args = ap.parse_args()

    cfg = StreamConfig()
    model = joblib.load(args.model)
    data_dir = Path(args.data_dir)

    # -----------------------
    # Build REF covariance (2x2) from UNSHIFTED TRAIN signals
    # -----------------------
    train_df = pd.read_csv(args.train_labels)

    S_ref_list = []
    for _, row in train_df.iterrows():
        p = resolve_path(data_dir, row["file"])
        if not p.exists():
            raise FileNotFoundError(f"Missing train file: {p} (from {args.train_labels})")
        df = pd.read_csv(p)
        if "ch0_f" not in df.columns or "ch1_f" not in df.columns:
            raise ValueError(f"{p} missing ch0_f/ch1_f. Columns: {list(df.columns)}")
        x0 = df["ch0_f"].to_numpy(dtype=np.float64)
        x1 = df["ch1_f"].to_numpy(dtype=np.float64)
        S_ref_list.append(np.column_stack([x0, x1]))

    S_ref = np.vstack(S_ref_list)
    C_ref = cov2(S_ref)

    # -----------------------
    # Evaluate SHIFTED
    # -----------------------
    test_df = pd.read_csv(args.test_labels)

    y_true, y_pred = [], []
    pred_counts = Counter()

    tag = args.align.upper()
    for _, row in test_df.iterrows():
        p = resolve_path(data_dir, row["file"])
        if not p.exists():
            raise FileNotFoundError(f"Missing test file: {p} (from {args.test_labels})")

        df = pd.read_csv(p)
        x0 = df["ch0_f"].to_numpy(dtype=np.float64)
        x1 = df["ch1_f"].to_numpy(dtype=np.float64)

        S = np.column_stack([x0, x1])

        if args.align == "none":
            S_use = S
        elif args.align == "sigwhiten":
            S_use = sigwhiten_align(S)
        else:  # sigcoral
            S_use = sigcoral_align(S, C_ref)

        feats = extract_features_from_signals(S_use[:, 0], S_use[:, 1], cfg)
        X = feats.to_numpy(dtype=np.float64)

        preds = model.predict(X)
        trial_pred = majority_vote(preds)

        y_true.append(row["label"])
        y_pred.append(trial_pred)
        pred_counts[trial_pred] += 1

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    acc = float((y_true == y_pred).mean())
    print(f"[{tag} | dual-channel] Trial accuracy: {acc:.3f}\n")

    print("Predicted label counts:")
    print(dict(pred_counts), "\n")

    cm, cm_norm = confusion_tables(y_true, y_pred)
    print("Confusion matrix (counts):")
    print(cm, "\n")
    print("Confusion matrix (row-normalized):")
    print(cm_norm)


if __name__ == "__main__":
    main()
