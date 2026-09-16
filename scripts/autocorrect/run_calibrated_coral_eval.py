# scripts/autocorrect/run_calibrated_coral_eval.py
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


def resolve_path(data_dir: Path, file_field: str) -> Path:
    p = Path(str(file_field))
    cand = data_dir / p
    if cand.exists():
        return cand
    return data_dir / p.name


def extract_dual_features(csv_path: Path, cfg: StreamConfig) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    x0 = df["ch0_f"].to_numpy(dtype=np.float64)
    x1 = df["ch1_f"].to_numpy(dtype=np.float64)

    f0 = build_feature_table(x0, cfg.fs_hz, cfg.win_sec, cfg.hop_sec, extract_window_features)\
        .drop(columns=["start_idx"], errors="ignore").add_prefix("ch0_")
    f1 = build_feature_table(x1, cfg.fs_hz, cfg.win_sec, cfg.hop_sec, extract_window_features)\
        .drop(columns=["start_idx"], errors="ignore").add_prefix("ch1_")

    return pd.concat([f0, f1], axis=1)


def _sqrtm_psd(C: np.ndarray) -> np.ndarray:
    w, V = np.linalg.eigh(C)
    w = np.maximum(w, 1e-12)
    return (V * np.sqrt(w)) @ V.T


def _invsqrtm_psd(C: np.ndarray) -> np.ndarray:
    w, V = np.linalg.eigh(C)
    w = np.maximum(w, 1e-12)
    return (V * (1.0 / np.sqrt(w))) @ V.T


def cov(X: np.ndarray) -> np.ndarray:
    X0 = X - X.mean(axis=0, keepdims=True)
    C = (X0.T @ X0) / max(1, (len(X0) - 1))
    # ridge for stability
    C = C + 1e-3 * np.eye(C.shape[0])
    return C


def coral_fit(X_src: np.ndarray, X_tgt: np.ndarray):
    """
    Fit CORAL transform that maps TARGET -> SOURCE.
    Returns (mu_src, mu_tgt, A) where:
      X_aligned = (X - mu_tgt) @ A + mu_src
    """
    mu_s = X_src.mean(axis=0)
    mu_t = X_tgt.mean(axis=0)
    Cs = cov(X_src)
    Ct = cov(X_tgt)

    A = _invsqrtm_psd(Ct) @ _sqrtm_psd(Cs)
    return mu_s, mu_t, A


def coral_apply(X: np.ndarray, mu_s: np.ndarray, mu_t: np.ndarray, A: np.ndarray) -> np.ndarray:
    return (X - mu_t) @ A + mu_s


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
    ap.add_argument("--data_dir", required=True)

    ap.add_argument("--unshifted_labels", required=True, help="reference (unshifted) labels csv")
    ap.add_argument("--shifted_calib_labels", required=True, help="small shifted calibration labels csv")
    ap.add_argument("--shifted_eval_labels", required=True, help="shifted eval labels csv (the rest)")

    ap.add_argument("--max_calib_trials", type=int, default=0,
                    help="optional cap on calibration trials (0 = use all rows in calib file)")

    args = ap.parse_args()

    cfg = StreamConfig()
    model = joblib.load(args.model)
    data_dir = Path(args.data_dir)

    # -----------------------------
    # Build SOURCE feature pool (unshifted)
    # -----------------------------
    src_df = pd.read_csv(args.unshifted_labels)
    Xs_list = []
    for _, row in src_df.iterrows():
        p = resolve_path(data_dir, row["file"])
        Xs_list.append(extract_dual_features(p, cfg))
    X_src = pd.concat(Xs_list, ignore_index=True).to_numpy(dtype=np.float64)

    # -----------------------------
    # Build TARGET feature pool (shifted calibration)
    # -----------------------------
    calib_df = pd.read_csv(args.shifted_calib_labels)
    if args.max_calib_trials and args.max_calib_trials > 0:
        calib_df = calib_df.iloc[: args.max_calib_trials].copy()

    Xt_list = []
    for _, row in calib_df.iterrows():
        p = resolve_path(data_dir, row["file"])
        Xt_list.append(extract_dual_features(p, cfg))
    X_tgt = pd.concat(Xt_list, ignore_index=True).to_numpy(dtype=np.float64)

    # Fit CORAL from calibration pool
    mu_s, mu_t, A = coral_fit(X_src, X_tgt)

    # -----------------------------
    # Evaluate on SHIFTED eval set
    # -----------------------------
    eval_df = pd.read_csv(args.shifted_eval_labels)

    trial_files = []
    y_true, y_pred = [], []
    pred_counts = Counter()

    for _, row in eval_df.iterrows():
        p = resolve_path(data_dir, row["file"])
        trial_files.append(str(row["file"]))
        feats = extract_dual_features(p, cfg)
        X = feats.to_numpy(dtype=np.float64)

        X_aligned = coral_apply(X, mu_s, mu_t, A)

        preds = model.predict(X_aligned)
        trial_pred = majority_vote(preds)

        y_true.append(row["label"])
        y_pred.append(trial_pred)
        pred_counts[trial_pred] += 1

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    acc = float((y_true == y_pred).mean())
    print(f"[CALIBRATED-CORAL | dual-channel] Trial accuracy: {acc:.3f}\n")

    print("Predicted label counts:")
    print(dict(pred_counts), "\n")

    cm, cm_norm = confusion_tables(y_true, y_pred)
    print("Confusion matrix (counts):")
    print(cm, "\n")
    print("Confusion matrix (row-normalized):")
    print(cm_norm)

    # Misclassified trials list
    errors = []
    for f, yt, yp in zip(trial_files, y_true, y_pred):
        if yt != yp:
            errors.append((f, yt, yp))

    if errors:
        err_df = pd.DataFrame(errors, columns=["file", "true_label", "predicted_label"])

        print("\nMisclassified trials:")
        print(err_df.to_string(index=False))

        # Save to CSV (recommended)
        out_dir = Path("graphs")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "misclassified_calibrated_coral.csv"
        err_df.to_csv(out_path, index=False)
        print(f"\nSaved: {out_path}")
    else:
        print("\nNo misclassified trials.")



if __name__ == "__main__":
    main()
