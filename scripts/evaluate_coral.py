# scripts/evaluate_coral.py
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
        f"Could not resolve file '{file_field}' under data_dir='{data_dir}'. Tried: {cand}"
    )


def sliding_windows(x: np.ndarray, fs: float, win_sec: float, hop_sec: float):
    win = int(win_sec * fs)
    hop = int(hop_sec * fs)
    for i in range(0, len(x) - win + 1, hop):
        yield x[i : i + win]


def majority_vote(preds):
    return Counter(preds).most_common(1)[0][0]


def drop_amp_cols(X_df: pd.DataFrame) -> pd.DataFrame:
    # MUST match what you consider “amp-heavy”
    drop_suffixes = ("__rms", "__mav", "__wl")
    drop_cols = [c for c in X_df.columns if c.endswith(drop_suffixes)]
    return X_df.drop(columns=drop_cols)


def trial_to_feature_df(
    csv_path: Path,
    cfg: StreamConfig,
    channels: list[str],
    drop_amp: bool,
) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    chan_frames = []
    for ch in channels:
        if ch not in df.columns:
            raise ValueError(
                f"{csv_path} missing column '{ch}'. Available: {list(df.columns)}"
            )

        x = df[ch].to_numpy(dtype=np.float64)
        feats = [
            extract_window_features(xw, cfg.fs_hz)
            for xw in sliding_windows(x, cfg.fs_hz, cfg.win_sec, cfg.hop_sec)
        ]
        ft = pd.DataFrame(feats).add_prefix(f"{ch}__")
        chan_frames.append(ft)

    X_df = pd.concat(chan_frames, axis=1)

    if drop_amp:
        X_df = drop_amp_cols(X_df)

    return X_df


def cov_and_mean(X: np.ndarray, eps: float = 1e-6):
    # X: (n, d)
    mu = X.mean(axis=0, keepdims=True)
    Xc = X - mu
    C = (Xc.T @ Xc) / max(1, (Xc.shape[0] - 1))
    d = C.shape[0]
    C = C + eps * np.eye(d)
    return C, mu


def sqrtm_psd(A: np.ndarray):
    w, V = np.linalg.eigh(A)
    w = np.clip(w, 1e-12, None)
    return (V * np.sqrt(w)) @ V.T


def invsqrtm_psd(A: np.ndarray):
    w, V = np.linalg.eigh(A)
    w = np.clip(w, 1e-12, None)
    return (V * (1.0 / np.sqrt(w))) @ V.T


def coral_fit(Xs: np.ndarray, Xt: np.ndarray):
    """
    Learn CORAL transform mapping TARGET -> SOURCE in feature space.
    Returns A, mu_s, mu_t such that:
      X_aligned = (X - mu_t) @ A + mu_s
    """
    Cs, mu_s = cov_and_mean(Xs)
    Ct, mu_t = cov_and_mean(Xt)
    A = invsqrtm_psd(Ct) @ sqrtm_psd(Cs)
    return A, mu_s, mu_t


def coral_apply(X: np.ndarray, A: np.ndarray, mu_s: np.ndarray, mu_t: np.ndarray):
    return (X - mu_t) @ A + mu_s


def stack_features_from_labels(
    data_dir: Path,
    labels_csv: Path,
    cfg: StreamConfig,
    channels: list[str],
    drop_amp: bool,
    verbose: bool = False,
) -> pd.DataFrame:
    labels = pd.read_csv(labels_csv)
    frames = []
    for _, row in labels.iterrows():
        fpath = resolve_trial_path(data_dir, row["file"])
        if verbose:
            print(f"[STACK] {row['file']} -> {fpath}")
        X_df = trial_to_feature_df(fpath, cfg, channels, drop_amp)
        frames.append(X_df)
    if not frames:
        raise RuntimeError(f"No trials found in {labels_csv}")
    return pd.concat(frames, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--use_coral",
        action="store_true",
        help="Apply CORAL domain alignment before evaluation",
    )

    ap.add_argument("--model", required=True)
    ap.add_argument("--channels", nargs="+", default=["ch0_f", "ch1_f"])
    ap.add_argument("--drop_amp", action="store_true", help="drop rms/mav/wl features")

    # Source (unshifted) distribution (only needed if --use_coral)
    ap.add_argument("--source_data_dir", required=True)
    ap.add_argument("--source_label_map", required=True)

    # Target calibration + evaluation (shifted)
    ap.add_argument("--target_data_dir", required=True)
    ap.add_argument("--calib_label_map", required=True)
    ap.add_argument("--eval_label_map", required=True)

    args = ap.parse_args()

    cfg = StreamConfig()
    model = joblib.load(args.model)

    # ---- peel off scaler + classifier if Pipeline ----
    if hasattr(model, "named_steps") and "scaler" in model.named_steps:
        scaler = model.named_steps["scaler"]
        clf = model.named_steps.get("rf", None) or model.named_steps.get("clf", None)
        if clf is None:
            raise ValueError(
                f"Pipeline steps are {list(model.named_steps.keys())}, expected 'rf' or 'clf'"
            )
    else:
        scaler = None
        clf = model

    allowed = {"ch0_f", "ch1_f"}
    for ch in args.channels:
        if ch not in allowed:
            raise ValueError(f"Unknown channel '{ch}'. Allowed: {sorted(allowed)}")

    source_data_dir = Path(args.source_data_dir)
    target_data_dir = Path(args.target_data_dir)

    # ---- Build SOURCE columns template (always, to lock feature order) ----
    print("\n=== Building SOURCE feature template (column order) ===")
    Xs_df = stack_features_from_labels(
        source_data_dir,
        Path(args.source_label_map),
        cfg,
        args.channels,
        args.drop_amp,
        verbose=False,
    )
    cols = list(Xs_df.columns)

    # ---- CORAL stats (only if requested) ----
    A = mu_s = mu_t = None
    if args.use_coral:
        print("=== Building TARGET CALIB feature cloud (for CORAL) ===")
        Xtcal_df = stack_features_from_labels(
            target_data_dir,
            Path(args.calib_label_map),
            cfg,
            args.channels,
            args.drop_amp,
            verbose=False,
        ).reindex(columns=cols)

        Xs = Xs_df.to_numpy(dtype=np.float64)
        Xtcal = Xtcal_df.to_numpy(dtype=np.float64)

        # IMPORTANT: do CORAL in the SAME feature space the classifier expects.
        # Your classifier expects scaled features if scaler exists.
        if scaler is not None:
            Xs = scaler.transform(Xs)
            Xtcal = scaler.transform(Xtcal)

        print(f"[CORAL] source windows: {Xs.shape} | target calib windows: {Xtcal.shape}")
        A, mu_s, mu_t = coral_fit(Xs, Xtcal)
    else:
        print("=== CORAL disabled: running baseline eval (no alignment) ===")

    # ---- Evaluate ----
    print("\n=== Evaluating on TARGET EVAL split ===")
    eval_labels = pd.read_csv(args.eval_label_map)

    y_true = []
    y_pred = []

    for _, row in eval_labels.iterrows():
        fpath = resolve_trial_path(target_data_dir, row["file"])
        true_label = row["label"]

        X_df = trial_to_feature_df(fpath, cfg, args.channels, args.drop_amp).reindex(
            columns=cols
        )
        X = X_df.to_numpy(dtype=np.float64)

        # Match training preprocessing:
        # 1) scale (if pipeline had scaler)
        if scaler is not None:
            X = scaler.transform(X)

        # 2) optional CORAL alignment
        if args.use_coral:
            X = coral_apply(X, A, mu_s, mu_t)

        # 3) predict with classifier
        preds = clf.predict(X)
        trial_pred = majority_vote(preds)

        print(f"{fpath.name}: true={true_label}, pred={trial_pred}")

        y_true.append(true_label)
        y_pred.append(trial_pred)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    acc = (y_true == y_pred).mean()
    tag = "CORAL" if args.use_coral else "baseline"
    print(f"\nTrial-level accuracy ({tag}): {acc:.3f}")

    classes = sorted(set(y_true))
    cm = pd.crosstab(
        pd.Series(y_true, name="true"),
        pd.Series(y_pred, name="pred"),
        dropna=False,
    ).reindex(index=classes, columns=classes, fill_value=0)

    print("Confusion matrix:")
    print(cm)
    print("Classes:", classes)
    print("Channels:", args.channels)
    print("drop_amp:", args.drop_amp)
    print("uses_scaler:", scaler is not None)


if __name__ == "__main__":
    main()
