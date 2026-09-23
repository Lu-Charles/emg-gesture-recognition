"""Freeze checked whole-trial splits and feature inputs; never fit or score a model."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import warnings

import numpy as np
import pandas as pd

from src.features import extract_window_features
from src.filters import preprocess_emg

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def quality_reasons(signal, quality):
    if signal.ndim != 2 or signal.shape[1] != 3:
        return ["invalid_signal_shape"]
    if len(signal) < quality["minimum_samples"]:
        return ["too_short"]
    if not np.isfinite(signal).all():
        return ["nonfinite_raw"]
    reasons = []
    dt = np.diff(signal[:, 0])
    expected = 1e6 / quality["fs_hz"]
    if abs(np.median(dt) / expected - 1) > quality["median_interval_tolerance_fraction"]:
        reasons.append("wrong_median_sampling_interval")
    if np.any((dt < expected * quality["interval_min_fraction"]) |
              (dt > expected * quality["interval_max_fraction"])):
        reasons.append("timestamp_discontinuity")
    if np.any(np.ptp(signal[:, 1:], axis=0) == 0):
        reasons.append("constant_raw_channel")
    return reasons


def checked_features(signal, cfg):
    preprocessing = dict(cfg["preprocessing"])
    preprocessing["fs"] = preprocessing.pop("fs_hz")
    channels = [preprocess_emg(signal[:, i+1], **preprocessing)
                for i in range(2)]
    if not all(np.isfinite(x).all() for x in channels):
        raise ValueError("nonfinite_filtered_signal")
    rows = []
    for start in range(0, len(signal)-cfg["window_samples"]+1, cfg["hop_samples"]):
        row = []
        for x in channels:
            window = x[start:start+cfg["window_samples"]]
            if np.ptp(window) == 0:
                raise ValueError("constant_filtered_window")
            with warnings.catch_warnings():
                warnings.simplefilter("error", RuntimeWarning)
                feats = extract_window_features(window, cfg["preprocessing"]["fs_hz"])
            if list(feats) != cfg["features_per_channel"]:
                raise RuntimeError("Feature schema changed")
            row.extend(feats.values())
        if not np.isfinite(row).all():
            raise ValueError("nonfinite_features")
        rows.append(row)
    return np.asarray(rows, dtype=np.float64)


def validate_identity(records):
    for field in ["trial_id", "signal_sha256", "channels_sha256"]:
        values = [r[field] for r in records]
        if len(values) != len(set(values)):
            raise ValueError(f"Duplicate {field}; resolve before splitting")
    for r in records:
        if Path(r["trial_id"]).stem.rsplit("_", 1)[0] != r["label"]:
            raise ValueError("Conflicting trial label")


def make_splits(records, cfg, source_assignments):
    validate_identity(records)
    rows = [dict(r, role="excluded" if r["exclusion_reason"] else "", calibration_rank=0)
            for r in records]
    expected_source = {r["trial_id"] for r in rows if r["session"] in cfg["source_sessions"]}
    if set(source_assignments) != expected_source:
        raise ValueError("Source manifest coverage mismatch")
    for r in rows:
        if r["session"] in cfg["source_sessions"] and not r["exclusion_reason"]:
            if source_assignments[r["trial_id"]] not in ["source_train", "source_val", "source_test"]:
                raise ValueError("Invalid source role")
            r["role"] = source_assignments[r["trial_id"]]
    rng = np.random.default_rng(cfg["split_seed"])
    reserve = max(cfg["calibration_budgets_per_class"])
    for session in cfg["target_sessions"]:
        for label in cfg["labels"]:
            pool = sorted([r for r in rows if r["session"] == session and
                           r["label"] == label and not r["exclusion_reason"]],
                          key=lambda r: r["trial_id"])
            if len(pool) <= reserve:
                raise ValueError(f"Insufficient trials for calibration and scoring: {session}/{label}")
            for rank, i in enumerate(rng.permutation(len(pool)), start=1):
                pool[i]["role"] = "target_calibration" if rank <= reserve else "target_test"
                pool[i]["calibration_rank"] = rank if rank <= reserve else 0
    if any(not r["role"] for r in rows):
        raise ValueError("Unassigned trial")
    for role in ["source_train", "source_val", "source_test"]:
        if {r["label"] for r in rows if r["role"] == role} != set(cfg["labels"]):
            raise ValueError(f"Missing class in {role}")
    return rows


def source_manifest_assignments(root, cfg):
    result = {}
    for part in ["train", "val", "test"]:
        df = pd.read_csv(root / cfg["source_split_root"] / part / "labels.csv")
        for r in df.to_dict("records"):
            p = Path(r["file"])
            if len(p.parts) != 3 or p.parts[0] != "unshifted_fix":
                raise ValueError("Unexpected source path")
            trial_id = "/".join([cfg["participant"], *p.parts[1:]])
            if trial_id in result or p.stem.rsplit("_", 1)[0] != r["label"]:
                raise ValueError("Duplicate or conflicting source manifest identity")
            result[trial_id] = "source_" + part
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default="research/retrospective_rf_coral_protocol_v1.json")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    protocol = ROOT / args.protocol
    cfg = json.loads(protocol.read_text())
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=False)
    protected = [p for folder in ["src", "scripts", "data", "graphs", "outputs"]
                 for p in (ROOT/folder).rglob("*") if p.is_file() and "__pycache__" not in p.parts]
    protected += [protocol]
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in sorted(protected)}
    save_json(out/"input_sha256.json", hashes)
    save_json(out/"protocol.json", cfg)
    save_json(out/"environment.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(), "root": str(ROOT),
        "python": sys.version, "platform": platform.platform(), "git_revision": None,
        "packages": {p: importlib.metadata.version(p) for p in ["numpy", "pandas", "scipy", "scikit-learn"]},
    })
    assignments = source_manifest_assignments(ROOT, cfg)
    records, feature_arrays = [], {}
    sessions = cfg["source_sessions"] + cfg["target_sessions"]
    for session in sessions:
        paths = sorted((ROOT/"data/raw"/cfg["participant"]/session).glob("*.csv"))
        if not paths:
            raise ValueError(f"Missing session {session}")
        print(f"Checking {session}: {len(paths)} raw trials", flush=True)
        for p in paths:
            label = p.stem.rsplit("_", 1)[0]
            if label not in cfg["labels"]:
                raise ValueError(f"Unknown label: {p}")
            frame = pd.read_csv(p)
            signal = frame[["t_us", *cfg["raw_channels"]]].to_numpy(dtype=np.float64)
            reasons = quality_reasons(signal, cfg["quality"])
            trial_id = f"{cfg['participant']}/{session}/{p.name}"
            key = hashlib.sha256(trial_id.encode()).hexdigest()[:24]
            if not reasons:
                try:
                    feature_arrays[key] = checked_features(signal, cfg)
                except (ValueError, IndexError, FloatingPointError, RuntimeWarning) as exc:
                    reasons.append(f"feature_failure:{type(exc).__name__}:{exc}")
            records.append({
                "trial_id": trial_id, "raw_path": str(p.relative_to(ROOT)),
                "participant": cfg["participant"], "session": session, "label": label,
                "file_sha256": hashes[str(p.relative_to(ROOT))],
                "signal_sha256": hashlib.sha256(signal.tobytes()).hexdigest(),
                "channels_sha256": hashlib.sha256(signal[:, 1:].copy().tobytes()).hexdigest(),
                "samples": len(signal), "recorded_seconds": len(signal)/cfg["quality"]["fs_hz"],
                "median_interval_us": float(np.median(np.diff(signal[:, 0]))),
                "max_interval_us": float(np.max(np.diff(signal[:, 0]))),
                "exclusion_reason": ";".join(reasons),
                "feature_key": key if not reasons else "",
                "windows": len(feature_arrays[key]) if not reasons else 0,
            })
        print(f"Completed quality/features for {session}", flush=True)
    rows = make_splits(records, cfg, assignments)
    manifest = pd.DataFrame(rows)
    manifest.to_csv(out/"trial_manifest.csv", index=False)
    manifest[manifest.role == "excluded"].to_csv(out/"exclusions.csv", index=False)
    for budget in cfg["calibration_budgets_per_class"]:
        selected = manifest[(manifest.role == "target_calibration") & (manifest.calibration_rank <= budget)]
        selected.to_csv(out/f"calibration_{budget}_per_class.csv", index=False)
    manifest[manifest.role == "target_test"].to_csv(out/"target_scoring.csv", index=False)
    np.savez_compressed(out/"features.npz", **feature_arrays)
    summary = {
        "scope": "Input preparation only; no model fit, predictions or performance metrics",
        "raw_trials": len(rows), "eligible_trials": int((manifest.role != "excluded").sum()),
        "excluded_trials": int((manifest.role == "excluded").sum()),
        "feature_count": 2*len(cfg["features_per_channel"]),
        "windows": int(manifest.windows.sum()),
        "counts": manifest.groupby(["session", "role", "label"]).size().rename("trials").reset_index().to_dict("records"),
        "all_protected_files_unchanged": all(digest(ROOT/p) == h for p, h in hashes.items()),
        "protected_files": len(hashes),
    }
    save_json(out/"summary.json", summary)
    if not summary["all_protected_files_unchanged"]:
        raise RuntimeError("Protected files changed during preparation")
    save_json(out/"artifact_sha256.json", {p.name: digest(p) for p in sorted(out.iterdir()) if p.is_file()})
    print(json.dumps({k: v for k, v in summary.items() if k != "counts"}), flush=True)


if __name__ == "__main__":
    main()
