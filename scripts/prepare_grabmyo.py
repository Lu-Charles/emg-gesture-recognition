"""Freeze allocations and verify a small non-final public-data smoke subset."""
from __future__ import annotations
import argparse
import csv
import json
import platform
import sys
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

from src.grabmyo import (LABELS, allocations, checksum_index, fetch, read_forearm,
                         sha256, short_windows, amplitude_features)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("data/public/grabmyo/1.1.0"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    groups, rows = allocations()
    config = {"created_utc": datetime.now(timezone.utc).isoformat(), "dataset": "GRABMyo 1.1.0",
              "allocation_seed": 20260906, "allocation_algorithm": "ascending SHA256(seed:context:value)",
              "participant_groups": groups, "labels": LABELS,
              "status": "allocations frozen before public-data performance; pilot preprocessing",
              "physical_units": "mV", "channels": list(range(16)), "fs": 2048,
              "software_filter": None, "whole_trial_normalization": False,
              "window_samples": 512, "hop_samples": 256, "onset_exclusion_samples": 1024,
              "calibration_repetitions_per_class": [1, 2, 3],
              "calibration_recording_seconds": [85, 170, 255],
              "scope": "held-gesture offline study; not measured response latency",
              "python": sys.version, "platform": platform.platform(),
              "packages": {p: version(p) for p in ("numpy", "scipy", "wfdb", "scikit-learn")}}
    (args.out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    with (args.out / "trial_manifest.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    snapshot = args.out / "code_snapshot"
    snapshot.mkdir()
    for name in ("src/grabmyo.py", "scripts/prepare_grabmyo.py", "tests/test_grabmyo.py"):
        (snapshot / Path(name).name).write_bytes(Path(name).read_bytes())
    checksums = checksum_index(args.data)
    chosen = {groups["train"][0], groups["development"][0], groups["development"][1]}
    selected = [r for r in rows if r["participant"] in chosen and
                r["gesture"] in (1, 10, 17) and r["trial"] == 1]
    downloads, validation = [], []
    try:
        for row in selected:
            for suffix in (".hea", ".dat"):
                downloads.append(fetch(args.data, row["record"] + suffix, checksums))
            x, info = read_forearm(args.data, row["record"], checksums)
            w, starts = short_windows(x)
            f = amplitude_features(w)
            validation.append({**row, **info, "windows": len(w), "features_shape": list(f.shape),
                               "first_window_start": int(starts[0]), "last_window_end": int(starts[-1] + 512)})
            print(f"Verified {len(validation)}/{len(selected)} records", flush=True)
    except Exception as exc:
        (args.out / "failure.json").write_text(json.dumps({"type": type(exc).__name__, "error": str(exc)}, indent=2))
        raise
    finally:
        (args.out / "retrieval.json").write_text(json.dumps(downloads, indent=2) + "\n")
        (args.out / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    summary = {"completed_utc": datetime.now(timezone.utc).isoformat(), "manifest_trials": len(rows),
               "verified_records": len(validation), "final_participants_loaded": 0,
               "model_trained": False, "config_sha256": sha256(args.out / "config.json"),
               "manifest_sha256": sha256(args.out / "trial_manifest.csv"),
               "official_checksum_list_sha256": sha256(args.data / "SHA256SUMS.txt")}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
