"""Development-only LDA feasibility pilot; not a publication evaluation."""
from __future__ import annotations
import argparse
import csv
import gzip
import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.grabmyo import checksum_index, fetch, read_forearm, short_windows, amplitude_features, sha256


def fitting_rows(rows, participant, method, budget):
    pool = [r for r in rows if int(r["participant"]) == participant]
    source = [r for r in pool if r["role"] == "enrollment"]
    target = [r for r in pool if r["role"] == "calibration" and int(r["calibration_rank"]) <= budget]
    if method == "source_only":
        return source
    if method == "source_plus_calibration":
        return source + target
    if method == "calibration_only":
        return target
    raise ValueError(method)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("data/public/grabmyo/1.1.0"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    config = json.loads((args.preparation / "config.json").read_text())
    frozen = json.loads((args.preparation / "summary.json").read_text())
    if sha256(args.preparation / "trial_manifest.csv") != frozen["manifest_sha256"]:
        raise ValueError("Frozen trial manifest was changed")
    if sha256(args.preparation / "config.json") != frozen["config_sha256"]:
        raise ValueError("Frozen allocation config was changed")
    participants = config["participant_groups"]["development"][:2]
    all_rows = list(csv.DictReader((args.preparation / "trial_manifest.csv").open()))
    rows = [r for r in all_rows if int(r["participant"]) in participants and int(r["session"]) in (1, 2)]
    if any(r["group"] != "development" for r in rows):
        raise ValueError("Pilot may only access development participants")
    run_config = {"created_utc": datetime.now(timezone.utc).isoformat(), "scope": "development-only feasibility pilot",
                  "participants": participants, "source_session": 1, "target_session": 2,
                  "model": {"name": "StandardScaler + LDA", "solver": "lsqr", "shrinkage": "auto"},
                  "methods": ["source_only", "source_plus_calibration", "calibration_only"],
                  "budgets": [1, 2, 3], "features": ["MAV", "RMS", "mean waveform length per channel"],
                  "preparation": str(args.preparation.resolve()), "preparation_config": config,
                  "preparation_hashes": frozen, "files": {f: sha256(Path(f)) for f in
                      ("src/grabmyo.py", "scripts/pilot_grabmyo.py", "tests/test_grabmyo.py")}}
    (args.out / "config.json").write_text(json.dumps(run_config, indent=2) + "\n")
    snapshot = args.out / "code_snapshot"
    snapshot.mkdir()
    for name in run_config["files"]:
        (snapshot / Path(name).name).write_bytes(Path(name).read_bytes())
    checksums = checksum_index(args.data)
    start = time.perf_counter()
    downloaded = []
    tasks = [r["record"] + suffix for r in rows for suffix in (".hea", ".dat")]
    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            for item in pool.map(lambda name: fetch(args.data, name, checksums), tasks):
                downloaded.append(item)
                if len(downloaded) % 100 == 0:
                    print(f"Verified files {len(downloaded)}/{len(tasks)}", flush=True)
    except Exception as exc:
        (args.out / "failure.json").write_text(json.dumps({"type": type(exc).__name__, "error": str(exc)}))
        raise
    finally:
        (args.out / "retrieval.json").write_text(json.dumps(downloaded, indent=2) + "\n")
    retrieval_seconds = time.perf_counter() - start
    features, qc = {}, []
    for r in rows:
        x, info = read_forearm(args.data, r["record"], checksums)
        windows, starts = short_windows(x)
        features[r["record"]] = amplitude_features(windows)
        qc.append({"record": r["record"], **info, "windows": len(starts)})
    (args.out / "signal_validation.json").write_text(json.dumps(qc, indent=2) + "\n")
    np.savez_compressed(args.out / "features.npz", **features)
    predictions, metrics, access = [], [], []
    modeldir = args.out / "models"
    modeldir.mkdir()
    settings = [("source_only", 0)] + [(m, b) for m in ("source_plus_calibration", "calibration_only") for b in (1, 2, 3)]
    for p in participants:
        score = [r for r in rows if int(r["participant"]) == p and r["role"] == "scoring"]
        test_x = np.concatenate([features[r["record"]] for r in score])
        test_y = np.concatenate([np.full(len(features[r["record"]]), int(r["class_index"])) for r in score])
        for method, budget in settings:
            fit = fitting_rows(rows, p, method, budget)
            if {r["record"] for r in fit} & {r["record"] for r in score}:
                raise ValueError("Training/scoring trial overlap")
            train_x = np.concatenate([features[r["record"]] for r in fit])
            train_y = np.concatenate([np.full(len(features[r["record"]]), int(r["class_index"])) for r in fit])
            model = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))
            tick = time.perf_counter()
            model.fit(train_x, train_y)
            elapsed = time.perf_counter() - tick
            pred = model.predict(test_x)
            name = f"p{p}_{method}_b{budget}"
            joblib.dump(model, modeldir / (name + ".joblib"))
            entry = {"participant": p, "source_session": 1, "target_session": 2, "method": method,
                     "budget": budget, "calibration_recording_seconds": 85 * budget,
                     "macro_f1": float(f1_score(test_y, pred, average="macro", labels=list(range(17)), zero_division=0)),
                     "balanced_accuracy": float(balanced_accuracy_score(test_y, pred)),
                     "scoring_trials": len(score), "scoring_windows": len(test_y), "fit_seconds": elapsed}
            metrics.append(entry)
            access.append({"name": name, "fit": [r["record"] for r in fit], "score": [r["record"] for r in score]})
            cursor = 0
            for r in score:
                for index in range(len(features[r["record"]])):
                    predictions.append({"participant": p, "method": method, "budget": budget,
                                        "record": r["record"], "window_start_sample": 1024 + 256 * index,
                                        "truth": int(test_y[cursor]), "prediction": int(pred[cursor])})
                    cursor += 1
            print(json.dumps(entry), flush=True)
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (args.out / "data_access.json").write_text(json.dumps(access, indent=2) + "\n")
    with gzip.open(args.out / "predictions.csv.gz", "wt") as f:
        writer = csv.DictWriter(f, fieldnames=list(predictions[0]))
        writer.writeheader()
        writer.writerows(predictions)
    summary = {"completed_utc": datetime.now(timezone.utc).isoformat(), "scope": "development pilot, two people, day1 to day2",
               "records": len(rows), "scoring_windows_per_person": 2380, "models": len(metrics),
               "final_participants_accessed": 0, "retrieval_seconds": retrieval_seconds,
               "total_seconds": time.perf_counter() - start, "neural_model_trained": False}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    hashes = {str(p.relative_to(args.out)): sha256(p) for p in args.out.rglob("*") if p.is_file()}
    (args.out / "artifact_sha256.json").write_text(json.dumps(hashes, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
