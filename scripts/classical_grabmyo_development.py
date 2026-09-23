"""Amplitude-inclusive LDA/RF comparisons on all eight development participants."""
import argparse
import csv
import gzip
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from src.grabmyo_corpus import WindowCorpus, hash_stream
from scripts.train_shared_emg import roles


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    dev = WindowCorpus(args.cache, "development")
    expected = json.loads((args.cache / "summary.json").read_text())["groups"]["development"]
    if hash_stream(args.cache / "development_features.npy") != expected["features_sha256"]:
        raise ValueError("Changed features")
    participants = sorted({int(r["participant"]) for r in dev.rows})
    config = {"created_utc": datetime.now(timezone.utc).isoformat(), "scope": "eight-person public development comparison",
              "cache": str(args.cache.resolve()), "cache_summary_sha256": hash_stream(args.cache / "summary.json"),
              "participants": participants, "lda": {"solver": "lsqr", "shrinkage": "auto"},
              "rf": {"n_estimators": 300, "min_samples_leaf": 2, "max_features": "sqrt", "random_state": 42, "n_jobs": 4},
              "features": "MAV/RMS/mean waveform length for each of16 forearm channels;48 features",
              "methods": ["source_only", "source_plus_calibration", "calibration_only"],
              "target_sessions": [2, 3], "budgets": [1, 2, 3],
              "code_hashes": {f: hash_stream(f) for f in ("scripts/classical_grabmyo_development.py", "src/grabmyo_corpus.py", "scripts/train_shared_emg.py")}}
    (args.out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    models = args.out / "models"; models.mkdir()
    metrics, access = [], []
    start = time.perf_counter()
    def arrays(ids):
        return np.asarray(dev.features[ids]).reshape(-1, 48), np.repeat(dev.labels[ids], 35)
    with gzip.open(args.out / "predictions.csv.gz", "wt") as file:
        writer = csv.DictWriter(file, fieldnames=["name", "participant", "session", "record", "window_start_sample", "truth", "prediction"])
        writer.writeheader()
        for p in participants:
            source = roles(dev.rows, p, 1, "enrollment")
            for session in (2, 3):
                score = roles(dev.rows, p, session, "scoring")
                score_x, score_y = arrays(score)
                for family in ("LDA", "RF"):
                    for method, budget in [("source_only", 0)] + [(m, b) for m in ("source_plus_calibration", "calibration_only") for b in (1, 2, 3)]:
                        cal = roles(dev.rows, p, session, "calibration", budget) if budget else []
                        fit_ids = (source if method == "source_only" else cal if method == "calibration_only" else source + cal)
                        if set(fit_ids) & set(score):
                            raise ValueError("Fit/scoring trial overlap")
                        x, y = arrays(fit_ids)
                        estimator = (LinearDiscriminantAnalysis(**config["lda"]) if family == "LDA" else RandomForestClassifier(**config["rf"]))
                        model = make_pipeline(StandardScaler(), estimator)
                        tick = time.perf_counter(); model.fit(x, y); fit_seconds = time.perf_counter() - tick
                        pred = model.predict(score_x)
                        name = f"p{p}_s{session}_{family}_{method}_b{budget}"
                        joblib.dump(model, models / (name + ".joblib"), compress=3)
                        metrics.append({"name": name, "participant": p, "session": session, "family": family,
                                        "method": method, "budget": budget, "calibration_seconds": 85 * budget,
                                        "macro_f1": float(f1_score(score_y, pred, average="macro", labels=list(range(17)), zero_division=0)),
                                        "balanced_accuracy": float(balanced_accuracy_score(score_y, pred)),
                                        "fit_seconds": fit_seconds, "scoring_trials": len(score), "scoring_windows": len(pred)})
                        access.append({"name": name, "fit": [dev.rows[i]["record"] for i in fit_ids], "score": [dev.rows[i]["record"] for i in score]})
                        for index, (truth, prediction) in enumerate(zip(score_y, pred)):
                            trial, w = divmod(index, 35)
                            writer.writerow({"name": name, "participant": p, "session": session, "record": dev.rows[score[trial]]["record"],
                                             "window_start_sample": 1024 + 256 * w, "truth": int(truth), "prediction": int(prediction)})
                    print(json.dumps({"participant": p, "session": session, "family": family, "models": len(metrics)}), flush=True)
                    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
                    (args.out / "data_access.json").write_text(json.dumps(access, indent=2) + "\n")
    result = {"completed_utc": datetime.now(timezone.utc).isoformat(), "models": len(metrics),
              "total_seconds": time.perf_counter() - start, "final_participants_accessed": 0}
    (args.out / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    snapshot = args.out / "code_snapshot"; snapshot.mkdir()
    for name in config["code_hashes"]:
        (snapshot / Path(name).name).write_bytes(Path(name).read_bytes())
    (args.out / "artifact_sha256.json").write_text(json.dumps({str(p.relative_to(args.out)): hash_stream(p)
        for p in args.out.rglob("*") if p.is_file()}, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
