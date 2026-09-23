"""Read back pilot models/predictions and independently verify fit access and metrics."""
from __future__ import annotations
import argparse
import csv
import gzip
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score
from src.grabmyo import checksum_index, read_forearm, short_windows, amplitude_features, sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("data/public/grabmyo/1.1.0"))
    args = parser.parse_args()
    config = json.loads((args.run / "config.json").read_text())
    artifacts = json.loads((args.run / "artifact_sha256.json").read_text())
    for relative, expected in artifacts.items():
        if sha256(args.run / relative) != expected:
            raise ValueError(f"Changed artifact: {relative}")
    preparation = Path(config["preparation"])
    if sha256(preparation / "trial_manifest.csv") != config["preparation_hashes"]["manifest_sha256"]:
        raise ValueError("Changed original manifest")
    manifest = {r["record"]: r for r in csv.DictReader((preparation / "trial_manifest.csv").open())}
    features = np.load(args.run / "features.npz")
    checksums = checksum_index(args.data)
    max_error = 0.
    for record in features.files:
        row = manifest[record]
        if row["group"] != "development" or int(row["participant"]) not in config["participants"]:
            raise ValueError("Non-development record in pilot")
        x, _ = read_forearm(args.data, record, checksums)
        windows, _ = short_windows(x)
        error = float(np.max(np.abs(amplitude_features(windows) - features[record])))
        max_error = max(max_error, error)
    if max_error != 0:
        raise ValueError("Raw feature rebuild mismatch")
    with gzip.open(args.run / "predictions.csv.gz", "rt") as f:
        predictions = list(csv.DictReader(f))
    metrics = json.loads((args.run / "metrics.json").read_text())
    access = json.loads((args.run / "data_access.json").read_text())
    fixed_scoring = {}
    total_predictions = 0
    for m, a in zip(metrics, access, strict=True):
        p, method, budget = m["participant"], m["method"], m["budget"]
        name = f"p{p}_{method}_b{budget}"
        if a["name"] != name or set(a["fit"]) & set(a["score"]):
            raise ValueError("Wrong model or overlapping fit/scoring access")
        expected_source = {r for r, row in manifest.items() if int(row["participant"]) == p and row["session"] == "1"}
        expected_cal = {r for r, row in manifest.items() if int(row["participant"]) == p and row["session"] == "2"
                        and row["role"] == "calibration" and int(row["calibration_rank"]) <= budget}
        expected_fit = (expected_source if method == "source_only" else expected_cal if method == "calibration_only"
                        else expected_source | expected_cal)
        expected_score = {r for r, row in manifest.items() if int(row["participant"]) == p and row["session"] == "2" and row["role"] == "scoring"}
        if set(a["fit"]) != expected_fit or set(a["score"]) != expected_score:
            raise ValueError("Incorrect actual fit/scoring identities")
        if p in fixed_scoring and fixed_scoring[p] != a["score"]:
            raise ValueError("Scoring order/set changed between budgets")
        fixed_scoring[p] = a["score"]
        model = joblib.load(args.run / "models" / (name + ".joblib"))
        fit_x = np.concatenate([features[r] for r in a["fit"]])
        scaler = model.steps[0][1]
        np.testing.assert_allclose(scaler.mean_, fit_x.mean(axis=0), rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(scaler.var_, fit_x.var(axis=0), rtol=1e-12, atol=1e-12)
        if scaler.n_samples_seen_ != len(fit_x):
            raise ValueError("Scaler fit sample count mismatch")
        predicted = model.predict(np.concatenate([features[r] for r in a["score"]]))
        expected_y = np.concatenate([np.full(35, int(manifest[r]["class_index"])) for r in a["score"]])
        saved = [r for r in predictions if int(r["participant"]) == p and r["method"] == method and int(r["budget"]) == budget]
        expected_ids = [(record, start) for record in a["score"] for start in range(1024, 10240 - 512 + 1, 256)]
        if [(r["record"], int(r["window_start_sample"])) for r in saved] != expected_ids:
            raise ValueError("Prediction/window identity mismatch")
        np.testing.assert_array_equal(predicted, [int(r["prediction"]) for r in saved])
        np.testing.assert_array_equal(expected_y, [int(r["truth"]) for r in saved])
        np.testing.assert_allclose(m["macro_f1"], f1_score(expected_y, predicted, average="macro", labels=list(range(17)), zero_division=0))
        np.testing.assert_allclose(m["balanced_accuracy"], balanced_accuracy_score(expected_y, predicted))
        total_predictions += len(saved)
    if total_predictions != len(predictions) or len(metrics) != 14:
        raise ValueError("Missing or extra model/prediction entries")
    result = {"verified_utc": datetime.now(timezone.utc).isoformat(), "verified_models": len(metrics),
              "verified_prediction_rows": total_predictions, "raw_records_rebuilt": len(features.files),
              "max_raw_feature_error": max_error, "final_participants_accessed": 0,
              "fit_access_and_scaler_verified": True, "fixed_scoring_verified": True,
              "verifier_sha256": sha256(Path(__file__))}
    (args.run / "independent_validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
