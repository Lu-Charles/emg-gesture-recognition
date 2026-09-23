"""Verify public neural pilot checkpoints, enrollment scaling and scoring access."""
import argparse
import csv
import gzip
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import f1_score, balanced_accuracy_score
from src.emg_cnn import CompactEMGNet
from src.grabmyo import sha256, checksum_index, read_forearm, short_windows
from scripts.pilot_emg_cnn import predict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--preparation", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads((args.run / "config.json").read_text())
    for relative, expected in json.loads((args.run / "artifact_sha256.json").read_text()).items():
        if sha256(args.run / relative) != expected:
            raise ValueError(f"Changed artifact {relative}")
    manifest = {r["record"]: r for r in csv.DictReader((args.preparation / "trial_manifest.csv").open())}
    metrics = json.loads((args.run / "metrics.json").read_text())
    access = json.loads((args.run / "data_access.json").read_text())
    with gzip.open(args.run / "predictions.csv.gz", "rt") as f:
        predictions = list(csv.DictReader(f))
    checksums = checksum_index(Path("data/public/grabmyo/1.1.0"))
    device = config["device"]
    checked = 0
    for p in config["participants"]:
        records = {}
        for record, row in manifest.items():
            if int(row["participant"]) != p or row["session"] not in ("1", "2"):
                continue
            if row["group"] != "development":
                raise ValueError("Nondevelopment input")
            x, _ = read_forearm(Path("data/public/grabmyo/1.1.0"), record, checksums)
            windows, _ = short_windows(x)
            records[record] = windows.transpose(0, 2, 1).astype(np.float32)
        source = [r for r in records if manifest[r]["session"] == "1"]
        source_x = np.concatenate([records[r] for r in source])
        saved_scale = np.load(args.run / "models" / f"p{p}_scaler.npz")
        mean = source_x.mean(axis=(0, 2), keepdims=True, dtype=np.float64)
        scale = np.maximum(source_x.std(axis=(0, 2), keepdims=True, dtype=np.float64), 1e-8)
        np.testing.assert_array_equal(mean, saved_scale["mean"])
        np.testing.assert_array_equal(scale, saved_scale["scale"])
        base = torch.load(args.run / "models" / f"p{p}_source_only_b0.pt", map_location="cpu", weights_only=True)
        fixed_score = None
        for m, a in zip(metrics, access, strict=True):
            if m["participant"] != p:
                continue
            mode, budget = m["method"], m["budget"]
            expected_cal = {r for r in records if manifest[r]["session"] == "2" and manifest[r]["role"] == "calibration"
                            and int(manifest[r]["calibration_rank"]) <= budget}
            expected_score = {r for r in records if manifest[r]["role"] == "scoring"}
            if set(a["source"]) != set(source) or set(a["calibration"]) != expected_cal or set(a["score"]) != expected_score:
                raise ValueError("Incorrect source, calibration or scoring access")
            if (set(a["source"]) | set(a["calibration"])) & set(a["score"]):
                raise ValueError("Fit/scoring overlap")
            if fixed_score is not None and a["score"] != fixed_score:
                raise ValueError("Scoring changed between budgets")
            fixed_score = a["score"]
            state = torch.load(args.run / "models" / (a["name"] + ".pt"), map_location="cpu", weights_only=True)
            if mode == "head" and any(not torch.equal(state[k], base[k]) for k in base if k.startswith("encoder.")):
                raise ValueError("Head-only update changed encoder")
            model = CompactEMGNet().to(device)
            model.load_state_dict(state)
            x = ((np.concatenate([records[r] for r in a["score"]]) - mean) / scale).astype(np.float32)
            truth = np.concatenate([np.full(35, int(manifest[r]["class_index"])) for r in a["score"]])
            pred = predict(model, x, device)
            saved = [r for r in predictions if int(r["participant"]) == p and r["method"] == mode and int(r["budget"]) == budget]
            ids = [(r, s) for r in a["score"] for s in range(1024, 9729, 256)]
            if [(r["record"], int(r["window_start_sample"])) for r in saved] != ids:
                raise ValueError("Prediction identity mismatch")
            np.testing.assert_array_equal(pred, [int(r["prediction"]) for r in saved])
            np.testing.assert_array_equal(truth, [int(r["truth"]) for r in saved])
            np.testing.assert_allclose(m["macro_f1"], f1_score(truth, pred, average="macro", labels=list(range(17)), zero_division=0))
            np.testing.assert_allclose(m["balanced_accuracy"], balanced_accuracy_score(truth, pred))
            checked += len(saved)
    if len(metrics) != 14 or checked != len(predictions):
        raise ValueError("Wrong number of models/predictions")
    result = {"verified_utc": datetime.now(timezone.utc).isoformat(), "models": len(metrics), "prediction_rows": checked,
              "normalization_rebuilt_from_day1": True, "fixed_scoring_and_calibration_access_verified": True,
              "head_encoder_unchanged": True, "final_participants_accessed": 0,
              "verifier_sha256": sha256(Path(__file__))}
    (args.run / "independent_validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
