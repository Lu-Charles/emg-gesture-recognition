"""Verify raw-to-cache spot checks, scaling, access, checkpoints and predictions."""
import argparse
import csv
import gzip
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score
from src.emg_cnn import CompactEMGNet
from src.grabmyo import checksum_index, read_forearm, short_windows, amplitude_features
from src.grabmyo_corpus import WindowCorpus, hash_stream, training_moments
from scripts.pilot_emg_cnn import predict
from scripts.train_shared_emg import roles, validate_separation


def neural_name(row, mode=None, budget=None):
    suffix = f"_e{row['enrollment_epochs']}" if "enrollment_epochs" in row else ""
    return (f"p{row['participant']}_s{row['session']}_{row['initialization']}{suffix}_"
            f"{row['mode'] if mode is None else mode}_b{row['budget'] if budget is None else budget}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--kind", choices=["neural", "classical"], required=True)
    args = parser.parse_args()
    config = json.loads((args.run / "config.json").read_text())
    for relative, expected in json.loads((args.run / "artifact_sha256.json").read_text()).items():
        if hash_stream(args.run / relative) != expected:
            raise ValueError(f"Changed run artifact: {relative}")
    for original, expected in config["code_hashes"].items():
        if hash_stream(args.run / "code_snapshot" / Path(original).name) != expected:
            raise ValueError(f"Source snapshot differs from initial configuration: {original}")
    cache = Path(config["cache"])
    if hash_stream(cache / "summary.json") != config["cache_summary_sha256"]:
        raise ValueError("Changed corpus summary")
    train, dev = WindowCorpus(cache, "train"), WindowCorpus(cache, "development")
    validate_separation(train.rows, dev.rows)
    cache_summary = json.loads((cache / "summary.json").read_text())
    for group in ("train", "development"):
        for kind in ("signals", "features", "manifest"):
            suffix = "csv" if kind == "manifest" else "npy"
            if hash_stream(cache / f"{group}_{kind}.{suffix}") != cache_summary["groups"][group][f"{kind}_sha256"]:
                raise ValueError("Changed cache")
    raw = Path("data/public/grabmyo/1.1.0")
    checksums = checksum_index(raw)
    # Selection fixed by identity, covers every permitted participant and session.
    spot_checks = 0
    for corpus in (train, dev):
        for index, row in enumerate(corpus.rows):
            if row["gesture"] != "1" or row["trial"] != "1":
                continue
            x, _ = read_forearm(raw, row["record"], checksums)
            np.testing.assert_array_equal(corpus.signals[index], x.T.astype(np.float32))
            windows, _ = short_windows(x)
            np.testing.assert_array_equal(corpus.features[index], amplitude_features(windows))
            spot_checks += 1
    if args.kind == "neural":
        mean, scale, count = training_moments(train)
        saved = np.load(args.run / "training_scaler.npz")
        np.testing.assert_array_equal(mean, saved["mean"])
        np.testing.assert_array_equal(scale, saved["scale"])
        if count != saved["count"]:
            raise ValueError("Shared scaler sample count mismatch")
    metrics = json.loads((args.run / "metrics.json").read_text())
    access = {r["name"]: r for r in json.loads((args.run / "data_access.json").read_text())}
    records = defaultdict(list)
    with gzip.open(args.run / "predictions.csv.gz", "rt") as f:
        for row in csv.DictReader(f):
            name = neural_name(row) if args.kind == "neural" else row["name"]
            records[name].append(row)
    score_sets = {}
    checked = 0
    for index, metric in enumerate(metrics):
        p, session, budget, name = metric["participant"], metric["session"], metric["budget"], metric["name"]
        source = roles(dev.rows, p, 1, "enrollment")
        cal = roles(dev.rows, p, session, "calibration", budget) if budget else []
        score = roles(dev.rows, p, session, "scoring")
        truth = np.repeat(dev.labels[score], 35)
        score_records = [dev.rows[i]["record"] for i in score]
        actual = access[name]
        if actual["score"] != score_records:
            raise ValueError("Wrong scoring order/set")
        score_sets[f"p{p}_s{session}"] = score_records
        if args.kind == "neural":
            if actual["source"] != [dev.rows[i]["record"] for i in source] or actual["calibration"] != [dev.rows[i]["record"] for i in cal]:
                raise ValueError("Wrong neural source/calibration access")
            state = torch.load(args.run / "models" / (name + ".pt"), map_location="cpu", weights_only=True)
            if metric["mode"] == "head":
                base_name = neural_name(metric, mode="none", budget=0) + ".pt"
                base = torch.load(args.run / "models" / base_name, map_location="cpu", weights_only=True)
                if any(not torch.equal(state[k], base[k]) for k in state if k.startswith("encoder.")):
                    raise ValueError("Head-only update changed encoder")
            model = CompactEMGNet().to(config["device"]); model.load_state_dict(state)
            x, _ = dev.batch(dev.window_ids(score), mean, scale)
            pred = predict(model, x, config["device"])
        else:
            method = metric["method"]
            fit_ids = source if method == "source_only" else cal if method == "calibration_only" else source + cal
            if actual["fit"] != [dev.rows[i]["record"] for i in fit_ids]:
                raise ValueError("Wrong classical fit access")
            model = joblib.load(args.run / "models" / (name + ".joblib"))
            fit_x = np.asarray(dev.features[fit_ids]).reshape(-1, 48)
            np.testing.assert_allclose(model.steps[0][1].mean_, fit_x.mean(0), rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(model.steps[0][1].var_, fit_x.var(0), rtol=1e-12, atol=1e-12)
            if model.steps[0][1].n_samples_seen_ != len(fit_x):
                raise ValueError("Wrong scaler sample count")
            pred = model.predict(np.asarray(dev.features[score]).reshape(-1, 48))
        saved = records[name]
        expected = [(r, 1024 + 256 * w) for r in score_records for w in range(35)]
        if [(r["record"], int(r["window_start_sample"])) for r in saved] != expected:
            raise ValueError("Prediction identity mismatch")
        np.testing.assert_array_equal(pred, [int(r["prediction"]) for r in saved])
        np.testing.assert_array_equal(truth, [int(r["truth"]) for r in saved])
        np.testing.assert_allclose(metric["macro_f1"], f1_score(truth, pred, average="macro", labels=list(range(17)), zero_division=0))
        np.testing.assert_allclose(metric["balanced_accuracy"], balanced_accuracy_score(truth, pred))
        checked += len(pred)
        if (index + 1) % 28 == 0:
            print(json.dumps({"verified_models": index + 1, "kind": args.kind}), flush=True)
    if args.kind == "neural":
        expected_models = (len(config["development_participants"]) * len(config["target_sessions"])
                           * len(config["initializations"]) * (1 + 2 * len(config["calibration_budgets"]))
                           * len(config.get("enrollment_epoch_checkpoints", [config["enrollment_epochs"]])))
        expected_names = {neural_name({"participant": p, "session": s, "initialization": init,
                            "mode": mode, "budget": budget,
                            **({"enrollment_epochs": epoch} if "enrollment_epoch_checkpoints" in config else {})})
                          for p in config["development_participants"] for s in config["target_sessions"]
                          for init in config["initializations"]
                          for epoch in config.get("enrollment_epoch_checkpoints", [config["enrollment_epochs"]])
                          for mode, budget in [("none", 0)] + [(m, b) for m in ("head", "full") for b in config["calibration_budgets"]]}
        if set(access) != expected_names or {m["name"] for m in metrics} != expected_names:
            raise ValueError("Incomplete or unexpected experiment variants")
    else:
        expected_models = 224
    if len(metrics) != expected_models or len(records) != len(metrics) or len(access) != len(metrics):
        raise ValueError("Incorrect model/prediction counts")
    reproduced = 0
    if "enrollment_epoch_checkpoints" in config:
        prior = Path(config["prior_run"])
        if hash_stream(prior / "artifact_sha256.json") != config["prior_artifact_manifest_sha256"]:
            raise ValueError("Changed original run manifest")
        if score_sets != json.loads((prior / "independent_validation.json").read_text())["scoring_sets"]:
            raise ValueError("Convergence and original scoring sets differ")
        if hash_stream(args.run / "training_scaler.npz") != hash_stream(prior / "training_scaler.npz"):
            raise ValueError("Convergence scaler differs from original")
        old_metrics = {r["name"]: r for r in json.loads((prior / "metrics.json").read_text())}
        old_history = {r["participant"]: r["epoch_losses"] for r in json.loads((prior / "adaptation_history.json").read_text())
                       if r["stage"] == "enrollment" and r["initialization"] == "random"}
        for p in config["development_participants"]:
            source = roles(dev.rows, p, 1, "enrollment")
            previous_losses = []
            for epoch in config["enrollment_epoch_checkpoints"]:
                checkpoint = torch.load(args.run / "enrollment" / f"p{p}_e{epoch}.pt", map_location="cpu", weights_only=True)
                expected_steps = epoch * ((len(source) * 35 + 127) // 128)
                if any(int(s["step"]) != expected_steps for s in checkpoint["optimizer"]["state"].values()):
                    raise ValueError("Enrollment optimizer steps were reset or omitted")
                if len(checkpoint["epoch_losses"]) != epoch or checkpoint["epoch_losses"][:len(previous_losses)] != previous_losses:
                    raise ValueError("Enrollment history is not one continuous trajectory")
                if epoch == 20 and checkpoint["epoch_losses"] != old_history[p]:
                    raise ValueError("Epoch20 training history differs from original")
                previous_losses = checkpoint["epoch_losses"]
        for metric in metrics:
            if metric["enrollment_epochs"] != 20:
                continue
            old_name = neural_name({k: v for k, v in metric.items() if k != "enrollment_epochs"})
            state = torch.load(args.run / "models" / (metric["name"] + ".pt"), map_location="cpu", weights_only=True)
            old = torch.load(prior / "models" / (old_name + ".pt"), map_location="cpu", weights_only=True)
            if any(not torch.equal(state[k], old[k]) for k in state):
                raise ValueError(f"Epoch20 weights differ from original: {old_name}")
            if any(metric[key] != old_metrics[old_name][key] for key in ("macro_f1", "balanced_accuracy")):
                raise ValueError("Epoch20 metrics differ from original")
            reproduced += 1
    result = {"verified_utc": datetime.now(timezone.utc).isoformat(), "kind": args.kind, "models": len(metrics),
              "prediction_rows": checked, "raw_cache_spot_checks": spot_checks,
              "scaler_rebuilt": True, "fit_access_verified": True, "scoring_sets": score_sets,
              "final_participants_accessed": 0, "epoch20_models_exactly_reproduced": reproduced,
              "verifier_sha256": hash_stream(Path(__file__))}
    (args.run / "independent_validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "scoring_sets"}))


if __name__ == "__main__":
    main()
