"""Day1-trained CNN and day2 updates on the same two development participants.

This baseline is trained separately for each participant. It is not yet the
planned cross-participant pretrained encoder, and is not a final evaluation.
"""
import argparse
import copy
import csv
import gzip
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score
from src.emg_cnn import ChannelStandardizer, CompactEMGNet
from src.grabmyo import checksum_index, read_forearm, short_windows, sha256


def sync(device):
    if device == "mps":
        torch.mps.synchronize()


def fit(model, x, y, epochs, lr, seed, device):
    xt = torch.from_numpy(x).to(device)
    yt = torch.tensor(y, dtype=torch.long, device=device)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    rng = np.random.default_rng(seed)
    losses = []
    model.train()
    sync(device)
    start = time.perf_counter()
    for _ in range(epochs):
        order = rng.permutation(len(x))
        epoch_loss = 0.
        for j in range(0, len(order), 128):
            ids = torch.tensor(order[j:j + 128], dtype=torch.long, device=device)
            optimizer.zero_grad()
            loss = torch.nn.functional.cross_entropy(model(xt[ids]), yt[ids])
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach().cpu()) * len(ids)
        losses.append(epoch_loss / len(x))
    sync(device)
    if not np.isfinite(losses).all():
        raise ValueError("Nonfinite training loss")
    return {"epoch_losses": losses, "fit_seconds": time.perf_counter() - start}


def predict(model, x, device):
    model.eval()
    output = []
    with torch.no_grad():
        for j in range(0, len(x), 128):
            scores = model(torch.from_numpy(x[j:j + 128]).to(device))
            if not torch.isfinite(scores).all():
                raise ValueError("Nonfinite prediction logits")
            output.extend(scores.argmax(1).cpu().numpy())
    return np.asarray(output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--classical", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    prior = json.loads((args.classical / "config.json").read_text())
    if not (args.classical / "independent_validation.json").exists():
        raise ValueError("Finish classical independent verification first")
    preparation = Path(prior["preparation"])
    if sha256(preparation / "trial_manifest.csv") != prior["preparation_hashes"]["manifest_sha256"]:
        raise ValueError("Modified allocation manifest")
    participants = prior["participants"]
    rows = [r for r in csv.DictReader((preparation / "trial_manifest.csv").open())
            if int(r["participant"]) in participants and int(r["session"]) in (1, 2)]
    if any(r["group"] != "development" for r in rows):
        raise ValueError("Development participants only")
    torch.set_num_threads(4)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    config = {"created_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "two-development-person day1-trained baseline; no shared pretraining yet",
              "participants": participants, "seed": 42, "enrollment_epochs": 20,
              "enrollment_lr": .001, "update_epochs": 10, "update_lr": .0001,
              "batch_size": 128, "optimizer": "Adam", "loss": "cross_entropy",
              "methods": ["source_only", "head", "full"], "budgets": [1, 2, 3],
              "scaler_fit": "day1 enrollment only, fixed for all updates", "device": device,
              "torch": torch.__version__, "platform": platform.platform(),
              "classical_config_sha256": sha256(args.classical / "config.json"),
              "source_hashes": {f: sha256(Path(f)) for f in ("src/grabmyo.py", "src/emg_cnn.py", "scripts/pilot_emg_cnn.py")}}
    (args.out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    root = Path("data/public/grabmyo/1.1.0")
    checksums = checksum_index(root)
    all_metrics, all_predictions, all_access, training = [], [], [], []
    models = args.out / "models"
    models.mkdir()
    start = time.perf_counter()
    for p in participants:
        selected = [r for r in rows if int(r["participant"]) == p]
        windows = {}
        for r in selected:
            x, _ = read_forearm(root, r["record"], checksums)
            w, _ = short_windows(x)
            windows[r["record"]] = w.transpose(0, 2, 1).astype(np.float32)
        def arrays(subset):
            return (np.concatenate([windows[r["record"]] for r in subset]),
                    np.concatenate([np.full(35, int(r["class_index"])) for r in subset]))
        source = [r for r in selected if r["role"] == "enrollment"]
        score = [r for r in selected if r["role"] == "scoring"]
        source_x, source_y = arrays(source)
        score_x, score_y = arrays(score)
        scaler = ChannelStandardizer().fit(source_x)
        source_x, score_x = scaler.transform(source_x), scaler.transform(score_x)
        np.savez(models / f"p{p}_scaler.npz", mean=scaler.mean, scale=scaler.scale)
        torch.manual_seed(42)
        base = CompactEMGNet().to(device)
        training.append({"participant": p, "stage": "enrollment", **fit(base, source_x, source_y, 20, .001, 42, device)})
        for mode, budget in [("source_only", 0)] + [(m, b) for m in ("head", "full") for b in (1, 2, 3)]:
            model = copy.deepcopy(base)
            calibration = []
            if budget:
                calibration = [r for r in selected if r["role"] == "calibration" and int(r["calibration_rank"]) <= budget]
                model.set_update_mode(mode)
                cal_x, cal_y = arrays(calibration)
                training.append({"participant": p, "stage": mode, "budget": budget,
                                 **fit(model, scaler.transform(cal_x), cal_y, 10, .0001, 42, device)})
            actual_fit = {r["record"] for r in source + calibration}
            if actual_fit & {r["record"] for r in score}:
                raise ValueError("Fit/scoring trial overlap")
            if mode == "head":
                for a, b in zip(model.encoder.parameters(), base.encoder.parameters(), strict=True):
                    if not torch.equal(a, b):
                        raise ValueError("Encoder changed in head-only update")
            pred = predict(model, score_x, device)
            name = f"p{p}_{mode}_b{budget}"
            torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, models / (name + ".pt"))
            all_access.append({"name": name, "source": [r["record"] for r in source],
                               "calibration": [r["record"] for r in calibration], "score": [r["record"] for r in score]})
            result = {"participant": p, "method": mode, "budget": budget, "calibration_recording_seconds": 85 * budget,
                      "macro_f1": float(f1_score(score_y, pred, average="macro", labels=list(range(17)), zero_division=0)),
                      "balanced_accuracy": float(balanced_accuracy_score(score_y, pred)), "scoring_trials": len(score),
                      "scoring_windows": len(pred), "parameters": sum(v.numel() for v in model.parameters())}
            all_metrics.append(result)
            cursor = 0
            for r in score:
                for index in range(35):
                    all_predictions.append({"participant": p, "method": mode, "budget": budget, "record": r["record"],
                                            "window_start_sample": 1024 + 256 * index, "truth": int(score_y[cursor]),
                                            "prediction": int(pred[cursor])})
                    cursor += 1
            print(json.dumps(result), flush=True)
    for name, value in (("metrics", all_metrics), ("data_access", all_access), ("training", training)):
        (args.out / (name + ".json")).write_text(json.dumps(value, indent=2) + "\n")
    with gzip.open(args.out / "predictions.csv.gz", "wt") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_predictions[0]))
        writer.writeheader()
        writer.writerows(all_predictions)
    snapshot = args.out / "code_snapshot"
    snapshot.mkdir()
    for name in config["source_hashes"]:
        (snapshot / Path(name).name).write_bytes(Path(name).read_bytes())
    summary = {"completed_utc": datetime.now(timezone.utc).isoformat(), "models": len(all_metrics),
               "scope": config["scope"], "total_seconds": time.perf_counter() - start,
               "final_participants_accessed": 0, "shared_encoder_pretraining_completed": False}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.out / "artifact_sha256.json").write_text(json.dumps(
        {str(f.relative_to(args.out)): sha256(f) for f in args.out.rglob("*") if f.is_file()}, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
