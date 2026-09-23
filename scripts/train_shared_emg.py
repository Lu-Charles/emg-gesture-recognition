"""Pretrain on 20 people, then compare shared and random initialization on development.

The final participant group is inaccessible to this script. Fixed epochs and
identical enrollment/adaptation settings isolate the initialization comparison.
"""
from __future__ import annotations
import argparse
import copy
import csv
import gzip
import json
import platform
import resource
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score, balanced_accuracy_score
from src.emg_cnn import CompactEMGNet
from src.grabmyo_corpus import WindowCorpus, hash_stream, training_moments, epoch_blocks
from scripts.pilot_emg_cnn import fit, predict, sync


def roles(rows, participant, session, role, budget=None):
    return [i for i, r in enumerate(rows) if int(r["participant"]) == participant and int(r["session"]) == session
            and r["role"] == role and (budget is None or int(r["calibration_rank"]) <= budget)]


def validate_separation(train_rows, development_rows):
    if any(r["group"] != "train" for r in train_rows) or any(r["group"] != "development" for r in development_rows):
        raise ValueError("Wrong participant group")
    if {r["participant"] for r in train_rows} & {r["participant"] for r in development_rows}:
        raise ValueError("Representation and development participants overlap")


def pretrain(corpus, mean, scale, device, out, epochs=20, seed=42):
    torch.manual_seed(seed)
    model = CompactEMGNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    history = []
    start = time.perf_counter()
    for epoch in range(epochs):
        total_loss, correct = 0., 0
        seen = 0
        epoch_start = time.perf_counter()
        model.train()
        for block, (trial_ids, order) in enumerate(epoch_blocks(len(corpus.rows), seed, epoch)):
            x, y = corpus.batch(corpus.window_ids(trial_ids), mean, scale)
            xt = torch.from_numpy(x).to(device)
            yt = torch.from_numpy(y).to(device)
            for offset in range(0, len(order), 128):
                ids = torch.from_numpy(order[offset:offset + 128]).to(device)
                optimizer.zero_grad()
                logits = model(xt[ids])
                loss = torch.nn.functional.cross_entropy(logits, yt[ids])
                loss.backward()
                optimizer.step()
                total_loss += float(loss.detach().cpu()) * len(ids)
                correct += int((logits.argmax(1) == yt[ids]).sum().cpu())
                seen += len(ids)
            del xt, yt, x, y
            if (block + 1) % 16 == 0:
                print(json.dumps({"phase": "pretrain", "epoch": epoch + 1, "windows": seen,
                                  "seconds": time.perf_counter() - epoch_start}), flush=True)
        sync(device)
        if not np.isfinite(total_loss) or seen != len(corpus.rows) * 35:
            raise ValueError("Nonfinite pretraining loss or incorrect window count")
        row = {"epoch": epoch + 1, "training_loss": total_loss / seen,
               "training_accuracy_online": correct / seen, "seconds": time.perf_counter() - epoch_start,
               "scope": "training diagnostics, not evaluation"}
        history.append(row)
        (out / "pretraining_history.json").write_text(json.dumps(history, indent=2) + "\n")
        print(json.dumps(row), flush=True)
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, out / f"pretrain_epoch{epoch + 1}.pt")
    return model, time.perf_counter() - start


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    train, dev = WindowCorpus(args.cache, "train"), WindowCorpus(args.cache, "development")
    validate_separation(train.rows, dev.rows)
    summary = json.loads((args.cache / "summary.json").read_text())
    for group in ("train", "development"):
        for kind in ("signals", "features", "manifest"):
            suffix = "csv" if kind == "manifest" else "npy"
            if hash_stream(args.cache / f"{group}_{kind}.{suffix}") != summary["groups"][group][f"{kind}_sha256"]:
                raise ValueError("Changed cache artifact")
    torch.set_num_threads(4)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    participants = sorted({int(r["participant"]) for r in dev.rows})
    config = {"created_utc": datetime.now(timezone.utc).isoformat(), "scope": "development, shared versus random initialization",
              "cache": str(args.cache.resolve()), "cache_summary_sha256": hash_stream(args.cache / "summary.json"),
              "cache_config": train.config, "training_participants": summary["groups"]["train"]["participants"],
              "development_participants": participants, "seed": args.seed, "device": device, "torch": torch.__version__,
              "platform": platform.platform(), "hardware": subprocess.check_output(["/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip(),
              "pretraining_epochs": 20, "pretraining_lr": .001, "enrollment_epochs": 20, "enrollment_lr": .001,
              "pretraining_order": f"128-trial shuffled buffers, all windows shuffled once within buffer; seed{args.seed}+epoch",
              "update_epochs": 10, "update_lr": .0001, "optimizer": "Adam", "batch_size": 128,
              "scaler": "20-person training-window mean/std, fixed for all shared/random enrollment and updates",
              "initializations": ["shared", "random"], "update_modes": ["none", "head", "full"],
              "calibration_budgets": [1, 2, 3], "target_sessions": [2, 3], "model_selection": "fixed last epoch; no final data",
              "code_hashes": {f: hash_stream(f) for f in ("src/emg_cnn.py", "src/grabmyo_corpus.py", "scripts/train_shared_emg.py", "scripts/pilot_emg_cnn.py")}}
    (args.out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    start = time.perf_counter()
    mean, scale, count = training_moments(train)
    np.savez(args.out / "training_scaler.npz", mean=mean, scale=scale, count=count)
    shared, training_seconds = pretrain(train, mean, scale, device, args.out, seed=args.seed)
    models = args.out / "models"; models.mkdir()
    metrics, access, histories = [], [], []
    with gzip.open(args.out / "predictions.csv.gz", "wt") as prediction_file:
        writer = csv.DictWriter(prediction_file, fieldnames=["participant", "session", "initialization", "mode", "budget", "record", "window_start_sample", "truth", "prediction"])
        writer.writeheader()
        for p in participants:
            source_ids = roles(dev.rows, p, 1, "enrollment")
            source_x, source_y = dev.batch(dev.window_ids(source_ids), mean, scale)
            for initialization in ("shared", "random"):
                torch.manual_seed(args.seed)
                base = copy.deepcopy(shared) if initialization == "shared" else CompactEMGNet().to(device)
                base.set_update_mode("full")
                history = fit(base, source_x, source_y, 20, .001, args.seed, device)
                histories.append({"participant": p, "initialization": initialization, "stage": "enrollment", **history})
                for session in (2, 3):
                    score_ids = roles(dev.rows, p, session, "scoring")
                    score_x, score_y = dev.batch(dev.window_ids(score_ids), mean, scale)
                    for mode, budget in [("none", 0)] + [(m, b) for m in ("head", "full") for b in (1, 2, 3)]:
                        model = copy.deepcopy(base)
                        calibration_ids = roles(dev.rows, p, session, "calibration", budget) if budget else []
                        if set(source_ids + calibration_ids) & set(score_ids):
                            raise ValueError("Fit/scoring overlap")
                        if budget:
                            model.set_update_mode(mode)
                            cal_x, cal_y = dev.batch(dev.window_ids(calibration_ids), mean, scale)
                            histories.append({"participant": p, "session": session, "initialization": initialization, "stage": mode, "budget": budget,
                                              **fit(model, cal_x, cal_y, 10, .0001, args.seed, device)})
                        if mode == "head" and any(not torch.equal(a, b) for a, b in zip(model.encoder.parameters(), base.encoder.parameters(), strict=True)):
                            raise ValueError("Encoder changed in head-only update")
                        pred = predict(model, score_x, device)
                        name = f"p{p}_s{session}_{initialization}_{mode}_b{budget}"
                        torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, models / (name + ".pt"))
                        access.append({"name": name, "source": [dev.rows[i]["record"] for i in source_ids],
                                       "calibration": [dev.rows[i]["record"] for i in calibration_ids],
                                       "score": [dev.rows[i]["record"] for i in score_ids]})
                        result = {"name": name, "participant": p, "session": session, "initialization": initialization,
                                  "mode": mode, "budget": budget, "calibration_seconds": 85 * budget,
                                  "macro_f1": float(f1_score(score_y, pred, average="macro", labels=list(range(17)), zero_division=0)),
                                  "balanced_accuracy": float(balanced_accuracy_score(score_y, pred)),
                                  "scoring_trials": len(score_ids), "scoring_windows": len(pred)}
                        metrics.append(result)
                        for index, (truth, prediction) in enumerate(zip(score_y, pred)):
                            trial, window = divmod(index, 35)
                            writer.writerow({"participant": p, "session": session, "initialization": initialization,
                                             "mode": mode, "budget": budget, "record": dev.rows[score_ids[trial]]["record"],
                                             "window_start_sample": 1024 + 256 * window, "truth": int(truth), "prediction": int(prediction)})
                    print(json.dumps({"phase": "development", "participant": p, "session": session,
                                      "initialization": initialization, "models_saved": len(metrics)}), flush=True)
                    for file, content in (("metrics", metrics), ("data_access", access), ("adaptation_history", histories)):
                        (args.out / (file + ".json")).write_text(json.dumps(content, indent=2) + "\n")
    result = {"completed_utc": datetime.now(timezone.utc).isoformat(), "pretraining_seconds": training_seconds,
              "total_seconds": time.perf_counter() - start, "models": len(metrics), "training_people": 20,
              "development_people": len(participants), "training_windows": len(train.rows) * 35,
              "scaler_scalar_samples_per_channel": count, "peak_process_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              "device": device, "final_participants_accessed": 0, "scope": "development only"}
    (args.out / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    snapshot = args.out / "code_snapshot"; snapshot.mkdir()
    for name in config["code_hashes"]:
        (snapshot / Path(name).name).write_bytes(Path(name).read_bytes())
    (args.out / "artifact_sha256.json").write_text(json.dumps({str(p.relative_to(args.out)): hash_stream(p)
        for p in args.out.rglob("*") if p.is_file()}, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
