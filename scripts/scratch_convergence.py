"""Fixed 80-epoch scratch control; 20/40 checkpoints remain diagnostic."""
import argparse
import copy
import csv
import gzip
import json
import resource
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import f1_score, balanced_accuracy_score
from src.emg_cnn import CompactEMGNet
from src.emg_enrollment import enrollment_path
from src.grabmyo_corpus import WindowCorpus, hash_stream
from scripts.train_shared_emg import roles
from scripts.pilot_emg_cnn import fit, predict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    prior = json.loads((args.prior / "config.json").read_text())
    validation = json.loads((args.prior / "independent_validation.json").read_text())
    if validation["models"] != 224 or not validation["fit_access_verified"]:
        raise ValueError("Prior shared/random run must be verified")
    for name, expected in json.loads((args.prior / "artifact_sha256.json").read_text()).items():
        if hash_stream(args.prior / name) != expected:
            raise ValueError(f"Changed prior artifact: {name}")
    cache = Path(prior["cache"])
    dev = WindowCorpus(cache, "development")
    summary = json.loads((cache / "summary.json").read_text())
    if hash_stream(cache / "summary.json") != prior["cache_summary_sha256"]:
        raise ValueError("Changed cache summary")
    for kind, suffix in (("signals", "npy"), ("manifest", "csv")):
        if hash_stream(cache / f"development_{kind}.{suffix}") != summary["groups"]["development"][f"{kind}_sha256"]:
            raise ValueError("Changed development cache")
    args.out.mkdir(parents=True, exist_ok=False)
    files = ["scripts/scratch_convergence.py", "src/emg_enrollment.py", "src/emg_cnn.py",
             "src/grabmyo_corpus.py", "scripts/train_shared_emg.py", "scripts/pilot_emg_cnn.py"]
    config = {**prior, "created_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "development scratch convergence control; 80 epochs primary, 20/40 diagnostic",
              "prior_run": str(args.prior.resolve()), "prior_artifact_manifest_sha256": hash_stream(args.prior / "artifact_sha256.json"),
              "initializations": ["random"], "pretraining_epochs": 0,
              "enrollment_epochs": 80, "enrollment_epoch_checkpoints": [20, 40, 80],
              "model_selection": "fixed endpoint80; no per-person or target-score checkpoint selection",
              "code_hashes": {f: hash_stream(f) for f in files}}
    (args.out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    snapshot = args.out / "code_snapshot"; snapshot.mkdir()
    for name in files:
        (snapshot / Path(name).name).write_bytes(Path(name).read_bytes())
    (args.out / "training_scaler.npz").write_bytes((args.prior / "training_scaler.npz").read_bytes())
    with np.load(args.out / "training_scaler.npz") as scaler:
        mean, scale = scaler["mean"], scaler["scale"]
    models = args.out / "models"; models.mkdir()
    enrollment = args.out / "enrollment"; enrollment.mkdir()
    torch.set_num_threads(4)
    device = config["device"]
    metrics, accesses, histories = [], [], []
    start = time.perf_counter()
    with gzip.open(args.out / "predictions.csv.gz", "wt") as file:
        writer = csv.DictWriter(file, fieldnames=["participant", "session", "initialization", "enrollment_epochs", "mode", "budget", "record", "window_start_sample", "truth", "prediction"])
        writer.writeheader()
        for p in config["development_participants"]:
            source = roles(dev.rows, p, 1, "enrollment")
            source_x, source_y = dev.batch(dev.window_ids(source), mean, scale)
            torch.manual_seed(42)
            trajectory = CompactEMGNet().to(device)
            for checkpoint in enrollment_path(trajectory, source_x, source_y, [20, 40, 80], .001, 42, device):
                epoch = checkpoint["epoch"]
                torch.save(checkpoint, enrollment / f"p{p}_e{epoch}.pt")
                histories.append({"participant": p, "initialization": "random", "stage": "enrollment", "enrollment_epochs": epoch,
                                  "epoch_losses": checkpoint["epoch_losses"], "fit_seconds": checkpoint["fit_seconds"]})
                base = CompactEMGNet().to(device); base.load_state_dict(checkpoint["model"])
                for session in (2, 3):
                    score = roles(dev.rows, p, session, "scoring")
                    score_x, score_y = dev.batch(dev.window_ids(score), mean, scale)
                    for mode, budget in [("none", 0)] + [(m, b) for m in ("head", "full") for b in (1, 2, 3)]:
                        model = copy.deepcopy(base)
                        cal = roles(dev.rows, p, session, "calibration", budget) if budget else []
                        if set(source + cal) & set(score):
                            raise ValueError("Fit/scoring overlap")
                        if budget:
                            model.set_update_mode(mode)
                            x, y = dev.batch(dev.window_ids(cal), mean, scale)
                            histories.append({"participant": p, "session": session, "enrollment_epochs": epoch, "stage": mode, "budget": budget,
                                              **fit(model, x, y, 10, .0001, 42, device)})
                        pred = predict(model, score_x, device)
                        name = f"p{p}_s{session}_random_e{epoch}_{mode}_b{budget}"
                        torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, models / f"{name}.pt")
                        accesses.append({"name": name, "source": [dev.rows[i]["record"] for i in source],
                                         "calibration": [dev.rows[i]["record"] for i in cal], "score": [dev.rows[i]["record"] for i in score]})
                        metrics.append({"name": name, "participant": p, "session": session, "initialization": "random",
                                        "enrollment_epochs": epoch, "mode": mode, "budget": budget, "calibration_seconds": 85 * budget,
                                        "macro_f1": float(f1_score(score_y, pred, average="macro", labels=list(range(17)), zero_division=0)),
                                        "balanced_accuracy": float(balanced_accuracy_score(score_y, pred)), "scoring_trials": len(score), "scoring_windows": len(pred)})
                        for i, (truth, prediction) in enumerate(zip(score_y, pred)):
                            trial, w = divmod(i, 35)
                            writer.writerow({"participant": p, "session": session, "initialization": "random", "enrollment_epochs": epoch,
                                             "mode": mode, "budget": budget, "record": dev.rows[score[trial]]["record"],
                                             "window_start_sample": 1024 + 256 * w, "truth": int(truth), "prediction": int(prediction)})
                for name, value in (("metrics", metrics), ("data_access", accesses), ("adaptation_history", histories)):
                    (args.out / f"{name}.json").write_text(json.dumps(value, indent=2) + "\n")
                print(json.dumps({"participant": p, "enrollment_epochs": epoch, "last_training_loss": checkpoint["epoch_losses"][-1], "models_saved": len(metrics)}), flush=True)
    result = {"completed_utc": datetime.now(timezone.utc).isoformat(), "total_seconds": time.perf_counter() - start,
              "models": len(metrics), "development_people": len(config["development_participants"]),
              "peak_process_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              "final_participants_accessed": 0, "scope": "development only"}
    (args.out / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    (args.out / "artifact_sha256.json").write_text(json.dumps({str(p.relative_to(args.out)): hash_stream(p) for p in args.out.rglob("*") if p.is_file()}, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
