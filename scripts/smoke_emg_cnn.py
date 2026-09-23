"""Training-only throughput smoke on already verified public training data.

No validation/test accuracy is computed and these weights are not research results.
"""
import argparse
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from src.emg_cnn import ChannelStandardizer, CompactEMGNet
from src.grabmyo import checksum_index, read_forearm, short_windows, sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    root = Path("data/public/grabmyo/1.1.0")
    metadata = json.loads((args.preparation / "validation.json").read_text())
    selected = [r for r in metadata if r["group"] == "train"]
    if not selected or any(r["group"] != "train" for r in selected):
        raise ValueError("No permitted training records")
    torch.manual_seed(42)
    torch.set_num_threads(4)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    config = {"created_utc": datetime.now(timezone.utc).isoformat(), "scope": "training-only throughput smoke",
              "records": [r["record"] for r in selected], "seed": 42, "batch_size": 128,
              "steps": 30, "warmup_steps": 5, "optimizer": "Adam", "learning_rate": .001,
              "loss": "cross_entropy", "device": device, "torch": torch.__version__,
              "python": sys.version, "platform": platform.platform(),
              "hardware": subprocess.check_output(["/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip(),
              "source_hashes": {f: sha256(Path(f)) for f in ("src/emg_cnn.py", "src/grabmyo.py", "scripts/smoke_emg_cnn.py")}}
    (args.out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    checksums = checksum_index(root)
    windows, labels = [], []
    for row in selected:
        x, _ = read_forearm(root, row["record"], checksums)
        w, _ = short_windows(x)
        windows.append(w.transpose(0, 2, 1))
        labels.extend([row["class_index"]] * len(w))
    x = np.concatenate(windows)
    scaler = ChannelStandardizer().fit(x)
    xt = torch.from_numpy(scaler.transform(x)).to(device)
    yt = torch.tensor(labels, dtype=torch.long, device=device)
    model = CompactEMGNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    rng = np.random.default_rng(42)
    losses, durations = [], []
    for step in range(30):
        ids = torch.tensor(rng.integers(0, len(x), size=128), dtype=torch.long, device=device)
        if device == "mps":
            torch.mps.synchronize()
        start = time.perf_counter()
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(xt[ids]), yt[ids])
        loss.backward()
        optimizer.step()
        if device == "mps":
            torch.mps.synchronize()
        durations.append(time.perf_counter() - start)
        losses.append(float(loss.detach().cpu()))
    if not np.isfinite(losses).all() or not all(torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError("Nonfinite training loss/weights")
    result = {"completed_utc": datetime.now(timezone.utc).isoformat(), "records": len(selected),
              "windows": len(x), "classes_present": sorted(set(labels)), "output_classes": 17,
              "parameters": sum(p.numel() for p in model.parameters()), "device": device,
              "median_step_seconds_after_warmup": float(np.median(durations[5:])),
              "losses_training_only": losses, "step_seconds": durations,
              "interpretation": "throughput and finite-gradient smoke only; no accuracy/generalization evidence",
              "final_participants_accessed": 0}
    (args.out / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    snapshot = args.out / "code_snapshot"
    snapshot.mkdir()
    for name in config["source_hashes"]:
        (snapshot / Path(name).name).write_bytes(Path(name).read_bytes())
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
