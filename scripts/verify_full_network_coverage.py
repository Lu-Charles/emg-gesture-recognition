"""Replay all full-network development checkpoints on CPU from raw cached windows."""

import json
import time
from pathlib import Path
import numpy as np
import torch
from src.grabmyo_corpus import WindowCorpus, hash_stream
from src.emg_cnn import CompactEMGNet
from scripts.decisive_calibration import measures, vote

R = Path(__file__).resolve().parents[1]
O = R / "research/runs/20260916_full_network_coverage"


def main():
    assert (O / "complete.json").exists()
    torch.set_num_threads(1)
    c = WindowCorpus(R / "data/public/grabmyo/cache_20260906_v1", "development")
    sc = np.load(R / "research/runs/20260906_shared_pretraining_v2/training_scaler.npz")
    res = json.loads((O / "results.json").read_text())
    access = {z["name"]: z for z in json.loads((O / "access.json").read_text())}
    baseacc = {
        z["name"]: z
        for z in json.loads(
            (R / "research/runs/20260916_decisive_seed42/access.json").read_text()
        )
    }
    saved = np.load(O / "predictions.npz")
    start = time.perf_counter()
    windows = 0
    for p in res["config"]["participants"]:
        for session in [2, 3]:
            records = [
                z
                for z in res["records"]
                if z["participant"] == p and z["session"] == session
            ]
            first = access[records[0]["name"].split("_full_")[0]]
            ids = first["scoring"]
            x, y = c.batch(c.window_ids(ids), sc["mean"], sc["scale"])
            x = torch.from_numpy(x)
            truth = np.array([int(c.rows[i]["class_index"]) for i in ids])
            for rr in records:
                name = rr["name"]
                case = name.split("_full_")[0]
                a = access[case]
                b = baseacc[case]
                for key in ["source", "calibration", "scoring"]:
                    assert a[key] == b[key]
                model = CompactEMGNet()
                model.load_state_dict(
                    torch.load(
                        O / f"models/{name}.pt", weights_only=True, map_location="cpu"
                    )
                )
                model.eval()
                with torch.no_grad():
                    w = np.concatenate(
                        [
                            model(x[j : j + 257]).argmax(1).numpy()
                            for j in range(0, len(x), 257)
                        ]
                    )
                v = vote(w)
                np.testing.assert_array_equal(v, saved[name])
                m = measures(v, truth, rr["gestures"])
                for key, val in m.items():
                    assert val == rr[key], (name, key)
                windows += len(w)
            print(p, session, "verified", time.perf_counter() - start, flush=True)
    out = dict(
        passed=True,
        models=len(res["records"]),
        raw_scoring_windows=windows,
        trial_votes=len(res["records"]) * 68,
        seconds=time.perf_counter() - start,
        device="CPU replay of MPS-fitted checkpoints",
        code_sha256=hash_stream(__file__),
    )
    (O / "verification.json").write_text(json.dumps(out, indent=2) + "\n")
    print(out)


if __name__ == "__main__":
    main()
