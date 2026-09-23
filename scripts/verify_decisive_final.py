"""Independent checkpoint inference, trial roles and metrics for decisive screen."""

import argparse
import json
import copy
from pathlib import Path
import joblib
import numpy as np
import torch
from sklearn.metrics import f1_score
from src.emg_cnn import CompactEMGNet
from src.grabmyo_corpus import hash_stream
from src.final_coverage import FinalCorpus as WindowCorpus


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    args = ap.parse_args()
    run = args.run
    torch.set_num_threads(1)
    root = Path(__file__).resolve().parents[1]
    corpus = WindowCorpus(root / "data/public/grabmyo/cache_final_20260916", "final")
    data = json.loads((run / "results.json").read_text())
    assert (
        json.loads((run / "complete.json").read_text())["records"]
        == len(data["records"])
        == 12480
    )
    cfg = data["config"]
    access = {a["name"]: a for a in json.loads((run / "access.json").read_text())}
    prior = Path(cfg["prior"])
    scaler = np.load(prior / "training_scaler.npz")
    counts = dict(cases=0, checkpoints=0, votes=0, raw_embedding_windows=0)
    groups = {}
    summary = {}
    with np.load(run / "trial_predictions.npz") as saved:
        for p in cfg["participants"]:
            model = CompactEMGNet()
            model.load_state_dict(
                torch.load(run / f"models/p{p}_source.pt", weights_only=True)
            )
            model.eval()
            allids = [
                i for i, r in enumerate(corpus.rows) if int(r["participant"]) == p
            ]
            lookup = {i: j for j, i in enumerate(allids)}
            z = np.load(run / f"p{p}_embeddings.npy", mmap_mode="r")
            assert z.shape == (357, 35, 64)
            # Reconstruct every embedding with a distinct inference batch size.
            ids = corpus.window_ids(allids)
            recomputed = []
            with torch.no_grad():
                for start in range(0, len(ids), 257):
                    x, _ = corpus.batch(
                        ids[start : start + 257], scaler["mean"], scaler["scale"]
                    )
                    recomputed.append(model.encoder(torch.from_numpy(x)).numpy())
            np.testing.assert_allclose(
                np.concatenate(recomputed).reshape(z.shape), z, rtol=2e-5, atol=2e-6
            )
            counts["raw_embedding_windows"] += len(ids)
            standard, lda = joblib.load(run / f"models/p{p}_lda.joblib")
            for r in [r for r in data["records"] if r["participant"] == p]:
                base = f"p{p}_s{r['session']}_o{r['choice']}_k{r['k']}"
                a = access[base]
                source, cal, score = map(
                    set, [a["source"], a["calibration"], a["scoring"]]
                )
                assert len(source) == 119 and len(cal) == 2 and len(score) == 68
                assert not source & cal and not source & score and not cal & score
                assert {int(corpus.rows[i]["calibration_rank"]) for i in cal} == {1, 2}
                assert {int(corpus.rows[i]["class_index"]) for i in cal} == set(
                    r["gestures"]
                )
                for role, indices in [
                    ("enrollment", source),
                    ("calibration", cal),
                    ("scoring", score),
                ]:
                    assert all(
                        corpus.rows[i]["group"] == "final"
                        and corpus.rows[i]["role"] == role
                        and int(corpus.rows[i]["participant"]) == p
                        for i in indices
                    )
                truth = np.array(
                    [int(corpus.rows[i]["class_index"]) for i in a["scoring"]]
                )
                if r["method"].startswith("cnn_"):
                    head = copy.deepcopy(model.head)
                    if r["method"] != "cnn_frozen":
                        head.load_state_dict(
                            torch.load(
                                run / f"models/{r['name']}.pt", weights_only=True
                            )
                        )
                    x = torch.from_numpy(
                        np.array(z[[lookup[i] for i in a["scoring"]]].reshape(-1, 64))
                    )
                    with torch.no_grad():
                        pred = np.concatenate(
                            [head(block).argmax(1).numpy() for block in x.split(73)]
                        )
                else:
                    x = np.asarray(corpus.features[a["scoring"]]).reshape(-1, 48)
                    classifier = lda
                    if r["method"] == "lda_pooled":
                        classifier = joblib.load(
                            run / f"models/{base}_lda_pooled.joblib"
                        )
                    if r["method"] == "lda_gain":
                        sx = np.asarray(corpus.features[a["source"]]).reshape(-1, 48)
                        sy = np.repeat(
                            [int(corpus.rows[i]["class_index"]) for i in a["source"]],
                            35,
                        )
                        cx = np.asarray(corpus.features[a["calibration"]]).reshape(
                            -1, 48
                        )
                        cy = np.repeat(
                            [
                                int(corpus.rows[i]["class_index"])
                                for i in a["calibration"]
                            ],
                            35,
                        )
                        g = np.clip(
                            np.exp(
                                np.mean(
                                    [
                                        np.log(np.maximum(sx[sy == c].mean(0), 1e-12))
                                        - np.log(np.maximum(cx[cy == c].mean(0), 1e-12))
                                        for c in r["gestures"]
                                    ],
                                    axis=0,
                                )
                            ),
                            0.5,
                            2,
                        )
                        np.testing.assert_allclose(g, r["gain"], rtol=1e-12, atol=1e-12)
                        x = x * g
                    pred = classifier.predict(standard.transform(x))
                v = np.array(
                    [
                        np.bincount(x, minlength=17).argmax()
                        for x in pred.reshape(-1, 35)
                    ]
                )
                np.testing.assert_array_equal(v, saved[r["name"]])
                np.testing.assert_allclose(
                    np.mean(v == truth), r["accuracy"], atol=1e-14
                )
                np.testing.assert_allclose(
                    f1_score(
                        truth, v, labels=range(17), average="macro", zero_division=0
                    ),
                    r["macro_f1"],
                    atol=1e-14,
                )
                cm = np.zeros((17, 17), int)
                np.add.at(cm, (truth, v), 1)
                np.testing.assert_array_equal(cm, r["confusion"])
                key = (r["method"], r["steps"], r["k"])
                groups.setdefault(key, {}).setdefault(p, []).append(r)
                counts["checkpoints"] += 1
                counts["votes"] += len(v)
            print("verified participant", p, flush=True)
    for key, people in groups.items():
        fields = [
            "accuracy",
            "macro_f1",
            "other_active_accuracy",
            "other_into_calibration",
            "calibrated_recall",
            "rest_recall",
        ]
        pa = [
            dict(
                participant=p, **{f: float(np.mean([r[f] for r in rs])) for f in fields}
            )
            for p, rs in people.items()
        ]
        summary["|".join(map(str, key))] = dict(
            method=key[0],
            steps=key[1],
            k=key[2],
            participants=pa,
            **{f: float(np.mean([r[f] for r in pa])) for f in fields},
        )
    counts["cases"] = len(access)
    (run / "verification.json").write_text(
        json.dumps(
            dict(
                passed=True,
                counts=counts,
                final_participants_accessed=15,
                verifier_sha256=hash_stream(__file__),
                results_sha256=hash_stream(run / "results.json"),
            ),
            indent=2,
        )
        + "\n"
    )
    (run / "analysis.json").write_text(json.dumps(summary, indent=2) + "\n")
    (run / "verify_decisive_calibration.py").write_bytes(Path(__file__).read_bytes())
    print(json.dumps(counts))


if __name__ == "__main__":
    main()
