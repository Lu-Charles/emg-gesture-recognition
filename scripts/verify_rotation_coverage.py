"""Independent coefficient replay and calibration-only transformation audit."""

import argparse
import json
from pathlib import Path
import numpy as np
from src.grabmyo_corpus import hash_stream
from threadpoolctl import threadpool_limits

R = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--run", type=Path, default=R / "research/runs/20260916_rotation_coverage"
    )
    args = ap.parse_args()
    o = args.run
    r = json.loads((o / "results.json").read_text())
    cache = Path(
        r["config"].get("cache", R / "research/runs/20260907_sensor_shift_pilot_v2")
    )
    rows = json.loads((cache / "trial_manifest.json").read_text())
    x = np.load(cache / "tdar_rms_features.npy")
    a = np.load(o / "parameters.npz")
    v = np.load(o / "predictions.npz")
    acc = {z["name"]: z for z in json.loads((o / "access.json").read_text())}
    assert (o / "complete.json").exists()

    def rot(z, s):
        w = z.reshape(-1, z.shape[-1] // 8, 8)
        lo = int(np.floor(s))
        t = s - lo
        return (
            (1 - t) * w[:, :, (np.arange(8) - lo) % 8]
            + t * w[:, :, (np.arange(8) - lo - 1) % 8]
        ).reshape(z.shape)

    for sel in r["selections"]:
        p = sel["participant"]
        ids = next(z["source"] for z in acc.values() if z["name"].startswith(f"p{p}_"))
        sy = np.array([rows[i]["label"] for i in ids])
        u = np.array([x[ids][sy == k, :, 64:72].mean((0, 1)) for k in range(7)])
        np.testing.assert_allclose(u, sel["source_templates"])
        assert np.argsort(-u.sum(1), kind="stable")[:2].tolist() == sel["active"]
        scores = []
        for g in range(7):
            for h in range(7):
                q = np.log(np.maximum(u[g], 1e-12)) - np.log(np.maximum(u[h], 1e-12))
                score = (
                    min(
                        np.mean((q - q[(np.arange(8) - s) % 8]) ** 2)
                        for s in range(1, 8)
                    )
                    if g != h
                    else -np.inf
                )
                scores.append(score)
        assert list(np.unravel_index(np.argmax(scores), (7, 7))) == sel["identifiable"]
    for j, rr in enumerate(r["records"]):
        m = rr["method"]
        name = rr["name"]
        base = name.removesuffix("_" + m)
        z = acc[base]
        src, cal, score = z["source"], z["calibration"], z["score"]
        p = rr["participant"]
        assert not (
            set(src) & set(cal) or set(src) & set(score) or set(cal) & set(score)
        )
        assert len(src) == 14 and len(score) == 7
        assert all(
            rows[i]["subject"] == p and rows[i]["session"] == 0
            for i in src + cal + score
        )
        assert all(rows[i]["position"] == 0 and rows[i]["repetition"] < 2 for i in src)
        assert all(
            rows[i]["repetition"] == 2 and rows[i]["position"] == rr["position"]
            for i in score
        )
        assert all(
            rows[i]["repetition"] < 2 and rows[i]["position"] == rr["position"]
            for i in cal
        )
        assert p in r["config"]["participants"]
        tx = x[score].reshape(-1, 72).copy()
        truth = np.array([rows[i]["label"] for i in score])
        prefix = f"p{p}"
        if m == "pooled" or m.startswith("reference_"):
            logits = ((tx - a[prefix + "_mean"]) / a[prefix + "_scale"]) @ a[
                base + "_pooled_coef"
            ].T + a[base + "_pooled_intercept"]
        else:
            st = rr["state"]
            s = st["shift"]
            g = np.array(st["gain"])
            trans = rot(tx, s)
            trans[:, np.r_[0:8, 24:32, 64:72]] *= np.tile(g, 3)
            logits = ((trans - a[prefix + "_mean"]) / a[prefix + "_scale"]) @ a[
                prefix + "_coef"
            ].T + a[prefix + "_intercept"]
            if m not in ["frozen", "cosine_rotation"]:
                gs = sorted({rows[i]["label"] for i in cal})
                u = np.array(
                    [
                        x[[i for i in src if rows[i]["label"] == k], :, 64:72].mean(
                            (0, 1)
                        )
                        for k in gs
                    ]
                )
                b = np.array(
                    [
                        x[[i for i in cal if rows[i]["label"] == k], :, 64:72].mean(
                            (0, 1)
                        )
                        for k in gs
                    ]
                )
                d = np.log(np.maximum(u, 1e-12)) - np.log(np.maximum(rot(b, s), 1e-12))
                np.testing.assert_allclose(
                    np.clip(np.exp(d.mean(0)), 0.5, 2), g, rtol=1e-10, atol=1e-10
                )
                if m.startswith("profile"):
                    assert abs(np.mean((d - d.mean(0)) ** 2) - st["cost"]) < 1e-10
        w = logits.argmax(1)
        q = np.array(
            [np.bincount(ww, minlength=7).argmax() for ww in w.reshape(-1, 15)]
        )
        np.testing.assert_array_equal(q, v[name])
        assert abs(np.mean(q == truth) - rr["accuracy"]) < 1e-12
        assert abs(np.mean(w == np.repeat(truth, 15)) - rr["window_accuracy"]) < 1e-12
        assert (
            abs(sum(rows[i]["recorded_seconds"] for i in cal) - rr["recorded_seconds"])
            < 1e-12
        )
        if (j + 1) % 3430 == 0:
            print(j + 1, flush=True)
    out = dict(
        passed=True,
        records=len(r["records"]),
        cases=len(acc),
        trial_votes=len(r["records"]) * 7,
        participants=r["config"]["participants"],
        transform_gains_rederived=True,
        source_selection_checked=True,
        scope="fitted-coefficient replay; cache independently audited separately; no independent covariance refit",
        code_sha256=hash_stream(__file__),
    )
    (o / "verification.json").write_text(json.dumps(out, indent=2) + "\n")
    print(out)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
