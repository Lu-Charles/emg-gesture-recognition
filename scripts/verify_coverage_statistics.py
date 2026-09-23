"""Independent statistic reconstruction from checkpoint-verified confusion matrices."""

import json
from pathlib import Path
import numpy as np
from scipy import stats

R = Path(__file__).resolve().parents[1]
O = R / "research/expanded_study_20260916"


def main():
    s = json.loads((O / "statistics.json").read_text())
    by = {}
    count = 0
    for seed in [42, 0, 1]:
        run = R / f"research/runs/20260916_decisive_final_seed{seed}"
        data = json.loads((run / "results.json").read_text())["records"]
        for r in data:
            cm = np.array(r["confusion"])
            assert cm.shape == (17, 17) and np.all(cm.sum(1) == 4)
            gs = r["gestures"]
            other = [k for k in range(16) if k not in gs]
            di = cm.diagonal()
            metrics = {
                "accuracy": np.trace(cm) / cm.sum(),
                "calibrated_recall": sum(di[gs]) / sum(cm[gs].sum(1)),
                "other_active_accuracy": sum(di[other]) / sum(cm[other].sum(1)),
                "other_into_calibration": cm[np.ix_(other, gs)].sum() / cm[other].sum(),
                "rest_recall": cm[16, 16] / cm[16].sum(),
            }
            for k, v in metrics.items():
                assert abs(v - r[k]) < 1e-12
            if r["k"] == 2 and (
                (r["method"] == "cnn_replay" and r["steps"] == 100)
                or r["method"] == "cnn_frozen"
            ):
                by.setdefault((r["participant"], r["method"]), []).append(
                    metrics["accuracy"]
                )
            count += 1
    ps = sorted({p for p, m in by})
    diff = np.array(
        [np.mean(by[p, "cnn_replay"]) - np.mean(by[p, "cnn_frozen"]) for p in ps]
    )
    a = s["final_cnn_replay_k2_vs_frozen"]["accuracy"]
    np.testing.assert_allclose(diff, a["per_person"], rtol=1e-12, atol=1e-12)
    # Direct SeNic vote reconstruction of the primary comparison.
    r = json.loads(
        (R / "research/runs/20260916_rotation_final/results.json").read_text()
    )
    pred = np.load(R / "research/runs/20260916_rotation_final/predictions.npz")
    sel = {z["participant"]: z["active"] for z in r["selections"]}
    values = {}
    for z in r["records"]:
        if z["method"] == "cosine_gain" and z["k"] == 2:
            acc = np.mean(pred[z["name"]] == np.arange(7))
            assert acc == z["accuracy"]
            values.setdefault(z["participant"], {"all": [], "active": []})[
                "all"
            ].append(acc)
            if [z["g"], z["h"]] == sel[z["participant"]]:
                values[z["participant"]]["active"].append(acc)
    d = np.array(
        [
            np.mean(values[p]["active"]) - np.mean(values[p]["all"])
            for p in sorted(values)
        ]
    )
    b = s["final_selection_primary"]
    np.testing.assert_allclose(d, b["per_person"], rtol=1e-12, atol=1e-12)
    for v, reported in [(diff, a), (d, b)]:
        n = len(v)
        mean = v.mean()
        var = sum((v - mean) ** 2) / (n - 1)
        se = np.sqrt(var / n)
        p = 2 * stats.t.sf(abs(mean / se), n - 1)
        lo, hi = mean + np.array([-1, 1]) * stats.t.ppf(0.975, n - 1) * se
        assert abs(p - reported["p"]) < 1e-12
        np.testing.assert_allclose([lo, hi], reported["ci95"], atol=1e-12)
    pv = sorted([a["p"], b["p"]])
    expected = [min(1, 2 * pv[0]), max(min(1, 2 * pv[0]), pv[1])]
    got = sorted(z["holm_p"] for z in s["primary_tests"].values())
    np.testing.assert_allclose(expected, got, atol=1e-12)
    report = dict(
        passed=True,
        confusion_records=count,
        primary_participants=[len(diff), len(d)],
        t_intervals_independently_rederived=True,
        holm_verified=True,
        no_primary_holm_significant_at_05=all(z >= 0.05 for z in got),
    )
    (O / "statistics_verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(report)


if __name__ == "__main__":
    main()
