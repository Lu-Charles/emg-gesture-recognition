"""Post-hoc diagnostic of frozen CSL outputs; no fitting or checkpoint changes.

Permutation scores use scoring truth and are optimistic diagnostics, never valid
recognition performance. Each case remains a correlated repeated measurement.
"""

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "research/runs/20260907_csl_components_confirmatory"
OUTPUT = ROOT / "research/runs/20260909_csl_prediction_bias"
GESTURES = [8, 9, 12, 13, 16, 21, 23, 24]
METHODS = [
    "frozen",
    "gain_only",
    "spatial_only",
    "spatial_gain",
    "classifier_fine",
    "classifier_fine_gain",
]
NAMES = [
    "Frozen",
    "Gain",
    "Spatial",
    "Spatial + gain",
    "Fine-tuning",
    "Fine-tuning + gain",
]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def diagnose(y, pred, calibration):
    y, pred = np.asarray(y), np.asarray(pred)
    assert y.shape == pred.shape and set(y) == set(range(8))
    assert set(pred) <= set(range(8))
    cm = np.zeros((8, 8), dtype=int)
    np.add.at(cm, (y, pred), 1)
    other = y != calibration
    called = pred == calibration
    errors = pred != y
    correct_other = other & ~errors
    errors_to_calibration = other & called
    errors_elsewhere = other & errors & ~called
    assert sum(other) == sum(correct_other) + sum(errors_to_calibration) + sum(
        errors_elsewhere
    )
    ri, ci = linear_sum_assignment(-cm)
    precision = np.diag(cm) / np.maximum(cm.sum(0), 1)
    recall = np.diag(cm) / cm.sum(1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-30)
    return dict(
        accuracy=float(np.mean(~errors)),
        macro_f1=float(f1.mean()),
        calibrated_recall=float(np.mean(pred[~other] == calibration)),
        calibrated_precision=(
            float(np.mean(y[called] == calibration)) if called.any() else None
        ),
        predicted_calibrated_fraction=float(called.mean()),
        other_correct=float(correct_other.sum() / other.sum()),
        other_to_calibrated=float(errors_to_calibration.sum() / other.sum()),
        other_to_wrong_other=float(errors_elsewhere.sum() / other.sum()),
        other_errors_to_calibrated_fraction=(
            float(errors_to_calibration.sum() / (other & errors).sum())
            if (other & errors).any()
            else None
        ),
        all_predictions_calibrated=bool(called.all()),
        at_least_90pct_predictions_calibrated=bool(called.mean() >= 0.9),
        oracle_case_permutation_accuracy=float(cm[ri, ci].sum() / len(y)),
        confusion=cm.tolist(),
    )


def main():
    OUTPUT.mkdir(exist_ok=False)
    manifest = json.loads((INPUT / "manifest.json").read_text())
    results = json.loads((INPUT / "results.json").read_text())["results"]
    assert len(results) == 3840
    assert json.loads((INPUT / "verification.json").read_text())["passed"]
    identities = [e["id"] for e in manifest]
    assert len(set(identities)) == len(identities)
    for e in manifest:
        assert e["label"] == GESTURES.index(e["gesture"])
        assert Path(e["path"]).stem == f"gest{e['gesture']}"
        assert (
            e["id"]
            == f"p{e['participant']}_s{e['session']}_g{e['gesture']:02}_r{e['rep']:02}"
        )
    cases, seen = [], set()
    frozen_predictions, aggregate = {}, {m: np.zeros((8, 8), int) for m in METHODS}
    for r in results:
        key = (
            r["participant"],
            r["source_session"],
            r["target_session"],
            r["gesture"],
            r["method"],
        )
        assert key not in seen
        seen.add(key)
        entries = [
            e
            for e in manifest
            if e["participant"] == r["participant"]
            and e["session"] == r["target_session"]
            and e["rep"] > 0
        ]
        y = np.array([e["label"] for e in entries])
        pred = np.array(r["metrics"]["trial_predictions"])
        cal = GESTURES.index(r["gesture"])
        d = diagnose(y, pred, cal)
        assert np.isclose(d["accuracy"], r["metrics"]["trial_accuracy"], atol=1e-14)
        assert np.isclose(
            d["other_correct"], r["metrics"]["unseen_trial_accuracy"], atol=1e-14
        )
        if r["method"] == "frozen":
            pair = key[:3]
            if pair in frozen_predictions:
                np.testing.assert_array_equal(frozen_predictions[pair], pred)
            frozen_predictions[pair] = pred
        aggregate[r["method"]] += np.array(d["confusion"])
        cases.append(
            dict(
                participant=key[0],
                source_session=key[1],
                target_session=key[2],
                gesture=key[3],
                method=key[4],
                **d,
            )
        )
    # Replay every saved frame-to-trial vote independently, without loading models.
    files = sorted(INPUT.glob("subject*/s*_to_s*/g*/predictions.npz"))
    assert len(files) == 640
    checked_frames = 0
    for file in files:
        p = int(file.parent.parent.parent.name.removeprefix("subject"))
        src, tar = map(int, file.parent.parent.name[1:].split("_to_s"))
        g = int(file.parent.name[1:])
        entries = [
            e
            for e in manifest
            if e["participant"] == p and e["session"] == tar and e["rep"] > 0
        ]
        local = {
            r["method"]: r
            for r in cases
            if (
                r["participant"],
                r["source_session"],
                r["target_session"],
                r["gesture"],
            )
            == (p, src, tar, g)
        }
        with np.load(file) as saved:
            assert saved["trial_ids"].tolist() == [e["id"] for e in entries]
            np.testing.assert_array_equal(
                saved["lengths"], [e["frames"] for e in entries]
            )
            truth = saved["labels"]
            offsets = np.r_[0, np.cumsum(saved["lengths"])]
            assert offsets[-1] == len(truth)
            for i, e in enumerate(entries):
                assert np.all(truth[offsets[i] : offsets[i + 1]] == e["label"])
            for method in METHODS:
                frames = saved[method]
                assert len(frames) == len(truth)
                votes = []
                for lo, hi in zip(offsets[:-1], offsets[1:]):
                    block = frames[lo:hi]
                    counts = np.bincount(block, minlength=8)
                    ties = np.flatnonzero(counts == counts.max())
                    votes.append(int(block[np.flatnonzero(np.isin(block, ties))[0]]))
                d = diagnose([e["label"] for e in entries], votes, GESTURES.index(g))
                assert d == {k: local[method][k] for k in d}
                checked_frames += len(frames)
    fields = [
        k
        for k in cases[0]
        if k
        not in (
            "confusion",
            "participant",
            "source_session",
            "target_session",
            "gesture",
            "method",
        )
    ]
    participants, summary = {}, {}
    for method in METHODS:
        rows = [r for r in cases if r["method"] == method]
        assert len(rows) == 640
        pa = []
        for person in range(2, 6):
            group = [r for r in rows if r["participant"] == person]
            assert len(group) == 160
            pa.append(
                dict(
                    participant=person,
                    **{
                        k: float(np.mean([r[k] for r in group if r[k] is not None]))
                        for k in fields
                    },
                )
            )
        participants[method] = pa
        cm = aggregate[method]
        ri, ci = linear_sum_assignment(-cm)
        summary[method] = dict(
            **{k: float(np.mean([r[k] for r in pa])) for k in fields},
            exactly_collapsed_cases=sum(r["all_predictions_calibrated"] for r in rows),
            near_collapsed_cases=sum(
                r["at_least_90pct_predictions_calibrated"] for r in rows
            ),
            pooled_identity_accuracy=float(np.trace(cm) / cm.sum()),
            pooled_oracle_permutation_accuracy=float(cm[ri, ci].sum() / cm.sum()),
            pooled_oracle_prediction_to_truth={
                str(GESTURES[j]): GESTURES[i] for i, j in zip(ri, ci)
            },
        )
    report = dict(
        timestamp=datetime.now(timezone.utc).isoformat(),
        scope="post-hoc existing-output diagnostic; no new training or final evaluation",
        cases=len(cases),
        participants=participants,
        summary=summary,
        checks=dict(
            manifest_entries=len(manifest),
            prediction_archives=len(files),
            frame_predictions_voted=checked_frames,
            label_mapping_consistent=True,
            identical_frozen_predictions_across_calibration_choices=True,
        ),
        source_hashes={
            str(p): digest(p)
            for p in [
                INPUT / "manifest.json",
                INPUT / "results.json",
                INPUT / "verification.json",
                Path(__file__),
            ]
        },
        caveats=[
            "Four independent participants; cases correlated.",
            "Oracle label permutations use scoring labels and are diagnostics only.",
            "Mapping checks verify filenames and pipeline consistency, not the original human gesture execution.",
            "Precision means average defined case precision; undefined cases omitted within participant.",
        ],
    )
    (OUTPUT / "analysis.json").write_text(json.dumps(report, indent=2) + "\n")
    (OUTPUT / "confusions.json").write_text(
        json.dumps({m: a.tolist() for m, a in aggregate.items()}, indent=2) + "\n"
    )
    with (OUTPUT / "cases.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[k for k in cases[0] if k != "confusion"])
        writer.writeheader()
        writer.writerows(
            {k: v for k, v in r.items() if k != "confusion"} for r in cases
        )
    (OUTPUT / "audit_csl_prediction_bias.py").write_text(Path(__file__).read_text())
    fig, ax = plt.subplots(figsize=(10, 5.5), layout="constrained")
    left = np.zeros(6)
    for key, label, color in [
        ("other_correct", "Correct gesture", "#2c7a78"),
        ("other_to_calibrated", "Incorrectly called calibration gesture", "#c46b35"),
        ("other_to_wrong_other", "Another wrong gesture", "#a4aebc"),
    ]:
        values = np.array([summary[m][key] * 100 for m in METHODS])
        ax.barh(NAMES, values, left=left, color=color, label=label)
        for i, v in enumerate(values):
            if v > 5:
                ax.text(
                    left[i] + v / 2,
                    i,
                    f"{v:.1f}%",
                    ha="center",
                    va="center",
                    color="white" if key != "other_to_wrong_other" else "#202020",
                )
        left += values
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Outcomes on gestures absent from calibration (%)")
    ax.set_title(
        "Adaptation pulls other gestures toward the calibration label\nSaved CSL trial votes · equal weighting of four participants",
        loc="left",
    )
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), frameon=False, ncol=1)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(OUTPUT / "prediction_bias.png", dpi=180)
    fig.savefig(OUTPUT / "prediction_bias.svg")
    plt.close(fig)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
