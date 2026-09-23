"""Participant-level statistics for the frozen primary contrasts and disclosed controls."""

import json
from pathlib import Path
import numpy as np
from scipy.stats import t, ttest_1samp

R = Path(__file__).resolve().parents[1]
O = R / "research/expanded_study_20260916"


def interval(d):
    """Summarize one difference per participant, including the seeded bootstrap."""
    d = np.asarray(d, float)
    count = len(d)
    mean = float(d.mean())
    standard_error = d.std(ddof=1) / np.sqrt(count)
    half_width = t.ppf(0.975, count - 1) * standard_error
    bootstrap_means = (
        np.random.default_rng(20260916).choice(d, (10000, count), replace=True).mean(1)
    )
    return dict(
        n=count,
        mean=mean,
        ci95=[float(mean - half_width), float(mean + half_width)],
        bootstrap95=np.quantile(bootstrap_means, [0.025, 0.975]).tolist(),
        p=(
            float(ttest_1samp(d, 0).pvalue)
            if standard_error
            else (1.0 if mean == 0 else 0.0)
        ),
        positive=int(sum(d > 1e-12)),
        negative=int(sum(d < -1e-12)),
        per_person=d.tolist(),
    )


def paired(a, b, key="accuracy"):
    """Average repeated conditions within each person before taking paired differences."""
    participants = sorted(set(record["participant"] for record in a))
    assert participants == sorted(set(record["participant"] for record in b))
    differences = [
        np.mean([record[key] for record in a if record["participant"] == person])
        - np.mean([record[key] for record in b if record["participant"] == person])
        for person in participants
    ]
    return dict(participants=participants, metric=key, **interval(differences))


def grouped(records, fields, metrics):
    """Report condition means with equal weight for each participant."""
    summaries = []
    for keys in sorted(
        {tuple(record[field] for field in fields) for record in records}
    ):
        condition_records = [
            record
            for record in records
            if tuple(record[field] for field in fields) == keys
        ]
        participants = sorted({record["participant"] for record in condition_records})
        values = {
            metric: [
                float(
                    np.mean(
                        [
                            record[metric]
                            for record in condition_records
                            if record["participant"] == person
                        ]
                    )
                )
                for person in participants
            ]
            for metric in metrics
        }
        summaries.append(
            dict(
                zip(fields, keys),
                participants=participants,
                per_person=values,
                **{
                    metric: float(np.mean(person_values))
                    for metric, person_values in values.items()
                },
            )
        )
    return summaries


def summarize_neural_runs(study):
    """Summarize each CNN seed and retain the records for the precision diagnostic."""
    all_neural = {}
    for cohort in ["development", "final"]:
        records = []
        for seed in [42, 0, 1]:
            folder = (
                R
                / "research/runs"
                / f"20260916_decisive_{'final_' if cohort=='final' else ''}seed{seed}"
            )
            assert json.loads((folder / "verification.json").read_text())["passed"]
            a = json.loads((folder / "results.json").read_text())["records"]
            records.extend(
                [dict(z, seed=seed) for z in a if z["method"].startswith("cnn_")]
            )
        all_neural[cohort] = records
        study[cohort + "_neural"] = grouped(
            records,
            ["method", "steps", "k"],
            [
                "accuracy",
                "other_active_accuracy",
                "other_into_calibration",
                "calibrated_recall",
                "rest_recall",
                "macro_f1",
            ],
        )
        primary = [z for z in records if z["steps"] in [0, 100]]
        for k in [1, 2]:
            for m in ["cnn_naive", "cnn_replay", "cnn_l2"]:
                a = [z for z in primary if z["k"] == k and z["method"] == m]
                b = [z for z in primary if z["k"] == k and z["method"] == "cnn_frozen"]
                study[f"{cohort}_{m}_k{k}_vs_frozen"] = {
                    metric: paired(a, b, metric)
                    for metric in [
                        "accuracy",
                        "other_active_accuracy",
                        "other_into_calibration",
                    ]
                }
        study[cohort + "_replay_coverage"] = paired(
            [z for z in primary if z["k"] == 2 and z["method"] == "cnn_replay"],
            [z for z in primary if z["k"] == 1 and z["method"] == "cnn_replay"],
        )
    return all_neural


def summarize_rotation_runs(study):
    """Summarize SeNic rotation methods and the source-selected gesture pairs."""
    for cohort in ["development", "final"]:
        folder = (
            R
            / "research/runs"
            / (
                "20260916_rotation_coverage"
                if cohort == "development"
                else "20260916_rotation_final"
            )
        )
        assert json.loads((folder / "verification.json").read_text())["passed"]
        r = json.loads((folder / "results.json").read_text())
        a = r["records"]
        sels = {s["participant"]: s for s in r["selections"]}
        study[cohort + "_rotation"] = grouped(
            a,
            ["method", "k"],
            [
                "accuracy",
                "window_accuracy",
                "other_accuracy",
                "other_into_calibration",
                "calibrated_recall",
                "recorded_seconds",
            ],
        )
        selected = []
        for z in a:
            if z["k"] == 2:
                for rule in ["active", "diverse", "identifiable", "fixed"]:
                    if [z["g"], z["h"]] == sels[z["participant"]][rule]:
                        selected.append(dict(z, selector=rule))
        study[cohort + "_rotation_selection"] = grouped(
            selected,
            ["method", "selector"],
            ["accuracy", "other_accuracy", "recorded_seconds"],
        )
        active = [
            z
            for z in selected
            if z["method"] == "cosine_gain" and z["selector"] == "active"
        ]
        random = [z for z in a if z["method"] == "cosine_gain" and z["k"] == 2]
        study[cohort + "_selection_primary"] = paired(active, random)
        study[cohort + "_rotation_coverage_contrast"] = paired(
            random, [z for z in a if z["method"] == "cosine_gain" and z["k"] == 1]
        )
        study[cohort + "_rotation_vs_frozen"] = paired(
            random, [z for z in a if z["method"] == "frozen" and z["k"] == 2]
        )
        study[cohort + "_rotation_gain_vs_cosine"] = paired(
            random, [z for z in a if z["method"] == "cosine_rotation" and z["k"] == 2]
        )


def summarize_classical_runs(study):
    """Summarize amplitude-preserving classical baselines and gesture selection."""
    for cohort in ["development", "final"]:
        folder = (
            R
            / "research/runs"
            / (
                "20260916_classical_selection"
                if cohort == "development"
                else "20260916_classical_final"
            )
        )
        assert json.loads((folder / "verification.json").read_text())["passed"]
        r = json.loads((folder / "results.json").read_text())
        a = r["records"]
        sels = {s["participant"]: s for s in r["selections"]}
        study[cohort + "_classical"] = grouped(
            a,
            ["method", "k"],
            ["accuracy", "other_active_accuracy", "other_into_calibration", "macro_f1"],
        )
        selected = [
            dict(z, selector=rule)
            for z in a
            for rule in ["active", "diverse", "fixed"]
            if [z["g"], z["h"]] == sels[z["participant"]][rule]
        ]
        study[cohort + "_classical_selection"] = grouped(
            selected, ["method", "selector"], ["accuracy", "other_active_accuracy"]
        )


def main():
    O.mkdir(exist_ok=True)
    study = {}
    all_neural = summarize_neural_runs(study)
    summarize_rotation_runs(study)
    summarize_classical_runs(study)
    folder = R / "research/runs/20260916_full_network_coverage"
    assert json.loads((folder / "verification.json").read_text())["passed"]
    r = json.loads((folder / "results.json").read_text())
    study["development_full_network"] = grouped(
        r["records"],
        ["method", "steps", "k"],
        [
            "accuracy",
            "other_active_accuracy",
            "other_into_calibration",
            "calibrated_recall",
        ],
    )
    ref = json.loads(
        (R / "research/runs/20260916_rotation_final_reference/results.json").read_text()
    )
    study["final_rotation_reference"] = grouped(
        ref, ["method"], ["accuracy", "window_accuracy", "recorded_seconds"]
    )
    # Confirmatory family of two primary hypotheses: Holm step-down adjustment.
    keys = ["final_cnn_replay_k2_vs_frozen", "final_selection_primary"]
    p = [study[keys[0]]["accuracy"]["p"], study[keys[1]]["p"]]
    order = np.argsort(p)
    adjusted = np.zeros(2)
    prev = 0.0
    for rank, i in enumerate(order):
        prev = max(prev, min(1.0, (2 - rank) * p[i]))
        adjusted[i] = prev
    study["primary_tests"] = {
        k: dict(raw_p=p[i], holm_p=float(adjusted[i])) for i, k in enumerate(keys)
    }
    precision = {}
    for rr in all_neural["final"]:
        if rr["k"] == 2 and rr["steps"] in [0, 100]:
            cm = np.array(rr["confusion"])
            gs = rr["gestures"]
            entry = precision.setdefault(rr["method"], dict(correct=0, predicted=0))
            entry["correct"] += int(cm.diagonal()[gs].sum())
            entry["predicted"] += int(cm[:, gs].sum())
    study["posthoc_calibrated_prediction_correctness"] = {
        m: dict(
            **v,
            fraction=v["correct"] / v["predicted"],
            scope="pooled exact-label correctness conditional on predicting either calibrated gesture; post-hoc descriptive",
        )
        for m, v in precision.items()
    }
    (O / "statistics.json").write_text(json.dumps(study, indent=2) + "\n")
    print(
        json.dumps(
            dict(
                grabmyo=study["final_cnn_replay_k2_vs_frozen"],
                senic=study["final_selection_primary"],
                primary=study["primary_tests"],
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
