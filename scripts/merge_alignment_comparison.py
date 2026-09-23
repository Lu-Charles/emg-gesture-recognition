"""Check CUDA/MPS bridge controls before combining descriptive development tables."""

import argparse
import json
from pathlib import Path
import numpy as np
from scripts.summarize_alignment_comparison import interval


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--small", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    original = json.loads((args.original / "results.json").read_text())
    followup = json.loads((args.small / "results.json").read_text())
    assert (args.original / "complete.json").exists() and (
        args.small / "complete.json"
    ).exists()
    for key in [
        "seed",
        "participants",
        "sessions",
        "choices",
        "k",
        "steps",
        "inputs",
        "learning_rate",
        "batch_size",
    ]:
        assert original["config"][key] == followup["config"][key], key
    assert json.loads((args.original / "access.json").read_text()) == json.loads(
        (args.small / "access.json").read_text()
    )
    original_records = {record["name"]: record for record in original["records"]}
    original_predictions = np.load(args.original / "window_predictions.npz")
    followup_predictions = np.load(args.small / "window_predictions.npz")
    differences = []
    window_differences = 0
    bridge = []
    for record in followup["records"]:
        if record["method"] not in ["frozen", "source_ce"]:
            continue
        original_record = original_records[record["name"]]
        bridge.append(record["name"])
        changed_windows = int(
            np.count_nonzero(
                original_predictions[record["name"]]
                != followup_predictions[record["name"]]
            )
        )
        window_differences += changed_windows
        if original_record["confusion"] != record["confusion"]:
            differences.append(
                dict(
                    name=record["name"],
                    window_differences=changed_windows,
                    accuracy_difference=record["accuracy"]
                    - original_record["accuracy"],
                )
            )
    bridge_report = dict(
        identical_trial_metrics=not differences,
        bridge_evaluations=len(bridge),
        compared_windows=len(bridge) * 68 * 35,
        window_differences=window_differences,
        changed_trial_confusions=differences,
        original_device=original["config"]["device"],
        original_torch=original["config"]["torch"],
        followup_device=followup["config"]["device"],
        followup_torch=followup["config"]["torch"],
    )
    (args.out / "bridge_verification.json").write_text(
        json.dumps(bridge_report, indent=2) + "\n"
    )
    if differences:
        raise ValueError(
            "Trial metrics differ across bridge controls; report separately or rerun matched controls"
        )
    records = original["records"] + [
        record for record in followup["records"] if record["method"].startswith("coral")
    ]
    metrics = ["accuracy", "other_active_accuracy", "calibrated_recall"]
    participants = original["config"]["participants"]
    steps = original["config"]["steps"]
    methods = ["target_ce", "replay", "source_ce"] + [
        f"coral_{w}" for w in ["0.0001", "0.001", "0.01", "0.1", "1", "10"]
    ]
    conditions = [("frozen", 0)] + [(m, s) for m in methods for s in steps]
    summary = []
    vectors = {}
    contrasts = []
    for k in [1, 2]:
        for method, step in conditions:
            subset = [
                record
                for record in records
                if record["k"] == k
                and record["method"] == method
                and record["steps"] == step
            ]
            entry = dict(k=k, method=method, steps=step)
            for metric in metrics:
                person_values = np.array(
                    [
                        np.mean(
                            [
                                record[metric]
                                for record in subset
                                if record["participant"] == person
                            ]
                        )
                        for person in participants
                    ]
                )
                assert all(
                    sum(record["participant"] == person for record in subset) == 8
                    for person in participants
                )
                vectors[k, method, step, metric] = person_values
                entry[metric] = interval(person_values)
            summary.append(entry)
        for method, step in conditions[1:]:
            for control in ["frozen", "source_ce", "replay"]:
                if method == control:
                    continue
                control_steps = 0 if control == "frozen" else step
                contrasts.append(
                    dict(
                        k=k,
                        method=method,
                        steps=step,
                        control=control,
                        **{
                            metric: interval(
                                vectors[k, method, step, metric]
                                - vectors[k, control, control_steps, metric]
                            )
                            for metric in metrics
                        },
                    )
                )
    result = dict(
        participants=participants,
        distinct_evaluations=len(records),
        conditions=summary,
        contrasts=contrasts,
        scope="Adaptive development-only comparison; all six weights retained; no confirmatory hypothesis test.",
        bridge=bridge_report,
    )
    (args.out / "combined_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = [
        "# Complete development comparison",
        "",
        result["scope"],
        "",
        "Eight participants; one initialization seed; four fixed gesture choices and two target sessions.",
        "Additional smaller weights were chosen after the original loss diagnostics showed dominance of the alignment term.",
        "Original run: RunPod CUDA/PyTorch "
        + original["config"]["torch"]
        + ". Follow-up: local MPS/PyTorch "
        + followup["config"]["torch"]
        + ".",
        f"All {len(bridge)} repeated frozen/source-only trial confusion matrices match; {window_differences} window labels differ across devices.",
        "",
        "| K | Method | Steps | Overall (%) | Omitted active (%) | Calibrated recall (%) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for record in summary:
        lines.append(
            f"| {record['k']} | {record['method']} | {record['steps']} | "
            + " | ".join(f"{record[m]['mean']:.2f}" for m in metrics)
            + " |"
        )
    lines += [
        "",
        "All settings are descriptive. Paired participant differences and unadjusted intervals are in combined_summary.json.",
        "These results adapt a published objective to this project’s backbone and data access; they do not reproduce the original Hyser/VGG accuracy.",
        "",
    ]
    (args.out / "COMPARISON.md").write_text("\n".join(lines))
    print(
        json.dumps(
            {
                "bridge": bridge_report,
                "conditions": len(summary),
                "distinct_evaluations": len(records),
            }
        )
    )


if __name__ == "__main__":
    main()
