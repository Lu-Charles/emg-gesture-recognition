"""Participant-level descriptive summaries; all prescribed settings retained."""

import argparse
import json
from pathlib import Path
import numpy as np
from scipy.stats import t


def interval(values):
    """Return the participant mean and t interval in percentage points."""
    percentages = np.asarray(values, dtype=float) * 100
    mean = float(percentages.mean())
    half_width = (
        float(
            t.ppf(0.975, len(percentages) - 1)
            * percentages.std(ddof=1)
            / np.sqrt(len(percentages))
        )
        if len(percentages) > 1
        else None
    )
    return dict(
        mean=mean,
        ci95=None if half_width is None else [mean - half_width, mean + half_width],
        participants=percentages.tolist(),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads((args.run / "results.json").read_text())
    config = data["config"]
    records = data["records"]
    assert (args.run / "complete.json").exists()
    participants = config["participants"]
    metrics = ["accuracy", "other_active_accuracy", "calibrated_recall"]
    conditions = [("frozen", 0)] + [
        (m, s) for m in config["methods"] for s in config["steps"]
    ]
    vectors = {}
    rows = []
    contrasts = []
    for k in [1, 2]:
        for method, steps in conditions:
            subset = [
                record
                for record in records
                if record["k"] == k
                and record["method"] == method
                and record["steps"] == steps
            ]
            row = dict(k=k, method=method, steps=steps)
            for metric in metrics:
                values = [
                    np.mean(
                        [
                            record[metric]
                            for record in subset
                            if record["participant"] == person
                        ]
                    )
                    for person in participants
                ]
                assert all(
                    sum(record["participant"] == person for record in subset)
                    == len(config["sessions"]) * len(config["choices"])
                    for person in participants
                )
                vectors[k, method, steps, metric] = np.array(values)
                row[metric] = interval(values)
            rows.append(row)
        for method, steps in conditions[1:]:
            for control in ["frozen", "source_ce", "replay"]:
                if method == control:
                    continue
                control_steps = 0 if control == "frozen" else steps
                if (k, control, control_steps, metrics[0]) not in vectors:
                    continue
                contrasts.append(
                    dict(
                        k=k,
                        method=method,
                        steps=steps,
                        control=control,
                        **{
                            metric: interval(
                                vectors[k, method, steps, metric]
                                - vectors[k, control, control_steps, metric]
                            )
                            for metric in metrics
                        },
                    )
                )
    diagnostics = []
    for method, steps in conditions[1:]:
        selected = [
            record["diagnostics"]
            for record in records
            if record["method"] == method and record["steps"] == steps
        ]
        item = dict(method=method, steps=steps)
        for metric in [
            "source_ce",
            "target_ce",
            "coral_raw",
            "update_l2",
            "seconds_including_prior_checkpoint_scoring",
        ]:
            values = [
                record[metric] for record in selected if record[metric] is not None
            ]
            if values:
                item[metric] = dict(
                    min=float(np.min(values)),
                    median=float(np.median(values)),
                    max=float(np.max(values)),
                )
        if method.startswith("coral"):
            values = [
                config["methods"][method]
                * record["coral_raw"]
                / max(record["source_ce"], 1e-30)
                for record in selected
            ]
            item["weighted_alignment_to_source_ce_ratio"] = dict(
                min=float(np.min(values)),
                median=float(np.median(values)),
                max=float(np.max(values)),
            )
        diagnostics.append(item)
    result = dict(
        scope=config["scope"],
        participants=participants,
        unit="percentage points; participant averages over sessions and choices",
        warning="Reused development data, one seed; descriptive intervals, no confirmatory testing or selected winner.",
        conditions=rows,
        contrasts=contrasts,
        diagnostics=diagnostics,
    )
    (args.run / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = [
        "# Published-objective development comparison",
        "",
        result["warning"],
        "",
        "Means below first average the sessions and fixed gesture choices within each participant.",
        "This adapts the Yuan/Deep CORAL objective to our backbone and task; it is not a Hyser or author-code reproduction.",
        "",
        "| K | Method | Steps | Overall (%) | Omitted active (%) | Calibrated recall (%) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for record in rows:
        lines.append(
            f"| {record['k']} | {record['method']} | {record['steps']} | "
            + " | ".join(f"{record[m]['mean']:.2f}" for m in metrics)
            + " |"
        )
    lines += [
        "",
        "## Paired descriptive differences",
        "",
        "| K | Method | Steps | Control | Overall difference, pp (95% interval) | Omitted difference, pp |",
        "|---|---|---:|---|---:|---:|",
    ]
    for record in contrasts:
        if not record["method"].startswith("coral"):
            continue
        bounds = record["accuracy"]["ci95"]
        interval_text = "" if bounds is None else f" ({bounds[0]:.2f}, {bounds[1]:.2f})"
        lines.append(
            f"| {record['k']} | {record['method']} | {record['steps']} | {record['control']} | {record['accuracy']['mean']:.2f}{interval_text} | {record['other_active_accuracy']['mean']:.2f} |"
        )
    lines += [
        "",
        "All tested weights and checkpoints are shown. No final participants were accessed.",
        "Saved checkpoints and window predictions permit independent replay; verification files record its actual scope.",
        "",
    ]
    (args.run / "SUMMARY.md").write_text("\n".join(lines))
    print(
        json.dumps(
            {
                "conditions": len(rows),
                "contrasts": len(contrasts),
                "participants": participants,
            }
        )
    )


if __name__ == "__main__":
    main()
