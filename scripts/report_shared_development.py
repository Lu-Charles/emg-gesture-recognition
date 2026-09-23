"""Report verified eight-person development results, preserving all comparisons."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--neural", type=Path, default=Path("research/runs/20260906_shared_pretraining_v2"))
    parser.add_argument("--classical", type=Path, default=Path("research/runs/20260906_classical_development_v1"))
    parser.add_argument("--out", type=Path, default=Path("research/runs/20260906_shared_development_report_v1"))
    parser.add_argument("--report", type=Path, default=Path("research/SHARED_PRETRAINING_RESULTS.md"))
    args = parser.parse_args()
    neural, classical = args.neural, args.classical
    nv = json.loads((neural / "independent_validation.json").read_text())
    cv = json.loads((classical / "independent_validation.json").read_text())
    if nv["scoring_sets"] != cv["scoring_sets"]:
        raise ValueError("Neural/classical scoring sets differ")
    nr = json.loads((neural / "metrics.json").read_text())
    cr = json.loads((classical / "metrics.json").read_text())
    out = args.out
    out.mkdir(exist_ok=False)
    methods = [
        ("LDA: day 1 + calibration", "LDA", "source_plus_calibration"),
        ("LDA: calibration only", "LDA", "calibration_only"),
        ("RF: day 1 + calibration", "RF", "source_plus_calibration"),
        ("RF: calibration only", "RF", "calibration_only"),
        ("CNN random: full update", "random", "full"),
        ("CNN shared: full update", "shared", "full"),
        ("CNN random: head update", "random", "head"),
        ("CNN shared: head update", "shared", "head"),
    ]
    def selected(family, method, budget):
        if family in ("LDA", "RF"):
            key = "source_only" if budget == 0 and method != "calibration_only" else method
            return [r for r in cr if r["family"] == family and r["method"] == key and r["budget"] == budget]
        key = "none" if budget == 0 else method
        return [r for r in nr if r["initialization"] == family and r["mode"] == key and r["budget"] == budget]
    summary = []
    table = []
    for label, family, method in methods:
        cells = []
        for budget in (0, 1, 2, 3):
            rows = selected(family, method, budget)
            if not rows:
                cells.append("—"); continue
            if len(rows) != 16:
                raise ValueError("Expected eight people and two target sessions")
            value = float(np.mean([r["macro_f1"] for r in rows]))
            cells.append(f"{value:.3f}")
            summary.append({"method": label, "budget": budget, "calibration_seconds": 85 * budget,
                            "mean_macro_f1": value, "participant_session_pairs": len(rows)})
        table.append("| " + " | ".join([label] + cells) + " |")
    paired = []
    for mode in ("head", "full"):
        for budget in (0, 1, 2, 3):
            a = {(r["participant"], r["session"]): r["macro_f1"] for r in selected("shared", mode, budget)}
            b = {(r["participant"], r["session"]): r["macro_f1"] for r in selected("random", mode, budget)}
            for (p, s), value in a.items():
                paired.append({"participant": p, "session": s, "mode": mode, "budget": budget,
                               "shared_minus_random_macro_f1": value - b[(p, s)]})
    # Resample people, keeping both later sessions together. These intervals
    # describe development variation; they are not final confirmatory tests.
    contrasts = []
    for budget in (0, 1, 2, 3):
        shared = {(r["participant"], r["session"]): r["macro_f1"] for r in selected("shared", "full", budget)}
        for label, family, method in (("random CNN", "random", "full"), ("pooled LDA", "LDA", "source_plus_calibration"),
                                      ("calibration-only LDA", "LDA", "calibration_only"), ("pooled RF", "RF", "source_plus_calibration")):
            other = {(r["participant"], r["session"]): r["macro_f1"] for r in selected(family, method, budget)}
            if not other:
                continue
            differences = {key: shared[key] - value for key, value in other.items()}
            people = sorted({p for p, s in differences})
            person_differences = np.array([np.mean([v for (person, s), v in differences.items() if person == p]) for p in people])
            boot = np.random.default_rng(20260906).choice(person_differences, (10000, len(people)), replace=True).mean(1)
            low, high = np.quantile(boot, [.025, .975])
            contrasts.append({"comparator": label, "budget": budget, "mean_shared_full_minus_comparator": float(person_differences.mean()),
                              "participant_bootstrap_95_low": float(low), "participant_bootstrap_95_high": float(high),
                              "better_session_pairs": sum(v > 0 for v in differences.values()), "total_session_pairs": len(differences),
                              "worst_session_difference": min(differences.values())})
    (out / "paired_baseline_comparisons.json").write_text(json.dumps(contrasts, indent=2) + "\n")
    for file, rows in (("summary.csv", summary), ("paired_pretraining_effects.csv", paired)):
        with (out / file).open("w") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.9))
    colors = ["#167d8d", "#167d8d", "#99721d", "#99721d", "#aaa0c5", "#6548a3"]
    for i, (label, family, method) in enumerate(methods[:6]):
        entries = [r for r in summary if r["method"] == label]
        axes[0].plot([r["calibration_seconds"] for r in entries], [r["mean_macro_f1"] for r in entries],
                     label=label, color=colors[i], marker="o", markersize=3,
                     linestyle="--" if method == "calibration_only" else "-")
    for mode, color in (("full", "#6548a3"), ("head", "#c06071")):
        entries = [float(np.mean([r["shared_minus_random_macro_f1"] for r in paired if r["mode"] == mode and r["budget"] == b])) for b in (0, 1, 2, 3)]
        axes[1].plot([0, 85, 170, 255], entries, marker="o", label=f"{mode.capitalize()} update", color=color)
    axes[0].set_ylabel("Mean macro-F1")
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Accuracy versus calibration cost", fontsize=11)
    axes[1].set_ylabel("Shared minus random initialization (macro-F1)")
    axes[1].axhline(0, color="#888888", linestyle="--", linewidth=1)
    axes[1].set_title("Effect of pretraining with matched settings", fontsize=11)
    for ax in axes:
        ax.set_xlabel("New-day calibration recording (seconds)")
        ax.set_xticks([0, 85, 170, 255]); ax.grid(axis="y", color="#eeeeee")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(loc="upper center", bbox_to_anchor=(.5, -.22), fontsize=8, ncol=2, frameon=False)
    axes[1].legend(loc="upper center", bbox_to_anchor=(.5, -.22), fontsize=9, ncol=2, frameon=False)
    fig.suptitle("Public-data development results: 8 people, 2 later days", fontsize=13)
    fig.subplots_adjust(bottom=.31, top=.83, wspace=.33)
    fig.savefig(out / "development_comparison.png", dpi=180, facecolor="white")
    fig.savefig(out / "development_comparison.svg", facecolor="white")
    ns = json.loads((neural / "summary.json").read_text())
    report = """# Shared pretraining: development results

This experiment uses GRABMyo v1.1.0 public data only. It is development evidence, not the final evaluation or a demonstrated novel method. The 15 final participants remain untouched.

The compact CNN has 40,689 parameters. It learns from all three days of the 20 representation-training participants, then enrolls each of eight separate development participants using their day-1 recordings. Shared and random initialization use identical training-group normalization, enrollment settings and later-day adaptation settings. Neither shared normalization nor representation training sees development participants.

Each target day uses one, two or three whole calibration trials per class and the same four scoring trials per class for every method. Day-1 enrollment uses 595 seconds of recorded data. Additional calibration costs below exclude rest and interaction overhead. Predictions use 250 ms held-gesture observations; this does not establish live-control latency or clinical benefit.

## Mean macro-F1

Equal weighting across eight participants, each with two target sessions. Windows within trials and sessions within participants are correlated.

| Method | 0 s | 85 s | 170 s | 255 s |
| --- | ---: | ---: | ---: | ---: |
""" + "\n".join(table) + "\n\n"
    report += "## Effect of pretraining\n\n"
    for budget in (0, 1, 2, 3):
        differences = [r["shared_minus_random_macro_f1"] for r in paired if r["mode"] == "full" and r["budget"] == budget]
        report += f"- {85 * budget} seconds, full-update comparison: mean difference {np.mean(differences):+.4f}; positive for {sum(v > 0 for v in differences)}/16 participant/session pairs.\n"
    report += "\n## Paired baseline comparisons\n\nShared CNN with full updating minus each comparator. Intervals resample the eight participants, retaining both target days together (10,000 resamples, seed 20260906). These descriptive intervals reflect participant variation within this development sample; they do not cover model-seed uncertainty or establish a final result.\n\n| Comparator | Calibration | Mean difference | 95% participant bootstrap interval | Better session pairs |\n| --- | ---: | ---: | ---: | ---: |\n"
    for row in contrasts:
        report += f"| {row['comparator']} | {85 * row['budget']} s | {row['mean_shared_full_minus_comparator']:+.3f} | [{row['participant_bootstrap_95_low']:+.3f}, {row['participant_bootstrap_95_high']:+.3f}] | {row['better_session_pairs']}/16 |\n"
    report += f"""
These are descriptive development comparisons; no significance or population-wide claim is made. Retain all per-participant results and compare against both pooled and calibration-only classical baselines. Do not choose a winner separately for each final participant.

## Execution and verification

Shared pretraining completed in {ns['pretraining_seconds'] / 60:.2f} minutes on Apple M3/MPS. The neural run including development enrollment and adaptation took {ns['total_seconds'] / 60:.2f} minutes, excluding retrieval and cache creation. Peak process RSS was {ns['peak_process_rss_bytes'] / 1e9:.2f} GB; this is a process measurement, not total machine or GPU peak memory.

An earlier attempt was interrupted before completing an epoch because scattered memory-mapped reads were slow. The completed run used shuffled buffers of 128 trials, with each window visited once per epoch. The interrupted configuration and source remain preserved.

Readback verification checked all saved model predictions, fit/calibration/scoring identities and metrics. It rebuilt shared scaling from the training corpus, verified complete cache hashes, and compared raw/cache values for every permitted participant and session. Head-only encoder weights stayed unchanged. Classical and neural scoring sets match exactly.

"""
    report += f"![Development comparisons]({(out / 'development_comparison.png').resolve()})\n\n"
    report += f"- [Neural run]({neural.resolve()})\n- [Classical run]({classical.resolve()})\n- [Paired effects]({(out / 'paired_pretraining_effects.csv').resolve()})\n"
    args.report.write_text(report)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
