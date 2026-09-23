"""Summarize verified scratch enrollment control without choosing checkpoints."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.grabmyo_corpus import hash_stream


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=Path("research/SCRATCH_CONVERGENCE_RESULTS.md"))
    args = parser.parse_args()
    config = json.loads((args.run / "config.json").read_text())
    prior = Path(config["prior_run"])
    validation = json.loads((args.run / "independent_validation.json").read_text())
    if validation["models"] != 336 or validation["epoch20_models_exactly_reproduced"] != 112:
        raise ValueError("Complete verification and exact epoch20 reproduction first")
    rows = json.loads((args.run / "metrics.json").read_text())
    previous = json.loads((prior / "metrics.json").read_text())
    if validation["scoring_sets"] != json.loads((prior / "independent_validation.json").read_text())["scoring_sets"]:
        raise ValueError("Changed comparison scoring sets")
    args.out.mkdir(parents=True, exist_ok=False)
    def select(epoch, mode, budget, shared=False):
        mode = "none" if budget == 0 else mode
        return [r for r in (previous if shared else rows) if r["mode"] == mode and r["budget"] == budget
                and (r["initialization"] == "shared" if shared else r["enrollment_epochs"] == epoch)]
    summary, table, paired = [], [], []
    for mode in ("full", "head"):
        for epoch, shared in ((20, False), (40, False), (80, False), (20, True)):
            label = f"{'Shared' if shared else 'Scratch'}, {epoch} enrollment epochs, {mode} update"
            cells = []
            for budget in (0, 1, 2, 3):
                selected = select(epoch, mode, budget, shared)
                if len(selected) != 16:
                    raise ValueError("Missing participant/session pairs")
                mean = float(np.mean([r["macro_f1"] for r in selected]))
                cells.append(f"{mean:.3f}")
                summary.append({"model": "shared" if shared else "scratch", "enrollment_epochs": epoch,
                                "mode": mode, "budget": budget, "calibration_seconds": 85 * budget, "mean_macro_f1": mean})
            table.append("| " + " | ".join([label] + cells) + " |")
        for budget in (0, 1, 2, 3):
            scratch80 = {(r["participant"], r["session"]): r["macro_f1"] for r in select(80, mode, budget)}
            for label, comparison in (("scratch80 minus scratch20", select(20, mode, budget)),
                                      ("scratch80 minus shared20", select(20, mode, budget, True))):
                for r in comparison:
                    paired.append({"contrast": label, "mode": mode, "budget": budget, "participant": r["participant"],
                                   "session": r["session"], "macro_f1_difference": scratch80[(r["participant"], r["session"])] - r["macro_f1"]})
    intervals = []
    for contrast in sorted({r["contrast"] for r in paired}):
        for budget in (0, 1, 2, 3):
            chosen = [r for r in paired if r["contrast"] == contrast and r["mode"] == "full" and r["budget"] == budget]
            people = sorted({r["participant"] for r in chosen})
            values = np.array([np.mean([r["macro_f1_difference"] for r in chosen if r["participant"] == p]) for p in people])
            boot = np.random.default_rng(20260906).choice(values, (10000, len(people)), replace=True).mean(1)
            low, high = np.quantile(boot, [.025, .975])
            intervals.append({"contrast": contrast, "budget": budget, "mean": float(values.mean()),
                              "participant_bootstrap_95_low": float(low), "participant_bootstrap_95_high": float(high),
                              "positive_pairs": sum(r["macro_f1_difference"] > 0 for r in chosen)})
    for name, data in (("summary", summary), ("paired_differences", paired), ("bootstrap_comparisons", intervals)):
        (args.out / f"{name}.json").write_text(json.dumps(data, indent=2) + "\n")
        with (args.out / f"{name}.csv").open("w") as f:
            writer = csv.DictWriter(f, fieldnames=list(data[0])); writer.writeheader(); writer.writerows(data)
    history = json.loads((args.run / "adaptation_history.json").read_text())
    final_histories = [r for r in history if r["stage"] == "enrollment" and r["enrollment_epochs"] == 80]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.7))
    for epoch, model, color in ((20, "scratch", "#bbb0d0"), (40, "scratch", "#9972ad"), (80, "scratch", "#58398b"), (20, "shared", "#18878a")):
        data = [r for r in summary if r["model"] == model and r["enrollment_epochs"] == epoch and r["mode"] == "full"]
        axes[0].plot([r["calibration_seconds"] for r in data], [r["mean_macro_f1"] for r in data], marker="o", color=color,
                     label=f"{model.capitalize()}: {epoch} enrollment epochs")
    axes[0].set(xlabel="New-day calibration recording (seconds)", ylabel="Mean macro-F1", ylim=(0, 1), xticks=[0, 85, 170, 255])
    for r in final_histories:
        axes[1].plot(range(1, 81), r["epoch_losses"], alpha=.65, linewidth=1, label=f"p{r['participant']}")
    axes[1].set(xlabel="Day-1 enrollment epoch", ylabel="Training cross-entropy loss", yscale="log")
    axes[0].set_title("Later-day performance: full updates", fontsize=11)
    axes[1].set_title("Scratch enrollment diagnostics", fontsize=11)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="y", alpha=.15)
        ax.legend(loc="upper center", bbox_to_anchor=(.5, -.24), ncol=2 if ax == axes[0] else 4, fontsize=8, frameon=False)
    fig.suptitle("Longer scratch training: eight development participants")
    fig.subplots_adjust(bottom=.31, top=.84, wspace=.30)
    fig.savefig(args.out / "scratch_convergence.png", dpi=180)
    fig.savefig(args.out / "scratch_convergence.svg")
    run_summary = json.loads((args.run / "summary.json").read_text())
    report = """# Longer scratch-training control

Public GRABMyo development evidence only. The 15 final participants remain untouched. This control tests whether a longer enrollment budget reduces the advantage attributed to shared pretraining; it does not establish an optimal scratch model or a novel method.

Each of eight development participants has one uninterrupted scratch-training trajectory, with Adam state and shuffle sequence preserved through 80 epochs. Epoch 80 is the predeclared primary endpoint, 40 is diagnostic, and 20 checks exact reproduction. Architecture, seed 42, normalization, day-1 data, calibration budgets and later-day scoring sets match the earlier run. Shared models below retain their original 20 enrollment epochs and additional population pretraining. This comparison is not matched total computation.

At each enrollment checkpoint, head-only and full updates restart independently from that checkpoint and use the original ten calibration epochs at learning rate 0.0001. Day-1 enrollment still costs 595 recorded seconds; longer training adds computation, not additional participant recordings. Calibration costs exclude rest and interaction overhead. These are offline 250 ms held-gesture observations.

## Mean macro-F1

Equal participant weighting, two target sessions per participant. All endpoints are reported; no per-participant checkpoint selection.

| Model | 0 s | 85 s | 170 s | 255 s |
| --- | ---: | ---: | ---: | ---: |
""" + "\n".join(table)
    report += "\n\n## Paired full-update comparisons\n\nIntervals resample eight participants with both sessions retained together (10,000 resamples; seed 20260906). Descriptive development uncertainty only; it excludes neural-seed uncertainty. Positive values favor scratch with 80 enrollment epochs.\n\n| Contrast | Calibration | Mean difference | 95% participant interval | Positive pairs |\n| --- | ---: | ---: | ---: | ---: |\n"
    for r in intervals:
        report += f"| {r['contrast']} | {85*r['budget']} s | {r['mean']:+.3f} | [{r['participant_bootstrap_95_low']:+.3f}, {r['participant_bootstrap_95_high']:+.3f}] | {r['positive_pairs']}/16 |\n"
    report += f"\n## Validation and runtime\n\nAll 336 checkpoint/adaptation variants and 799,680 repeated prediction rows were verified. The 112 epoch-20 models exactly reproduce the earlier weights and metrics. Verification checked fit/scoring access, fixed scoring sets, raw/cache records, training-group normalization and source/artifact hashes. Checkpoint tests show that evaluation snapshots cannot mutate the training trajectory and Adam is not restarted.\n\nThe run took {run_summary['total_seconds']/60:.2f} minutes on Apple M3/MPS, excluding initial input validation; peak process RSS {run_summary['peak_process_rss_bytes']/1e9:.2f} GB. It did not repeat population pretraining. Declining training loss alone does not prove improved generalization or full convergence.\n\n"
    report += f"![Longer training control]({(args.out/'scratch_convergence.png').resolve()})\n\n[Run artifacts]({args.run.resolve()}) · [Original comparison]({Path('research/SHARED_PRETRAINING_RESULTS.md').resolve()})\n"
    args.report.write_text(report)
    (args.out / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    (args.out / "artifact_sha256.json").write_text(json.dumps({p.name: hash_stream(p) for p in args.out.iterdir() if p.is_file()}, indent=2) + "\n")
    print(json.dumps([r for r in summary if r["mode"] == "full"], indent=2))


if __name__ == "__main__":
    main()
