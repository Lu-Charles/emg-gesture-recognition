"""Generate a descriptive report from the verified public development pilots."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    root = Path("research/runs")
    classical = root / "20260906_grabmyo_classical_pilot_v1"
    neural = root / "20260906_grabmyo_neural_pilot_v1"
    for run in (classical, neural):
        if not (run / "independent_validation.json").exists():
            raise ValueError("Verify both pilots before reporting")
    lda = json.loads((classical / "metrics.json").read_text())
    cnn = json.loads((neural / "metrics.json").read_text())
    out = root / "20260906_grabmyo_pilot_report_v1"
    out.mkdir(parents=True, exist_ok=False)
    methods = [
        ("LDA: day 1 + calibration", lda, "source_plus_calibration", "#167d8d", "o"),
        ("LDA: calibration only", lda, "calibration_only", "#8b6515", "s"),
        ("CNN: full update", cnn, "full", "#6548a3", "o"),
        ("CNN: head update", cnn, "head", "#bf5666", "s"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8), sharey=True)
    for ax, p in zip(axes, (3, 7)):
        for label, data, method, color, marker in methods:
            entries = sorted([r for r in data if r["participant"] == p and r["method"] == method], key=lambda r:r["budget"])
            if method != "calibration_only":
                entries = [r for r in data if r["participant"] == p and r["method"] == "source_only"] + entries
            ax.plot([r["calibration_recording_seconds"] for r in entries], [r["macro_f1"] for r in entries],
                    color=color, marker=marker, linewidth=1.8, markersize=4, label=label)
        ax.set_title(f"Participant {p}", fontsize=11)
        ax.set_xlabel("New-day calibration recording (seconds)", fontsize=9)
        ax.set_xticks([0, 85, 170, 255])
        ax.set_ylim(0, 1)
        ax.grid(axis="y", color="#e6e6e6")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Macro-F1 on fixed day-2 scoring trials")
    fig.suptitle("Public EMG development pilot: day 1 to day 2", fontsize=13)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=2, frameon=False, fontsize=9)
    fig.subplots_adjust(bottom=.24, top=.83, wspace=.15)
    fig.savefig(out / "calibration.png", dpi=180, facecolor="white")
    fig.savefig(out / "calibration.svg", facecolor="white")
    rows = []
    for label, data, method, _, _ in methods:
        values = []
        for budget in (0, 1, 2, 3):
            key = "source_only" if budget == 0 and method != "calibration_only" else method
            found = [r["macro_f1"] for r in data if r["method"] == key and r["budget"] == budget]
            values.append(f"{np.mean(found):.3f}" if found else "—")
        rows.append("| " + " | ".join([label] + values) + " |")
    report = """# Public-data development pilot

This is a two-person development experiment, not the final study or a publication-ready result. It uses GRABMyo v1.1.0 only. Charles's custom recordings are excluded.

Day 1 enrollment contains 119 five-second recordings/person. Day 2 has a fixed 68-trial scoring set/person and a separate calibration pool. Each trial contributes 35 correlated 250 ms windows; uncertainty cannot be estimated by treating these as independent participants.

The CNN is a 40,689-parameter model trained separately on each person's day 1. Shared pretraining across the 20 training participants has not been run. Standardization uses day 1 only and stays fixed during neural updates. LDA uses MAV/RMS/mean-waveform-length features and shrinkage; its scaler fits only its permitted training data.

## Average participant macro-F1

| Method | 0 seconds | 85 seconds | 170 seconds | 255 seconds |
| --- | ---: | ---: | ---: | ---: |
""" + "\n".join(rows) + """

Calibration recording time excludes interaction overhead. Initial day 1 enrollment costs 595 seconds/person. Head/full CNN updates start from the identical day 1 checkpoint for each budget. LDA and CNN use identical scoring trials and calibration allocations. No settings were selected on final participants; none of their signal data were accessed.

## Interpretation

Adding new-day calibration improves the pooled LDA baseline for both development participants. The initial CNN improves with updating but remains behind LDA at each matched budget in this pilot. Full neural updating performs better than head-only updating here. These results support studying calibration efficiency but do not demonstrate a neural advantage, novelty, or the need for an update safeguard.

Next: train the shared encoder on the separate 20-person training group, then compare it on development participants using the same calibration/scoring sets. Keep the strong classical baseline and all current negative results.

## Verification

Eight public-data tests and three neural risk tests passed. Separate readback verification rebuilt raw features/scaling, checked actual training/calibration/scoring access, regenerated predictions from saved models, and recomputed metrics. Each method family has 14 models and 33,320 saved scoring predictions, including repeated scoring across budgets. No clinical or live-control validation was performed.

"""
    image_path = (out / "calibration.png").resolve()
    report += f"![Per-participant development curves]({image_path})\n\n"
    for name, path in (("Classical run", classical), ("Neural run", neural)):
        report += f"- [{name}]({path.resolve()})\n"
    Path("research/GRABMYO_PILOT_RESULTS.md").write_text(report)
    print(str(image_path))


if __name__ == "__main__":
    main()
