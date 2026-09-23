"""Reproduce selected means and Holm correction from saved participant summaries.

Python standard library only. This is not raw-data inference or model fitting.
Run from anywhere: python reproduce_summary.py
"""

import json
import statistics
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    data = json.loads((root / "statistics.json").read_text())
    rows = []
    for record in data["final_neural"]:
        if record["k"] == 2 and (
            record["method"] == "cnn_frozen"
            or (
                record["steps"] == 100
                and record["method"] in ["cnn_naive", "cnn_replay"]
            )
        ):
            mean = statistics.mean(record["per_person"]["accuracy"])
            assert abs(mean - record["accuracy"]) < 1e-12
            rows.append(
                {
                    "method": record["method"],
                    "participants": len(record["participants"]),
                    "accuracy_percent": 100 * mean,
                }
            )
    for key in ["final_selection_primary", "final_rotation_coverage_contrast"]:
        record = data[key]
        mean = statistics.mean(record["per_person"])
        assert abs(mean - record["mean"]) < 1e-12
        rows.append(
            {
                "contrast": key,
                "participants": record["n"],
                "difference_percentage_points": 100 * mean,
            }
        )
    ordered = sorted(data["primary_tests"].items(), key=lambda test: test[1]["raw_p"])
    previous_adjusted = 0
    for i, (name, test) in enumerate(ordered):
        adjusted = max(previous_adjusted, min(1, (len(ordered) - i) * test["raw_p"]))
        assert abs(adjusted - test["holm_p"]) < 1e-12
        previous_adjusted = adjusted
    print(
        json.dumps(
            {
                "passed": True,
                "results": rows,
                "primary_tests": data["primary_tests"],
                "scope": "Saved-summary arithmetic only; confidence intervals and raw predictions are not reconstructed by this script.",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
