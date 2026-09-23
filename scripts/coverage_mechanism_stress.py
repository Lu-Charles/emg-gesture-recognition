"""Post-hoc synthetic stress test of the conditional shared-gain claim, not human evidence."""

from pathlib import Path
import json
from datetime import datetime, timezone
import numpy as np
from scripts.rotation_coverage import profile_rotation
from src.grabmyo_corpus import hash_stream

O = Path(__file__).resolve().parents[1] / "research/runs/20260916_mechanism_stress"


def main():
    O.mkdir(exist_ok=False)
    cfg = dict(
        scope="post-hoc synthetic mechanism qualification; final human protocol unaffected",
        timestamp=datetime.now(timezone.utc).isoformat(),
        seed=20260916,
        replicates=500,
        scenarios=["shared_diverse", "shared_proportional", "gesture_specific_gains"],
        channels=8,
        gestures=7,
        gain_log_uniform=[-0.3, 0.3],
        misspecification_log_sd=0.5,
        code_sha256=hash_stream(__file__),
    )
    (O / "protocol.json").write_text(json.dumps(cfg, indent=2) + "\n")
    rng = np.random.default_rng(cfg["seed"])
    records = []
    for trial in range(cfg["replicates"]):
        u = np.exp(rng.normal(0, 0.6, (7, 8)))
        true = int(rng.integers(-4, 4))
        gain = np.exp(rng.uniform(-0.3, 0.3, 8))
        noise = rng.normal(0, 0.5, (7, 8))
        for scenario in cfg["scenarios"]:
            source = u.copy()
            if scenario == "shared_proportional":
                source[1] = 2 * source[0]
            target = np.roll(source / gain, -true, axis=1)
            if scenario == "gesture_specific_gains":
                target *= np.exp(noise)
            for k in [1, 2]:
                shift, g, cost = profile_rotation(source[:k], target[:k], False)
                corrected = np.roll(target, int(shift), axis=1) * g
                other = np.sqrt(
                    np.mean((np.log(corrected[k:]) - np.log(source[k:])) ** 2)
                )
                costs = []
                for s in range(-4, 4):
                    residual = np.log(source[:k]) - np.log(
                        np.roll(target[:k], s, axis=1)
                    )
                    costs.append(float(np.mean((residual - residual.mean(0)) ** 2)))
                admissible = sum(v <= min(costs) + 1e-10 for v in costs)
                if scenario == "shared_diverse" and k == 2:
                    assert shift == true and other < 1e-12 and admissible == 1
                if k == 1 or scenario == "shared_proportional":
                    assert admissible == 8
                records.append(
                    dict(
                        trial=trial,
                        scenario=scenario,
                        k=k,
                        true_shift=true,
                        estimated_shift=shift,
                        correct=shift == true,
                        template_cost=cost,
                        ambiguous_minima=admissible,
                        other_log_rms_error=float(other),
                    )
                )
    summary = [
        dict(
            scenario=s,
            k=k,
            correct=float(
                np.mean(
                    [
                        r["correct"]
                        for r in records
                        if r["scenario"] == s and r["k"] == k
                    ]
                )
            ),
            other_log_rms_error=float(
                np.mean(
                    [
                        r["other_log_rms_error"]
                        for r in records
                        if r["scenario"] == s and r["k"] == k
                    ]
                )
            ),
            ambiguous_minima=float(
                np.mean(
                    [
                        r["ambiguous_minima"]
                        for r in records
                        if r["scenario"] == s and r["k"] == k
                    ]
                )
            ),
        )
        for s in cfg["scenarios"]
        for k in [1, 2]
    ]
    (O / "results.json").write_text(
        json.dumps(dict(config=cfg, records=records, summary=summary), indent=2) + "\n"
    )
    (O / "verification.json").write_text(
        json.dumps(
            dict(
                passed=True,
                known_transform_checks=500,
                ambiguous_cases=2000,
                scope="asserted generator-ground-truth properties; simulation only",
            )
        )
        + "\n"
    )
    print(summary)


if __name__ == "__main__":
    main()
