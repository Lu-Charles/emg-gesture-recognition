# Fixed-signal gesture coverage: initial development screen

Completed 2026-09-09T18:46:56.137018+00:00. Development evidence only: eight GRABMyo people, two later days per person, 16 predetermined subset orders. No final participants accessed.

This screen uses amplitude-inclusive LDA and a simple source/target feature-mean gain correction. It is not SAL, a new algorithm, a CNN comparison or a publishability result.

| Active calibration gestures | Frozen accuracy | Gain correction | Gain, shrinkage 1 | Gain, shrinkage 4 | Archived recording seconds |
|---|---:|---:|---:|---:|---:|
| 1 | 54.14% | 54.03% | 59.50% | 57.08% | 5 |
| 2 | 54.14% | 60.04% | 60.81% | 57.40% | 10 |
| 4 | 54.14% | 64.34% | 61.43% | 57.67% | 20 |
| 8 | 54.14% | 66.11% | 62.21% | 57.68% | 40 |
| 16 | 54.14% | 67.02% | 62.25% | 57.90% | 80 |

All conditions use exactly 16 nonoverlapping 250-ms calibration windows: four seconds of analyzed signal, distributed among active gestures. Rest is not calibrated and remains in the fixed 17-class scoring vocabulary. Each score is an equal participant mean after averaging days and orders. Every target scoring set contains the same four trials per class, with majority voting over the existing 35 windows per trial. Lowest label resolves a vote tie in this new screen, unlike the preserved CSL first-occurrence rule.

Unregularized gain correction improves from 54.03% at one gesture to 67.02% at sixteen, with frozen at 54.14%. Shrinkage 1 helps the single-gesture case (59.50%) but reduces the sixteen-gesture result (62.25%). This motivates testing whether regularization should depend on available coverage; it does not establish such a policy. No setting has been promoted to final evaluation.

The K=16 condition touches 80 seconds of archived recordings, compared with five seconds for K=1. Therefore the result isolates selected signal volume, not collection burden. Coverage changes both class identity and per-class observation duration. Existing trial allocations permit separate whole-recording cost controls; see the protocol.

## Actual validation and reproducibility

Four tests passed: collapse versus perfect recall/poor precision, scrambled permutation versus collapse, exact signal budget and nonoverlapping windows, and scoring/final-cohort access rejection. The run enforces the existing development participant IDs, calibration rank 1 and whole-trial role separation.

The separate verifier reproduced accuracy and macro-F1 for all 5,120 saved records (348,160 trial predictions), checked fixed scoring identities and frozen predictions across budgets, and independently reconstructed 35 amplitude-feature windows in each of 32 selected raw trials. Raw features agree within the stated floating-point tolerance. This does not independently replay all fitted LDA model predictions; that remains a reproducibility requirement before final claims. No optimization-based adaptation or final hypothesis test was performed.

Commands actually run from .:

```sh
.venv-public/bin/python -B -m unittest tests.test_coverage_diagnostics -v
.venv-public/bin/python -B -m scripts.pilot_gesture_coverage
.venv-public/bin/python -B -m scripts.verify_coverage_development
```

Run artifacts: research/runs/20260909_coverage_development_v1/protocol.json, access_manifest.json, results.json, trial_predictions.npz, verification.json, environment.json and source snapshots. The pilot refuses to overwrite its default run directory. Raw inputs are the existing versioned development cache; hashes are in verification.json.

Next: use these exact development splits for ordinary CNN fine-tuning and source replay with equal target-example exposure, and add a strong TDAR+RMS baseline before evaluating source-only gesture selection.
