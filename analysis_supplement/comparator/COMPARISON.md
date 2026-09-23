# Complete development comparison

Adaptive development-only comparison; all six weights retained; no confirmatory hypothesis test.

Eight participants; one initialization seed; four fixed gesture choices and two target sessions.
Additional smaller weights were chosen after the original loss diagnostics showed dominance of the alignment term.
Original run: RunPod CUDA/PyTorch 2.8.0+cu128. Follow-up: local MPS/PyTorch 2.14.0.
All 384 repeated frozen/source-only trial confusion matrices match; 12 window labels differ across devices.

| K | Method | Steps | Overall (%) | Omitted active (%) | Calibrated recall (%) |
|---|---|---:|---:|---:|---:|
| 1 | frozen | 0 | 77.48 | 76.25 | 84.38 |
| 1 | target_ce | 25 | 76.84 | 74.40 | 99.22 |
| 1 | target_ce | 100 | 73.53 | 70.73 | 99.61 |
| 1 | replay | 25 | 78.10 | 75.83 | 99.22 |
| 1 | replay | 100 | 78.52 | 76.25 | 99.61 |
| 1 | source_ce | 25 | 77.78 | 76.46 | 84.77 |
| 1 | source_ce | 100 | 78.17 | 76.80 | 84.77 |
| 1 | coral_0.0001 | 25 | 77.90 | 76.48 | 84.77 |
| 1 | coral_0.0001 | 100 | 78.24 | 76.95 | 83.59 |
| 1 | coral_0.001 | 25 | 77.69 | 76.22 | 85.16 |
| 1 | coral_0.001 | 100 | 78.19 | 76.82 | 84.77 |
| 1 | coral_0.01 | 25 | 77.21 | 75.57 | 86.72 |
| 1 | coral_0.01 | 100 | 77.87 | 76.41 | 85.16 |
| 1 | coral_0.1 | 25 | 76.88 | 75.16 | 87.50 |
| 1 | coral_0.1 | 100 | 76.63 | 75.03 | 85.16 |
| 1 | coral_1 | 25 | 76.61 | 74.87 | 87.11 |
| 1 | coral_1 | 100 | 73.71 | 71.88 | 82.81 |
| 1 | coral_10 | 25 | 76.56 | 74.82 | 87.11 |
| 1 | coral_10 | 100 | 72.66 | 70.68 | 82.81 |
| 2 | frozen | 0 | 77.48 | 76.73 | 76.95 |
| 2 | target_ce | 25 | 80.35 | 77.20 | 96.68 |
| 2 | target_ce | 100 | 77.16 | 73.24 | 98.83 |
| 2 | replay | 25 | 80.95 | 77.99 | 96.29 |
| 2 | replay | 100 | 81.41 | 78.40 | 97.66 |
| 2 | source_ce | 25 | 77.78 | 76.93 | 77.34 |
| 2 | source_ce | 100 | 78.17 | 77.29 | 77.34 |
| 2 | coral_0.0001 | 25 | 77.92 | 77.01 | 77.15 |
| 2 | coral_0.0001 | 100 | 78.19 | 77.34 | 77.15 |
| 2 | coral_0.001 | 25 | 77.76 | 76.73 | 77.73 |
| 2 | coral_0.001 | 100 | 78.35 | 77.37 | 78.32 |
| 2 | coral_0.01 | 25 | 77.02 | 75.92 | 77.15 |
| 2 | coral_0.01 | 100 | 77.96 | 76.90 | 78.12 |
| 2 | coral_0.1 | 25 | 76.68 | 75.45 | 77.54 |
| 2 | coral_0.1 | 100 | 76.42 | 75.25 | 76.56 |
| 2 | coral_1 | 25 | 76.45 | 75.17 | 77.54 |
| 2 | coral_1 | 100 | 73.64 | 72.29 | 73.83 |
| 2 | coral_10 | 25 | 76.42 | 75.14 | 77.54 |
| 2 | coral_10 | 100 | 72.56 | 70.93 | 73.63 |

All settings are descriptive. Paired participant differences and unadjusted intervals are in combined_summary.json.
These results adapt a published objective to this project’s backbone and data access; they do not reproduce the original Hyser/VGG accuracy.
