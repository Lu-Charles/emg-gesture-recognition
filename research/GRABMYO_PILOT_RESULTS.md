# Public-data development pilot

This is a two-person development experiment, not the final study or a publication-ready result. It uses GRABMyo v1.1.0 only. Charles's custom recordings are excluded.

Day 1 enrollment contains 119 five-second recordings/person. Day 2 has a fixed 68-trial scoring set/person and a separate calibration pool. Each trial contributes 35 correlated 250 ms windows; uncertainty cannot be estimated by treating these as independent participants.

The CNN is a 40,689-parameter model trained separately on each person's day 1. Shared pretraining across the 20 training participants has not been run. Standardization uses day 1 only and stays fixed during neural updates. LDA uses MAV/RMS/mean-waveform-length features and shrinkage; its scaler fits only its permitted training data.

## Average participant macro-F1

| Method | 0 seconds | 85 seconds | 170 seconds | 255 seconds |
| --- | ---: | ---: | ---: | ---: |
| LDA: day 1 + calibration | 0.738 | 0.804 | 0.831 | 0.849 |
| LDA: calibration only | — | 0.752 | 0.824 | 0.844 |
| CNN: full update | 0.601 | 0.708 | 0.748 | 0.766 |
| CNN: head update | 0.601 | 0.640 | 0.688 | 0.716 |

Calibration recording time excludes interaction overhead. Initial day 1 enrollment costs 595 seconds/person. Head/full CNN updates start from the identical day 1 checkpoint for each budget. LDA and CNN use identical scoring trials and calibration allocations. No settings were selected on final participants; none of their signal data were accessed.

## Interpretation

Adding new-day calibration improves the pooled LDA baseline for both development participants. The initial CNN improves with updating but remains behind LDA at each matched budget in this pilot. Full neural updating performs better than head-only updating here. These results support studying calibration efficiency but do not demonstrate a neural advantage, novelty, or the need for an update safeguard.

Next: train the shared encoder on the separate 20-person training group, then compare it on development participants using the same calibration/scoring sets. Keep the strong classical baseline and all current negative results.

## Verification

Eight public-data tests and three neural risk tests passed. Separate readback verification rebuilt raw features/scaling, checked actual training/calibration/scoring access, regenerated predictions from saved models, and recomputed metrics. Each method family has 14 models and 33,320 saved scoring predictions, including repeated scoring across budgets. No clinical or live-control validation was performed.

![Per-participant development curves](runs/20260906_grabmyo_pilot_report_v1/calibration.png)

- [Classical run](runs/20260906_grabmyo_classical_pilot_v1)
- [Neural run](runs/20260906_grabmyo_neural_pilot_v1)
