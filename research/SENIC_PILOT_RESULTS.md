# SeNic calibration-efficiency pilot

Development results only. The selected route is a focused public-data study of brief recalibration after real electrode rotation. No new algorithm, independent final result or publication readiness is established.

## Protocol

Six development participants: h0,h1,h14,h18,h24,h25; session0 only. Source position0 repetitions0,1 enroll the classifier; repetition2 scores every condition. Target calibration uses one labeled gesture (all seven choices averaged), one repetition of all seven gestures, or two repetitions. All method/budget comparisons use identical scoring files. Positions1–10 include designed rotations and two random positions. The other rotation participants remain reserved; fatigue participants are excluded.

Raw eight-channel Myo counts at200Hz; fixed3–5second interval; 250ms windows with125ms step,15windows/trial. Costs below include entire CSV durations, not just the2seconds analyzed. Three recordings were shorter than6seconds, so the initial run stopped before model fitting; all trials received the same shorter interval in v2. No selective trial exclusion.

## Main results

Equal participant weighting, averaged across ten target positions. One-gesture results average every gesture choice, not the best-performing gesture. Accuracy is per held-gesture window, not online control success.

| Features | Method | Calibration trials | Recorded seconds | Accuracy | Macro F1 |
|---|---|---:|---:|---:|---:|
| amplitude | Frozen LDA | 0 | 0.0 | 23.75% | 0.1761 |
| amplitude | Source + target refit | 1 | 7.5 | 28.59% | 0.2144 |
| amplitude | Source + target refit | 7 | 52.3 | 62.43% | 0.5909 |
| amplitude | Source + target refit | 14 | 104.8 | 76.24% | 0.7458 |
| amplitude | RMS spatial registration | 1 | 7.5 | 57.58% | 0.5305 |
| amplitude | RMS spatial registration | 7 | 52.3 | 64.21% | 0.6037 |
| amplitude | RMS spatial registration | 14 | 104.8 | 64.52% | 0.6075 |
| amplitude | Target-only refit | 7 | 52.3 | 79.14% | 0.7726 |
| amplitude | Target-only refit | 14 | 104.8 | 86.68% | 0.8573 |
| tdar_rms | Frozen LDA | 0 | 0.0 | 27.89% | 0.2208 |
| tdar_rms | Source + target refit | 1 | 7.5 | 32.02% | 0.2498 |
| tdar_rms | Source + target refit | 7 | 52.3 | 68.06% | 0.6674 |
| tdar_rms | Source + target refit | 14 | 104.8 | 79.46% | 0.7873 |
| tdar_rms | Target-only refit | 7 | 52.3 | 80.19% | 0.7874 |
| tdar_rms | Target-only refit | 14 | 104.8 | 88.00% | 0.8728 |

## Participant results, TDAR + RMS

| Participant | Frozen | 7 trials | 14 trials |
|---|---:|---:|---:|
| h0 | 19.24% | 84.19% | 91.90% |
| h1 | 24.86% | 71.43% | 82.19% |
| h14 | 17.14% | 88.67% | 92.95% |
| h18 | 13.52% | 75.24% | 82.57% |
| h24 | 48.29% | 78.67% | 88.48% |
| h25 | 44.29% | 82.95% | 89.90% |

## Interpretation

The unshifted held-out reference averages 90.95% accuracy. At target positions, seven-trial target-only recalibration improves accuracy by 52.30 percentage points over frozen, with at least10points improvement for 6/6people. The effect is large, but ordinary supervised refitting is an established method.

Seven trials save 50.1% of the fourteen-trial recording duration but lose 7.81 accuracy points. The exploratory paired participant-bootstrap interval for seven minus fourteen is [-9.43, -6.19] points. The pre-outcome3point short-budget screen does not pass. No participant is within3points. Do not claim essentially unchanged performance at half the recording cost.

Single-gesture spatial registration is substantially better than the frozen amplitude baseline and a one-gesture source-plus-target refit, but remains below all-gesture target-only retraining. This baseline is simple circular interpolation of amplitude features using labeled RMS templates; it is not a literal reproduction of Li/Xu and cannot establish superiority over published shift correction.

As a planned descriptive regime check, small-rotation participants h24,h25 at positions1–8 are shown separately below. These are only two people, so their result is not a general small-shift conclusion. No significance test or model choice is based on this subgroup.

| Small-rotation subgroup method | Accuracy |
|---|---:|
| amplitude, RMS spatial registration, 1 trials | 56.67% |
| tdar_rms, Frozen LDA, 0 trials | 51.96% |
| tdar_rms, Target-only refit, 7 trials | 79.35% |
| tdar_rms, Target-only refit, 14 trials | 87.56% |

## Validation and limitations

Three risk tests pass. Independent audit verified 1386 raw hashes with no duplicate files, 606 whole-trial allocations, training-only scalers for 1332 saved models, and all 209,160 saved prediction rows in 1992 files. All15aggregate groups were recalculated. Reserved-participant signals and Charles recordings were not accessed.

Successful pilot wall time: 13.39seconds; independent audit: 5.55seconds. These exclude implementation, downloads and the failed first extraction attempt. This is a classical feature/LDA experiment, not neural training.

Only six development people and one scoring repetition per position. Multiple rotation positions do not create additional independent participants. Position order can confound angle with elapsed time/fatigue; approximately360-degree positions are physically near the original orientation. No cross-day conclusion from this run, no rest/unknown-gesture rejection task, no real-time prosthesis demonstration. Exact final cohort completeness remains to audit. Raw data redistribution rights are not established; provide retrieval instructions rather than copying raw data into a public package.

## Next action

Complete a matched-budget compact CNN and applicable published rotation-correction/augmentation comparison on these same development splits. Preserve the14trial reference and adverse short-budget result. Continue this calibration-efficiency study without claiming a winning new method; freeze the final protocol before reserved-participant evaluation.

## Provenance

- Dataset: https://github.com/BoZhuBo/SeNic at a4c12f7daab28a80d557677ae8dbcef0d7871ba2.
- Dataset paper: https://doi.org/10.1109/TNSRE.2022.3173708.
- Prior adaptive correction: https://doi.org/10.1109/JBHI.2020.3012698; synchronous gesture correction: https://pmc.ncbi.nlm.nih.gov/articles/PMC7070560/.
- Audit metadata and download hashes: research/runs/20260907_sensor_shift_audit/.
- Config, code snapshots, environment, trial/fit ledgers, features, models and predictions: research/runs/20260907_sensor_shift_pilot_v2.
- Source snapshots and hashes preserve the evaluated implementation.
