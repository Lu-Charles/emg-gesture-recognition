# SeNic neural and rotation-augmentation comparison

Development evidence only: six people, session0, seed42. The study remains calibration efficiency after real electrode rotation. This stage tests stronger comparators; it does not introduce a new method or establish final publication readiness.

## Current interpretation

The amplitude-inclusive classical baseline remains strongest among the tested configurations: 80.19% at seven trials and 88.00% at fourteen. After 600 steps, plain CNN target fine-tuning reaches 69.56%/79.81%, and target-only scratch training reaches 70.68%/81.38%. Every extended fit exceeds 95% training accuracy, so the original fine-tuning shortfall was partly an optimization issue, but longer training did not close the held-out gap. These are six-person development results, not a universal ranking.

At the original matched step counts, nominal ±45-degree augmentation improves aggregate frozen accuracy from 27.27% to 33.83%, but reduces calibrated accuracy and fits its own data poorly in many cases. Its superiority or failure is unresolved until a matched longer-training control; this implementation is an adaptation of the published principle. The initial figure calibration_comparison.png uses 150/300-step results only.

Next: one bounded matched 600-step augmentation control on these development participants, retaining the current plain-CNN and LDA references. Use fitting diagnostics to assess adequacy and preserve all outcomes. Final cohort signals remain unscored. This is a calibration benchmark in development; a distinct publication contribution is not yet established.

## Methods and scope

Same original enroll/calibration/scoring trials and3–5second signal interval as the classical pilot. The CNN retains amplitude using an explicit log1pMAV path alongside temporal convolutions. Fixed300steps for enrollment/scratch and150steps for fine-tuning, Adam.001, minibatch64. No scoring-based model selection. One scalar RMS is fitted on permitted unique training samples. Fine-tuning keeps source normalization; target-only scratch fits target normalization.

Primary families: plainCNN and identicalCNN with circular channel permutations−1,0,+1 during training, approximating±45degrees. A separate zero-calibration sensitivity uses all8cyclic shifts. This follows the published ABSDA permutation principle, but changes the sensor, representation, backbone and training; it is not a reproduction of original HD-EMG scores. Physical ruler angles never enter a model. See SENIC_NEURAL_METHODS.md.

This neural stage covers0/7/14trials. One-gesture cases remain classical-only. Fine-target reuses the source model but updates on target data; fine-pooled updates using source+target; scratch-target trains anew using target data. All methods have identical target recording budgets and scoring files.

## Initial fixed-step target-position results

Equal participant weighting, then ten positions per participant. Recording time includes complete calibration files. Window accuracy is not real-time control success.

| Model | Training strategy | Trials | Recording seconds | Accuracy | Macro F1 |
|---|---|---:|---:|---:|---:|
| TDAR+RMS LDA | frozen | 0 | 0.0 | 27.89% | 0.2208 |
| TDAR+RMS LDA | target_only | 7 | 52.3 | 80.19% | 0.7874 |
| TDAR+RMS LDA | target_only | 14 | 104.8 | 88.00% | 0.8728 |
| cnn | fine_pooled | 7 | 52.3 | 54.43% | 0.5245 |
| cnn | fine_pooled | 14 | 104.8 | 61.16% | 0.6004 |
| cnn | fine_target | 7 | 52.3 | 61.79% | 0.6021 |
| cnn | fine_target | 14 | 104.8 | 68.52% | 0.6773 |
| cnn | frozen | 0 | 0.0 | 27.27% | 0.2267 |
| cnn | scratch_target | 7 | 52.3 | 70.35% | 0.6944 |
| cnn | scratch_target | 14 | 104.8 | 80.89% | 0.8018 |
| cnn_roll360_zero_only | frozen | 0 | 0.0 | 33.05% | 0.2926 |
| cnn_roll45 | fine_pooled | 7 | 52.3 | 48.05% | 0.4491 |
| cnn_roll45 | fine_pooled | 14 | 104.8 | 52.92% | 0.5060 |
| cnn_roll45 | fine_target | 7 | 52.3 | 55.19% | 0.5269 |
| cnn_roll45 | fine_target | 14 | 104.8 | 59.76% | 0.5756 |
| cnn_roll45 | frozen | 0 | 0.0 | 33.83% | 0.2994 |
| cnn_roll45 | scratch_target | 7 | 52.3 | 57.67% | 0.5661 |
| cnn_roll45 | scratch_target | 14 | 104.8 | 67.22% | 0.6570 |

## Paired comparisons

Exploratory participant bootstrap:10000draws, seed42; intervals condition on this development sample and the fixed scoring repetitions. No correction for multiple comparisons; do not use these as confirmatory tests. All specified comparisons are shown.

| Comparison | Accuracy difference, points | 95% interval, points | People improving |
|---|---:|---|---:|
| cnn/fine_target:7 minus14 | -6.73 | [-8.02, -5.54] | 0/6 |
| cnn/fine_pooled:7 minus14 | -6.73 | [-7.43, -6.00] | 0/6 |
| cnn/scratch_target:7 minus14 | -10.54 | [-12.25, -8.97] | 0/6 |
| cnn_roll45/fine_target:7 minus14 | -4.57 | [-6.30, -2.87] | 0/6 |
| cnn_roll45/fine_pooled:7 minus14 | -4.87 | [-5.90, -3.87] | 0/6 |
| cnn_roll45/scratch_target:7 minus14 | -9.56 | [-11.11, -8.22] | 0/6 |
| roll45 minus plain:frozen/0 | +6.56 | [+4.62, +8.33] | 6/6 |
| roll45 minus plain:fine_target/7 | -6.60 | [-11.98, -2.03] | 1/6 |
| roll45 minus plain:fine_target/14 | -8.76 | [-13.73, -4.94] | 0/6 |
| roll45 minus plain:fine_pooled/7 | -6.38 | [-10.48, -2.38] | 1/6 |
| roll45 minus plain:fine_pooled/14 | -8.24 | [-12.75, -4.35] | 0/6 |
| roll45 minus plain:scratch_target/7 | -12.68 | [-19.16, -7.08] | 0/6 |
| roll45 minus plain:scratch_target/14 | -13.67 | [-20.40, -8.40] | 0/6 |

## Designed shifts within45degrees

Computed from circular mean changes in all8recorded channel angles relative to source position0. Positions9/10 are excluded here as the separate random regime. Near360-degree returns may have small net displacement and remain included. Numbers of positions differ between participants; each person has equal weight within a stratum. Full strata and position mappings are in angle_strata.json.

| Model | Strategy | Trials | Accuracy | Participants |
|---|---|---:|---:|---:|
| cnn | fine_pooled | 7 | 73.21% | 6 |
| cnn | fine_pooled | 14 | 76.69% | 6 |
| cnn | fine_target | 7 | 75.34% | 6 |
| cnn | fine_target | 14 | 79.94% | 6 |
| cnn | frozen | 0 | 51.51% | 6 |
| cnn | scratch_target | 7 | 69.94% | 6 |
| cnn | scratch_target | 14 | 78.99% | 6 |
| cnn_roll360_zero_only | frozen | 0 | 29.46% | 6 |
| cnn_roll45 | fine_pooled | 7 | 60.22% | 6 |
| cnn_roll45 | fine_pooled | 14 | 62.78% | 6 |
| cnn_roll45 | fine_target | 7 | 62.68% | 6 |
| cnn_roll45 | fine_target | 14 | 64.98% | 6 |
| cnn_roll45 | frozen | 0 | 48.79% | 6 |
| cnn_roll45 | scratch_target | 7 | 56.77% | 6 |
| cnn_roll45 | scratch_target | 14 | 64.86% | 6 |

## Fitting diagnostics

Accuracy here is on the unaugmented fitting windows, not held-out performance. Low values limit interpretation of that trained baseline; they are not evidence against the original published method. The final training loss for augmented models includes randomly shifted windows and need not agree with clean fitting accuracy.

| Model | Stage | Fits | Mean fit accuracy | Minimum | Fits below95% |
|---|---|---:|---:|---:|---:|
| cnn | enrollment | 6 | 99.92% | 99.52% | 0 |
| cnn | fine_target | 120 | 91.53% | 67.62% | 63 |
| cnn | fine_pooled | 120 | 91.10% | 74.76% | 78 |
| cnn | scratch_target | 120 | 99.94% | 98.10% | 0 |
| cnn_roll45 | enrollment | 6 | 92.86% | 89.52% | 5 |
| cnn_roll45 | fine_target | 120 | 76.84% | 40.95% | 106 |
| cnn_roll45 | fine_pooled | 120 | 75.47% | 48.33% | 120 |
| cnn_roll45 | scratch_target | 120 | 95.67% | 78.57% | 40 |
| cnn_roll360_zero_only | enrollment | 6 | 62.30% | 52.38% | 6 |

## Validation and runtime

Six risk tests passed before training. Independent verification checked all 1386 raw trials, 186 allocations, 738 saved checkpoints and fitting accuracies, 918 prediction files/96,390 prediction rows, and 15aggregate groups. Training-only normalization, source-history and target-budget ledgers were checked. No reserved-participant signal values or Charles-device signals were decoded or scored. Reserved archives were separately downloaded and listed for file-structure verification.

Training run wall time: 540.77seconds (9.01minutes). Audit: 14.89seconds. Excludes implementation/literature review. CPU1thread; source/fine/scratch times are separately logged.

## Interpretation limits

Same-day results only; physical rotation cannot establish cross-day robustness. Six independent development people; only one scoring repetition per position; overlapping windows do not increase participant count. Fixed position order can confound angle with time/fatigue. The ±45degree augmentation has intentionally limited synthetic support; poor results at larger angles do not refute its intended range. All8shift exposure is an explicitly wider sensitivity. Sparse8channel permutation approximates a real rotation and lacks the spatial resolution of the original64channel sensor.

Preserve all outcomes, including inferior neural or augmentation results. Existing classical7trial accuracy80.19% versus14trial88.00% remains the reference. No method should be selected using reserved-subject results. The report tables describe the tested configurations; any next-stage selection and reason must be recorded separately before final evaluation.

## Provenance

- Methods: research/SENIC_NEURAL_METHODS.md.
- ABSDA source: https://unbscholar.dspace.lib.unb.ca/server/api/core/bitstreams/54407f52-28ec-4abf-95ef-33f55e98ff41/content.
- SeNic: https://github.com/BoZhuBo/SeNic at a4c12f7daab28a80d557677ae8dbcef0d7871ba2.
- Full configs, source snapshots, inputs, fitted-model ledger, predictions and audit: research/runs/20260907_senic_neural_v1.
- Source snapshots and hashes preserve the evaluated implementation.

## Additional optimization check (post-hoc)

The initial plainCNN fine-tuning runs often failed to fit their own calibration data at150steps, while scratch models fit well. After inspecting those fitting diagnostics and initial scores, all360plainCNN fine-target/fine-pooled/scratch-target cases were rerun to600steps. Same initialization/source model, optimizer, minibatch sequence, normalization and trials. Every original150/300step checkpoint was reproduced bit-for-bit before extending training. No case or checkpoint was chosen for its test performance. This is an explicitly post-hoc optimization diagnostic, not a new prespecified winning experiment.

The augmented models retain their original fixed-step results; they were not retrained in this check. Therefore,600step plain models must not be used to claim an equal-compute advantage against150/300step augmented models.

| Strategy | Trials | Initial accuracy | 600step accuracy | Change, points | Mean fit accuracy | Fits below95% |
|---|---:|---:|---:|---:|---:|---:|
| fine_target | 7 | 61.79% | 69.56% | +7.76 | 100.00% | 0/60 |
| fine_target | 14 | 68.52% | 79.81% | +11.29 | 99.91% | 0/60 |
| fine_pooled | 7 | 54.43% | 61.60% | +7.17 | 99.98% | 0/60 |
| fine_pooled | 14 | 61.16% | 72.40% | +11.24 | 99.53% | 0/60 |
| scratch_target | 7 | 70.35% | 70.68% | +0.33 | 100.00% | 0/60 |
| scratch_target | 14 | 80.89% | 81.38% | +0.49 | 99.99% | 0/60 |

Audit passed:360extended checkpoints and training accuracies,360exact original optimization prefixes,37,800saved scoring predictions. Run time 669.15seconds (11.15minutes); audit 3.96seconds. Part of the run overlapped the main audit/reporting on this machine; timings are observed shared-machine wall time.

The classical TDAR+RMS references remain80.19% at7trials and88.00% at14trials. Compare those with the whole table, not only the initially underoptimized CNN. Remaining fitting shortfalls or poor generalization must be disclosed. Longer fitting alone does not establish a distinct paper contribution.

Diagnostic outputs: research/runs/20260907_senic_convergence_v2.
