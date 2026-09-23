# First open-set EMG baseline results

Completed classical development screen; no neural open-set model or final evaluation yet. This establishes baseline behavior, not novelty or a successful new method. All results below are pooled over eight development participants with equal person weight; each person contributes two target sessions and six calibration orders. Windows/orders are not independent people.

## No new-day calibration

| Method | Model fit data | Rejection score | Correct commands accepted | Unfamiliar gestures accepted as commands | People meeting residual-problem screen |
| --- | --- | --- | ---: | ---: | ---: |
| none | source_plus_target | distance | 64.04% | 66.96% | 2/8 |
| none | source_plus_target | probability | 65.85% | 53.21% | 2/8 |

## Equal105-second new-day recording budget

| Method | Model fit data | Rejection score | Correct commands accepted | Unfamiliar gestures accepted as commands | People meeting residual-problem screen |
| --- | --- | --- | ---: | ---: | ---: |
| both | source_plus_target | distance | 92.87% | 83.59% | 8/8 |
| both | source_plus_target | probability | 90.06% | 59.53% | 8/8 |
| both | target_only | distance | 92.75% | 80.35% | 8/8 |
| both | target_only | probability | 90.09% | 62.52% | 8/8 |
| model_only | source_plus_target | distance | 95.34% | 88.00% | 8/8 |
| model_only | source_plus_target | probability | 81.35% | 48.51% | 6/8 |
| model_only | target_only | distance | 94.72% | 81.24% | 8/8 |
| model_only | target_only | probability | 89.97% | 60.91% | 7/8 |
| threshold_only | source_plus_target | distance | 80.94% | 92.31% | 6/8 |
| threshold_only | source_plus_target | probability | 79.92% | 79.98% | 6/8 |

`model_only` updates/refits the classifier and retains the original model's numeric threshold. `threshold_only` leaves the source classifier fixed and uses all three new-day rounds to set its threshold. `both` uses two rounds for the classifier and a disjoint third round for threshold setting. `source_plus_target` includes day1 model-fitting data; `target_only` refits from the permitted new-day model-fitting rounds. The source classifier uses five day1 trials per known class, with a separate sixth trial for its threshold.

Probability and distance use the same fitted LDA classifier. Distance is Mahalanobis distance to its predicted class centroid, using its shrinkage covariance; this is a simple distance baseline, not Gao's learned prototype method.95% known calibration-score retention is fixed before outcomes. Actual test acceptance can differ. An unfamiliar gesture predicted as rest does not issue a command and does not count as unknown false activation; rest false activation is reported separately.

## Interpretation limits and decision

The threshold-transfer hypothesis is not established by these results. In the independent day1 reference, probability rejection correctly accepts89.17% of active known commands but also accepts59.46% of unfamiliar gestures. The joint source-plus-target probability baseline on later days reaches90.06% correct acceptance with59.53% unknown false acceptance. Thus a major open-set limitation already exists within day1; these figures do not show personalization causing it.

An explicitly post-hoc oracle-threshold summary of the planned full curves further tests whether simply changing the cutoff could suffice. For the105-second target-only model with probability scores, even a test-informed threshold chosen separately for each person/session/order to retain at least80% correct commands still accepts27.96% of unfamiliar gestures on average among feasible cases. 96/96 such replay cases can reach80%. This is an optimistic diagnostic using test labels, not a deployable method or a new result selected for submission. The corresponding source-plus-target model-only diagnostic is31.08%; all methods and infeasible cases are saved in posthoc_oracle_summary.csv. This indicates limited score separation in these classical models, beyond a stale cutoff alone.

The last column checks the prespecified first residual-problem condition within each participant's average: at least80% correctly accepted known active commands alongside at least10% unknown false acceptance. It is a descriptive screen for each baseline, not an overall pass. The six-of-eight-person requirement must be assessed against the strongest simple/neural competitors; this first classical stage does not settle it. Per-session results are available separately. The alternative source-to-target deterioration criterion requires comparison at comparable correct acceptance, not a raw change between unrelated operating points.

Do not claim a method contribution from improving a stale threshold alone. Threshold-only, joint and target-only controls are included to prevent that interpretation. No new learned rule has been fitted. The next planned comparison is a freshly pretrained known-only compact CNN and appropriate rejection controls, keeping all held-out gesture/model exposure constraints. If strong simple methods remove the problem, preserve that result and stop this branch.

These are cued unfamiliar-gesture proxies, not activities-of-daily-living recordings, real-time command events or a clinical safety test. No false activations/hour or reaction latency is inferred. The source known/unknown vocabulary was chosen before these results. Existing17-class checkpoints and their normalization were not reused. Final15 participants and Charles's recordings were excluded.

## Verification and runtime

- 232 saved LDA models; 8064 operating-point rows, with complete scores and thresholds.
- 1344 allocations checked for participant/known-class access, disjoint fit/threshold/score trials and exact recording budgets.
- 72 raw-signal windows independently checked using scalar features and a separate Burg implementation; maximum AR difference 2.32e-12.
- 695520 saved window predictions independently reproduced from fitted discriminant coefficients; all operating-point metrics/counts recalculated; 3472 curve points checked.
- Six risk-focused tests passed in0.916 seconds. Earlier v1 stopped before model fitting because of a tuple-indexing error; its artifacts/log are preserved, and a regression test covers the fix.
- Completed v2 runtime 44.48 seconds, including 31.26 seconds feature extraction and 5.65 seconds summed fit/save time. Independent audit 7.08 seconds. These exclude implementation/setup/report time.

## Exact artifacts

Run: `research/runs/20260906_open_set_classical_v2`. `config.json` and `allocations.json` were written before extracting features or inspecting outcomes. `trial_manifest.json`, `model_access.json`, saved models and prediction files preserve access/provenance. `aggregate.csv`, `per_participant.csv`, `per_session.csv`, `per_class.csv`, `retention_sensitivity.csv` and `participant_intervals.csv` contain the full0/35/70/105-second results. Intervals use10000 participant-bootstrap draws, conditional on fixed procedures, from only eight independent people. `curve_index.json` links complete descriptive command tradeoff curves and standard OSCR/AUROC summaries; test-derived curve thresholds are not deployable calibration procedures.

Source snapshots and hashes are saved. Environment is `.venv-open-set`, with pinned librosa0.11.0 and complete dependency versions saved in `environment.txt`; earlier `.venv-public` unchanged.
