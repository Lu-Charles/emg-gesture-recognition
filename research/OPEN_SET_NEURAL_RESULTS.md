# Stronger open-set EMG baseline screen

Eight development participants, one training seed, two later sessions and six calibration orders. Twenty separate participants supply shared representation training; the final15 remain untouched. These are matched-backbone adaptations of published ideas, not original-benchmark reproductions or a new proposed method.

<!-- Assessment of the frozen 20260906_open_set_neural_v1 run only. -->
## Interpretation and continuation decision

The published PredIN adaptation supplies a substantially stronger baseline: with105seconds of joint calibration and source replay, correct command acceptance is92.05% versus the CNN's92.79%, while unfamiliar command acceptance falls from53.56% to39.71%. It improves unknown acceptance for all eight people; the post-hoc paired mean difference is−13.85percentage points, with a conditional participant-bootstrap interval of−21.49 to−6.93points. The mean correct-acceptance difference is−0.73points. This benefit belongs to an existing method, not a new contribution. The prespecified softmax control reaches92.02%/38.28%; it is reported as a control, not retrospectively substituted for the primary prototype score.

The residual-problem condition still holds for all eight people in each later session under joint calibrated PredIN, with either score:≥80% correct acceptance and≥10% unfamiliar acceptance. However, the day1 reference already has substantial unfamiliar acceptance. No distinct day-caused failure has been established. The optimistic, test-informed full-budget PredIN oracle still accepts20.72% unfamiliar gestures at≥80% correct commands using prototype scores, or18.79% with probability scores; all96replays are feasible. These are diagnostics, not calibrated deployment results.

The tempting calibration-allocation claim is not consistent across people. Joint PredIN/prototype calibration averages7.55points fewer unfamiliar commands than model-only calibration at the same105second budget, with0.24points more correct commands. Yet unfamiliar acceptance improves for only3/8people; the paired unknown-rate interval is−24.12 to+6.64points. Its mean benefit is therefore insufficient evidence for an always-joint rule. The probability control improves unknown acceptance for only2/8people. These three paired contrasts were selected after reading complete operating points and are explicitly exploratory; see paired_contrasts.json for both update strategies and each session separately. Comparing only with a stale threshold also cannot establish a new-method contribution under the original criteria.

Recommendation: retain a narrowly scoped empirical calibration study as provisional, and repeat only the fixed CNN/PredIN procedures with seeds0/1 before further investment. Check whether the participant-level reversals persist; do not train an allocation controller or search architectures from these outcomes. The residual-problem gate justifies this bounded reproducibility check, while the mitigation/novelty gate remains unmet. The current project does not yet support the8–9/10 aspiration. If the pattern is unstable or adds no distinct finding beyond existing work, stop expanding this branch as a standout-method project. Final15 remain locked.



## Independent day1 reference

| Family | Correct commands accepted | Unfamiliar commands accepted |
| --- | ---: | ---: |
| cnn | 89.76% | 48.50% |
| pl | 76.85% | 46.46% |
| predin | 89.23% | 45.96% |

## No new-day calibration

| Family | Update | Fit strategy | Correct commands accepted | Unfamiliar commands accepted | Residual screen |
| --- | --- | --- | ---: | ---: | ---: |
| cnn | none | target_finetune | 82.66% | 46.63% | 4/8 |
| pl | none | target_finetune | 72.22% | 44.79% | 1/8 |
| predin | none | target_finetune | 81.50% | 44.60% | 5/8 |

## Equal105-second new-day budget

| Family | Update | Fit strategy | Correct commands accepted | Unfamiliar commands accepted | Residual screen |
| --- | --- | --- | ---: | ---: | ---: |
| cnn | both | pooled_replay | 92.79% | 53.56% | 8/8 |
| cnn | both | target_finetune | 92.05% | 53.00% | 8/8 |
| cnn | model_only | pooled_replay | 88.39% | 44.51% | 6/8 |
| cnn | model_only | target_finetune | 87.93% | 44.95% | 7/8 |
| cnn | threshold_only | target_finetune | 89.81% | 61.15% | 7/8 |
| pl | both | pooled_replay | 77.41% | 44.82% | 2/8 |
| pl | both | target_finetune | 78.18% | 47.70% | 2/8 |
| pl | model_only | pooled_replay | 77.81% | 47.24% | 2/8 |
| pl | model_only | target_finetune | 77.61% | 47.25% | 3/8 |
| pl | threshold_only | target_finetune | 78.03% | 60.21% | 3/8 |
| predin | both | pooled_replay | 92.05% | 39.71% | 8/8 |
| predin | both | target_finetune | 92.22% | 43.04% | 8/8 |
| predin | model_only | pooled_replay | 91.82% | 47.26% | 7/8 |
| predin | model_only | target_finetune | 91.79% | 47.58% | 8/8 |
| predin | threshold_only | target_finetune | 88.25% | 54.77% | 7/8 |

CNN uses maximum softmax probability. PL and PredIN use maximum prototype similarity, with softmax controls retained in the complete tables. All use the preregistered95% known calibration-score retention. Correct command acceptance excludes rest; unfamiliar gestures predicted as rest do not issue an active command. Rest false activation, wrong known commands and rejection are saved separately.

`model_only` fine-tunes using three target rounds but keeps the original numeric threshold. `threshold_only` uses all three rounds for the fixed source model's threshold. `both` uses two fitting rounds and one disjoint threshold round. `target_finetune` starts from the enrolled neural model and updates on target data; it is not training from scratch. `pooled_replay` also reuses day1 fitting data. No extra target recordings are granted to replay. Known fitting/threshold/scoring trials are disjoint throughout. Order permutations are repeated allocations, not independent people.

The residual screen counts people averaging≥80% correct acceptance and≥10% unfamiliar command acceptance at this operating point. This is a baseline limitation screen; six people passing it does not establish novelty or a successful mitigation.

## Optimistic threshold-separation diagnostic

| Family | Update | Fit strategy | Feasible fraction | Unknown acceptance at ≥80% correct |
| --- | --- | --- | ---: | ---: |
| cnn | both | pooled_replay | 100.0% | 24.17% |
| cnn | both | target_finetune | 100.0% | 24.69% |
| cnn | model_only | pooled_replay | 100.0% | 23.28% |
| cnn | model_only | target_finetune | 100.0% | 23.45% |
| cnn | threshold_only | target_finetune | 81.2% | 26.48% |
| pl | both | pooled_replay | 87.5% | 52.68% |
| pl | both | target_finetune | 87.5% | 49.66% |
| pl | model_only | pooled_replay | 87.5% | 44.13% |
| pl | model_only | target_finetune | 87.5% | 44.50% |
| pl | threshold_only | target_finetune | 62.5% | 47.04% |
| predin | both | pooled_replay | 100.0% | 21.18% |
| predin | both | target_finetune | 100.0% | 22.90% |
| predin | model_only | pooled_replay | 100.0% | 20.72% |
| predin | model_only | target_finetune | 100.0% | 22.25% |
| predin | threshold_only | target_finetune | 81.2% | 23.83% |

These thresholds use scoring labels to attain at least80% correct known commands separately per participant/session/order. They are explicitly test-informed oracle diagnostics, never deployable calibration or training inputs. This diagnostic was introduced after the earlier classical results and retained for this comparison. Infeasible cases cannot reach80% even with no rejection; their unknown rates are missing rather than zero. Feasibility differences limit comparisons of feasible-only means. A score-separation problem already present on day1 is not a distinctive day-shift contribution.

## Fidelity and training limitations

[PredIN author manuscript v2](https://arxiv.org/html/2407.19753v2) supplies the prototype, compactness, triplet and two-branch inconsistency equations. The local implementation interprets the unusual compactness equation literally: half vectorL2 norm below vectorL1 norm1, otherwise vectorL1 minus0.5. No official author code was located. The published work uses different backbones, SGD and100epochs; this screen uses the project's compact encoder plus128-dimensional embedding, Adam,20 shared epochs,20 enrollment epochs and10 update epochs. All settings and final-checkpoint selection were fixed before scoring. Thus unfavorable results do not establish that the original published method fails. See NEURAL_OPEN_SET_METHODS.md for complete choices and departures. PL is the PredIN manuscript's single-prototype baseline, not an exact reproduction of Gao's personalized method.

- cnn: training CE 0.5988 → 0.0446; total loss 0.5988 → 0.0446. PredIN CE sums two branches. These are training losses, not held-out accuracy or proof of convergence.
- pl: training CE 1.2577 → 0.2658; total loss 40.1632 → 0.4496. PredIN CE sums two branches. These are training losses, not held-out accuracy or proof of convergence.
- predin: training CE 2.5120 → 0.1884; total loss 81.0906 → 0.1837. PredIN CE sums two branches. These are training losses, not held-out accuracy or proof of convergence.

After viewing PL's poor recognition, an explicitly post-hoc training-fit diagnostic scored all enrollment fitting trials without rejection. It changes no settings or selected checkpoints:

- cnn: in-sample enrollment accuracy 99.93% mean (99.76–100.00% across people); 0/8 people have a known class with less than20% training recall.
- pl: in-sample enrollment accuracy 87.88% mean (82.61–99.59% across people); 7/8 people have a known class with less than20% training recall.
- predin: in-sample enrollment accuracy 99.45% mean (96.08–100.00% across people); 0/8 people have a known class with less than20% training recall.

The single-prototype PL adaptation severely confuses known classes on its own fitting data for most people. It therefore cannot serve as evidence that a well-trained original PL method has been defeated. Compact-backbone/training-schedule and literal-equation fidelity remain limitations. This diagnostic uses training data, not additional held-out or final data. CPU fit diagnostics and artifact auditing overlapped portions of PredIN execution, so wall times reflect the actual shared-machine workload rather than isolated hardware benchmarks. Audit elapsed time includes waiting for the last family to complete.

Only one vocabulary and seed are tested. No unseen-gesture diversity, second-dataset result, final-participant result, daily-life safety claim or8–9/10 research claim is supported yet. Comparisons with LDA also differ in access to20-person representation training; architecture superiority cannot be isolated from that difference.

## Verification and measured runtime

Five focused loss/access/gradient tests passed before the run. Independent audit checked all1344allocation ledgers, recomputed normalization from raw known-only signals with overlap multiplicities, checked every saved prediction's labels and all reported metric counts/cutoffs, and reproduced nine fixed windows per checkpoint using direct functional CPU operations. This is sampled numerical checkpoint verification, not replay of every neural prediction or optimizer update. Complete curves were spot-checked. CPU/MPS tolerances and exact counts are in validation.json.

Total experiment wall time: 39.70 minutes, excluding implementation and verification. Per-family timing:

- cnn: 7.42 minutes total, 4.32 minutes shared training, 232 saved calibrated models.
- pl: 11.01 minutes total, 5.72 minutes shared training, 232 saved calibrated models.
- predin: 21.05 minutes total, 11.75 minutes shared training, 232 saved calibrated models.

Run: `research/runs/20260906_open_set_neural_v1`. Full aggregate, participant/session/class, retention-sensitivity, participant-bootstrap intervals, source and oracle tables are saved alongside per-family models/scores/loss histories. Bootstrap intervals resample eight people, conditional on this fixed vocabulary/procedure; overlapping windows are not independent samples. Config, source snapshots, hashes, environment and access manifests preserve provenance. No custom data or final15 signal access.
