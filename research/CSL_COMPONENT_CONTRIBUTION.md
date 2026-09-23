# Completed contribution: calibration success versus cross-gesture transfer

The study now supports a defensible, narrowly scoped empirical paper. All primary experiments, fixed implementation controls and supplementary search seeds are complete and independently verified. The novelty is modest and specific; this is not a new state-of-the-art adaptation architecture.

## What the paper establishes

For the tested public single-gesture HD-EMG recipe, accurate recognition of later trials of the calibrated gesture does not establish transfer to the other gestures. Spatial-plus-gain adaptation reaches **96.99% on held-out calibrated-class trials but 34.51% on the other seven classes**, versus **60.71%** for the frozen model. Spatial-only adaptation changes seven parameters with the recognizer fixed and exhibits the same distinction: **96.63% versus 45.77%**.

This is a participant-separated component study: one development participant, four reserved participants, all 80 ordered session pairs and all eight calibration choices, with identical scoring trials across conditions. All classes were present during source training. The seven “other” classes are absent only from target calibration.

Gain correction raises overall accuracy from **60.71% to 62.80%**, while its other-class mean is **59.34%**. That small other-class difference is uncertain across four people and is not presented as a population-level harmful effect. It illustrates why an aggregate improvement alone does not establish transfer across the vocabulary.

## What the controls add

- **Exact-loss identity preference:** selecting the existing identity candidate when its computed loss exactly equals the minimum improves other-class means by **1.80 points for spatial adaptation** and **2.45 points for spatial + gain**. The selected initial loss is identical in all 1,280 matched comparisons; all 1,111 non-intervened checkpoints are tensor-identical. The control also harms 20/640 and 35/640 cases, respectively. It is a measured implementation effect, not a guarantee of improvement, and both cohort means remain below frozen.
- **Processing order:** resampling raw amplitudes before normalization and applying gain after resampling yields other-class means of **35.65% and 28.85%**. This alternative does not resolve the transfer gap. It changes normalization and gain placement together.
- **Search sampling:** on the fixed session pair 1→2, spatial-only means are **50.50%, 56.10% and 56.35%** for seeds 42–44, exceeding that pair's frozen mean of **45.49%**. Combined means are **36.11%, 37.60% and 35.47%**. Individual cases can vary substantially across seeds. This positive spatial-only result must be retained; neither universal failure nor full-cohort seed invariance is supported.

The contribution is the matched attribution of gain/spatial behavior, the separation of calibrated-class generalization from cross-class transfer, and controlled examination of specific implementation choices. The recommendation is an evaluation practice: preserve a frozen comparator and assess the intended vocabulary beyond the calibration subset.

## Relationship to prior work

[Pereira et al.'s SAL manuscript](https://arxiv.org/html/2409.08058v2) proposes limited-class adaptation as future work; its [later public implementation](https://github.com/joao-binenbojm/spatial-adaptation-layer/tree/7c5a075a58dff06566b82965f8235ba744377fb8) already contains the single-gesture components evaluated here. We credit that implementation and disclose the corrected provider geometry and exact configuration. We do not claim to reproduce or refute the manuscript's headline results.

[Yuan et al.](https://www.pure.ed.ac.uk/ws/portalfiles/portal/484438014/YuanEtalIEEETMRB2024TowardsHighlyFlexible.pdf) already investigates subset/unknown-gesture calibration across users and discusses cross-gesture amplitude shifts. [Shi et al.](https://pubmed.ncbi.nlm.nih.gov/38427548/) also studies unknown calibration gestures. [Zhang et al.](https://www.sciencedirect.com/science/article/pii/S174680942500922X) describes single-gesture fine-tuning in accessible publisher highlights; its full methods were unavailable in this audit. [Li et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC13306682/) provides further prior art for adapting inputs to a frozen recognizer. None of the general ideas of limited-class calibration, amplitude correction or frozen-classifier adaptation is claimed as new.

## Strength and limits

The strongest submission is a focused empirical biosignal/ML paper with a reproducibility and implementation-audit contribution. The evidence is substantially stronger than the earlier single-pair feasibility tests. The contribution does not require a new architecture or a claim that an adaptation method wins.

Generalization remains limited by four independent people, one dataset, eight author-selected classes, a fixed primary source seed and offline processing. Natural session variation is not controlled electrode displacement. Three seconds of labeled calibration additionally uses 90 seconds of target rest. The all-class reference reaches 89.14% but receives more labels and more gradient updates. The identity rule's improvement is modest, not a competitive recognition breakthrough.

The scientific stopping criterion has been reached: a specific claim, matched comparisons, independent participant evaluation, preserved counterexamples, fixed controls and verified outputs. Next is venue-specific manuscript preparation and human review, not another speculative method pivot or tuning on the evaluated cohort. Acceptance and internship impact have not been established by these experiments.

## Deliverables

- `CSL_PAPER_DRAFT.md`: complete research manuscript draft with result tables and figures.
- `CSL_SUBMISSION_ABSTRACT.md`: 223-word abstract.
- `CSL_COMPONENT_METHODS.md`: exact methodological details and limitations.
- `CSL_CONTROL_FINDINGS.md`: complete control results.
- `REPRODUCE_CSL_COMPONENT_STUDY.md`: full-workspace audit instructions and analysis review bundle.
- `runs/20260907_csl_components_confirmatory/`: frozen protocol, manifest, primary/control analyses, predictions, checkpoints and verification reports.
