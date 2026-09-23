# Cross-session calibration from one gesture: study methods

This document describes the frozen experiment. It contains no provisional performance claims. The machine-readable protocols and exact source snapshots are authoritative for reproduction.

## Question and scope

We examine whether a calibration procedure that recognizes one gesture also preserves recognition of the other gestures. All eight classes are present during source training; the other seven are absent only from target calibration. This is not recognition of new gesture classes. The study separates channel-wise gain correction from spatial alignment in a public HD-EMG adaptation implementation. It is an empirical evaluation of existing components, not a new architecture or a claim to have invented calibration from a subset of classes.

The primary comparison uses natural changes between recording sessions. Those changes may include electrode placement, contact, effort and other factors. The experiment does not isolate physical electrode displacement, and an estimated affine transform is not treated as a measured displacement.

## Data and participant separation

CSL-HDEMG contains five participants recorded in five sessions. Dataset reference: [Amma et al., CHI 2015, DOI 10.1145/2702123.2702501](https://www.csl.uni-bremen.de/cms/publications/bibtexbrowser.php?bib=csl_all_publications.bib&key=amma2015advancing). Participant 1, sessions 1 and 2, supplied development data. Participants 2–5 were reserved until the evaluation design was frozen. We use the eight gesture IDs selected in the authors' later single-gesture configuration: 8, 9, 12, 13, 16, 21, 23 and 24. We do not claim results for the complete 26-gesture task.

For every reserved participant, we evaluate all 20 ordered source–target session pairs. Each source recognizer is trained independently using all available trials in that source session. In the target session, the first actual trial of a gesture supplies calibration; later trials supply scoring. We repeat this for every possible calibration gesture, without selecting a favorable gesture from test results. The scoring set includes all eight gesture classes and is identical across methods, calibration choices and the all-class reference.

The reserved data contain 1,598 active trials and 600 rest trials. Two active files—participant 4, session 4, gestures 8 and 9—contain nine rather than ten repetitions. This was discovered during raw-data checks, before any model was trained on the reserved participants. We retained all actual trials and never filled a missing repetition by duplication. Target session 4 for participant 4 therefore has 70 scoring trials; other target sessions have 72. Counts and trial identities accompany every result.

## Signal processing

Each raw trial has 192 channels and 6,144 samples, recorded at 2,048 Hz. The provider's documented mapping removes every eighth channel, rearranges the remaining 168 channels into a 7 × 24 array, and reverses spatial rows. Time order is preserved. This corrects a discrepancy in the inspected public loader and is disclosed as a departure from that loader.

We preserve the later author recipe's trial-wise mean centering, causal band-pass and band-stop filters, activity segmenter, 1,025-sample RMS window and spatial 3 × 3 median filter with zero padding. Its literal `mean_square` configuration spelling does not activate the separate baseline-subtraction branch; our primary experiment retains that behavior. No normalization statistic is fitted to scoring trials. The segmenter does use the target session's 30 rest trials.

The primary task is offline segmented-trial recognition. The RMS operation and full-trial vote do not establish an online latency or a causal streaming capability.

## Models and matched components

We pin the public author implementation to revision `7c5a075a58dff06566b82965f8235ba744377fb8`. The recognizer consists of batch normalization, input dropout and an eight-class linear classifier. Source training uses 15 epochs, Adam with learning rate 0.001, batch size 128, mean cross-entropy, dropout probability 0.5 and a three-epoch learning-rate warm-up. The final checkpoint is used; target scores never select an epoch.

Six conditions share the same source checkpoint:

| Condition | Target-session update |
|---|---|
| Frozen | None |
| Gain | Channel-wise source/target mean ratio, clamped to [0.5, 2] |
| Spatial | Seven bounded affine parameters; classifier and normalization statistics fixed |
| Spatial + gain | The author's combined affine and gain procedure |
| Fine-tuning | Classifier and batch-normalization parameters/statistics updated |
| Fine-tuning + gain | The same classifier update preceded by gain correction |

Gain statistics use only the calibration gesture and the corresponding source-class recordings. Spatial adaptation uses one RMS prototype from the calibration trial. It evaluates 16,384 Latin-hypercube candidates plus identity using calibration loss, then performs 500 Adam updates at learning rate 0.01. Search and scoring use bicubic interpolation; gradient updates use bilinear interpolation, following the author recipe. Batch normalization and dropout are in evaluation mode during spatial adaptation. No learnable additive baseline is included in this later recipe.

The combined author procedure derives gain from the spatially transformed target mean but applies it to the incoming signal before normalization and resampling. We preserve this ordering. Consequently, the factorial comparison measures the behavior of these implemented components; it does not identify a physiological interaction between amplitude change and electrode displacement.

Fine-tuning uses 150 epochs at learning rate 0.001, batch size 128 and a 30-epoch warm-up. As context, we also fine-tune using the first trial of all eight gestures. This reference uses the same epoch count and therefore more gradient updates. It measures the performance of that practical calibration recipe; its difference from single-class fine-tuning is not attributed solely to data quantity.

The primary seed is 42. Candidate evaluation is accelerated by applying `torch.func.vmap` to the unchanged author forward function. Both gain conditions were checked against all 16,385 scalar candidate evaluations on development data; their selected initial candidates matched and losses agreed within the recorded numerical tolerances. The implementation and source hashes are saved.

## Outcomes and analysis

Each scoring trial receives a majority vote over its frame predictions. Ties follow the first occurrence among tied labels. The primary outcome is accuracy on the seven classes absent from calibration. We separately report accuracy over all eight classes, held-out repetitions of the calibration class, and the calibration recording itself.

Negative transfer means overall scoring accuracy is strictly below the frozen model on the same trials. A prespecified analysis counts negative-transfer cases among updates with at least 99% calibration-frame accuracy. These counts are descriptive repeated measurements, not independent Bernoulli trials.

We average the eight calibration choices within each pair, then the 20 pairs within each participant. The four participant means are the independent units. Paired contrasts measure gain versus frozen, spatial versus frozen, spatial-plus-gain versus gain, and the gain/spatial factorial interaction. We report participant values and descriptive 95% bootstrap intervals from 10,000 participant resamples with seed 2026. With four people, these intervals are coarse and cannot support broad population claims. Overlapping frames do not increase the independent sample size.

Before inspecting cohort performance summaries, we also fixed a supplementary search-seed check: seeds 43 and 44 on session 1→2 for all four reserved participants and all eight calibration choices. It changes only spatial candidate sampling and retains both seeds, source checkpoints, preprocessing and scoring trials. This checks a specific source of initialization sensitivity; it is not a second independent cohort or a general multi-seed retraining study.

## Verification and recording cost

The raw archive and extracted files have integrity hashes. Whole-trial manifests record channel mapping, labels, repetitions, frame boundaries and roles. Checkpoint verification independently recomputes predictions using different inference batch sizes, checks trial votes and calibration fit, and verifies that frozen model state remains unchanged. Selected raw frames are independently reconstructed using the provider's array mapping and a separate median-filter implementation.

One-gesture calibration uses three seconds of labeled signal. The inherited segmenter additionally uses 90 seconds of target rest, so the study does not demonstrate a three-second total setup. The all-class reference uses 24 seconds of labeled signal plus the same rest data. These are recorded signal durations and exclude instructions, transitions and electrode setup.

Reproducibility relies on saved code snapshots, hashes, configurations and environment versions. The upstream commit is recorded separately. Raw data and third-party source are not automatically cleared for public redistribution. No manuscript or dataset has been submitted or published by this task.

## Source-fidelity details for reviewers

The later public configuration is distinct from the SAL manuscript's earlier all-class experiments. It uses gain correction and disables the learnable additive baseline. We therefore identify it by its exact code revision and configuration rather than claiming to reproduce the manuscript's headline numbers.

Normalized translation bounds are ±8/23 horizontally and ±8/6 vertically. Scaling is bounded by [1/1.1, 1.1], and both shear parameters by ±0.1. The literal rotation bound is ±15/180 radians; we preserve that numerical expression and do not describe it as ±15 degrees. The choice of spatial search range and gain/warp ordering may influence the findings. Neither was optimized on the independent results.

Mean RMS templates estimate a gain mask at each of 168 positions. This should not be confused with a single global amplitude scale or a uniquely identified physiological parameter. Our use of “gain component” refers to this exact clamped template-ratio procedure.

## Additional source-based implementation controls

Two further controls were fixed before inspecting the primary cohort's performance summaries. Both use all 80 ordered session pairs and all eight calibration choices, reusing the primary source checkpoints and scoring trials.

The input-order control resamples raw amplitudes before frozen batch normalization. In its gain condition, it applies gain after resampling. This checks an explicit alternative ordering of existing components. It changes both gain placement and normalization placement, so its effects cannot be assigned to gain order alone.

The identity-tie control changes only the initial candidate selection: if the existing identity candidate has exactly the minimum computed calibration loss, select it; otherwise retain the original first minimum. It uses no tolerance, additional candidate or new optimizer setting. Subsequent updates are unchanged. Its initial measured calibration loss is therefore identical to the original choice, but its final target accuracy need not be.

Both controls were tested on development data and retained even when they performed poorly. See `CSL_CALIBRATION_AMBIGUITY.md` for the algebra and synthetic tests, and the separate input-order and identity-tie protocols for frozen execution details. These are implementation controls within this study, not independent participant cohorts.
