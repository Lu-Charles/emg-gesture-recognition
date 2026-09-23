# Development methods

This describes the current public-data development experiment. Participant and trial allocations are fixed; final model settings will be frozen after development. No final-participant results are available.

## Dataset and partitioning

We use GRABMyo version 1.1.0, containing recordings from 43 healthy participants across three days, with 17 gesture/rest classes and seven five-second repetitions per class and day. Signals are sampled at 2048 Hz. The corrected class ordering follows the release's MotionSequence.txt. We select the 16 forearm signals F1–F16 using the authors' conversion guidance. Physical values are decoded in millivolts using each header's gain and baseline. [Dataset and release documentation](https://physionet.org/content/grabmyo/1.1.0/)

Participant IDs were ranked by SHA256 with seed 20260906 before inspecting public-data model performance. Twenty participants form the representation-training group, eight form the development group, and 15 are reserved for final evaluation. Exact IDs and trial assignments are recorded in the allocation manifest. The current pipeline permits access only to training and development participants.

All three days from the training group can be used for representation learning. For a development participant, all day-1 trials are available for enrollment. On each later day, three whole trials per class form a calibration pool, and four form the scoring set. Calibration budgets are nested at one, two and three trials per class. The scoring set is identical across budgets, initializations and methods. Trial membership is assigned before window extraction.

## Signal observations

The pipeline retains the recordings' documented acquisition filtering and adds no software filter or whole-trial normalization. Each trial supplies 512-sample observations at a 256-sample hop, beginning at sample 1024. This gives 35 windows per trial, each spanning 250 ms. The initial half-second exclusion is a declared development choice; it is not an estimate of movement onset or controller reaction time.

Neural inputs use float32 physical signals. Classical features are computed in float64 and include mean absolute value, root mean square amplitude and mean absolute first difference for each channel, giving 48 features. Amplitude is retained in all primary methods.

## Shared encoder and matched control

The CNN has three convolutional layers: 16-to-32 channels with kernel 9 and stride 4; 32-to-64 with kernel 7 and stride 4; and 64-to-64 with kernel 5 and stride 2. ReLU follows each convolution. Global average pooling and a 17-class linear head give 40,689 parameters. Convolutions operate on a complete available observation; this does not claim samplewise streaming within that window.

Shared pretraining uses cross-entropy loss, Adam at a learning rate of 0.001, batch size 128 and 20 epochs. The last epoch is used without selection on final results. All 249,900 training windows are included once per epoch. To reduce disk access costs, training permutes trials into buffers of up to 128 trials, then shuffles windows within each buffer. The run seed plus the epoch index determines the ordering. Seed 42 is the original run; the robustness experiment specifies additional seeds 0 and 1, without changing data or hyperparameters. The same run seed controls neural initialization and enrollment/adaptation shuffle order. Tests check complete coverage, uniqueness and the final partial buffer.

One mean and standard deviation per channel are estimated from training-group windows only. These statistics are fixed throughout enrollment, adaptation and scoring. The shared-initialization and random-initialization models use exactly the same normalization. This differs from the earlier two-person pilot's per-person normalization and prevents attributing a normalization change to pretraining.

Each development participant is enrolled by updating the full network on day 1 for 20 epochs at learning rate 0.001. Later-day comparisons include no update, head-only updating and full-network updating. Adaptation uses 10 epochs at learning rate 0.0001 and the separate calibration data. Every budget starts from the same enrolled model. No scoring samples are used for normalization or updates.

## Classical comparisons

LDA uses automatic shrinkage and the least-squares solver. Random Forest uses 300 trees, minimum leaf size 2, square-root feature sampling and seed 42. Each uses the amplitude-inclusive features and a scaler fitted only on its permitted training set. Comparisons include day-1 data alone, day-1 data plus calibration, and calibration alone. The latter two receive the same target calibration trials as neural adaptation.

The shared neural model additionally uses the separate representation-training group. The matched random-initialization control isolates the effect of this pretraining; the classical comparisons assess performance against practical alternatives at the same new-day calibration cost. A separate, completed seed 42 control extends scratch enrollment continuously to 40 and 80 epochs with uninterrupted Adam state; this reduces but does not eliminate the original pretraining gap. The 20-epoch scratch baseline is therefore a matched-computation control, not a claim of optimal scratch convergence. A richer published TDAR plus retained-RMS LDA comparison remains unimplemented; see TDAR_IMPLEMENTATION_NOTES.md.

## Outcomes and limitations

We record macro-F1 and balanced accuracy for every participant/session pair. Development summaries weight participants equally and retain both later days. Repeated windows are correlated and are not treated as independent participants. Final uncertainty estimates should resample participants, preserving their two sessions together.

Day-1 enrollment uses 595 seconds of recorded signal per participant. New-day calibration budgets use 85, 170 and 255 seconds of recorded signal; rest and interaction overhead are additional. These are offline held-gesture experiments, with no custom-device data, clinical outcome, measured electrode displacement or live-control claim.

Runs retain configurations, source snapshots, dataset and cache hashes, model checkpoints, training-access manifests and scoring predictions. Separate verification reloads models, regenerates predictions, recomputes metrics and checks normalization, calibration/scoring separation and fixed scoring identities. All interrupted attempts and negative comparisons remain in the experiment log.

## Adaptive calibration duration

The stopping study reuses the shared CNN with full updates. Each balanced round acquires one reserved five-second trial from each of 17 classes, adding 85 seconds of recorded signal. The first round is mandatory. A controller may stop after one or two rounds; otherwise the full three-round budget is acquired.

The original pilot uses calibration ranks 123. The order check evaluates all six global permutations of these three ranks. These permutations reuse the same calibration pool and scoring trials; they do not cover arbitrary independent per-class orderings or new trial allocations. For each acquired subset, the recognizer resets to its day-1 enrolled state and trains for the same ten epochs, with trials supplied in canonical manifest order. Thus acquisition order changes availability, without also changing the fitting order for an identical subset. A shared full-budget model is used across all six orders.

After each of the first two rounds, decision features are computed only on the newly acquired round. We apply both the preceding checkpoint, which has not seen that round, and the updated checkpoint. The eight inputs are round index; preceding-checkpoint classification error, negative log-likelihood and standard deviation of whole-trial mean negative log-likelihood; updated-checkpoint predictive entropy and error; fraction of changed hard predictions; and mean total variation between preceding and updated class-probability vectors. Updated-checkpoint measurements are in-sample diagnostics, not validation estimates. No scoring labels or future-round measurements enter a held-out stopping decision.

The learned controller standardizes the eight inputs and fits ridge regression to average remaining gain per additional round: the difference between full-budget and current-budget macro-F1 divided by the number of remaining rounds. It does not predict only the immediate next round's gain. Stop when the predicted average gain is at most a selected threshold. Targets are available only from other participants during policy fitting.

Simple controllers threshold preceding error, updated entropy, or probability change. Each budget's numeric threshold is a quantile of that feature among policy-training participants. A selected fixed-budget family and the three fixed 85/170/255-second routines are retained as controls. The learned model is not required to outperform simple rules.

## Nested policy selection and uncertainty

Outer evaluation leaves out one of the eight development participants. Inner leave-one-participant-out selection uses only the other seven participants. Every session, decision state and calibration-order replay of a person remains in the same fold. Ridge regularization values are 1, 10, and 100; gain thresholds are -1, 0, .0025, .005, .01, .02, .04, .08, and 1. Simple-rule quantiles are -1, 0, .1, .25, .5, .75, .9, 1, and 2, where outside-range values explicitly mean never or always stop. Fixed routines select one/two/three rounds.

Select the candidate with the smallest mean recording cost among those with inner mean macro-F1 loss at most .01 relative to full calibration; break ties by smaller loss, then fixed candidate ordering. If no candidate qualifies, use full calibration. This empirical mean-loss constraint does not guarantee protection for a new individual. Save all candidate tables and each reached stop/continue decision.

The initial order check has two separately declared tracks: apply original-order policies unchanged to all orders of their held-out participant; and refit policies on all orders of other participants under the same nested procedure. The additional training-seed check makes the all-order-trained learned and pre-error policies the principal pair while retaining frozen-original controls. Each seed's policy is trained using that seed's recognizer; no best seed is selected after viewing scores.

The descriptive equal-cost control interpolates between adjacent fixed budgets at the policy's observed mean recording duration, representing the expected performance of a label-independent randomized fixed routine. Its cost match is retrospective analysis, not a deployed selector fitted on held-out labels. The screening criteria are at least 20% target-day recording reduction, mean F1 loss no greater than .01, and positive advantage over this equal-cost mixture. Report the actual continuous values as well as pass/fail; the 20% cutoff alone does not establish practical superiority.

Uncertainty calculations resample participants with their sessions and orders retained. Across seeds, retain every seed for each resampled participant. There are eight independent development people, not 96 per-seed replays or 24 independent person-seed observations. The bootstrap is descriptive and conditional on the fitted recognizers and policies; it does not repeat all model selection or remove the influence of earlier development-based project choices. Record worst losses and replays losing more than .02, including how many people they involve. All 15 final participants remain reserved for evaluation after the practical procedure and remaining baseline choices are frozen.
