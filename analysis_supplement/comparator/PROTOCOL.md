# Published-objective development comparison

Development comparison conducted September 22, 2026 in America/Los_Angeles (September 23 UTC).

## Question and scope

Does source-supervised covariance alignment provide a useful brief-calibration alternative to the existing frozen, supervised-update and replay controls on our cross-session task? This tests a published method family before deciding whether a new method is worth developing. No new-method or confirmatory superiority claim.

First stage: existing eight GRABMyo development participants (3, 5, 7, 20, 21, 22, 24, 38), seed42 enrollment checkpoints, both target sessions, four preselected cyclic choices (0,4,8,12), K1/K2. These choices are fixed before this run and are a feasibility subset, not exhaustive gesture evaluation. 128 cases. All cases score the same existing 68 whole trials per target session. Calibration uses two five-second recordings, with the first recording shared between paired coverage conditions. Whole-trial roles and final-participant exclusion are enforced.

## Relation to published work

[Yuan et al., IEEE TMRB, DOI 10.1109/TMRB.2024.3504737](https://www.pure.ed.ac.uk/ws/portalfiles/portal/484438014/YuanEtalIEEETMRB2024TowardsHighlyFlexible.pdf), Section III F and Figure 3c: labeled source classification plus covariance alignment of unlabeled target and source latent features. Section III G studies gesture subsets. Figure 3 and relevant method pages inspected visually and as text.

Implement the standard Deep CORAL covariance penalty: squared Frobenius difference between unbiased feature covariance matrices divided by 4*d^2. No target labels enter this objective. Update encoder and classifier jointly from each participant's enrolled checkpoint, resetting before every case/method. Include source-only continued training to distinguish source supervision from alignment effects.

This is a **backbone-matched adaptation of a published objective**, not an exact replication of the paper's reported accuracy. Differences: GRABMyo cross-session personal enrollment rather than Hyser inter-user transfer; 16-channel CompactEMGNet with 64 latent features and categorical CE rather than 256-channel VGG16-BN with 128 latent features; original window/preprocessing/splits retained; two recordings rather than three examples per gesture; class-balanced source sampling; explicit optimizer/step schedule below. The paper does not specify all optimizer and domain-weight details in the inspected methods; these choices are ours. Its printed classification equation resembles binary CE although outputs are multiclass; we retain our standard categorical CE. Do not silently call this an author-code reproduction.

## Fixed comparison

- Frozen enrolled model.
- Full-network target-only categorical CE.
- Full-network target CE plus equally weighted source CE (existing replay definition).
- Full-network source CE only, no alignment.
- Source CE + CORAL at weights 0.1, 1, 10, all reported.
- Adam learning rate 0.0001; checkpoints at 25 and 100 steps; batch64 per domain. Same target sampling and same class-balanced source sampling across applicable methods. Frozen model has zero update cost; compute/source access differences reported.

Use amplitude-preserving original scaler and no batch normalization changes. Record source CE, raw/weighted alignment loss and update norms to diagnose inactive/unstable objectives. Per-case checkpoints and predictions allow separate scoring replay. Training never receives scoring tensors or labels. Labels identifying prompted gestures are used only to select permitted trials and calculate evaluation metrics; CORAL receives their signals without labels.

## Interpretation and next-stage decision

Aggregate sessions and choices within each participant before reporting means or descriptive paired intervals. Report overall and omitted-active-gesture accuracy, calibrated recall and update cost. No inferential win declaration or new primary hypothesis based on this development screen. All weights/checkpoints remain visible, including failures. Compare the same seed, case subset and update scope; do not compare this subset with a differently averaged historical headline.

If a published objective works, that strengthens the benchmark but is not our algorithmic novelty. Advance an extension only after identifying a distinct contribution and a useful benefit/retention/cost trade-off beyond controls. Use fresh data for any new confirmatory evaluation after freezing the full protocol. Do not tune on the already-used final cohorts.

## Execution and cost

Initial plan used local MPS while RunPod had USD2.19 and the CLI lacked an API key. Charles explicitly requested RunPod and authorized spending as needed; the console now shows USD12.19. Local work is limited to implementation, unit tests and smoke verification. Main comparison will use one RTX3090, advertised GPU USD0.50/hour plus 40GB container disk USD0.006/hour, with an initial under-USD5 target. Retrieve and checksum all results before stopping the pod. No other pods are modified. No public code/data publication is authorized here.

Charles clarified that maximizing publication prospects is the priority. This bounded comparator addresses a concrete review weakness; do not expand into a major method redesign without useful evidence and a distinct contribution.


## 2026-09-23T03:12:54.495571+00:00 — Preserved first run; loss-scale follow-up frozen

Main RunPod comparison completed128cases1664records in439.33s. Every saved evaluation replayed on CUDA (3960320windows);1603downloaded files SHA256verified. First pod stopped and console confirmed USD0/hour before this extension. Initial frozen77.48%,K2 replay80.95/81.41% at25/100steps. CORAL0.1/1/10 did not improve on frozen. These are development observations, not final superiority evidence.

Loss diagnostics show even weight0.1 has median weighted alignment/source-CE ratio143.13 at25steps and15.88 at100steps. This is a material fairness concern: the first grid does not establish a well-balanced or well-tuned comparator. Extend **before additional runs** to fixed weights0.0001,0.001,0.01, keeping every original result. Repeat source-only and frozen as cross-run controls; unchanged128cases,checkpoints,optimizer,initialization,scoring and seed. This is an explicit adaptive development extension motivated by loss scale after viewing first results, not a new prespecified primary analysis. All six weights reported; no final-data access and no architecture/method redesign. Stop this bounded comparator screen after the extension unless a correctness failure requires repair.


### 2026-09-23T03:23:57.603207+00:00 — Execution fallback

The first RunPod run and full CUDA replay completed and were retrieved; CPU sample replay also passed208checkpoints495040windows. The EMG pod was confirmed stopped at USD0/hour, screenshot runpod_stopped.png. On restart, the console reported its GPU unavailable. An automatic-migration option was clicked but no migration completion was observed; browser control then repeatedly detached/timed out. No second cloud experiment was launched. Proceed locally on MPS for the fixed smaller-weight grid, with matched frozen/source-only bridge controls to quantify cross-platform differences. Do not silently pool runs until bridge verification passes. Recheck the account state when browser access returns; do not modify the unrelated tandem pod visible in the same account. Local smaller-grid smoke2cases10records23800window predictions passed.


### 2026-09-23T03:29:53.098725+00:00 — Local throughput correction and bridge criterion

Interrupted local small-grid v1 for avoidable MPS synchronization overhead, preserving its first9completedcases81records and partial10thcase in research/runs/20260923_alignment_small_seed42_mps_v1. No results used to change scientific settings. Combined per-parameter finite-gradient predicates into one host synchronization; gradients/optimizer unchanged. Local smoke v2 produced9model state dictionaries exactly equal to smoke v1, including the source initialization. Smoke v2 completed2cases10records in4.74s (earlier smoke106.47s under slower system conditions; no isolated speedup estimate claimed).

Restarted full small-grid experiment as research/runs/20260923_alignment_small_seed42_mps_v2. Before combining with CUDA results, require matching input hashes, access manifests and exactly identical trial confusion matrices for every repeated frozen/source-only evaluation (384bridge evaluations). Count any window-label differences separately. If trial metrics differ, do not silently pool. Charles reports another ChatGPT using RunPod; avoid further competing browser actions or modifying its pod. CLI key still unconfigured; account migration status pending verification.

Primary-source cross-check: Sun and Saenko, Deep CORAL (2016), Sections3–4, https://arxiv.org/html/1607.01719v1 . The covariance formula matches the implementation. The original authors describe balancing the losses approximately at training completion. This supports the loss-scale diagnostic; our chosen grid remains a development choice rather than a claim of globally optimal tuning.
