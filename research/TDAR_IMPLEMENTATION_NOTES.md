# TDAR recognition control — implementation notes

Status: source review only; no TDAR model or feature extraction run yet. Prepared while the fixed seed check trains. These notes do not alter the running neural or stopping protocols.

The planned control is LDA with TDAR plus retained RMS: MAV, zero crossings, slope sign changes, waveform length, four autoregressive coefficients, and RMS on each of16 channels (144 features). Keep the exact512-sample windows, participant/trial separation, calibration budgets and scoring trials. Compare source-only, source-plus-calibration and calibration-only fitting; retain existing basic-feature and neural results even if the richer control wins.

[Campbell et al. (2021)](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2021.657958/full) defines TDAR as the four Hudgins time-domain features plus fourth-order AR and evaluates handcrafted features with LDA. This supports the feature family, not an exact replication of that paper's dataset or complete pipeline. RMS is our disclosed addition.

The [LibEMG implementation](https://github.com/LibEMG/libemg/blob/main/libemg/feature_extractor.py), read September6, uses strict positive/negative crossings (zeros do not count), slope-product >=0 at the default threshold, summed absolute waveform differences, and librosa LPC order4 with the leading coefficient omitted. The slope equality convention counts flat triples; preserve and explicitly test any chosen tie convention. The online documentation's short AR equation is insufficient to reproduce the estimator; use actual implementation/API and pin its version before extraction. Source main is mutable and has not yet been pinned locally.

Next implementation choices to record before scoring: exact LPC dependency/version and degenerate-signal behavior, floating-point precision, slope/crossing thresholds, feature ordering and scaler fit access. Prefer existing published-tool conventions over introducing tuned thresholds. Verify small hand-calculated examples, synthetic AR recovery/sign convention, constant/zero stability, and independent comparisons against the pinned library on a fixed identity-based sample. Do not claim optimality or exact paper replication. No final-participant signals are needed for this development control.


## 2026-09-06T22:43:24.873221+00:00 — implemented for the first open-set classical stage

Implemented in src/open_set_emg.py:144feature-major columns (MAV,ZC,SSC,WL,AR1–4,RMS),float64rawmV,strictZC,nonnegativeSSCproduct,unnormalizedWL,librosa0.11.0Burgdenominatorcoefficients,no demeaning. AllzeroARexplicitlyzero;nonfiniteoutputraises. Fixed independent window sample acrossall8developmentpeople/3sessions matches scalarfeature+independentBurg calculation,maxARerror2.32e-12. Hand-calculated features,zero/constant behavior andsyntheticARsign/recovery tested. This follows reviewedfeatureconventions, not a pinnedfullLibEMG or Campbellpaper numerical reproduction. Run research/runs/20260906_open_set_classical_v2. It does not retroactively complete the prior17-class stopping-study TDARcontrol.
