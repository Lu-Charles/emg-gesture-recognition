# Legacy EMG audit — T1

Date: 2026-09-06 UTC. Project: `.`.

The existing pipeline is usable as a reference for the public-data work, with
important safeguards needed before reuse. Filtering is reproducible and the saved
split groups checked here are disjoint. Feature edge cases, malformed split input,
and pseudo-label handling have confirmed defects. No legacy code, recordings,
models, or figures were modified.

## Evidence and reproduction

Run directory: `research/runs/20260906T1_legacy_audit`.

From the project root:

```sh
venv/bin/python -B -m scripts.audit_legacy --out research/runs/20260906T1_legacy_audit
venv/bin/python -B research/runs/20260906T1_legacy_audit/analyze_inventory.py
```

The first command was run successfully. For another run, supply a **new** output
directory: the audit refuses to overwrite an existing one. The second command
summarizes the saved inventory and recorded raw-signal hashes.

- `config.json`: exact environment, preprocessing, source manifests, selection,
  RF settings, adaptation access, and scope.
- `input_sha256.json`: hashes of 978 protected files, including the new audit
  script and existing code/configuration/data/models/figures. None changed during
  the run. File hashes identify the code and inputs for this run.
- `signal_inventory.csv`, `processed_reproduction.csv`, `directory_inventory.json`:
  full signal inventory, timing checks, and filtering reconstruction errors.
- `manifest_resolution.json`, `split_checks.json`, `inventory_assessment.json`:
  label-path candidates, partition checks, and raw-signal identity checks.
- `diagnostics.json`: six behavioral checks, including four confirmed failures.
- `smoke_manifest.csv`, `feature_schema.json`, `window_counts.json`,
  `trial_predictions.csv`, `window_predictions.csv`, `smoke_metrics.json`:
  exact smoke selection, representations, predictions, and metrics.
- `summary.json`: execution summary and preservation check.

Runtime: Python 3.13.0; NumPy 2.4.1; SciPy 1.17.0; pandas 2.3.3;
scikit-learn 1.8.0; joblib 1.5.3; pyserial 3.5. Machine: Mac15,12,
Apple M3, 16 GiB RAM, macOS 15.6.1 arm64. Existing environment imports succeeded.
All 40 Python source files in the inventory parsed successfully. This does not
mean every script or hardware path was executed.

## Data and timing

The current raw tree contains one participant identifier, S01:

| Session folder | Raw trials | Interpretation of folder name |
|---|---:|---|
| 20260131 | 96 | unshifted |
| 20260201 | 6 | unshifted |
| 20260201_shift1cm | 57 | custom shifted condition |
| 20260207_shift1p5cm | 60 | custom shifted condition |
| 20260208_shift1p5cm_light | 30 | custom shifted/light condition |

These are local folder annotations, not independently verified measurements of
electrode position. Labels are `extend`, `fist`, and `rest`; raw channels are
`ch0` and `ch1`, and timestamps are recorded as `t_us`. Electrode anatomy and
calibration into physical voltage units are not documented in this checkout.

There are 249 current raw and 249 fixed processed CSVs, 159 older processed CSVs,
203 archived CSVs, and 12 uncontrolled archived CSVs: **872 signal files** total.
Copies across processing stages are not independent recordings. Archived trees
also contain S02 files; their metadata and historical models are not a validated
cross-participant benchmark. There are 29 label/split manifests and 19 saved model
files. Existing model binaries were hashed, not loaded or used for this smoke run.

Every current raw trial has a median timestamp interval of 2,000 microseconds
(500 Hz). Three files have a large interval between their **first and second**
rows:

| Raw file under `data/raw/S01/` | First interval |
|---|---:|
| 20260201_shift1cm/fist_011.csv | 120.991 s |
| 20260208_shift1p5cm_light/extend_002.csv | 255.021 s |
| 20260208_shift1p5cm_light/rest_001.csv | 2.112 s |

These are timestamp discontinuities, not proof of their cause. Filtering and
windowing currently treat rows as uniformly sampled and ignore timestamps.
Do not calculate duration naively across these jumps or silently bridge them in
a future analysis. The 18 trials selected for the smoke test have none of these
jumps. No raw data were repaired or removed.

All 215 archived signal CSVs have median intervals corresponding to 800 Hz;
many also have large timestamp intervals. The current 500 Hz configuration must
not be applied to those files without a separately verified protocol. This
audit does not reconstruct the notebook's earlier sampling-frequency mistake.

Recomputing all 249 fixed processed trials using current code reproduces both
filtered channels with maximum absolute error **2.84e-14**. Their timestamp and
raw-channel columns exactly match the corresponding raw CSVs. Current filtering
uses median DC removal, a 60 Hz notch (Q=30), and a fourth-order 20–200 Hz
Butterworth bandpass via forward/backward filtering at 500 Hz. Reproduction
establishes consistency with current code, not acquisition validity.

## Splits and evaluation

Four train/validation/test groups and four calibration/evaluation pairs have no
within-group overlap by either recorded filename or raw timestamp/channel hash.
The fixed unshifted source split is 60/21/21 trials, balanced across the three
labels. Every manifest row has a matching signal somewhere in the inventory;
this suffix-based audit is not proof that every script's configured data root
resolves correctly. Raw, old processed, and fixed processed copies must remain
distinguished by explicit roots.

The two stored 1.5 cm calibration budgets use different scoring trials: the
five-trials-per-class budget scores 45 trials, the three-trials-per-class budget
scores 51, and only 38 scoring trials are shared. These cannot directly support
a paired fixed-scoring-set budget comparison. The new public-data protocol
should reserve scoring trials first and nest calibration budgets separately.

Filename/label mismatches were confined to the two explicitly shuffled-label
control manifests. Those controls should remain separate from ordinary training.

`scripts/evaluate_coral.py` fits alignment using a separate calibration manifest
and correctly transforms target features toward source statistics after applying
the source scaler. A synthetic anisotropic test verified the direction (relative
covariance error 7.51e-7). The script itself does not reject overlapping manifests,
so the new runner must enforce disjointness before feature extraction.

`scripts/autocorrect/run_autocorrect_eval.py` estimates signal alignment from the
whole trial being scored. That grants access to scoring signals and is a
different protocol from fitting solely on separate calibration trials. It must
not be pooled with limited-calibration results without explicitly declaring the
different data access. No scoring labels enter that alignment computation.

## Confirmed defects and other reuse constraints

| Finding | Evidence | Consequence |
|---|---|---|
| Zero-signal spectral extraction crashes | `src/features.py:41`; zero window raises IndexError | Reject or handle degenerate windows explicitly before reuse |
| Constant-signal Higuchi feature becomes NaN | `src/features.py:84`; constant-window check fails | Require a documented finite-value policy; do not silently change historic features |
| Conflicting labels can leak one file across splits | `scripts/split_by_file.py:21`; synthetic conflicting-label test creates five overlapping file IDs | Validate unique trial identity and one label per trial before allocation |
| Pseudo-labels store class indices rather than class values | `src/adapt.py:15`; predicted `fist` stored as integer 1 | Self-calibration is incompatible with the string gesture labels; keep deferred |

A single-row CORAL calibration check remained finite but produced a transform
norm about 14,967 and a held-out output norm about 44,923. Finite numbers alone
do not establish stability. Small-budget alignment needs explicit minimum sample,
conditioning, regularization, and fallback rules developed without scoring data.

Other code-review findings, not executed as hardware tests:

- Training has a basename fallback while evaluation uses stricter paths. Repeated
  gesture filenames across sessions make implicit fallback unsafe for new data.
- Training/evaluation restrict inputs to two named channels. Keep the feature
  primitives but generalize channel handling for the public dataset.
- The RF retains RMS, MAV, and waveform length in its primary 18-feature,
  two-channel representation. `train.py` has no no-amplitude option even though
  evaluation supports it; the no-amplitude model training provenance is incomplete.
- Feature tables do not carry participant/session/trial identity after stacking;
  model artifacts omit a complete preprocessing/feature schema contract.
- Whole-trial offline filtering and short-window live filtering have different
  boundary behavior. Live prediction supports one channel while default training
  uses two. The live application is not validated for the dual-channel models.
- `capture.py` checks duration only after the stream yields a valid row; the
  advertised no-data timeout does not interrupt a silent stream.
- Several plotting scripts contain hard-coded confusion matrices. Their images
  are historical reported results, not freshly generated prediction evidence.
- Requirements are unpinned and omit matplotlib despite plotting/capture imports.
  Existing ignores omit `data/processed_fix`, archives, and future public datasets.
  Exclude these data and artifact directories from source control.

## Bounded smoke test — not a benchmark

Used fresh RF training with amplitude features: six source trials, 2,003 training
windows, 18 features, seed 42. Windows are 250 ms (125 samples), with 50 ms
(25 sample) stride, split by whole trial before windowing. Calibration uses six
separate target trials, two per gesture, totaling **102.312 seconds of recorded
signal**; this is not measured user interaction time. Target calibration labels
determine balanced selection, but CORAL fitting uses only their signal features.
Source train and test trials here come from 20260131; target trials come from the
custom 20260201_shift1cm folder. Selection was fixed by sorted names before scoring.

| Scoring set / method | Trials correct | Balanced accuracy | Macro-F1 |
|---|---:|---:|---:|
| Source / RF | 3/3 | 1.000 | 1.000 |
| Shifted target / RF | 3/3 | 1.000 | 1.000 |
| Same shifted target / RF + CORAL | 2/3 | 0.667 | 0.556 |

CORAL changed the selected rest trial to a fist prediction. Preserve this result;
do not tune against these three trials. The sample is far too small and restricted
to one person to support a performance or novelty claim. No uncertainty interval
is reported for this smoke test. Historical saved-model metrics were not
reproduced, and no GRABMyo experiment was run.

RF fitting took about 0.24 seconds and the audit plus smoke took about 8.26 seconds
on this machine; these are one-run elapsed timings, not deployment benchmarks.
Peak memory was not measured. Raw predictions and the exact selection are saved.

## Reuse and next milestone

Reuse the per-channel amplitude/spectral primitives after addressing degenerate
inputs, the whole-trial windowing approach, RF constructor, and the target-to-source
CORAL formulation. Build strict dataset metadata, explicit channel/sample-rate
handling, split manifests with fixed scoring trials, and result persistence around
them. Preserve the custom pipeline as a separately identified legacy workflow.

T1 is complete. T2 is next: verify official GRABMyo metadata and implement a loader
plus leakage-checked split manifests for a small documented smoke subset. Harden
any reused feature routines before relying on them. Do not assume the legacy
500 Hz filters or two-channel naming apply to public data. T0's separate planning
acknowledgment remains pending and does not block local implementation.
