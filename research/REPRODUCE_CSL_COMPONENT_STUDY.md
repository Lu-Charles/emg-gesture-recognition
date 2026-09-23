# Reproducing and auditing the CSL component study

Canonical project directory: `.`.

This is a local research package. It has not been published, submitted or pushed. The project is not a Git repository; the run protocols record SHA256 hashes and preserve script snapshots. A separate upstream repository supplies the credited author code at a pinned commit.

## Evidence locations

| Item | Location relative to this project |
|---|---|
| Earlier author-recipe component checks and single-case replay | `research/runs/20260907_csl_author_recipe_v1/` |
| Eight calibration choices on development participant 1 | `research/runs/20260907_csl_components_development/` |
| Reserved-participant protocol, data manifest and model outputs | `research/runs/20260907_csl_components_confirmatory/` |
| Prespecified spatial-search seed sensitivity | `research/runs/20260907_csl_search_seed_sensitivity/` |
| Complete methods and caveats | `research/CSL_COMPONENT_METHODS.md` |
| Related-work distinction | `research/CSL_COMPONENT_CONTRIBUTION.md` |
| Historical decisions and failed attempts | `research/DECISIONS.md`, `research/EXPERIMENTS.md` |

The primary run's `protocol.json` is frozen before model evaluation. Earlier versions and outcome-blind execution/data-QC amendments are retained. `manifest.json` identifies each raw trial, its raw hash, gesture/repetition/session, processed frame boundaries and cache hash. The two missing repetitions are listed in `preprocessing.json`; no duplicate trial was inserted.

Each participant directory contains five source checkpoints, one directory for each ordered session pair, one directory for each calibration gesture, and the all-class reference. Adapted checkpoints include separate flags needed to reproduce the input transform and gain behavior. `results.json` files contain actual metrics and training logs; `predictions.npz` files contain saved predictions, labels and scoring-trial identities.

## Audit commands for this completed local workspace

Use the existing `.venv-public` environment from the canonical project directory. These commands read model/data artifacts and write audit or report files; they do not contact authors or submit anything.

```sh
.venv-public/bin/python -m scripts.verify_csl_author_recipe
.venv-public/bin/python -m scripts.verify_csl_components development
.venv-public/bin/python -m scripts.verify_csl_components final
.venv-public/bin/python -m scripts.report_csl_components
```

The report command requires a completed, passing final verification. Check the actual run status before invoking it. Historical audit files should be copied before a rerun if their exact original timings are to be retained. The primary final verifier uses an inference batch size different from training-time scoring and checks frame predictions, majority votes, calibration fit, allowed-data gain means and frozen parameter/buffer state. It also independently reconstructs predetermined raw frames with a separate spatial median filter.

## Rebuilding experiments

Obtain CSL-HDEMG through the provider's registration process and retain its usage conditions. The original five split archive files are expected in the user's Downloads folder, named `cslhdemgsplit.zip.partaa` through `cslhdemgsplit.zip.partae`. `scripts.prepare_csl_sal` audits the archive without creating another combined 12 GB copy. Its saved archive-member hashes are reused by later preprocessing.

The author source snapshot is pinned to `7c5a075a58dff06566b82965f8235ba744377fb8` in the public `joao-binenbojm/spatial-adaptation-layer` repository. Its file ledger is at `research/runs/20260907_csl_author_recipe_v1/source/sha256.json`. This snapshot is a later single-gesture implementation, not a confirmed release of the manuscript's original experiments.

The main execution stages are:

```sh
.venv-public/bin/python -m scripts.csl_author_recipe check
.venv-public/bin/python -m scripts.csl_author_recipe prepare
.venv-public/bin/python -m scripts.csl_author_recipe run
.venv-public/bin/python -m scripts.csl_component_study
.venv-public/bin/python -m scripts.csl_confirmatory prepare
.venv-public/bin/python -m scripts.csl_confirmatory run
.venv-public/bin/python -m scripts.csl_search_sensitivity run
```

Use a separate copy of the workspace and separate output directories for a fresh replication; do not overwrite the evidence used in the paper. The current research scripts use fixed local output paths and are not yet a portable installation package. Frozen protocols are required before running the later stages, and source/data bootstrap artifacts must already be present. A public distribution should provide the wrappers and retrieval instructions without assuming permission to redistribute third-party source or raw recordings.

`environment.json` records the actual Python, NumPy, SciPy, PyTorch and plotting versions, CPU execution and worker configuration. Source training and adaptation use the settings in the protocols, with no score-selected epochs. Four computational workers read immutable code and data, write separate participant directories, and return results to a coordinator; they do not edit repository code concurrently.

## Interpretation rules

Use the four participant summaries as the independent observations. Session pairs, gesture choices and frames are repeated measurements. “Other gestures” means gestures present during source training but absent from target calibration. The results concern this eight-class, offline, natural-session-variation experiment.

Report the 90 seconds of target rest used by segmentation alongside the three seconds of labeled single-gesture signal. The all-class reference has both more labeled classes and more optimizer updates. Do not present either the rest cost or optimization difference as absent.

## Loading an individual checkpoint

Use `scripts.load_csl_checkpoint.load_checkpoint(path)` with the absolute `.pt` path. It reads the adjacent state metadata and handles the input-order variant explicitly. The lower-level helper in the original recipe verifier builds the original author forward function and is not a general loader for the input-order control. An unrecognized forward variant raises an error.

The input-order and identity-tie experiments are separately logged at `research/runs/20260907_csl_input_order/` and `research/runs/20260907_csl_identity_ties/`. Their verification commands are:

```sh
.venv-public/bin/python -m scripts.verify_csl_components input-order
.venv-public/bin/python -m scripts.verify_csl_components identity
.venv-public/bin/python -m scripts.verify_csl_search_sensitivity
.venv-public/bin/python -m scripts.report_csl_controls
```

These commands require the corresponding completed experiments. `scripts.finish_csl_study` records the actual order and status of the frozen stages in the primary run's `stage_status.json`.

## Portable analysis review bundle

`research/artifacts/csl_review_bundle/` contains the manuscript, figures, per-case records, protocols and verification reports. Its `reproduce_analysis.py` recomputes primary participant means, paired contrasts, descriptive bootstrap intervals, high-fit negative-transfer counts, control means and search-seed means from the included records. It requires NumPy only and checks numerical agreement to absolute tolerance 1e-12.

This is an analysis-only bundle: it does not retrain models or repeat the original checkpoint/raw-signal verification. Raw recordings, processed signal caches, model checkpoints and third-party source remain in the full local project. The bundle is prepared locally and has not been submitted or published.
