# Reproducibility

## What can be checked from this archive

1. **Saved-summary arithmetic:** `python analysis_supplement/reproduce_summary.py` verifies selected participant means and the primary Holm correction using the Python standard library.
2. **Method and protocol tests:** install `requirements-review.txt`, then run the five focused test commands in [README.md](README.md#installation-and-tests). The broader legacy test collection has additional dependencies and is outside this focused check. Synthetic fixtures exercise the implemented methods and separation rules; the optional full-cache test is explicitly skipped when its manifest is absent.
3. **Source integrity:** `MANIFEST_SHA256.json` records the SHA-256 of each release file except the manifest itself. `software_validation.json` records what was actually tested. `readability_validation.json` covers the earlier comparison with preserved source.

These checks do not retrain the study or reconstruct every prediction from raw recordings.

## What a full repeat requires

| Input | Availability |
|---|---|
| GRABMyo 1.1.0 raw records and provider checksums | Download from PhysioNet under its terms |
| SeNic pinned raw archive | Download from the provider; use the included archive verification ledger |
| CSL-HDEMG records | Obtain under the original provider's terms |
| Shared-pretraining checkpoints and participant-enrolled models | Not included; stable retrieval route pending |
| Development caches, complete run JSON files, and prediction banks | Not included; stable retrieval route pending |
| Documented final protocol | Included under `research/runs/20260625_confirmatory_protocol/` |
| Historical package versions | `environment_public.json`, `environment_features.json`, and comparator run records |

`ORIGINAL_REPRODUCTION_NOTES.md` preserves the earlier workflow order, with a correction at its start explaining which files are absent from this archive. Some paths and rendering fonts are specific to the original workstation. The original expanded-manuscript rendering scripts do not build the current Sensors manuscript.

Before a complete repeat, arrange the missing artifacts, reconstruct the documented directory layout, and use fresh output directories. Use `python -m scripts.<name>` from the project root so local imports resolve. Preserve the documented participant allocation, trial roles, preprocessing, seeds, calibration budgets and scoring sets. Do not use already-scored final participants to tune methods.

## Environments and numerical scope

The review requirements pin the original public analysis environment. They do not combine it with the feature-extraction environment or reproduce the original CUDA image. Burg autoregressive features used librosa 0.11.0 with the versions in `environment_features.json`. The comparator ran with CUDA/PyTorch 2.8.0 and MPS/PyTorch 2.14.0; hardware and run settings are recorded separately.

The readability revision was compared with the preserved implementation using synthetic checkpoints and saved-study statistics. This establishes consistency for those checks, not identical training across hardware or a clean-machine repeat of the entire paper.

## Dates and manuscript version

The final manuscript and supplementary document are maintained separately; they are not distributed in this code repository. [PROVENANCE.md](PROVENANCE.md) documents the protocol version, included records, and verification scope.
