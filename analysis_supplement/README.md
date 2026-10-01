# Analysis supplement

This folder contains the evaluation protocol, participant-level numerical summaries, table values, comparator results, and verification records. The summary script reconstructs selected means and the primary Holm adjustment from saved results.

Run `python analysis_supplement/reproduce_summary.py` from the repository root with Python 3; only the standard library is required. This check does not rerun neural training, reconstruct raw-data predictions, or independently calculate t intervals. The saved statistics verification report documents a separate check against confusion matrices and SeNic predictions.

The included software test log records 24 tests: 23 passed and one skipped because the optional GRABMyo final cache was unavailable. `software_validation.json` identifies the tested code revision, the log checksum, and the validation scope. This is a retained validation record, not a new test run performed during packaging.

Raw EMG recordings, trained model banks, and complete prediction archives are not included. These materials support numerical consistency and software checks, but do not constitute a full experimental reproduction. Obtain GRABMyo 1.1.0 from https://physionet.org/content/grabmyo/1.1.0/ and SeNic from its documented source at revision https://github.com/BoZhuBo/SeNic/tree/a4c12f7daab28a80d557677ae8dbcef0d7871ba2, subject to the providers' terms.

The internal comparator work diary is omitted from this publication-facing folder. The numerical comparison, all six tested alignment coefficients, bridge checks, and run provenance remain available in `comparator/` and `../comparator_run_records/`. `manifest.json` records the checksums of the files in this folder.
