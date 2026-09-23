# Gesture coverage and vocabulary retention during brief EMG calibration

Research code for an empirical study of how brief calibration affects recognition of the full EMG gesture vocabulary. The analyses cover GRABMyo session changes, SeNic electrode positions, and a CSL-HDEMG diagnostic. The package includes the documented evaluation protocol, analysis scripts, participant summaries, and tests.

The manuscript and supplementary document are maintained separately from this code repository. The numerical analysis supplement remains included under `analysis_supplement/`. The two corrected primary comparisons did not establish superiority.

## Main findings

At the same two-recording calibration budget, recognizing the prompted gestures well did not guarantee retention of the full vocabulary.

| Final GRABMyo comparison | Overall trial accuracy | Calibrated-gesture recall |
|---|---:|---:|
| Frozen recognizer | 81.37% | 81.34% |
| Ordinary head update | 77.09% | 98.15% |
| Head update with source replay | 82.44% | 96.55% |

These are two-gesture results at the primary 100-step endpoint, averaged within each of 15 final participants across three seeds. Replay improved overall accuracy by 1.07 percentage points but reduced omitted-gesture accuracy by 0.81 points. Neither of the study's two primary superiority comparisons passed Holm correction.

For 24 final SeNic participants, two distinct gestures improved rotation-plus-gain accuracy by 5.80 points over repeating one gesture at the same recording count; this was a descriptive secondary comparison. The supplementary covariance-alignment study reused eight development participants and did not establish an alignment-specific advantage. Complete summaries and the fixed protocol are included below.

## Quick start: verify saved summaries

Python 3 is sufficient for this check; no packages or datasets are required.

```sh
python analysis_supplement/reproduce_summary.py
```

The command prints JSON containing `"passed": true`, selected participant means, and the two-test Holm correction. It reads bundled summaries without fitting models or reconstructing predictions. The supplementary comparison is in `analysis_supplement/comparator/combined_summary.json`; its values are percentages, and its intervals use eight participant averages.

## Installation and tests

Use Python 3.13. From this directory:

```sh
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-review.txt
python -m unittest discover -s tests -p "test_alignment_comparator.py" -v
python -m unittest discover -s tests -p "test_final_coverage.py" -v
python -m unittest discover -s tests -p "test_decisive_calibration.py" -v
python -m unittest discover -s tests -p "test_rotation_coverage.py" -v
python -m unittest discover -s tests -p "test_coverage_diagnostics.py" -v
```

On Windows, activate with `.venv\Scripts\activate` instead. The documented validation platform and actual outcome are recorded in `software_validation.json`; other platforms have not been tested here.

The focused paper suites check covariance calculations and gradients, unchanged source checkpoints, replay reproducibility, calibration budgets, collapse diagnostics, rotation ambiguity, and trial separation. Unit tests use synthetic inputs. A separate integration test uses the full GRABMyo manifest when available and reports an explicit skip otherwise. Run with normal Python: legacy experimental safeguards use assertions and must not be disabled with `-O` or `PYTHONOPTIMIZE`.

`requirements-review.txt` supports the review tests and analysis imports. Feature extraction additionally uses librosa 0.11.0 in the separately recorded feature environment. Historical PDF rendering also needs ReportLab and external fonts. Neither step is part of the quick check above. See [Reproducibility](REPRODUCIBILITY.md) before starting an experiment.

Earlier hardware, acquisition, and public-data pilot scripts are retained as research context. Their tests and optional dependencies extend beyond the focused paper review environment. `research/AUDIT.md` records known legacy limitations. The commands above validate the paper suites.

## Repository structure

| Component | Implementation |
|---|---|
| Calibration schedules and head adaptation | `scripts/decisive_calibration.py`: `schedules`, `fit_head` |
| Feature covariance alignment | `src/alignment_comparator.py` |
| Full-network adaptation and controls | `scripts/run_alignment_comparison.py` |
| Rotation and gain estimation | `scripts/rotation_coverage.py`: `profile_rotation` |
| Trial and cohort validation | `src/final_coverage.py`, `scripts/pilot_gesture_coverage.py` |
| Participant-level statistics | `scripts/analyze_coverage_study.py` |
| Saved results and supplementary comparison | `analysis_supplement/` |
| Documented protocol and run provenance | `research/runs/`, `comparator_run_records/` |
| Automated checks | `tests/` |

Run experiment commands as modules from this directory, for example `python -m scripts.run_alignment_comparison --help`. Many historical scripts expect the original project layout and pretrained models. They are not standalone raw-data reproduction commands.

## Data and experiment provenance

Comparator records use September 23, 2026 UTC, corresponding to September 22 evening in America/Los_Angeles. The inspected JSON execution timestamps precede their containing commits. GitHub repository creation and upload times describe separate events. See [PROVENANCE.md](PROVENANCE.md) for the scope of the date and checksum checks.

- GRABMyo 1.1.0: <https://physionet.org/content/grabmyo/1.1.0/>.
- SeNic, pinned revision: <https://github.com/BoZhuBo/SeNic/tree/a4c12f7daab28a80d557677ae8dbcef0d7871ba2>.
- CSL-HDEMG: obtain from the original provider under its access terms.

Follow each provider's license and citation requirements. Raw data, trained model banks, and complete prediction archives are not distributed here. Stable access to the necessary model and prediction artifacts remains to be arranged.

The current local protocol documents the analysis, lists June 25, 2026 as its finalization date, and is not an external preregistration. The manuscript uses the same date and protocol checksum and describes comparisons as protocol-defined. The protocol copies and checksum files identify the current version; see [PROVENANCE.md](PROVENANCE.md) for verification scope. The supplementary comparator reused development participants; all six tested alignment coefficients are reported. The readability edits preserve calculation order, seeds, function parameters, and result keys. `readability_validation.json` records comparisons with the original implementation. Historical run records retain the hashes of the code actually executed.

## Citation, reuse, and support

`CITATION.cff` identifies this software candidate. The manuscript has not yet been assigned a publication DOI. Original project code is available under the [MIT License](LICENSE), selected by Charles Lu. Dataset rights and third-party dependency licenses remain with their respective providers.

For a problem report, give the command, Python/package versions, full traceback, and expected behavior to the corresponding author. Do not include restricted raw data. Proposed changes should include a focused test and identify any effect on data splits, calibration access, or reported results. Scientific changes require a new run record; preserve existing results.
