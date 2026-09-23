> Archive correction (September 22, 2026 Pacific time; September 23 UTC): These are historical instructions for a larger local bundle. The checkpoints, PDF, figures and complete artifact index described below are **not included in this code archive**. References to supplied checkpoints do not establish availability. Follow README.md and REPRODUCIBILITY.md for this package's supported checks and missing prerequisites.

# EMG coverage study: local reproducibility bundle

This bundle accompanies the expanded manuscript. It contains the frozen protocol, code and dependencies from this workflow, audited participant statistics, figures, local artifact index, verification reports, and three small shared-pretraining checkpoints. Raw datasets, per-case adaptation model banks and exhaustive result JSON/prediction archives remain in the canonical project and are not included here. Nothing has been published or submitted.

## Rebuild the document from audited summaries

Use the recorded Python environments in environment_public.json and environment_features.json. The public environment supplies PyTorch/scikit-learn/matplotlib; the feature environment supplies librosa0.11.0 for the pinned Burg implementation. From this bundle's root, run `python -m scripts.write_coverage_manuscript`. PDF rendering additionally needs reportlab/Pillow, Liberation Serif and DejaVu Sans fonts; the renderer's default font directory points to the bundled runtime on the author's Mac and must be adjusted on another machine. The PDF is already included for immediate reading.

## Repeat final evaluation from permitted public downloads

This is an outline of the dependency order, not a claim that this bundle was rerun on a clean machine. Output scripts deliberately reject existing run directories. Use a fresh copy when repeating. Keep the frozen protocol unchanged; altered methods on already-examined final participants are exploratory.

1. Retrieve GRABMyo1.1.0 and its official SHA256SUMS.txt into data/public/grabmyo/1.1.0. The included src/grabmyo.py records source URLs and label/trial allocation. Run `python -m scripts.cache_final_coverage grabmyo`. For SeNic, run `python -m scripts.retrieve_senic_final_archives`, which uses the included pinned Git-blob ledger and downloads only the24frozen participants. Then run `python -m scripts.cache_final_coverage senic` in the feature environment. No raw data are redistributed in this bundle.
2. Run `python -m scripts.enroll_final_coverage` with the supplied verified shared-pretraining checkpoints. This uses only each final person's day1 source recordings. Their model inference and calibration follow later.
3. For each seed42,0,1, run decisive_final with `--seed` and `--out research/runs/20260916_decisive_final_seed<seed>`, then verify_decisive_final with `--run` that directory. Inspect failures rather than ignoring them.
4. In the feature environment run tdar_final_cache, then run classical_final_selection and verify_classical_final in the public environment.
5. Run rotation_final, rotation_final_reference and verify_rotation_coverage with `--run research/runs/20260916_rotation_final`; verify the cached features and archive identities as recorded in the audit ledgers.
6. The complete aggregate-analysis program also reads the development-stage runs and full-network control. Those full artifacts are indexed in artifact_index.json rather than shipped. With those available, run analyze_coverage_study, verify_coverage_statistics and write_coverage_manuscript. This distinction prevents presenting a summary-only report rebuild as a fresh scientific replication.

## Interpretation boundaries

No primary superiority claim survived the two-test Holm correction. The activity selector did not reproduce its development advantage. Secondary coverage and omitted-gesture effects are descriptive. Data subsets, scores and optimization settings must not be silently changed. Public-device offline findings do not establish custom-hardware or clinical performance. The discrete log-ratio rotation selector is implemented; the older continuous Schur-information proposal is not, and its general superiority was not tested.
