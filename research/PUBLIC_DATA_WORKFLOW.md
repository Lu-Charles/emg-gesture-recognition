# Public-data workflow

Run from `.`. This pipeline uses GRABMyo v1.1.0 only. Charles's recordings are excluded. It lives alongside the historical project without changing the historical environment.

## Environment and provenance

Use `.venv-public/bin/python`. Exact dependencies are in `requirements-public.lock.txt`. Public data are stored in `data/public/grabmyo/1.1.0/`, excluded from Git. The original official metadata/checksum retrieval is logged under `research/runs/20260906_grabmyo_metadata/`. Downloads verify against that published checksum list and keep cached originals. Current dataset license is CC BY4.0; cite the dataset paper and release in any manuscript.

## Fixed allocations

`research/runs/20260906_grabmyo_preparation_v1/` contains the fixed participant allocation and complete trial manifest. Do not regenerate it under another seed to improve results. Training, development and final participants are disjoint. Final signal data must stay outside development runs.

Each development/final participant uses day-1 enrollment. Days 2 and 3 have three calibration trials and four scoring trials per class. Scoring is identical for all budgets. The initial pilot used only development participants 3 and 7, day 1 to day 2; the complete development pipeline below uses all eight development participants and both later days.

## Reproduce development pilots

Use a new output directory for every run; scripts refuse to overwrite one. Existing outputs remain evidence.

```sh
.venv-public/bin/python -B -m unittest discover -s tests -p 'test_grabmyo.py' -v
.venv-public/bin/python -B -m unittest discover -s tests -p 'test_emg_cnn.py' -v
.venv-public/bin/python -B -m scripts.pilot_grabmyo --preparation research/runs/20260906_grabmyo_preparation_v1 --out research/runs/my_classical_run
.venv-public/bin/python -B -m scripts.verify_grabmyo_pilot --run research/runs/my_classical_run
.venv-public/bin/python -B -m scripts.pilot_emg_cnn --classical research/runs/my_classical_run --out research/runs/my_neural_run
.venv-public/bin/python -B -m scripts.verify_emg_cnn_pilot --run research/runs/my_neural_run --preparation research/runs/20260906_grabmyo_preparation_v1
```

These are development pilots, not the final experiment. The initial neural pilot trains a separate model on each person's day 1; shared pretraining across the 20 training participants is implemented in the complete pipeline below. Saved configurations distinguish these scopes. Report 250 ms held-gesture window metrics, not live-control performance. Calibration budgets are 85/170/255 seconds of recorded data across 17 classes.

Full pretraining needs a batched or memory-mapped signal loader; do not materialize every overlapping window for the20-person corpus in RAM at once. The small resident-batch throughput smoke excludes that data-loading cost.

## Shared pretraining and complete development comparison

The shared pipeline is now implemented. It uses all 20 representation-training participants and all eight development participants, with both later days. It downloads no final-participant signals. The official AWS mirror supports reusable connections; every downloaded file is checked against the published SHA256 list.

Use fresh output paths when reproducing. These example commands run sequentially to avoid competing for memory and to make timings interpretable:

```sh
.venv-public/bin/python -B -m scripts.cache_grabmyo --preparation research/runs/20260906_grabmyo_preparation_v1 --out data/public/grabmyo/my_cache
.venv-public/bin/python -B -m scripts.train_shared_emg --cache data/public/grabmyo/my_cache --out research/runs/my_shared_run
.venv-public/bin/python -B -m scripts.classical_grabmyo_development --cache data/public/grabmyo/my_cache --out research/runs/my_classical_development
.venv-public/bin/python -B -m scripts.verify_shared_development --run research/runs/my_shared_run --kind neural
.venv-public/bin/python -B -m scripts.verify_shared_development --run research/runs/my_classical_development --kind classical
.venv-public/bin/python -B -m scripts.report_shared_development --neural research/runs/my_shared_run --classical research/runs/my_classical_development --out research/runs/my_shared_report --report research/MY_SHARED_REPORT.md
```

The completed cache at `data/public/grabmyo/cache_20260906_v1/` can be reused instead of rebuilding it. It contains whole-trial physical signals and classical features; overlapping neural windows are created in bounded buffers. Shared and random initialization use the same training-group normalization and development settings. See `PUBLIC_METHODS.md` for the full design. Checkpoints, caches and raw data stay out of Git.

## Longer scratch-enrollment control

The convergence control reuses the verified training-group scaler and runs a single scratch trajectory through 80 enrollment epochs per development participant. Checkpoints at 20/40/80 retain uninterrupted Adam state and shuffle order. Epoch 80 is the fixed primary endpoint; earlier checkpoints are retained as diagnostics. Population pretraining is not repeated.

```sh
.venv-public/bin/python -B -m unittest tests.test_emg_enrollment tests.test_shared_emg -v
.venv-public/bin/python -B -m scripts.scratch_convergence --prior research/runs/20260906_shared_pretraining_v2 --out research/runs/my_scratch_control
.venv-public/bin/python -B -m scripts.verify_shared_development --run research/runs/my_scratch_control --kind neural
.venv-public/bin/python -B -m scripts.report_scratch_convergence --run research/runs/my_scratch_control --out research/runs/my_scratch_report --report research/MY_SCRATCH_RESULTS.md
```

Verification additionally requires exact reproduction of the original 112 scratch model weights and metrics at epoch 20, preserved enrollment-loss prefixes, and uninterrupted optimizer step counts. Longer enrollment adds computation while using the same participant recordings. It does not establish an optimally tuned scratch baseline.

## Adaptive calibration duration pilot (completed 2026-09-06)

Run from `.`. Use new output names when repeating; existing runs are preserved. These commands reuse the verified shared recognizer checkpoints.

```sh
.venv-public/bin/python -B -m unittest tests.test_calibration_stopping tests.test_shared_emg -v
.venv-public/bin/python -B -m scripts.pilot_calibration_stopping --prior research/runs/20260906_shared_pretraining_v2 --out research/runs/20260906_stopping_observations_v1
.venv-public/bin/python -B -m scripts.evaluate_calibration_stopping --features research/runs/20260906_stopping_observations_v1 --out research/runs/20260906_stopping_policy_v1
.venv-public/bin/python -B -m scripts.verify_calibration_stopping --observations research/runs/20260906_stopping_observations_v1 --policies research/runs/20260906_stopping_policy_v1
.venv-public/bin/python -B -m scripts.report_calibration_stopping --run research/runs/20260906_stopping_policy_v1 --out research/runs/20260906_stopping_report_v1
```

Report: CALIBRATION_STOPPING_RESULTS.md. No final-participant signals are accessible through this pilot. All hyperparameter fitting groups participants; multiple order episodes must retain that grouping in later experiments.

## Six-order robustness (completed 2026-09-06)

From `.`; choose new output directories to rerun. Existing source/model/config snapshots remain in each completed run.

```sh
.venv-public/bin/python -B -m unittest tests.test_calibration_stopping tests.test_shared_emg -v
.venv-public/bin/python -B -m scripts.calibration_order_observations --prior research/runs/20260906_shared_pretraining_v2 --original-observations research/runs/20260906_stopping_observations_v1 --out research/runs/20260906_stopping_order_observations_v1
.venv-public/bin/python -B -m scripts.evaluate_calibration_stopping --features research/runs/20260906_stopping_order_observations_v1 --frozen-policies research/runs/20260906_stopping_policy_v1 --out research/runs/20260906_stopping_order_policy_v1
.venv-public/bin/python -B -m scripts.verify_stopping_orders --observations research/runs/20260906_stopping_order_observations_v1 --policies research/runs/20260906_stopping_order_policy_v1
.venv-public/bin/python -B -m scripts.report_stopping_orders --run research/runs/20260906_stopping_order_policy_v1 --out research/runs/20260906_stopping_order_report_v1
```

`order` identifies correlated calibration replay episodes. Participant is the grouping unit throughout fitting, selection and uncertainty analysis. Primary frozen-policy robustness and the separate all-order-training extension are both retained. Final participants remain inaccessible.

## Additional training seeds (completed and verified, 2026-09-06)

The recognizer now accepts `--seed`, default42. The same seed controls neural initialization and pretraining/enrollment/update shuffle order. Downstream observations derive it from the verified recognizer config. Fixed data, normalization and policy selection settings are unchanged.

The exact seed0 training command launched separately:

```sh
.venv-public/bin/python -B -m scripts.train_shared_emg --cache data/public/grabmyo/cache_20260906_v1 --out research/runs/20260906_shared_seed0_v1 --seed 0
```

After that recognizer completes, the following commands run remaining steps sequentially, with per-stage logs and exit-status ledgers. The seed1 command includes training. These are the current run names and must not overwrite completed outputs; for a fresh reproduction use new names in the runner or invoke its listed component commands with new output directories.

```sh
.venv-public/bin/python -B -m scripts.run_seed_stopping --seed 0 --existing-recognizer
.venv-public/bin/python -B -m scripts.run_seed_stopping --seed 1
.venv-public/bin/python -B -m scripts.report_stopping_seeds --runs research/runs/20260906_stopping_order_policy_v1 research/runs/20260906_seed0_order_policy_v1 research/runs/20260906_seed1_order_policy_v1 --out research/runs/20260906_stopping_seed_report_v1
```

The runner stops on any failed stage; inspect and preserve its log before retrying. It verifies the recognizer, extracts original-order states, fits and verifies original policies, expands to all orders, fits all-order policies and verifies them. The report requires all three seeds and matching non-seed protocols, retains all outcomes, and groups uncertainty by the eight participants. It does not access final data.

The two new seed runs and combined report are complete. Report arithmetic can be checked with `.venv-public/bin/python -B -m scripts.verify_stopping_seed_report --run research/runs/20260906_stopping_seed_report_v1`. TRAINING_SEED_RESULTS.md separates the prespecified comparison from a post-hoc simple-policy mixture diagnostic. No model choices were changed for that diagnostic.
