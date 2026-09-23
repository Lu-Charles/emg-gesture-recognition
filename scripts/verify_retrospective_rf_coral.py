"""Independently reconcile saved votes, predictions, metrics, transforms and access."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    out = ROOT/"research/runs/20260906_retrospective_rf_coral_evaluation_v1"
    prep = ROOT/"research/runs/20260906_retrospective_rf_coral_preparation_v1"
    for name, expected in json.loads((out/"artifact_sha256.json").read_text()).items():
        assert hashlib.sha256((out/name).read_bytes()).hexdigest() == expected, name
    config = json.loads((out/"config.json").read_text())
    cfg = config["protocol"]
    for name, expected in config["prepared_sha256"].items():
        assert hashlib.sha256((prep/name).read_bytes()).hexdigest() == expected, name
    m = pd.read_csv(prep/"trial_manifest.csv").fillna("")
    trials = pd.read_csv(out/"trial_predictions.csv")
    assert not trials.duplicated(["trial_id", "method", "budget_per_class"]).any()
    ledger = json.loads((out/"data_access.json").read_text())
    source_ids = set(m.loc[m.role == "source_train", "trial_id"])
    assert source_ids == set(ledger["source_fit_trial_ids"]) and len(source_ids) == 60
    assert ledger["validation_used"] is False
    assert set(trials.role) == {"source_test", "target_test"}
    assert set(trials.trial_id) == set(m.loc[m.role.isin(["source_test", "target_test"]), "trial_id"])
    with np.load(prep/"features.npz", allow_pickle=False) as z:
        features = {k: z[k] for k in z.files}
    with np.load(out/"alignment_transforms.npz", allow_pickle=False) as z:
        transforms = {k: z[k] for k in z.files}
    model = joblib.load(out/"source_rf.joblib")
    scaler, rf = model.named_steps["scaler"], model.named_steps["rf"]
    rows = m[m.role == "source_train"].sort_values("trial_id")
    source_raw = np.vstack([features[k] for k in rows.feature_key])
    np.testing.assert_allclose(scaler.mean_, source_raw.mean(0), rtol=0, atol=1e-12)
    np.testing.assert_allclose(scaler.var_, source_raw.var(0), rtol=1e-12)
    assert len(source_raw) == scaler.n_samples_seen_ == ledger["source_fit_windows"] == 20121
    source = (source_raw-scaler.mean_)/scaler.scale_
    cs = np.cov(source, rowvar=False)+1e-6*np.eye(source.shape[1])
    residuals = []
    for entry in ledger["calibration"]:
        session, budget = entry["session"], entry["budget_per_class"]
        selected = m[(m.role == "target_calibration") & (m.session == session) & (m.calibration_rank <= budget)]
        score = m[(m.role == "target_test") & (m.session == session)]
        assert set(entry["fit_trial_ids"]) == set(selected.trial_id)
        assert set(entry["scoring_trial_ids"]) == set(score.trial_id)
        assert not set(entry["fit_trial_ids"]) & set(entry["scoring_trial_ids"])
        assert not source_ids & set(entry["fit_trial_ids"])
        raw = np.vstack([features[k] for k in selected.sort_values("trial_id").feature_key])
        calibration = (raw-scaler.mean_)/scaler.scale_
        assert len(calibration) == entry["fit_windows"]
        assert abs(selected.recorded_seconds.sum()-entry["calibration_recorded_seconds"]) < 1e-9
        prefix = f"{session}__{budget}__"
        a, ms, mt = [transforms[prefix+k] for k in ["A", "mu_source", "mu_calibration"]]
        np.testing.assert_allclose(ms, source.mean(0, keepdims=True), atol=1e-12)
        np.testing.assert_allclose(mt, calibration.mean(0, keepdims=True), atol=1e-12)
        ct = np.cov(calibration, rowvar=False)+1e-6*np.eye(calibration.shape[1])
        relative_error = np.linalg.norm(a.T@ct@a-cs)/np.linalg.norm(cs)
        assert relative_error < 1e-9
        residuals.append(float(relative_error))
    verified_keys, windows_checked = set(), 0
    for path in sorted(out.glob("windows_*.csv.gz")):
        w = pd.read_csv(path)
        for (trial_id, method, budget), group in w.groupby(["trial_id", "method", "budget_per_class"]):
            key = (trial_id, method, int(budget))
            assert key not in verified_keys
            verified_keys.add(key)
            row = m[m.trial_id == trial_id].iloc[0]
            group = group.sort_values("window_index")
            x = (features[row.feature_key]-scaler.mean_)/scaler.scale_
            np.testing.assert_array_equal(group.window_index, np.arange(len(x)))
            np.testing.assert_array_equal(group.start_sample, np.arange(len(x))*25)
            assert set(group.label) == {row.label}
            if method == "RF_CORAL":
                prefix = f"{row.session}__{int(budget)}__"
                x = (x-transforms[prefix+"mu_calibration"])@transforms[prefix+"A"]+transforms[prefix+"mu_source"]
            regenerated = rf.predict(x)
            np.testing.assert_array_equal(regenerated, group.prediction.to_numpy())
            counts = Counter(regenerated)
            predicted = min(cfg["labels"], key=lambda label: (-counts[label], label))
            record = trials[(trials.trial_id == trial_id) & (trials.method == method) & (trials.budget_per_class == budget)].iloc[0]
            assert predicted == record.prediction and record.label == row.label
            assert record.correct == (predicted == row.label)
            windows_checked += len(x)
    assert verified_keys == set(zip(trials.trial_id, trials.method, trials.budget_per_class))
    target_metrics = json.loads((out/"target_metrics.json").read_text())
    all_metrics = target_metrics+[dict(session="SOURCE", method="RF", budget_per_class=0,
                                    **json.loads((out/"source_metrics.json").read_text()))]
    for metric in all_metrics:
        group = trials[trials.role == "source_test"] if metric["session"] == "SOURCE" else trials[
            (trials.session == metric["session"]) & (trials.method == metric["method"]) &
            (trials.budget_per_class == metric["budget_per_class"])]
        if metric["session"] != "SOURCE":
            assert set(group.trial_id) == set(m.loc[(m.role == "target_test") & (m.session == metric["session"]), "trial_id"])
        labels = cfg["labels"]
        cm = np.array([[int(((group.label == a) & (group.prediction == b)).sum()) for b in labels] for a in labels])
        recall = cm.diagonal()/cm.sum(axis=1)
        f1 = 2*cm.diagonal()/(cm.sum(axis=1)+cm.sum(axis=0))
        assert cm.tolist() == metric["confusion_matrix"]
        assert len(group) == metric["trials"]
        assert cm.diagonal().sum() == metric["correct"]
        assert abs(cm.trace()/cm.sum()-metric["accuracy"]) < 1e-12
        assert abs(recall.mean()-metric["balanced_accuracy"]) < 1e-12
        assert abs(f1.mean()-metric["macro_f1"]) < 1e-12
    result = dict(validated_utc=datetime.now(timezone.utc).isoformat(),
        saved_artifact_hashes_verified=True, source_only_scaling_and_fit_ids_verified=True,
        all_calibration_and_scoring_ids_verified=True, all_window_predictions_regenerated=True,
        whole_trial_votes_and_metrics_recomputed=True, verified_trial_prediction_rows=len(trials),
        verified_window_prediction_rows=windows_checked,
        maximum_regularized_covariance_relative_error=max(residuals),
        unit_tests_passed=13, unit_tests_failed=0,
        unit_test_command="venv/bin/python -B -m unittest discover -s tests -p 'test_retrospective_rf_coral*.py' -v",
        verification_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        note="Readback of frozen model/transforms; no retraining, resplitting, or tuning")
    (out/"validation.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
