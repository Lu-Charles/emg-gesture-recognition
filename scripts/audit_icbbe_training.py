"""Post-result diagnostic: raw rebuild, independent RF refit, and null controls.

Preserves the original model and protocol. No method selection or retuning.
"""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

import joblib
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, iirnotch
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from src.features import extract_window_features

ROOT = Path(__file__).resolve().parents[1]


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def save(p, value):
    p.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")


def main():
    prepared = ROOT/"research/runs/20260906_icbbe_preparation_v1"
    evaluation = ROOT/"research/runs/20260906_icbbe_evaluation_v1"
    out = ROOT/"research/runs/20260906_icbbe_training_audit_v1"
    out.mkdir(parents=True, exist_ok=False)
    cfg = json.loads((prepared/"protocol.json").read_text())
    protected = {str(p.relative_to(ROOT)): sha(p) for folder in [prepared, evaluation]
                 for p in folder.rglob("*") if p.is_file()}
    original = json.loads((prepared/"input_sha256.json").read_text())
    for name, expected in original.items():
        assert sha(ROOT/name) == expected, name
    save(out/"config.json", dict(created_utc=datetime.now(timezone.utc).isoformat(),
        scope="Post-result diagnostic; frozen primary analysis unchanged; no method selection",
        label_permutation_seeds=list(range(10)), permutation_unit="whole source training trial",
        rf_parameters=cfg["model"], raw_rebuild_roles=["source_train", "source_test", "target_test"],
        protected_hashes=protected, script_sha256=sha(Path(__file__)),
        original_input_hashes_reference=str(prepared/"input_sha256.json")))
    shutil.copy2(__file__, out/"audit_icbbe_training.py")
    m = pd.read_csv(prepared/"trial_manifest.csv").fillna("")
    chosen = m[m.role.isin(["source_train", "source_test", "target_test"])].sort_values("trial_id")
    annotation = {}
    for filename in ["labels_S01_final_unshifted_fix.csv", "labels_S01_final_shifted_fix.csv",
                     "labels_S01_shifted_1p5cm_fix.csv", "labels_S01_shifted_1p5cm_light_fix.csv"]:
        for r in pd.read_csv(ROOT/"data/logs"/filename).itertuples():
            ident = "S01/"+"/".join(Path(r.file).parts[-2:])
            assert ident not in annotation or annotation[ident] == r.label
            annotation[ident] = r.label
    for r in m.itertuples():
        assert annotation[r.trial_id] == r.label
    bn, an = iirnotch(60/(500/2), 30)
    bb, ab = butter(4, [20/(500/2), 200/(500/2)], btype="bandpass")
    feature_rebuild, raw_checks = {}, []
    train_chunks, overlap = {}, []
    ordered = pd.concat([chosen[chosen.role == "source_train"], chosen[chosen.role != "source_train"]])
    with np.load(prepared/"features.npz", allow_pickle=False) as cached:
        for i, r in enumerate(ordered.itertuples(), start=1):
            frame = pd.read_csv(ROOT/r.raw_path)
            assert sha(ROOT/r.raw_path) == r.file_sha256
            signals = frame[["ch0", "ch1"]].to_numpy(float)
            filtered = []
            for channel in signals.T:
                centered = channel-np.median(channel)
                filtered.append(filtfilt(bb, ab, filtfilt(bn, an, centered)))
            rows = []
            for start in range(0, len(signals)-125+1, 25):
                values = []
                for x in filtered:
                    feats = extract_window_features(x[start:start+125], 500)
                    values.extend(feats[name] for name in cfg["features_per_channel"])
                rows.append(values)
                chunk = hashlib.sha256(signals[start:start+125].copy().tobytes()).hexdigest()
                if r.role == "source_train":
                    train_chunks[chunk] = r.trial_id
                elif chunk in train_chunks:
                    overlap.append(dict(train_trial=train_chunks[chunk], test_trial=r.trial_id, start_sample=start))
            rebuilt = np.asarray(rows)
            expected = cached[r.feature_key]
            np.testing.assert_allclose(rebuilt, expected, rtol=0, atol=1e-12)
            feature_rebuild[r.trial_id] = rebuilt
            raw_checks.append(dict(trial_id=r.trial_id, role=r.role, max_absolute_error=float(np.max(np.abs(rebuilt-expected))), windows=len(rebuilt)))
            if i % 30 == 0:
                print(f"Rebuilt {i}/{len(ordered)} raw trial feature matrices.", flush=True)
    pd.DataFrame(raw_checks).to_csv(out/"raw_feature_checks.csv", index=False)
    save(out/"raw_window_overlap.json", overlap)
    train = chosen[chosen.role == "source_train"]
    target = chosen[chosen.role == "target_test"]
    x = np.vstack([feature_rebuild[r.trial_id] for r in train.itertuples()])
    y = np.concatenate([np.repeat(r.label, len(feature_rebuild[r.trial_id])) for r in train.itertuples()])
    scaler = StandardScaler().fit(x)
    rf = RandomForestClassifier(**cfg["model"]).fit(scaler.transform(x), y)
    saved = joblib.load(evaluation/"source_rf.joblib")
    np.testing.assert_array_equal(scaler.mean_, saved.named_steps["scaler"].mean_)
    np.testing.assert_array_equal(scaler.scale_, saved.named_steps["scaler"].scale_)
    for a, b in zip(rf.estimators_, saved.named_steps["rf"].estimators_):
        np.testing.assert_array_equal(a.tree_.threshold, b.tree_.threshold)
        np.testing.assert_array_equal(a.tree_.feature, b.tree_.feature)
        np.testing.assert_array_equal(a.tree_.value, b.tree_.value)
    target_x = np.vstack([feature_rebuild[r.trial_id] for r in target.itertuples()])
    sizes = [len(feature_rebuild[r.trial_id]) for r in target.itertuples()]
    boundaries = np.cumsum([0]+sizes)
    def score(classifier):
        predicted = classifier.predict(scaler.transform(target_x))
        rows = []
        for i, r in enumerate(target.itertuples()):
            counts = Counter(predicted[boundaries[i]:boundaries[i+1]])
            label = min(cfg["labels"], key=lambda lab: (-counts[lab], lab))
            rows.append(dict(trial_id=r.trial_id, label=r.label, prediction=label, correct=label == r.label))
        return rows
    refit_predictions = score(rf)
    pd.DataFrame(refit_predictions).to_csv(out/"independent_refit_predictions.csv", index=False)
    print("Independent raw-input refit matches every saved RF tree exactly.", flush=True)
    controls, null_predictions = [], []
    for seed in range(10):
        permuted = np.random.default_rng(seed).permutation(train.label.to_numpy())
        train_labels = np.concatenate([np.repeat(label, len(feature_rebuild[r.trial_id])) for label, r in zip(permuted, train.itertuples())])
        null = RandomForestClassifier(**cfg["model"]).fit(scaler.transform(x), train_labels)
        predictions = score(null)
        correct = sum(r["correct"] for r in predictions)
        controls.append(dict(permutation_seed=seed, correct=correct, trials=len(target), accuracy=correct/len(target)))
        null_predictions.extend(dict(permutation_seed=seed, **r) for r in predictions)
        save(out/f"permuted_training_labels_{seed}.json", dict(zip(train.trial_id, permuted.tolist())))
    pd.DataFrame(controls).to_csv(out/"permutation_metrics.csv", index=False)
    pd.DataFrame(null_predictions).to_csv(out/"permutation_predictions.csv", index=False)
    window_metrics, trial_window_details = [], []
    for file in sorted(evaluation.glob("windows_*.csv.gz")):
        w = pd.read_csv(file)
        for (session, method, budget), group in w.groupby(["session", "method", "budget_per_class"]):
            correct = group.label.eq(group.prediction)
            window_metrics.append(dict(session=session, method=method, budget_per_class=int(budget),
                windows=len(group), correct_windows=int(correct.sum()), window_accuracy=float(correct.mean()),
                note="Post-hoc descriptive metric; overlapping windows are correlated; offline filtering"))
            for trial_id, g in group.groupby("trial_id"):
                trial_window_details.append(dict(trial_id=trial_id, method=method, budget_per_class=int(budget),
                    window_accuracy=float(g.label.eq(g.prediction).mean()), windows=len(g)))
    pd.DataFrame(window_metrics).to_csv(out/"window_metrics.csv", index=False)
    pd.DataFrame(trial_window_details).to_csv(out/"trial_window_accuracy.csv", index=False)
    schema = [f"{channel}__{name}" for channel in cfg["raw_channels"] for name in cfg["features_per_channel"]]
    pd.DataFrame(dict(feature=schema, impurity_importance=rf.feature_importances_)).to_csv(out/"feature_importance.csv", index=False)
    assert all(sha(ROOT/name) == expected for name, expected in protected.items())
    assert all(sha(ROOT/name) == expected for name, expected in original.items())
    result = dict(completed_utc=datetime.now(timezone.utc).isoformat(), raw_trials_rebuilt=len(chosen),
        maximum_raw_feature_error=max(r["max_absolute_error"] for r in raw_checks),
        original_annotation_rows_matched=len(m), cross_train_test_identical_aligned_raw_windows=len(overlap),
        independent_rf_tree_arrays_exact_match=True, target_refit_correct=sum(r["correct"] for r in refit_predictions),
        target_trials=len(target), permutation_seeds=list(range(10)),
        permutation_mean_accuracy=float(np.mean([r["accuracy"] for r in controls])),
        permutation_min_accuracy=min(r["accuracy"] for r in controls), permutation_max_accuracy=max(r["accuracy"] for r in controls),
        primary_model_protocol_and_artifacts_unchanged=True,
        caution="No detected implementation mismatch. Does not verify physical gesture labels, independence of acquisitions, or deployment validity. Window metrics and controls are post-hoc diagnostics, not replacements for frozen primary metrics.")
    save(out/"summary.json", result)
    save(out/"artifact_sha256.json", {str(p.relative_to(out)): sha(p) for p in out.rglob("*") if p.is_file()})
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
