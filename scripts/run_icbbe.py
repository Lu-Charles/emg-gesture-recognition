"""Run the frozen retrospective RF/CORAL protocol against checked S1 artifacts."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import sys
import time
import traceback

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score

from scripts.prepare_icbbe import ROOT, digest, save_json, validate_identity
from scripts.evaluate_coral import coral_fit, coral_apply, cov_and_mean
from src.model import make_rf


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_inputs(prepared):
    for name, expected in json.loads((prepared/"artifact_sha256.json").read_text()).items():
        require(digest(prepared/name) == expected, f"Prepared artifact changed: {name}")
    old_hashes = json.loads((prepared/"input_sha256.json").read_text())
    for name, expected in old_hashes.items():
        require(digest(ROOT/name) == expected, f"Frozen input/code changed: {name}")
    cfg = json.loads((prepared/"protocol.json").read_text())
    require(cfg == json.loads((ROOT/"research/icbbe_protocol_v1.json").read_text()), "Protocol mismatch")
    require(cfg["coral"]["ridge"] == 1e-6 and cfg["coral"]["eigenvalue_floor"] == 1e-12,
            "Legacy alignment supports only the frozen ridge/floor")
    m = pd.read_csv(prepared/"trial_manifest.csv").fillna("")
    validate_identity(m.to_dict("records"))
    valid = m[m.role != "excluded"]
    require(valid.feature_key.is_unique, "Repeated feature identity")
    require(set(valid.label) == set(cfg["labels"]), "Unexpected label set")
    with np.load(prepared/"features.npz", allow_pickle=False) as data:
        features = {k: data[k] for k in data.files}
    require(set(features) == set(valid.feature_key), "Feature cache coverage mismatch")
    for r in valid.to_dict("records"):
        x = features[r["feature_key"]]
        require(x.shape == (r["windows"], 18) and np.isfinite(x).all(), "Invalid feature matrix")
    scoring = pd.read_csv(prepared/"target_scoring.csv")
    require(set(scoring.trial_id) == set(m.loc[m.role == "target_test", "trial_id"]), "Scoring set changed")
    for session in cfg["target_sessions"]:
        require(set(m.loc[(m.role == "target_test") & (m.session == session), "label"]) == set(cfg["labels"]), "Missing scoring class")
    previous = set()
    for budget in cfg["calibration_budgets_per_class"]:
        c = calibration_rows(m, cfg, budget)
        saved = pd.read_csv(prepared/f"calibration_{budget}_per_class.csv")
        require(set(c.trial_id) == set(saved.trial_id), "Calibration set changed")
        require(not set(c.trial_id) & set(scoring.trial_id), "Calibration/scoring overlap")
        require(previous <= set(c.trial_id), "Calibration budgets are not nested")
        previous = set(c.trial_id)
    return cfg, m, features


def calibration_rows(manifest, cfg, budget):
    require(budget in cfg["calibration_budgets_per_class"], "Unplanned budget")
    rows = manifest[(manifest.role == "target_calibration") & (manifest.calibration_rank <= budget)]
    counts = rows.groupby(["session", "label"]).size()
    require(len(counts) == len(cfg["target_sessions"])*len(cfg["labels"]) and (counts == budget).all(),
            "Unbalanced/incomplete calibration allocation")
    return rows


def stack(rows, features):
    require(len(rows) > 0, "Empty trial selection")
    return np.vstack([features[k] for k in rows.feature_key])


def fit_source(manifest, features, cfg):
    rows = manifest[manifest.role == "source_train"].sort_values("trial_id")
    require(set(rows.label) == set(cfg["labels"]), "Source training lacks classes")
    model = make_rf(cfg["model"]["random_state"])
    actual = model.named_steps["rf"].get_params()
    require(all(actual[k] == v for k, v in cfg["model"].items()), "RF settings changed")
    x = stack(rows, features)
    y = np.concatenate([np.repeat(r.label, len(features[r.feature_key])) for r in rows.itertuples()])
    model.fit(x, y)
    scaler = model.named_steps["scaler"]
    require(int(scaler.n_samples_seen_) == len(x), "Scaler sample count differs from source training")
    require(np.allclose(scaler.mean_, x.mean(axis=0)), "Scaler is not source-only")
    return model, scaler.transform(x), rows


def fit_alignment(source, calibration, minimum_windows=20):
    require(source.ndim == calibration.ndim == 2 and source.shape[1] == calibration.shape[1],
            "Alignment feature shapes differ")
    require(len(source) >= minimum_windows and len(calibration) >= minimum_windows, "Insufficient alignment windows")
    require(np.isfinite(source).all() and np.isfinite(calibration).all(), "Nonfinite alignment input")
    transform = coral_fit(source, calibration)
    require(all(np.isfinite(v).all() for v in transform), "Nonfinite CORAL transform")
    diagnostics = {}
    for name, x in [("source", source), ("calibration", calibration)]:
        covariance, _ = cov_and_mean(x)
        eigenvalues = np.linalg.eigvalsh(covariance)
        require((eigenvalues > 0).all(), "Regularized covariance is not positive definite")
        diagnostics[name+"_cov_eigenvalues"] = eigenvalues.tolist()
        diagnostics[name+"_cov_condition"] = float(eigenvalues[-1]/eigenvalues[0])
    diagnostics["transform_spectral_norm"] = float(np.linalg.norm(transform[0], 2))
    diagnostics["source_windows"] = len(source)
    diagnostics["calibration_windows"] = len(calibration)
    return transform, diagnostics


def vote(predictions, labels):
    require(len(predictions) > 0 and set(predictions) <= set(labels), "Invalid predicted labels")
    counts = np.array([np.count_nonzero(predictions == label) for label in labels])
    return labels[int(counts.argmax())], bool(np.count_nonzero(counts == counts.max()) > 1)


def score_trials(model, rows, features, cfg, method, budget, transform=None):
    trial_results, window_results = [], []
    for r in rows.sort_values("trial_id").itertuples():
        x = model.named_steps["scaler"].transform(features[r.feature_key])
        if transform is not None:
            x = coral_apply(x, *transform)
        require(np.isfinite(x).all(), "Nonfinite scoring features; no fallback permitted")
        predictions = model.named_steps["rf"].predict(x)
        predicted, tied = vote(predictions, cfg["labels"])
        meta = dict(trial_id=r.trial_id, session=r.session, role=r.role, label=r.label, method=method,
                    budget_per_class=budget)
        trial_results.append(dict(meta, prediction=predicted, correct=bool(predicted == r.label),
                                  vote_tied=tied, windows=len(x)))
        window_results.append(pd.DataFrame(dict(meta, window_index=np.arange(len(x)),
                                               start_sample=np.arange(len(x))*cfg["hop_samples"],
                                               prediction=predictions)))
    return pd.DataFrame(trial_results), pd.concat(window_results, ignore_index=True)


def metrics(trials, labels):
    cm = confusion_matrix(trials.label, trials.prediction, labels=labels)
    return {"trials": len(trials), "correct": int((trials.label == trials.prediction).sum()),
            "accuracy": float(accuracy_score(trials.label, trials.prediction)),
            "balanced_accuracy": float(balanced_accuracy_score(trials.label, trials.prediction)),
            "macro_f1": float(f1_score(trials.label, trials.prediction, labels=labels, average="macro", zero_division=0)),
            "confusion_matrix": cm.tolist(), "class_order": labels,
            "per_class_recall": {label: float(cm[i, i]/cm[i].sum()) for i, label in enumerate(labels)},
            "tied_votes": int(trials.vote_tied.sum())}


def run(prepared, out):
    started = time.perf_counter()
    cfg, m, features = load_inputs(prepared)
    print("Frozen input hashes, calibration budgets, and scoring sets verified.", flush=True)
    code_files = sorted([*ROOT.glob("src/**/*.py"), *ROOT.glob("scripts/**/*.py"), *ROOT.glob("tests/test_icbbe*.py")])
    code_hashes = {str(p.relative_to(ROOT)): digest(p) for p in code_files}
    prepared_hashes = {p.name: digest(p) for p in sorted(prepared.iterdir()) if p.is_file()}
    save_json(out/"config.json", {"created_utc": datetime.now(timezone.utc).isoformat(), "protocol": cfg,
              "prepared_path": str(prepared), "prepared_sha256": prepared_hashes, "code_sha256": code_hashes,
              "git_revision": None, "root": str(ROOT), "python": sys.version, "platform": platform.platform(),
              "packages": {p: importlib.metadata.version(p) for p in ["numpy", "pandas", "scipy", "scikit-learn", "joblib"]}})
    for p in code_files:
        dest = out/"code_snapshot"/p.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dest)
    m.to_csv(out/"trial_manifest.csv", index=False)
    fit_started = time.perf_counter()
    model, source, train_rows = fit_source(m, features, cfg)
    fit_seconds = time.perf_counter()-fit_started
    joblib.dump(model, out/"source_rf.joblib")
    ledger = {"source_fit_trial_ids": train_rows.trial_id.tolist(), "source_fit_windows": len(source),
              "source_scaler_windows": int(model.named_steps["scaler"].n_samples_seen_),
              "validation_used": False, "calibration": []}
    print(f"Trained frozen source RF on {len(train_rows)} trials / {len(source)} windows.", flush=True)
    all_trials, summaries = [], []
    source_test = m[m.role == "source_test"]
    tr, win = score_trials(model, source_test, features, cfg, "RF", 0)
    all_trials.append(tr)
    win.to_csv(out/"windows_source_test.csv.gz", index=False, compression="gzip")
    source_metrics = metrics(tr, cfg["labels"])
    save_json(out/"source_metrics.json", source_metrics)
    transforms, diagnostics = {}, []
    for session in cfg["target_sessions"]:
        test = m[(m.role == "target_test") & (m.session == session)]
        baseline, win = score_trials(model, test, features, cfg, "RF", 0)
        all_trials.append(baseline)
        win.to_csv(out/f"windows_{session}_RF.csv.gz", index=False, compression="gzip")
        base_metric = metrics(baseline, cfg["labels"])
        summaries.append(dict(session=session, method="RF", budget_per_class=0, calibration_trials=0,
                              calibration_seconds=0.0, **base_metric))
        for budget in cfg["calibration_budgets_per_class"]:
            c = calibration_rows(m, cfg, budget)
            c = c[c.session == session].sort_values("trial_id")
            require(not set(c.trial_id) & set(test.trial_id), "Calibration/scoring overlap")
            calibration = model.named_steps["scaler"].transform(stack(c, features))
            transform, diagnostic = fit_alignment(source, calibration, cfg["coral"]["minimum_calibration_windows"])
            for name, value in zip(["A", "mu_source", "mu_calibration"], transform):
                transforms[f"{session}__{budget}__{name}"] = value
            diagnostics.append(dict(session=session, budget_per_class=budget, **diagnostic))
            ledger["calibration"].append(dict(session=session, budget_per_class=budget,
                fit_trial_ids=c.trial_id.tolist(), scoring_trial_ids=test.trial_id.tolist(),
                fit_windows=len(calibration), calibration_recorded_seconds=float(c.recorded_seconds.sum())))
            tr, win = score_trials(model, test, features, cfg, "RF_CORAL", budget, transform)
            all_trials.append(tr)
            win.to_csv(out/f"windows_{session}_CORAL_{budget}.csv.gz", index=False, compression="gzip")
            paired = baseline[["trial_id", "correct"]].merge(tr[["trial_id", "correct"]], on="trial_id", validate="one_to_one", suffixes=("_baseline", "_coral"))
            metric = metrics(tr, cfg["labels"])
            summaries.append(dict(session=session, method="RF_CORAL", budget_per_class=budget,
                calibration_trials=len(c), calibration_seconds=float(c.recorded_seconds.sum()),
                baseline_correct_to_wrong=int((paired.correct_baseline & ~paired.correct_coral).sum()),
                baseline_wrong_to_correct=int((~paired.correct_baseline & paired.correct_coral).sum()),
                delta_macro_f1=metric["macro_f1"]-base_metric["macro_f1"], **metric))
        print(f"Completed all frozen budgets for {session}.", flush=True)
    pd.concat(all_trials, ignore_index=True).to_csv(out/"trial_predictions.csv", index=False)
    save_json(out/"target_metrics.json", summaries)
    pd.DataFrame(summaries).drop(columns=["confusion_matrix", "class_order", "per_class_recall"]).to_csv(out/"target_metrics.csv", index=False)
    save_json(out/"data_access.json", ledger)
    save_json(out/"alignment_diagnostics.json", diagnostics)
    np.savez_compressed(out/"alignment_transforms.npz", **transforms)
    require(all(digest(ROOT/p) == h for p, h in code_hashes.items()), "Code changed during run")
    require(all(digest(prepared/p) == h for p, h in prepared_hashes.items()), "Prepared artifacts changed during run")
    old_hashes = json.loads((prepared/"input_sha256.json").read_text())
    require(all(digest(ROOT/p) == h for p, h in old_hashes.items()), "Protected input changed during run")
    save_json(out/"completion.json", {"status": "completed", "completed_utc": datetime.now(timezone.utc).isoformat(),
        "total_elapsed_seconds": time.perf_counter()-started, "source_fit_seconds": fit_seconds,
        "target_comparisons": len(summaries), "code_prepared_and_protected_inputs_unchanged": True,
        "scope": "One frozen retrospective run, one participant; no tuning or population significance test"})
    save_json(out/"artifact_sha256.json", {str(p.relative_to(out)): digest(p) for p in sorted(out.rglob("*")) if p.is_file()})
    print(pd.DataFrame(summaries)[["session", "method", "budget_per_class", "correct", "trials", "macro_f1"]].to_string(index=False), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prepared", default="research/runs/20260906_icbbe_preparation_v1")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = ROOT/args.out
    out.mkdir(parents=True, exist_ok=False)
    try:
        run(ROOT/args.prepared, out)
    except Exception:
        (out/"failure.txt").write_text(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
