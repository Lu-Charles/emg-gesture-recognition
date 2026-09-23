"""Read-only legacy diagnostics and a bounded smoke run; never a benchmark.

Run from the project root with the existing environment. Outputs go to a new
directory, which must not already exist. Legacy code/data/models are not edited.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score

from src.config import StreamConfig
from src.features import extract_window_features, higuchi_fd
from src.filters import preprocess_emg
from src.model import make_rf
from src.adapt import SelfCalibrator
from scripts.split_by_file import stratified_split_files
from scripts.evaluate_coral import coral_fit, coral_apply, trial_to_feature_df, majority_vote


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def save(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    root = Path.cwd()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    cfg = StreamConfig()
    source_files = sorted([*root.glob('*.py'), *root.glob('src/**/*.py'), *root.glob('scripts/**/*.py')])
    protected = sorted(set(source_files + [root/'requirements.txt', root/'.gitignore'] + [
        p for folder in ['data', 'graphs', 'outputs'] for p in (root/folder).rglob('*') if p.is_file()
    ]))
    hashes = {str(p.relative_to(root)): digest(p) for p in protected}
    save(out/'input_sha256.json', hashes)
    save(out/'config.json', {
        'scope': 'legacy audit plus smoke test; not development or final evaluation',
        'created_utc': datetime.now(timezone.utc).isoformat(), 'root': str(root),
        'git_revision': None, 'git_note': 'input_sha256.json identifies code and data',
        'python': sys.version, 'executable': sys.executable, 'platform': platform.platform(),
        'machine': platform.machine(), 'processor': platform.processor(),
        'packages': {p: importlib.metadata.version(p) for p in ['numpy','scipy','pandas','scikit-learn','joblib','pyserial','matplotlib']},
        'preprocessing': asdict(cfg), 'channels': ['ch0_f', 'ch1_f'],
        'dataset': 'local custom EMG; S01 smoke only; not GRABMyo',
        'seed': 42, 'selection': 'sorted file names, first 2/class train; first 1/class source test and target eval; all 6 existing 1cm calibration trials',
        'source_train_manifest': 'data/splits_fix/S01_unshifted_seed42/train/labels.csv',
        'source_test_manifest': 'data/splits_fix/S01_unshifted_seed42/test/labels.csv',
        'calibration_manifest': 'data/logs/labels_S01_shift1cm_calib.csv',
        'target_eval_manifest': 'data/logs/labels_S01_shift1cm_eval.csv',
        'model': make_rf(42).named_steps['rf'].get_params(),
        'coral': 'scripts.evaluate_coral; source-scaled features; covariance ridge 1e-6; target calibration signals only, no scoring data in fit',
        'metrics_unit': 'majority vote per whole trial; no uncertainty estimate for tiny single-person smoke sample',
    })
    print('Hash snapshot complete; checking all local CSV signals.', flush=True)
    inventory = Counter()
    timings = []
    raw_signal_hashes = {}
    raw_frames = {}
    processed_checks = []
    parse_errors = []
    for p in source_files:
        try:
            ast.parse(p.read_text(), filename=str(p))
        except SyntaxError as exc:
            parse_errors.append({'file': str(p.relative_to(root)), 'error': str(exc)})
    for p in sorted((root/'data').rglob('*.csv')):
        rel = str(p.relative_to(root))
        inventory[str(p.parent.relative_to(root))] += 1
        frame = pd.read_csv(p)
        if not {'t_us', 'ch0', 'ch1'}.issubset(frame.columns):
            continue
        signal = frame[['t_us', 'ch0', 'ch1']].to_numpy(dtype=np.float64)
        raw_signal_hashes[rel] = hashlib.sha256(signal.tobytes()).hexdigest()
        dt = np.diff(signal[:, 0])
        positive = dt[dt > 0]
        span = signal[-1, 0] - signal[0, 0]
        median_dt = float(np.median(positive)) if len(positive) else None
        timings.append({
            'file': rel, 'samples': len(frame), 'finite': bool(np.isfinite(frame.to_numpy(dtype=float)).all()),
            'nonpositive_intervals': int((dt <= 0).sum()), 'median_interval_us': median_dt,
            'median_rate_hz': 1e6 / median_dt if median_dt else None,
            'elapsed_rate_hz': float((len(frame)-1)*1e6/span) if span > 0 else None,
            'duration_s': float(span/1e6),
            'max_interval_us': float(dt.max()) if len(dt) else None,
            'intervals_over_1p5_median': int((dt > 1.5*median_dt).sum()) if median_dt else None,
        })
        if 'data/raw/' in rel:
            raw_frames[(p.parent.name, p.name)] = frame
    for p in sorted((root/'data/processed_fix').rglob('*.csv')):
        frame = pd.read_csv(p)
        raw = raw_frames[(p.parent.name, p.name)]
        row = {'file': str(p.relative_to(root)), 'raw_columns_equal': bool(np.array_equal(frame[['t_us','ch0','ch1']], raw[['t_us','ch0','ch1']]))}
        for ch in ['ch0','ch1']:
            expected = preprocess_emg(raw[ch].to_numpy(float), cfg.fs_hz, cfg.bandpass_lo, cfg.bandpass_hi, cfg.notch_hz, cfg.notch_q)
            row[ch + '_max_abs_error'] = float(np.max(np.abs(expected-frame[ch+'_f'].to_numpy(float))))
        processed_checks.append(row)
    pd.DataFrame(timings).to_csv(out/'signal_inventory.csv', index=False)
    pd.DataFrame(processed_checks).to_csv(out/'processed_reproduction.csv', index=False)
    save(out/'directory_inventory.json', dict(inventory))
    save(out/'signal_sha256.json', raw_signal_hashes)

    manifests = {}
    for p in sorted((root/'data').rglob('*.csv')):
        if 'labels' not in p.name:
            continue
        df = pd.read_csv(p)
        if not {'file', 'label'}.issubset(df):
            continue
        manifests[str(p.relative_to(root))] = df
    resolution = []
    candidates = sorted(raw_signal_hashes)
    for name, df in manifests.items():
        for row in df.to_dict('records'):
            matches = [p for p in candidates if p.endswith('/' + row['file'])]
            resolution.append({'manifest': name, **row, 'suffix_matches': matches})
    save(out/'manifest_resolution.json', resolution)
    split_checks = []

    def compare(name, groups):
        paths = {k: set(df.file) for k, df in groups.items()}
        keys = list(paths)
        split_checks.append({
            'name': name,
            'counts': {k: len(df) for k, df in groups.items()},
            'class_counts': {k: df.label.value_counts().to_dict() for k,df in groups.items()},
            'duplicate_rows': {k: int(df.file.duplicated().sum()) for k,df in groups.items()},
            'overlaps': {a+' / '+b: sorted(paths[a]&paths[b]) for i,a in enumerate(keys) for b in keys[i+1:]},
        })
    for base in ['data/splits', 'data/splits/S01', 'data/splits/S02', 'data/splits_fix/S01_unshifted_seed42']:
        compare(base, {s: manifests[f'{base}/{s}/labels.csv'] for s in ['train','val','test']})
    for a,b in [
        ('labels_S01_shift1cm_calib.csv','labels_S01_shift1cm_eval.csv'),
        ('labels_S01_shifted_calib.csv','labels_S01_shifted_eval.csv'),
        ('labels_S01_shifted_1p5cm_calib_fix.csv','labels_S01_shifted_1p5cm_eval_fix.csv'),
        ('labels_S01_shifted_1p5cm_fix_calib.csv','labels_S01_shifted_1p5cm_fix_eval.csv'),
    ]:
        compare(a, {'calibration': manifests['data/logs/'+a], 'evaluation': manifests['data/logs/'+b]})
    save(out/'split_checks.json', split_checks)

    # Diagnostic assertions describe required behavior; failures are preserved as
    # audit findings without modifying the functions under examination.
    diagnostics = []
    def check(name, fn):
        try:
            details = fn()
            diagnostics.append({'name': name, 'passed': True, 'details': details})
        except Exception as exc:
            diagnostics.append({'name': name, 'passed': False, 'error': f'{type(exc).__name__}: {exc}'})
    def finite_zero():
        values = extract_window_features(np.zeros(125), 500)
        assert np.isfinite(list(values.values())).all(), values
        return values
    def constant_hfd():
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            value = higuchi_fd(np.ones(125))
        assert np.isfinite(value), f'higuchi_fd(constant)={value}'
        return value
    def conflicting_split():
        df = pd.DataFrame([{'file':f'f{i}.csv','label':lab} for lab in ['a','b'] for i in range(8)])
        try:
            parts = stratified_split_files(df, 0.2, 0.2, 42)
        except ValueError:
            return 'Rejected conflicting labels'
        sets = [set(p.file) for p in parts]
        overlap = sorted(set.union(*(sets[i]&sets[j] for i in range(3) for j in range(i+1,3))))
        assert not overlap, f'Conflicting-label input accepted; files cross partitions: {overlap}'
    def pseudo_labels():
        class Model:
            classes_ = np.array(['extend','fist','rest'])
            def predict_proba(self, x): return np.tile([0.,1.,0.], (len(x),1))
        c = SelfCalibrator()
        c.ingest(Model(), np.ones((2,3)))
        assert list(c.yb) == ['fist','fist'], f'Buffered labels: {list(c.yb)}'
    def coral_direction():
        rng = np.random.default_rng(42)
        xs = rng.normal(size=(2000,4)) @ np.diag([1.,2.,3.,4.]) + 2.
        xt = rng.normal(size=(2000,4)) @ np.diag([4.,3.,2.,1.]) - 3.
        a, ms, mt = coral_fit(xs, xt)
        aligned = coral_apply(xt,a,ms,mt)
        error = float(np.linalg.norm(np.cov(aligned,rowvar=False)-np.cov(xs,rowvar=False))/np.linalg.norm(np.cov(xs,rowvar=False)))
        assert error < 1e-4, error
        assert np.allclose(aligned.mean(0), xs.mean(0))
        return {'relative_covariance_error':error}
    def coral_small():
        xs = np.arange(24.).reshape(6,4)
        xt = np.ones((1,4))
        a, ms, mt = coral_fit(xs,xt)
        z = coral_apply(np.array([[1.,2.,3.,4.]]),a,ms,mt)
        assert np.isfinite(z).all()
        return {'transform_norm':float(np.linalg.norm(a)), 'held_out_output_norm':float(np.linalg.norm(z)), 'note':'finite does not imply statistically stable with one calibration row'}
    for name,fn in [('zero_signal_features_finite',finite_zero),('constant_hfd_finite',constant_hfd),('conflicting_labels_do_not_leak_trials',conflicting_split),('pseudo_labels_preserve_classes',pseudo_labels),('coral_maps_target_to_source',coral_direction),('coral_single_row_finite',coral_small)]:
        check(name,fn)
    save(out/'diagnostics.json',diagnostics)
    print('Signal and split audit complete; starting bounded RF/CORAL smoke test.',flush=True)

    data_dir = root/'data/processed_fix/S01'
    def select(path,n):
        df = manifests[path].sort_values('file')
        return df.groupby('label',sort=True).head(n).reset_index(drop=True)
    sets = {
        'train':select('data/splits_fix/S01_unshifted_seed42/train/labels.csv',2),
        'source_test':select('data/splits_fix/S01_unshifted_seed42/test/labels.csv',1),
        'calibration':manifests['data/logs/labels_S01_shift1cm_calib.csv'],
        'target_test':select('data/logs/labels_S01_shift1cm_eval.csv',1),
    }
    selected = pd.concat([df.assign(partition=k) for k,df in sets.items()],ignore_index=True)
    selected.to_csv(out/'smoke_manifest.csv',index=False)
    assert not selected.file.duplicated().any()
    signal_ids = [raw_signal_hashes[str((data_dir/f).relative_to(root))] for f in selected.file]
    assert len(signal_ids) == len(set(signal_ids)), 'Selected trials duplicate underlying raw signals'
    cache = {}
    windows = {}
    for f in selected.file:
        frame = trial_to_feature_df(data_dir/f,cfg,['ch0_f','ch1_f'],False)
        assert np.isfinite(frame.to_numpy()).all(), f
        cache[f] = frame
        windows[f] = len(frame)
    save(out/'feature_schema.json',list(next(iter(cache.values())).columns))
    save(out/'window_counts.json',windows)
    def stack(df): return np.vstack([cache[f].to_numpy() for f in df.file])
    xtrain = stack(sets['train'])
    ytrain = np.concatenate([np.repeat(row.label, windows[row.file]) for row in sets['train'].itertuples()])
    model = make_rf(42)
    train_start = time.perf_counter()
    model.fit(xtrain,ytrain)
    fit_seconds = time.perf_counter()-train_start
    scaler = model.named_steps['scaler']
    a, ms, mt = coral_fit(scaler.transform(xtrain),scaler.transform(stack(sets['calibration'])))
    predictions = []
    window_predictions = []
    metrics = {}
    for partition,method in [('source_test','baseline'),('target_test','baseline'),('target_test','coral')]:
        for row in sets[partition].itertuples():
            x = cache[row.file].to_numpy()
            if method == 'coral':
                pred = model.named_steps['rf'].predict(coral_apply(scaler.transform(x),a,ms,mt))
            else:
                pred = model.predict(x)
            predictions.append({'partition':partition,'method':method,'file':row.file,'true':row.label,'predicted':majority_vote(pred)})
            window_predictions.extend({'partition':partition,'method':method,'file':row.file,'start_idx':i*25,'true':row.label,'predicted':label} for i,label in enumerate(pred))
        group = [p for p in predictions if p['partition']==partition and p['method']==method]
        truth = [p['true'] for p in group]
        pred = [p['predicted'] for p in group]
        labels = ['extend','fist','rest']
        metrics[partition+'/'+method] = {
            'trials':len(group),'accuracy':accuracy_score(truth,pred),
            'balanced_accuracy':balanced_accuracy_score(truth,pred),
            'macro_f1':f1_score(truth,pred,labels=labels,average='macro',zero_division=0),
            'labels':labels,'confusion_matrix':confusion_matrix(truth,pred,labels=labels).tolist(),
        }
    pd.DataFrame(predictions).to_csv(out/'trial_predictions.csv',index=False)
    pd.DataFrame(window_predictions).to_csv(out/'window_predictions.csv',index=False)
    save(out/'smoke_metrics.json',metrics)
    changed = [p for p,h in hashes.items() if not (root/p).exists() or digest(root/p)!=h]
    summary = {
        'source_files_parsed':len(source_files),'syntax_errors':parse_errors,
        'signals_checked':len(timings),'raw_trials':len(raw_frames),
        'processed_fix_trials_recomputed':len(processed_checks),
        'processed_fix_max_error':max(max(row['ch0_max_abs_error'],row['ch1_max_abs_error']) for row in processed_checks),
        'processed_fix_raw_columns_equal':all(row['raw_columns_equal'] for row in processed_checks),
        'manifests_checked':len(manifests),'split_groups_checked':len(split_checks),
        'diagnostics_passed':sum(d['passed'] for d in diagnostics),'diagnostics_total':len(diagnostics),
        'train_shape':list(xtrain.shape),'fit_seconds':fit_seconds,
        'protected_files':len(hashes),'changed_protected_files':changed,
        'total_seconds':time.perf_counter()-started,'finished_utc':datetime.now(timezone.utc).isoformat(),
        'note':'Expected diagnostic failures describe legacy defects; smoke metrics are not research evidence of generalization.'
    }
    save(out/'summary.json',summary)
    assert not changed, changed
    print(json.dumps(summary,indent=2))
    print(json.dumps(metrics,indent=2))


if __name__ == '__main__':
    main()
