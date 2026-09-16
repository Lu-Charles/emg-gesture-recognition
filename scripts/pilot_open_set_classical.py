"""First open-set problem screen; known-only TDAR+RMS LDA with two rejectors."""
import argparse
import csv
import itertools
import json
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits
from src.grabmyo_corpus import WindowCorpus, hash_stream
from src.open_set_emg import (KNOWN, protocol_rows, select, allocations, validate_allocation,
                              tdar_rms, LDADistance, cutoff, measures, command_curve, feature_arrays)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    corpus = WindowCorpus(args.cache, 'development')
    rows = protocol_rows(corpus.rows)
    cache_summary = json.loads((args.cache/'summary.json').read_text())
    for kind, ext in [('signals', 'npy'), ('manifest', 'csv')]:
        if hash_stream(args.cache/f'development_{kind}.{ext}') != cache_summary['groups']['development'][kind+'_sha256']:
            raise ValueError('Changed public cache')
    people = sorted({int(r['participant']) for r in rows})
    config = dict(created_utc=datetime.now(timezone.utc).isoformat(), scope='development classical problem screen, no learned open-set representation',
                  cache=str(args.cache.resolve()), cache_summary_sha256=hash_stream(args.cache/'summary.json'),
                  participants=people, known_zero_based=list(KNOWN), unknown_zero_based=list(range(10)),
                  participant_split='unchanged20train/8development/15final; classical uses day1 known enrollment of development people',
                  source_fit_trials=[1,2,3,4,5], source_threshold_trials=[6], source_score_trials=[7],
                  target_roles='original3calibration/4scoring whole-trial manifest',
                  feature_order=['MAV','ZC','SSC','WL','AR1','AR2','AR3','AR4','RMS'],
                  feature_layout='feature-major,16channels within feature;144 columns',
                  feature_definition='float64 physical mV, strict ZC, SSC product>=0, WL sum, uncentered Burg librosa0.11.0 denominator AR coefficients; allzero AR=0',
                  windows=corpus.config, lda=dict(solver='lsqr', shrinkage='auto', priors=[1/7]*7),
                  rejectors=['probability', 'distance'], distance='negative Mahalanobis distance to predicted LDA mean using fitted shrinkage covariance',
                  model_fit_strategies=['source_plus_target','target_only'], model_only_threshold='frozen numeric cutoff from source model/source trial6',
                  retention=[.90,.95,.99], primary_retention=.95, threshold='lower empirical quantile; equal class/trial/window counts; acceptance >= cutoff',
                  target_budgets_seconds=[0,35,70,105], orders=list(itertools.permutations([1,2,3])),
                  source_recording_seconds=dict(fit=175, threshold=35), source_scoring_not_training_seconds=35,
                  problem_gate='provisional only until compact CNN/prototype-learning competitors checked; memo criteria unchanged',
                  seed=42, platform=platform.platform(), git_revision=None,
                  code_hashes={f:hash_stream(f) for f in ['scripts/pilot_open_set_classical.py','src/open_set_emg.py','src/grabmyo_corpus.py','src/grabmyo.py','tests/test_open_set_emg.py']})
    write_json(args.out/'config.json', config)
    write_json(args.out/'trial_manifest.json', rows)
    (args.out/'environment.txt').write_text(subprocess.check_output([__import__('sys').executable,'-m','pip','freeze'],text=True))
    snapshot=args.out/'code_snapshot';snapshot.mkdir()
    for name in config['code_hashes']:
        (snapshot/Path(name).name).write_bytes(Path(name).read_bytes())
    import librosa, inspect
    (snapshot/'librosa_lpc_source.txt').write_text(inspect.getsource(librosa.lpc))
    plans=[]
    for p in people:
        for s in (2,3):
            for a in allocations(rows,p,s):
                for strategy in (['source_plus_target','target_only'] if a['target_fit'] else ['source_plus_target']):
                    b=dict(a, participant=p, session=s, fit_strategy=strategy)
                    if strategy=='target_only': b['fit']=a['target_fit']
                    validate_allocation(b, rows)
                    plans.append(b)
    write_json(args.out/'allocations.json', plans)
    write_json(args.out/'protocol_validation.json',dict(passed=True, allocations=len(plans), participants=people,
               fit_threshold_score_disjoint=True, known_only_fitting=True, budget_counts_match=True,
               final_participants_accessed=0, validated_utc=datetime.now(timezone.utc).isoformat()))
    # Protocol and all settings are on disk before signal feature extraction or outcomes.
    print(json.dumps(dict(phase='protocol_frozen', allocations=len(plans))),flush=True)
    ids=[r['corpus_index'] for r in rows if r['open_role']!='unused']
    features=np.lib.format.open_memmap(args.out/'features.npy',mode='w+',dtype='float64',shape=(len(rows),35,144))
    features[:]=np.nan
    extract_start=time.perf_counter()
    for offset in range(0,len(ids),16):
        selected=ids[offset:offset+16]
        x,_=corpus.batch(corpus.window_ids(selected))
        features[selected]=tdar_rms(x).reshape(len(selected),35,144)
        if offset%256==0:
            print(json.dumps(dict(phase='features', trials_done=min(offset+16,len(ids)), trials_total=len(ids))),flush=True)
    features.flush()
    write_json(args.out/'feature_access.json',dict(trial_ids=ids,records=[rows[i]['record'] for i in ids],
              deterministic_no_fitting=True,seconds=time.perf_counter()-extract_start))
    def arrays(selected):
        return feature_arrays(features,corpus.labels,selected)
    models=args.out/'models';models.mkdir()
    predictions=args.out/'predictions';predictions.mkdir()
    trained={}; saved={}; metrics=[]; access=[]; source_metrics=[]
    def model_for(fit_ids):
        key=tuple(sorted(fit_ids))
        if key not in trained:
            name=f'm{len(trained):04d}'
            x,y=arrays(key);tick=time.perf_counter()
            model=LDADistance().fit(x,y)
            trained[key]=(name,model)
            joblib.dump(model,models/(name+'.joblib'),compress=3)
            access.append(dict(model=name,fit_ids=list(key),fit_records=[rows[i]['record'] for i in key],fit_seconds=time.perf_counter()-tick))
        return trained[key]
    def predictions_for(fit_ids,eval_ids):
        name,model=model_for(fit_ids)
        key=(name,tuple(eval_ids))
        if key not in saved:
            x,y=arrays(eval_ids);pred,sc=model.score(x)
            fname=f'{name}_e{len(saved):04d}.npz'
            np.savez_compressed(predictions/fname,trial_ids=eval_ids,y=y,pred=pred,**sc)
            saved[key]=(fname,y,pred,sc)
        return saved[key]
    for p in people:
        source=select(rows,p,1,'enroll_fit');source_q=select(rows,p,1,'source_threshold')
        source_test=select(rows,p,1,'source_score')
        qfile,_,_,source_scores=predictions_for(source,source_q)
        sfile,sy,sp,ss=predictions_for(source,source_test)
        for rejector in config['rejectors']:
            for retention in config['retention']:
                threshold=cutoff(source_scores[rejector],retention)
                source_metrics.append(dict(participant=p,rejector=rejector,retention=retention,threshold=threshold,
                       threshold_file=qfile,score_file=sfile,**measures(sy,sp,ss[rejector],threshold)))
        for a in [a for a in plans if a['participant']==p]:
            model_name,_=model_for(a['fit'])
            # Classifier-only really retains the *old model's* numeric cutoff.
            threshold_fit=source if a['method'] in ('none','model_only') else a['fit']
            threshold_file,_,_,qs=predictions_for(threshold_fit,a['threshold'])
            score_file,y,pred,sc=predictions_for(a['fit'],a['score'])
            for rejector in config['rejectors']:
                for retention in config['retention']:
                    threshold=cutoff(qs[rejector],retention)
                    metrics.append(dict(participant=p,session=a['session'],method=a['method'],budget=a['budget'],
                       seconds=35*a['budget'],order=''.join(map(str,a['order'])),fit_strategy=a['fit_strategy'],
                       rejector=rejector,retention=retention,threshold=threshold,model=model_name,
                       threshold_file=threshold_file,score_file=score_file,
                       **measures(y,pred,sc[rejector],threshold)))
        print(json.dumps(dict(phase='classical',participant=p,models=len(trained),metric_rows=len(metrics))),flush=True)
        write_json(args.out/'metrics.json',metrics);write_json(args.out/'model_access.json',access)
    write_json(args.out/'source_metrics.json',source_metrics)
    # Full descriptive curves are independent of threshold policy/allocation.
    curve_index=[]
    for (name,eval_ids),(fname,y,pred,sc) in saved.items():
        if not np.any(y<10): continue
        for rejector in config['rejectors']:
            thresholds,ccr,ufa=command_curve(y,pred,sc[rejector])
            cf=fname.replace('.npz','_'+rejector+'_curve.npz')
            np.savez_compressed(predictions/cf,threshold=thresholds,known_correct_acceptance=ccr,unknown_false_acceptance=ufa)
            ordered=np.argsort(-sc[rejector],kind='stable'); ordered_scores=sc[rejector][ordered]
            ends=np.r_[np.flatnonzero(np.diff(ordered_scores)!=0),len(y)-1]
            standard_ccr=np.r_[0.,np.cumsum(((y>=10)&(pred==y))[ordered])[ends]/(y>=10).sum()]
            standard_fpr=np.r_[0.,np.cumsum((y<10)[ordered])[ends]/(y<10).sum()]
            curve_index.append(dict(score_file=fname,rejector=rejector,curve_file=cf,
                         known_unknown_auroc=float(roc_auc_score(y>=10,sc[rejector])),
                         standard_oscr=float(np.trapezoid(standard_ccr,standard_fpr)),
                         area_definition='standard OSCR includes rest among known classes and counts any accepted unknown as false positive; separate command curves exclude rest predictions from commands'))
    write_json(args.out/'curve_index.json',curve_index)
    summary=dict(completed_utc=datetime.now(timezone.utc).isoformat(),total_seconds=time.perf_counter()-start,
                 feature_seconds=json.loads((args.out/'feature_access.json').read_text())['seconds'],
                 fit_seconds=sum(a['fit_seconds'] for a in access), models=len(trained),metric_rows=len(metrics),
                 prediction_files=len(saved),final_participants_accessed=0,unknown_model_fit_trials=0)
    write_json(args.out/'summary.json',summary)
    write_json(args.out/'artifact_sha256.json',{str(p.relative_to(args.out)):hash_stream(p) for p in args.out.rglob('*') if p.is_file()})
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=4): main()
