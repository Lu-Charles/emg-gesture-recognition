"""Development-only fixed-signal-budget coverage screen on existing GRABMyo cache.

Tests simple gain correction, not a new method or faithful SAL replication.
Four seconds of nonoverlapping calibration windows always; acquisition costs vary.
"""
import csv
import hashlib
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from src.grabmyo import allocations

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT/'data/public/grabmyo/cache_20260906_v1'
OUT = ROOT/'research/runs/20260909_coverage_development_v1'
KS = [1, 2, 4, 8, 16]


def selections(order, k, offset):
    if sorted(order) != list(range(16)) or k not in KS:
        raise ValueError('Expected permutation of 16 active classes and valid coverage')
    n = 16 // k
    # Select n nonoverlapping 250 ms windows within the same four-second interval.
    slots = (np.arange(n)*k + offset % k) % 16
    return [(int(g), [int(2*s) for s in slots]) for g in order[:k]]


def validate_roles(rows, source, calibration, score):
    groups = [set(source), set(calibration), set(score)]
    if any(groups[i] & groups[j] for i in range(3) for j in range(i)):
        raise ValueError('Whole-trial leakage')
    if any(not group for group in groups):
        raise ValueError('Empty role')
    if len({rows[i]['participant'] for group in groups for i in group}) != 1:
        raise ValueError('Mixed participants')
    for ids,role in zip(groups,['enrollment','calibration','scoring']):
        if any(rows[i]['group'] != 'development' or rows[i]['role'] != role for i in ids):
            raise ValueError('Unauthorized cohort or trial role')
    if {rows[i]['session'] for i in source} != {1}:
        raise ValueError('Invalid source session')
    target = {rows[i]['session'] for i in calibration + score}
    if len(target) != 1 or target & {1}:
        raise ValueError('Mismatched target sessions')


def main():
    OUT.mkdir(exist_ok=False)
    rows=list(csv.DictReader((CACHE/'development_manifest.csv').open()))
    for row in rows:
        for key in ['participant','session','gesture','class_index','trial','calibration_rank']:
            row[key]=int(row[key])
    groups,_=allocations()
    assert sorted({r['participant'] for r in rows}) == sorted(groups['development'])
    assert not set(groups['final']) & {r['participant'] for r in rows}
    x=np.load(CACHE/'development_features.npy',mmap_mode='r')
    assert x.shape == (len(rows),35,48)
    seed=20260909
    order=np.random.default_rng(seed).permutation(16)
    orders=[np.roll(order,i).tolist() for i in range(16)]
    config=dict(timestamp=datetime.now(timezone.utc).isoformat(),scope='development screen; no final participant access',
        participants=groups['development'],reserved_participants=groups['final'],dataset='GRABMyo1.1.0',
        channels='F1-F16 physical mV',source_session=1,target_sessions=[2,3],calibration_rank=1,
        vocabulary=17,active_gestures=16,rest_class_index=16,k=KS,orders=orders,seed=seed,
        calibration_unique_seconds=4,calibration_windows=16,window_samples=512,fs=2048,
        calibration_interval_samples=[1024,9216],rest_calibration_access=False,
        scoring='Existing four whole trials per class; all 35 existing windows and majority vote',
        gain='48 positive amplitude features; log source-class mean minus log target-class mean, equal class mean; shrink by 1/(1+lambda), exponentiate, clip [.5,2]',
        lambdas=[0,1,4],source='source-session scaler and equal-prior shrinkage LDA, all 17 classes; no target fitting',
        recording_cost='5*k seconds from k archived recordings; four seconds of analyzed signal is not four seconds of acquisition',
        platform=platform.platform(),python=platform.python_version(),
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        manifest_sha256=hashlib.sha256((CACHE/'development_manifest.csv').read_bytes()).hexdigest())
    (OUT/'protocol.json').write_text(json.dumps(config,indent=2)+'\n')
    (OUT/'pilot_gesture_coverage.py').write_text(Path(__file__).read_text())
    start=time.perf_counter();results=[];access=[];predictions={}
    for person in sorted(groups['development']):
        source=[i for i,r in enumerate(rows) if r['participant']==person and r['role']=='enrollment']
        sx=np.asarray(x[source]).reshape(-1,48)
        sy=np.repeat([rows[i]['class_index'] for i in source],35)
        scaler=StandardScaler().fit(sx)
        lda=LinearDiscriminantAnalysis(solver='lsqr',shrinkage='auto',priors=np.full(17,1/17)).fit(scaler.transform(sx),sy)
        means=np.array([sx[sy==g].mean(0) for g in range(16)])
        for session in [2,3]:
            pool=[i for i,r in enumerate(rows) if r['participant']==person and r['session']==session and r['role']=='calibration' and r['calibration_rank']==1 and r['class_index']<16]
            score=[i for i,r in enumerate(rows) if r['participant']==person and r['session']==session and r['role']=='scoring']
            validate_roles(rows,source,pool,score)
            assert len(pool)==16 and len(score)==68
            lookup={rows[i]['class_index']:i for i in pool}
            tx=np.asarray(x[score]).reshape(-1,48)
            truth=np.array([rows[i]['class_index'] for i in score])
            frozen=lda.predict(scaler.transform(tx)).reshape(-1,35)
            frozen_votes=np.array([np.bincount(v,minlength=17).argmax() for v in frozen])
            access.append(dict(participant=person,session=session,source_records=[rows[i]['record'] for i in source],calibration_records={str(g):rows[i]['record'] for g,i in lookup.items()},scoring_records=[rows[i]['record'] for i in score]))
            for oi,order in enumerate(orders):
                for k in KS:
                    select=selections(order,k,oi)
                    deltas=[]
                    for g,indices in select:
                        target=np.asarray(x[lookup[g],indices]).mean(0)
                        deltas.append(np.log(np.maximum(means[g],1e-12))-np.log(np.maximum(target,1e-12)))
                    for method,lam in [('frozen',None),('gain',0),('gain_shrink1',1),('gain_shrink4',4)]:
                        if lam is None:
                            votes=frozen_votes
                        else:
                            gain=np.clip(np.exp(np.mean(deltas,axis=0)/(1+lam)),.5,2.)
                            assert np.isfinite(gain).all()
                            pred=lda.predict(scaler.transform(tx*gain)).reshape(-1,35)
                            votes=np.array([np.bincount(v,minlength=17).argmax() for v in pred])
                        selected=np.isin(truth,order[:k]);other=(truth<16)&~selected
                        r=dict(participant=person,session=session,order=oi,k=k,method=method,
                            recording_seconds=5*k,signal_seconds=4,
                            accuracy=float(np.mean(votes==truth)),balanced_accuracy=float(balanced_accuracy_score(truth,votes)),
                            macro_f1=float(f1_score(truth,votes,labels=range(17),average='macro',zero_division=0)),
                            selected_accuracy=float(np.mean(votes[selected]==truth[selected])),
                            other_active_accuracy=float(np.mean(votes[other]==truth[other])) if other.any() else None,
                            other_active_delta=float(np.mean(votes[other]==truth[other])-np.mean(frozen_votes[other]==truth[other])) if other.any() else None,
                            rest_recall=float(np.mean(votes[truth==16]==16)))
                        results.append(r)
                        predictions[f'p{person}_s{session}_o{oi}_k{k}_{method}']=votes
    summary=[]
    for k in KS:
        for method in ['frozen','gain','gain_shrink1','gain_shrink4']:
            group=[r for r in results if r['k']==k and r['method']==method]
            pa=[dict(participant=p,**{metric:float(np.mean([r[metric] for r in group if r['participant']==p])) for metric in ['accuracy','macro_f1','rest_recall']}) for p in sorted(groups['development'])]
            summary.append(dict(k=k,method=method,participants=pa,**{metric:float(np.mean([r[metric] for r in pa])) for metric in ['accuracy','macro_f1','rest_recall']}))
    (OUT/'results.json').write_text(json.dumps(dict(config=config,results=results,summary=summary,seconds=time.perf_counter()-start),indent=2)+'\n')
    (OUT/'access_manifest.json').write_text(json.dumps(access,indent=2)+'\n')
    np.savez_compressed(OUT/'trial_predictions.npz',**predictions)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    with threadpool_limits(limits=1):
        main()
