"""Frozen six-person development screen of real rotation recalibration budgets."""
import argparse
import csv
import hashlib
import itertools
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
import joblib
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, f1_score
from threadpoolctl import threadpool_limits
from src.senic import (DEVELOPMENT, GESTURES, FS, START, STOP, WINDOW, HOP,
                       parse_trial, trial_windows, rotate_features, estimate_rotation, validate_split)
from src.grabmyo import amplitude_features
from src.open_set_emg import tdar_rms


def digest(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v): Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--root',type=Path,default=Path('data/public/SeNic/extracted'))
    a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=False);tick=time.perf_counter()
    rows=[]
    for person in DEVELOPMENT:
        paths=sorted((a.root/f'h{person}'/'0').glob('*.csv'))
        rr=[parse_trial(p) for p in paths]
        expected=set(itertools.product(range(11),range(3),range(7)))
        actual={(r['position'],r['repetition'],r['label']) for r in rr}
        if actual != expected or len(rr)!=231: raise ValueError(f'Incomplete session h{person}')
        rows.extend(rr)
    for i,r in enumerate(rows):r['id']=i
    def ids(p,pos,reps,labels=range(7)):
        return [r['id'] for r in rows if r['subject']==p and r['position']==pos
                and r['repetition'] in reps and r['label'] in labels]
    plans=[]
    for p in DEVELOPMENT:
        source=ids(p,0,(0,1))
        for pos in range(11):
            score=ids(p,pos,(2,))
            sets=[[]] if pos==0 else [[],*[ids(p,pos,(0,),(g,)) for g in range(7)],
                                     ids(p,pos,(0,)),ids(p,pos,(0,1))]
            for cal in sets:
                validate_split(source,cal,score,rows)
                plans.append(dict(subject=p,position=pos,source=source,calibration=cal,score=score,
                                  trials=len(cal),gesture=rows[cal[0]]['gesture'] if len(cal)==1 else 'all'))
    config=dict(created_utc=datetime.now(timezone.utc).isoformat(),scope='development-only baseline feasibility; not final evidence or novel method',
        development=list(DEVELOPMENT),reserved_rotation_participants=[p for p in range(30) if p not in DEVELOPMENT],
        fatigue_participants_excluded=list(range(30,36)),session=0,source_position=0,source_repetitions=[0,1],
        scoring_repetition=2,calibration_trials=[0,1,7,14],one_gesture='all seven choices reported, never best-test selection',
        gestures=list(GESTURES),sampling_hz=FS,raw_units='Myo device counts, no mV claim',
        analysis_samples=[START,STOP],window=WINDOW,hop=HOP,extra_filter=None,
        cost='full duration of each recorded calibration CSV, including initial rest; 2s active analysis per trial',
        features=['amplitude_MAV_RMS_meanWL','TDAR_RMS'],lda=dict(solver='lsqr',shrinkage='auto',equal_priors=True),
        spatial_baseline='amplitude-only; least template cosine discrepancy over shifts -4..3.875 channels in 0.125 steps, RMS templates from allowed labeled calibration only; interpolate feature groups, not raw waveforms',
        comparisons=['frozen','source_plus_target','target_only_at_7_or_14','spatial_registration_amplitude'],
        ruler_angles='analysis strata only; never method input',
        interpretation='source accuracy and rotation loss establish validity/problem; 7-vs-14 trial paired accuracy/F1 and actual recording savings test diminishing returns; one-gesture spatial alignment exploratory baseline, not paper reproduction',
        screen='provisional short-calibration signal if seven-trial target-only mean accuracy within 3 percentage points of fourteen-trial target-only and improves frozen by >=10 points in >=4/6 people. Failure prompts diagnosis, not tuning final outcomes. Neither pass nor failure proves publication suitability.',
        seed=42,platform=platform.platform(),git_revision=None,
        code_hashes={str(p):digest(p) for p in ['scripts/pilot_senic.py','src/senic.py','src/grabmyo.py','src/open_set_emg.py','tests/test_senic.py']})
    write(a.out/'config.json',config);write(a.out/'allocations.json',plans)
    snap=a.out/'code_snapshot';snap.mkdir()
    for p in config['code_hashes']:(snap/Path(p).name).write_bytes(Path(p).read_bytes())
    (a.out/'environment.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
    write(a.out/'protocol_audit.json',dict(passed=True,allocations=len(plans),whole_trial_disjoint=True,
         identical_score_by_subject_position=True,reserved_signal_access=0))
    print('Protocol frozen:',len(rows),'trials,',len(plans),'allocations',flush=True)
    features={'amplitude':[], 'tdar_rms':[]}
    for r in rows:
        signal=np.loadtxt(r['path'],delimiter=',',dtype=np.float64)
        w,_=trial_windows(signal)
        r.update(samples=len(signal),recorded_seconds=len(signal)/FS,sha256=digest(r['path']),
                 windows=len(w),analysis_start=START,analysis_stop=STOP)
        features['amplitude'].append(amplitude_features(w))
        features['tdar_rms'].append(tdar_rms(w.transpose(0,2,1)))
        if (r['id']+1)%231==0:print('Features complete h'+str(r['subject']),flush=True)
    features={k:np.stack(v) for k,v in features.items()}
    for k,v in features.items():np.save(a.out/(k+'_features.npy'),v)
    write(a.out/'trial_manifest.json',rows)
    nwin=features['amplitude'].shape[1];labels=np.asarray([r['label'] for r in rows])
    modeldir=a.out/'models';modeldir.mkdir(); predir=a.out/'predictions';predir.mkdir()
    metrics=[];access=[];models={}
    def arrays(kind,ii):return features[kind][ii].reshape(-1,features[kind].shape[-1]),np.repeat(labels[ii],nwin)
    def fit(kind,ii):
        key=(kind,tuple(sorted(ii)))
        if key not in models:
            x,y=arrays(kind,list(key[1]));m=make_pipeline(StandardScaler(),LinearDiscriminantAnalysis(solver='lsqr',shrinkage='auto',priors=np.ones(7)/7))
            m.fit(x,y);name=f'm{len(models):04d}';joblib.dump(m,modeldir/(name+'.joblib'),compress=3)
            models[key]=(name,m);access.append(dict(model=name,feature=kind,fit_ids=list(key[1])))
        return models[key]
    for ai,plan in enumerate(plans):
        src,cal,score=plan['source'],plan['calibration'],plan['score']
        cost=sum(rows[i]['recorded_seconds'] for i in cal)
        for kind in features:
            methods=['frozen'] if not cal else ['source_plus_target']
            if len(cal)>=7:methods.append('target_only')
            if cal and kind=='amplitude':methods.append('spatial_registration')
            for method in methods:
                fitids=cal if method=='target_only' else src+cal if method=='source_plus_target' else src
                name,model=fit(kind,fitids);x,y=arrays(kind,score);shift=None;loss=None
                if method=='spatial_registration':
                    cls=sorted(set(labels[cal]));sa=[];ta=[]
                    for g in cls:
                        sa.append(features[kind][[i for i in src if labels[i]==g],:,8:16].mean((0,1)))
                        ta.append(features[kind][[i for i in cal if labels[i]==g],:,8:16].mean((0,1)))
                    shift,loss=estimate_rotation(np.asarray(sa),np.asarray(ta));x=rotate_features(x,shift)
                pred=model.predict(x);probs=model.predict_proba(x)
                trialpred=np.asarray([np.bincount(v,minlength=7).argmax() for v in pred.reshape(-1,nwin)])
                record=dict(subject=plan['subject'],position=plan['position'],feature=kind,method=method,
                    trials=plan['trials'],calibration_gesture=plan['gesture'],recorded_seconds=cost,
                    active_analysis_seconds=(STOP-START)/FS*len(cal),accuracy=float(accuracy_score(y,pred)),
                    macro_f1=float(f1_score(y,pred,labels=range(7),average='macro',zero_division=0)),
                    trial_accuracy=float(accuracy_score(labels[score],trialpred)),model=name,
                    allocation=ai,shift_channels=shift,registration_cost=loss,prediction_file=f'p{len(metrics):04d}.npz')
                np.savez_compressed(predir/record['prediction_file'],y=y,pred=pred,probabilities=probs,
                                    score_ids=score,trial_pred=trialpred)
                metrics.append(record)
        if ai%101==100:print('Scored allocations',ai+1,flush=True)
    write(a.out/'metrics.json',metrics);write(a.out/'model_access.json',access)
    groups=[]
    keys=sorted({(r['feature'],r['method'],r['trials']) for r in metrics})
    for kind,method,budget in keys:
        selected=[r for r in metrics if r['position']>0 and (r['feature'],r['method'],r['trials'])==(kind,method,budget)]
        people=[]
        for p in DEVELOPMENT:
            rr=[r for r in selected if r['subject']==p]
            people.append(dict(subject=p,**{k:float(np.mean([x[k] for x in rr])) for k in ['accuracy','macro_f1','trial_accuracy','recorded_seconds']}))
        groups.append(dict(feature=kind,method=method,trials=budget,participants=people,
                           **{k:float(np.mean([x[k] for x in people])) for k in ['accuracy','macro_f1','trial_accuracy','recorded_seconds']}))
    write(a.out/'summary.json',dict(groups=groups,wall_seconds=time.perf_counter()-tick,
           trial_count=len(rows),metric_rows=len(metrics),models=len(models),completed_utc=datetime.now(timezone.utc).isoformat()))
    print(json.dumps(dict(complete=True,seconds=time.perf_counter()-tick,metrics=len(metrics),models=len(models))),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
