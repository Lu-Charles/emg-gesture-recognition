"""Matched SeNic CNN / ABSDA-principle adaptation; no final participants."""
import argparse
import copy
import hashlib
import json
import platform
import subprocess
import sys
import time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import f1_score
from threadpoolctl import threadpool_limits
from src.senic import trial_windows,validate_split,DEVELOPMENT
from src.senic_neural import SeNicCNN,training_scale,fit,probabilities


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--classical',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=False);start=time.perf_counter()
    prior=json.loads((a.classical/'config.json').read_text());assert json.loads((a.classical/'independent_audit.json').read_text())['passed']
    rows=json.loads((a.classical/'trial_manifest.json').read_text());allplans=json.loads((a.classical/'allocations.json').read_text())
    assert set(prior['development'])==set(DEVELOPMENT)
    plans=[p for p in allplans if p['trials'] in (0,7,14)]
    for p in plans:validate_split(p['source'],p['calibration'],p['score'],rows)
    config=dict(created_utc=datetime.now(timezone.utc).isoformat(),scope='six-development-person, session0, neural comparator; no new method claim',
        classical=str(a.classical.resolve()),upstream_hashes={name:sha(a.classical/name) for name in ['config.json','trial_manifest.json','allocations.json','independent_audit.json']},
        development=list(DEVELOPMENT),seed=42,device='cpu',torch=torch.__version__,platform=platform.platform(),
        families={'cnn':[0],'cnn_roll45':[-1,0,1],'cnn_roll360_zero_only':list(range(8))},
        architecture='8->32 temporal Conv1d(k7,s2,p3),ReLU;32->64 Conv1d(k5,s2,p2),ReLU;temporal mean; concatenate log1p per-channel MAV;linear72->7',
        normalization='one scalar RMS from unique samples600:1000 of allowed fitting trials, across all channels; no centering/per-trial normalization',
        normalizer_roles='source scalar retained for fine-tuning; target-only scratch fits target scalar',
        source_steps=300,scratch_steps=300,update_steps=150,batch_size=64,lr=.001,weight_decay=.0001,optimizer='Adam',
        checkpoints='fixed last step; fitting accuracy/loss logged, never score-based selection',
        budgets=[0,7,14],one_gesture_scope='already tested classically; deferred neural one-gesture branch, not silently included in this screen',
        updates=['fine_target','fine_pooled','scratch_target'],fine_pooled='uniform sampling from concatenated source and target windows',
        augmentation='independent uniform shift per minibatch window; permutation of channel axis preserves within-channel waveform; same unaugmented batch ids/init/steps across primary families',
        fidelity='ABSDA permutation principle adapted to sparse8channel raw temporalCNN. +/-1 nominal45degrees vs paper +/-2 columns on16circumferential columns. Wide0..7 source-only sensitivity separately reported. No HD-EMG benchmark reproduction.',
        reference='https://unbscholar.dspace.lib.unb.ca/server/api/core/bitstreams/54407f52-28ec-4abf-95ef-33f55e98ff41/content',
        ruling='measured ruler angles forbidden as model input; descriptive +/-45degree and random-position strata planned',
        evaluation='identical original scoring ids,15windows,all gestures; fullCSV recording cost; compare all families/budgets and earlier strongLDA, no forced neural win',
        code_hashes={p:sha(p) for p in ['scripts/pilot_senic_neural.py','src/senic_neural.py','src/senic.py','tests/test_senic_neural.py']})
    write(a.out/'config.json',config);write(a.out/'allocations.json',plans);write(a.out/'trial_manifest.json',rows)
    snap=a.out/'code_snapshot';snap.mkdir()
    for p in config['code_hashes']:(snap/Path(p).name).write_bytes(Path(p).read_bytes())
    (a.out/'environment.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
    print('Frozen neural protocol',len(plans),'allocations',flush=True)
    raw=[];win=[]
    for row in rows:
        assert row['subject'] in DEVELOPMENT and row['session']==0 and sha(row['path'])==row['sha256']
        x=np.loadtxt(row['path'],delimiter=',');raw.append(x[600:1000])
        w,_=trial_windows(x);win.append(w.transpose(0,2,1))
    raw=np.stack(raw);win=np.stack(win).astype(np.float32);labels=np.array([r['label'] for r in rows])
    np.save(a.out/'raw_active.npy',raw);np.save(a.out/'windows.npy',win)
    write(a.out/'data_access.json',dict(trial_ids=list(range(len(rows))),reserved_signal_access=0,raw_verified=True))
    for name in ['models','predictions']:(a.out/name).mkdir()
    training=[];metrics=[];access=[];family_times={}
    def arrays(ids,scale):return (win[ids].reshape(-1,8,50)/scale).astype(np.float32),np.repeat(labels[ids],15)
    def save_model(model,kind,subject,position,budget,method,scale,normids,fitids,sourceids,parent,steps,record):
        name=f'm{len(access):04d}';torch.save(model.state_dict(),a.out/'models'/(name+'.pt'))
        info=dict(model=name,family=kind,subject=subject,position=position,trials=budget,method=method,scale=scale,
                  normalization_ids=normids,fit_ids=fitids,source_history=sourceids,parent_model=parent,steps=steps)
        access.append(info);training.append(dict(model=name,**record));return name
    def score(model,name,kind,method,plan,scale):
        x,y=arrays(plan['score'],scale);prob=probabilities(model,x);pred=prob.argmax(1)
        votes=np.array([np.bincount(z,minlength=7).argmax() for z in pred.reshape(7,15)])
        filename=f'p{len(metrics):04d}.npz';np.savez_compressed(a.out/'predictions'/filename,score_ids=plan['score'],y=y,pred=pred,probabilities=prob,trial_pred=votes)
        metrics.append(dict(family=kind,method=method,subject=plan['subject'],position=plan['position'],trials=plan['trials'],
            recorded_seconds=sum(rows[i]['recorded_seconds'] for i in plan['calibration']),active_analysis_seconds=2*len(plan['calibration']),
            accuracy=float(np.mean(pred==y)),macro_f1=float(f1_score(y,pred,labels=range(7),average='macro',zero_division=0)),
            trial_accuracy=float(np.mean(votes==labels[plan['score']])),model=name,prediction_file=filename))
    for kind,shifts in config['families'].items():
        ft=time.perf_counter()
        for person in DEVELOPMENT:
            pp=[p for p in plans if p['subject']==person];source=pp[0]['source'];scale=training_scale(raw,source)
            sx,sy=arrays(source,scale);torch.manual_seed(42);base=SeNicCNN();t=time.perf_counter()
            rec=fit(base,sx,sy,300,42,shifts=shifts);rec['seconds']=time.perf_counter()-t
            basename=save_model(base,kind,person,0,0,'enrollment',scale,source,source,[],None,300,rec)
            for p in pp:
                if p['trials']==0:score(base,basename,kind,'frozen',p,scale);continue
                if kind.endswith('zero_only'):continue
                for method in config['updates']:
                    cal=p['calibration'];is_scratch=method=='scratch_target'
                    if is_scratch:
                        torch.manual_seed(42);model=SeNicCNN();sc=training_scale(raw,cal);normids=cal;fitids=cal;steps=300
                    else:
                        model=copy.deepcopy(base);sc=scale;normids=source;fitids=source+cal if method=='fine_pooled' else cal;steps=150
                    x,y=arrays(fitids,sc);t=time.perf_counter();rec=fit(model,x,y,steps,42,shifts=shifts);rec['seconds']=time.perf_counter()-t
                    name=save_model(model,kind,person,p['position'],p['trials'],method,sc,normids,fitids,[] if is_scratch else source,
                                    None if is_scratch else basename,steps,rec)
                    score(model,name,kind,method,p,sc)
            write(a.out/'metrics.json',metrics);write(a.out/'model_access.json',access);write(a.out/'training.json',training)
            print(json.dumps(dict(family=kind,subject=person,models=len(access),metrics=len(metrics),elapsed_seconds=time.perf_counter()-start)),flush=True)
        family_times[kind]=time.perf_counter()-ft
    groups=[]
    for key in sorted({(m['family'],m['method'],m['trials']) for m in metrics}):
        selected=[m for m in metrics if m['position']>0 and (m['family'],m['method'],m['trials'])==key]
        values=[]
        for person in DEVELOPMENT:
            mm=[m for m in selected if m['subject']==person]
            values.append(dict(subject=person,**{k:float(np.mean([x[k] for x in mm])) for k in ['accuracy','macro_f1','trial_accuracy','recorded_seconds']}))
        groups.append(dict(family=key[0],method=key[1],trials=key[2],participants=values,
            **{k:float(np.mean([x[k] for x in values])) for k in ['accuracy','macro_f1','trial_accuracy','recorded_seconds']}))
    write(a.out/'summary.json',dict(groups=groups,wall_seconds=time.perf_counter()-start,family_seconds=family_times,
        models=len(access),metric_rows=len(metrics),completed_utc=datetime.now(timezone.utc).isoformat()))
    print('Completed neural screen',len(access),'models',time.perf_counter()-start,'seconds',flush=True)


if __name__=='__main__':
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):main()
