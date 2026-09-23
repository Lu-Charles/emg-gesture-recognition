"""Independently verify saved smoke-run trial boundaries, checkpoints and predictions."""
from collections import Counter
import csv
import hashlib
import importlib
import json
from pathlib import Path
import sys
import time
import warnings

import numpy as np
import torch
import scipy.io as sio
from scipy import signal

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'research/runs/20260907_csl_sal_smoke_v1'
AUDIT=ROOT/'research/runs/20260907_sal_code_audit'

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        while b:=f.read(8*1024*1024):h.update(b)
    return h.hexdigest()

def main():
    start=time.perf_counter();torch.set_num_threads(1)
    warnings.filterwarnings('ignore',message='Default grid_sample and affine_grid behavior')
    manifest=json.loads((OUT/'split_manifest.json').read_text())
    raw=json.loads((OUT/'raw_trial_manifest.json').read_text())
    results=json.loads((OUT/'results.json').read_text())
    cfg=json.loads((OUT/'config.json').read_text())
    assert cfg['participant']==1 and cfg['calibration_repetition_index']==0
    assert len(raw)==580 and len(manifest)==520
    assert len({e['id'] for e in raw})==580 and len({e['raw_sha256'] for e in raw})==580
    byid={e['id']:e for e in raw}
    roles={k:[e for e in manifest if e['role']==k] for k in ('source','calibration','score')}
    assert [len(roles[k]) for k in roles]==[260,26,234]
    for e in manifest:
        assert e['session'] in (1,2) and e['label']==e['gesture']-1
        assert e['role']==('source' if e['session']==1 else 'calibration' if e['rep']==0 else 'score')
        assert e['raw_sha256']==byid[e['id']]['raw_sha256']
        assert sha(Path(e['cache_path']))==e['cache_sha256']
        a=np.load(e['cache_path'],mmap_mode='r')
        assert a.shape==(e['frames'],1,7,24) and a.dtype==np.float32 and np.isfinite(a).all()
        assert e['frames']==e['end']-e['start'] and 0<=e['start']<e['end']<=e['shape'][1]
    # Recompute one predetermined trial per role using an independent explicit channel map.
    preprocessing_errors={}
    for role,entries in roles.items():
        e=entries[0];raw_trial=sio.loadmat(e['path'])['gestures'][e['rep'],0]
        assert hashlib.sha256(np.ascontiguousarray(raw_trial).tobytes()).hexdigest()==e['raw_sha256']
        filtered=signal.sosfilt(signal.butter(4,[20,380],btype='bandpass',fs=2048,output='sos'),raw_trial.T,axis=0)
        filtered=signal.sosfilt(signal.butter(4,[45,55],btype='bandstop',fs=2048,output='sos'),filtered,axis=0)
        rms=np.sqrt(signal.convolve(filtered**2,np.ones((307,1))/307,mode='same'))
        mapped=np.empty((e['frames'],1,7,24),dtype=np.float32)
        for row in range(7):
            for col in range(24):mapped[:,0,row,col]=rms[e['start']:e['end'],col*8+6-row]
        cached=np.load(e['cache_path']);np.testing.assert_array_equal(mapped,cached)
        preprocessing_errors[role]=float(np.max(np.abs(mapped-cached)))
    ledger=json.loads((AUDIT/'paper_era_snapshot_sha256.json').read_text())
    for name,expected in ledger['files'].items():assert sha(AUDIT/'paper_era_snapshot'/name)==expected
    sys.path.insert(0,str(AUDIT/'paper_era_snapshot'));net=importlib.import_module('networks')
    base=torch.load(OUT/'source.pt',map_location='cpu',weights_only=True)
    assert torch.count_nonzero(base['baseline'])==0
    for name,v in [('xshift',0),('yshift',0),('rot_theta',0),('xscale',1),('yscale',1),('xshear',0),('yshear',0)]:
        assert float(base['spatial_adapt.'+name])==v
    entries=roles['score'];lengths=[e['frames'] for e in entries]
    x=torch.from_numpy(np.concatenate([np.load(e['cache_path']) for e in entries]))
    y=np.concatenate([np.full(e['frames'],e['label']) for e in entries])
    checkpoints={};rows=[]
    for kind in cfg['conditions']:
        state=torch.load(OUT/('source.pt' if kind=='frozen' else kind+'.pt'),map_location='cpu',weights_only=True)
        if kind in ('sal_lbn','lbn_only'):
            for n,v in base.items():
                if n=='baseline' or (kind=='sal_lbn' and n.startswith('spatial_adapt.')):continue
                assert torch.equal(v,state[n]),(kind,n)
        m=net.LogisticRegressor(num_classes=26,input_shape=(7,24),channels=168,p_input=.5,baseline=True)
        m.load_state_dict(state);m.eval();pred=[]
        with torch.no_grad():
            for pos in range(0,len(x),1024):pred.append(m(x[pos:pos+1024]).argmax(1).numpy())
        pred=np.concatenate(pred)
        saved=np.load(OUT/(kind+'_predictions.npz'))
        np.testing.assert_array_equal(saved['predictions'],pred)
        np.testing.assert_array_equal(saved['labels'],y)
        np.testing.assert_array_equal(saved['lengths'],lengths)
        np.testing.assert_array_equal(saved['trial_ids'],[e['id'] for e in entries])
        trial_correct=0;begin=0
        for e,length in zip(entries,lengths):
            p=pred[begin:begin+length];counts=Counter(p.tolist());vote=max(counts,key=counts.get)
            trial_correct+=vote==e['label']
            rows.append(dict(condition=kind,trial_id=e['id'],label=e['label'],predicted_label=vote,frames=length,frame_accuracy=float(np.mean(p==e['label']))))
            begin+=length
        assert begin==len(pred)
        recomputed=dict(frame_accuracy=float(np.mean(pred==y)),trial_accuracy=trial_correct/len(entries),correct_trials=trial_correct)
        for key,v in recomputed.items():assert v==results['results'][kind][key]
        checkpoints[kind]=dict(predictions_verified=len(pred),**recomputed)
    with (OUT/'trial_predictions.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    report=dict(passed=True,scope='Independent artifact/prediction verification, not independent participant evaluation',raw_trials=580,active_trials=520,cache_hashes_checked=520,preprocessing_max_errors=preprocessing_errors,checkpoints=checkpoints,wall_seconds=time.perf_counter()-start)
    (OUT/'verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
