"""Reload all single-gesture checkpoints and independently verify scoring and state masks."""
import json
from pathlib import Path
import time
import warnings
import numpy as np
import torch
from scripts import csl_sal_smoke as core

OUT=core.ROOT/'research/runs/20260907_csl_single_gesture_v1'


def voted(p,entries):
    result=[];pos=0
    for e in entries:
        a=p[pos:pos+e['frames']];counts=np.bincount(a,minlength=26);winners=np.flatnonzero(counts==counts.max())
        result.append(int(a[np.flatnonzero(np.isin(a,winners))[0]]));pos+=len(a)
    assert pos==len(p)
    return np.array(result)


def main():
    began=time.perf_counter();torch.set_num_threads(1)
    warnings.filterwarnings('ignore',message='Default grid_sample and affine_grid behavior')
    cfg=json.loads((OUT/'config.json').read_text());records=json.loads((OUT/'models.json').read_text());report=json.loads((OUT/'results.json').read_text())
    assert len(records)==104
    assert {(r['calibration_label'],r['method']) for r in records}=={(g,m) for g in range(26) for m in cfg['methods']}
    assert cfg['steps']==500 and cfg['no_test_selection']
    ledger=json.loads((OUT/'artifact_sha256.json').read_text())
    for name,expected in ledger.items():assert core.digest(OUT/name)==expected
    parent=Path(cfg['parent']);assert core.digest(parent/'source.pt')==cfg['source_sha256']
    assert core.digest(parent/'split_manifest.json')==cfg['split_sha256']
    manifest=json.loads((parent/'split_manifest.json').read_text());cal={e['label']:e for e in manifest if e['role']=='calibration'}
    entries=[e for e in manifest if e['role']=='score'];truth_trial=np.array([e['label'] for e in entries])
    x=torch.from_numpy(np.concatenate([np.load(e['cache_path']) for e in entries]));truth=np.concatenate([np.full(e['frames'],e['label']) for e in entries])
    assert len(x)==305152 and len(entries)==234 and all(e['rep']!=0 for e in entries)
    nets,_,_=core.upstream();base=torch.load(parent/'source.pt',map_location='cpu',weights_only=True)
    frozen=np.load(parent/'frozen_predictions.npz')['predictions'];fv=voted(frozen,entries)
    checked=[]
    for index,row in enumerate(records):
        label=row['calibration_label'];e=cal[label]
        assert e['id']==row['calibration_trial_id'] and e['session']==2 and e['rep']==0
        assert e['frames']==row['calibration_frames'] and e['raw_seconds']==3.0
        state=torch.load(OUT/(row['id']+'.pt'),map_location='cpu',weights_only=True)
        method=row['method']
        allowed={n for n in base if (n=='baseline' or n.startswith('spatial_adapt.'))} if method=='sal_lbn' else {'baseline'} if method=='lbn_only' else {'fc.weight','fc.bias','bn.weight','bn.bias'}
        if method=='classifier_finetune':allowed.update({'bn.running_mean','bn.running_var','bn.num_batches_tracked'})
        for n,v in base.items():
            if n not in allowed:assert torch.equal(v,state[n]),(row['id'],n)
        m=nets.LogisticRegressor(num_classes=26,input_shape=(7,24),channels=168,p_input=.5,baseline=True);m.load_state_dict(state);m.eval()
        predictions=[]
        with torch.no_grad():
            for p in range(0,len(x),1024):predictions.append(m(x[p:p+1024]).argmax(1).numpy())
            cx=torch.from_numpy(np.load(e['cache_path']));fit=[]
            for p in range(0,len(cx),1024):fit.append(m(cx[p:p+1024]).argmax(1).numpy())
        pred=np.concatenate(predictions);fit=float(np.mean(np.concatenate(fit)==label))
        np.testing.assert_array_equal(pred,np.load(OUT/(row['id']+'_predictions.npz'))['predictions'])
        vote=voted(pred,entries);unseen=truth_trial!=label;assert unseen.sum()==225
        values=dict(trial_accuracy=float(np.mean(vote==truth_trial)),frame_accuracy=float(np.mean(pred==truth)),seen_trial_accuracy=float(np.mean(vote[~unseen]==truth_trial[~unseen])),unseen_trial_accuracy=float(np.mean(vote[unseen]==truth_trial[unseen])),fit_frame_accuracy=fit)
        for k,v in values.items():assert v==row[k],(row['id'],k,v,row[k])
        np.testing.assert_array_equal(vote,row['trial_predictions'])
        assert row['frozen_trial_accuracy']==float(np.mean(fv==truth_trial))
        assert row['frozen_unseen_accuracy']==float(np.mean(fv[unseen]==truth_trial[unseen]))
        checked.append(dict(id=row['id'],passed=True,predictions=len(pred)))
        if index%10==0:print('Verified',index+1,'of104',flush=True)
    for method in cfg['methods']:
        rows=[r for r in records if r['method']==method]
        for key in ('trial_accuracy','frame_accuracy','seen_trial_accuracy','unseen_trial_accuracy','fit_frame_accuracy'):
            assert float(np.mean([r[key] for r in rows]))==report['summary'][method][key]
    result=dict(passed=True,models=len(checked),predictions=sum(x['predictions'] for x in checked),score_trials=234,unseen_trials_per_choice=225,
        checks=['All26calibration gestures x4methods present','Previous source/split hashes unchanged','Same234scoretrials throughout','Allowed adaptation state only','Allsavedpredictions exactly replayed','Fit and seen/unseen votes recomputed','Aggregate metrics recomputed'],wall_seconds=time.perf_counter()-began)
    core.dump(OUT/'verification.json',result);print(json.dumps(result,indent=2))

if __name__=='__main__':main()
