"""Single-gesture development screen of original SAL, with fixed optimization budget.

Not a reproduction of the June2025 composite/template-search author pipeline.
"""
from collections import Counter
from copy import deepcopy
from datetime import datetime,timezone
import json
from pathlib import Path
import time
import warnings

import numpy as np
import torch
from torch import nn
from scripts import csl_sal_smoke as core

OUT=core.ROOT/'research/runs/20260907_csl_single_gesture_v1'
PARENT=core.OUT
METHODS=('sal_lbn','lbn_only','classifier_finetune','classifier_frozen_bn')


def mode(model,method):
    model.eval()
    for name,p in model.named_parameters():
        p.requires_grad = name=='baseline' or name.startswith('spatial_adapt.') if method=='sal_lbn' else name=='baseline' if method=='lbn_only' else name in ('bn.weight','bn.bias','fc.weight','fc.bias')
    model.input_dropout.train()
    if method=='classifier_finetune':model.bn.train()


def test_modes(nets):
    checks=[]
    for method in METHODS:
        torch.manual_seed(42);m=nets.LogisticRegressor(num_classes=26,input_shape=(7,24),channels=168,p_input=.5,baseline=True)
        before=deepcopy(m.state_dict());mode(m,method)
        opt=torch.optim.Adam([p for p in m.parameters() if p.requires_grad],lr=.001)
        opt.zero_grad();nn.functional.cross_entropy(m(torch.rand(20,1,7,24)),torch.zeros(20,dtype=torch.long)).backward();opt.step()
        allowed={n for n,p in m.named_parameters() if p.requires_grad}
        if method=='classifier_finetune':allowed.update({'bn.running_mean','bn.running_var','bn.num_batches_tracked'})
        changed={n for n,v in m.state_dict().items() if not torch.equal(v,before[n])}
        assert changed and changed<=allowed,(method,changed-allowed)
        if method!='classifier_finetune':assert torch.equal(before['bn.running_mean'],m.bn.running_mean)
        m.eval();x=torch.rand(3,1,7,24)
        with torch.no_grad():torch.testing.assert_close(m(x),m(x),rtol=0,atol=0)
        checks.append(dict(method=method,passed=True,changed_state=sorted(changed)))
    core.dump(OUT/'mask_tests.json',dict(passed=True,checks=checks))


def score(pred,truth,entries,label):
    boundaries=np.cumsum([0]+[e['frames'] for e in entries]);v=[]
    for a,b in zip(boundaries[:-1],boundaries[1:]):v.append(Counter(pred[a:b].tolist()).most_common(1)[0][0])
    y=np.array([e['label'] for e in entries]);v=np.array(v);unseen=y!=label
    return dict(frame_accuracy=float(np.mean(pred==truth)),trial_accuracy=float(np.mean(v==y)),
        seen_trial_accuracy=float(np.mean(v[~unseen]==y[~unseen])),unseen_trial_accuracy=float(np.mean(v[unseen]==y[unseen])),
        correct_trials=int(np.sum(v==y)),correct_unseen_trials=int(np.sum(v[unseen]==y[unseen])),trial_predictions=v.tolist())


def main():
    began=time.perf_counter();torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    warnings.filterwarnings('ignore',message='Default grid_sample and affine_grid behavior')
    assert not (OUT/'config.json').exists(),'Refusing to overwrite run'
    manifest=json.loads((PARENT/'split_manifest.json').read_text())
    ledger=json.loads((PARENT/'artifact_sha256.json').read_text())
    for name,expected in ledger.items():assert core.digest(PARENT/name)==expected,name
    for e in manifest:assert core.digest(Path(e['cache_path']))==e['cache_sha256']
    assert core.digest(Path(core.__file__))==json.loads((PARENT/'code_snapshot_sha256.json').read_text())['csl_sal_smoke.py']
    calibrations=[e for e in manifest if e['role']=='calibration'];scoring=[e for e in manifest if e['role']=='score']
    assert len(calibrations)==26 and len(scoring)==234
    assert {e['label'] for e in calibrations}==set(range(26))
    assert not {e['id'] for e in calibrations}&{e['id'] for e in scoring}
    assert all(e['rep']==0 and e['session']==2 for e in calibrations)
    cfg=dict(timestamp=datetime.now(timezone.utc).isoformat(),scope='One-class development diagnostic on original audited SAL; not the later author composite recipe',
        parent=str(PARENT),source_sha256=core.digest(PARENT/'source.pt'),split_sha256=core.digest(PARENT/'split_manifest.json'),
        participant=1,source_session=1,target_session=2,source_classes=26,calibration_classes=1,calibration_repetition=0,
        choices='All26gestures, equally weighted; no selection of the best calibration gesture',methods=list(METHODS),seed=42,
        steps=500,batch_size_max=1024,sampling='Independent deterministic randperm per step; all frames if trial shorter than batch; no scoring rows',
        optimizer='Adam',lr=.05,loss='sum cross entropy',weight_decay=0,dropout=.5,warmup_steps=250,
        budget_basis='A fixed 500-update optimization screen for all methods prevents update count from collapsing with fewer calibration trials; not exact author training settings',
        held_out='Same234trials/305152frames across every model, including unused calibration gestures; rep0 from other classes remains unused',
        reporting='Overall, calibration-class and other25class accuracies; compare with frozen source for each choice; final500step checkpoint only',
        recording_cost='3s labeled gesture plus90s target rest inherited from segmentation; not a3s total setup claim',
        no_test_selection=True,code_sha256=core.digest(Path(__file__)),
        no_new_author_recipe_claim='June2025 uses8CSLgestures, gain correction, bounded transform, prototype initial search and different optimizer/preprocessing. This isolates the stated one-class question with the original pipeline.')
    core.dump(OUT/'config.json',cfg);(OUT/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    core.dump(OUT/'calibration_allocations.json',[dict(choice=e['label'],raw_gesture=e['gesture'],trial_id=e['id'],raw_sha256=e['raw_sha256'],cache_sha256=e['cache_sha256'],frames=e['frames'],seconds=e['raw_seconds']) for e in calibrations])
    nets,_,_=core.upstream();test_modes(nets)
    base=torch.load(PARENT/'source.pt',map_location='cpu',weights_only=True)
    tx,ty,te=core.load_role(manifest,'score');truth=ty.numpy();frozen=np.load(PARENT/'frozen_predictions.npz')['predictions']
    records=[]
    for e in calibrations:
        label=e['label'];x=torch.from_numpy(np.load(e['cache_path']));y=torch.full((len(x),),label,dtype=torch.long)
        frozen_metrics=score(frozen,truth,te,label)
        for method in METHODS:
            ident=f'g{e["gesture"]:02}_{method}';started=time.perf_counter()
            m=nets.LogisticRegressor(num_classes=26,input_shape=(7,24),channels=168,p_input=.5,baseline=True)
            m.load_state_dict(base);mode(m,method);torch.manual_seed(42)
            gen=torch.Generator().manual_seed(42)
            trainable=[p for p in m.parameters() if p.requires_grad]
            opt=torch.optim.Adam(trainable,lr=.05,weight_decay=0)
            schedule=torch.optim.lr_scheduler.LinearLR(opt,start_factor=.01,end_factor=1,total_iters=250)
            losses=[]
            for step in range(500):
                ix=torch.randperm(len(x),generator=gen)[:min(1024,len(x))]
                opt.zero_grad();loss=nn.functional.cross_entropy(m(x[ix]),y[ix],reduction='sum');assert torch.isfinite(loss),(ident,step)
                loss.backward();opt.step();schedule.step();losses.append(float(loss.detach())/len(ix))
            assert all(torch.isfinite(p).all() for p in m.parameters()),ident
            permitted={n for n,p in m.named_parameters() if p.requires_grad}
            if method=='classifier_finetune':permitted.update({'bn.running_mean','bn.running_var','bn.num_batches_tracked'})
            for n,v in m.state_dict().items():
                if n not in permitted:assert torch.equal(base[n],v),(ident,n)
            m.eval();torch.save(m.state_dict(),OUT/(ident+'.pt'))
            fit=float(np.mean(core.predict(m,x)==label))
            pred=core.predict(m,tx);measures=score(pred,truth,te,label)
            np.savez_compressed(OUT/(ident+'_predictions.npz'),predictions=pred)
            row=dict(id=ident,method=method,calibration_gesture=e['gesture'],calibration_label=label,calibration_trial_id=e['id'],calibration_frames=len(x),
                trainable_scalars=sum(p.numel() for p in trainable),fit_frame_accuracy=fit,loss_first50=float(np.mean(losses[:50])),loss_last50=float(np.mean(losses[-50:])),
                loss_history=losses,**measures,frozen_trial_accuracy=frozen_metrics['trial_accuracy'],frozen_unseen_accuracy=frozen_metrics['unseen_trial_accuracy'],
                wall_seconds=time.perf_counter()-started)
            records.append(row);core.dump(OUT/'models.json',records)
            print(ident,'trial',round(measures['trial_accuracy'],4),'unseen',round(measures['unseen_trial_accuracy'],4),'fit',round(fit,4),flush=True)
    summary={}
    for method in METHODS:
        subset=[r for r in records if r['method']==method]
        summary[method]={key:float(np.mean([r[key] for r in subset])) for key in ('trial_accuracy','frame_accuracy','seen_trial_accuracy','unseen_trial_accuracy','fit_frame_accuracy')}
        summary[method].update(choices=len(subset),beats_frozen=sum(r['trial_accuracy']>r['frozen_trial_accuracy'] for r in subset),trial_min=min(r['trial_accuracy'] for r in subset),trial_max=max(r['trial_accuracy'] for r in subset))
    core.dump(OUT/'results.json',dict(summary=summary,models=len(records),predictions=len(records)*len(tx),wall_seconds=time.perf_counter()-began))
    core.dump(OUT/'artifact_sha256.json',{p.name:core.digest(p) for p in OUT.iterdir() if p.is_file() and p.name!='artifact_sha256.json'})
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
