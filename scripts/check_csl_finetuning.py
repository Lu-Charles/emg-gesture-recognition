"""Resolve classifier-only vs all-parameter fine-tuning on the frozen CSL smoke split."""
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import warnings

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from scripts import csl_sal_smoke as core

OUT=core.ROOT/'research/runs/20260907_csl_classifier_finetune_v1'
PARENT=core.OUT
ALLOWED={'bn.weight','bn.bias','fc.weight','fc.bias'}


def configure(model):
    model.train()
    for name,p in model.named_parameters():p.requires_grad=name in ALLOWED
    assert {n for n,p in model.named_parameters() if p.requires_grad}==ALLOWED


def unchanged_spatial(before,after):
    for name,value in before.items():
        if name=='baseline' or name.startswith('spatial_adapt.'):
            assert torch.equal(value,after[name]),name


def test_mask(nets):
    torch.manual_seed(42)
    m=nets.LogisticRegressor(num_classes=26,input_shape=(7,24),channels=168,p_input=.5,baseline=True)
    before=deepcopy(m.state_dict());configure(m)
    opt=torch.optim.Adam([p for p in m.parameters() if p.requires_grad],lr=.05)
    opt.zero_grad();nn.functional.cross_entropy(m(torch.rand(16,1,7,24)),torch.arange(16)).backward();opt.step()
    unchanged_spatial(before,m.state_dict())
    assert all(p.grad is None for n,p in m.named_parameters() if n not in ALLOWED)
    assert not torch.equal(before['fc.weight'],m.fc.weight)
    assert not torch.equal(before['bn.running_mean'],m.bn.running_mean)
    core.dump(OUT/'mask_test.json',dict(passed=True,checks=['SAL and baseline state unchanged','frozen parameters receive no gradient','classifier weights update','BN running statistics update'],trainable_names=sorted(ALLOWED)))


def main():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    warnings.filterwarnings('ignore',message='Default grid_sample and affine_grid behavior')
    OUT.mkdir(exist_ok=False)
    began=time.perf_counter()
    # Verify the exact implementation, data manifest, and previous outputs reused here.
    ledger=json.loads((PARENT/'artifact_sha256.json').read_text())
    for name,expected in ledger.items():assert core.digest(PARENT/name)==expected,name
    assert core.digest(Path(core.__file__))==json.loads((PARENT/'code_snapshot_sha256.json').read_text())['csl_sal_smoke.py']
    manifest=json.loads((PARENT/'split_manifest.json').read_text())
    for e in manifest:assert core.digest(Path(e['cache_path']))==e['cache_sha256']
    config=dict(timestamp=datetime.now(timezone.utc).isoformat(),scope='Pre-planned comparator-fidelity diagnostic after the original four-condition results; no hyperparameter search',
        parent=str(PARENT),parent_source_sha256=core.digest(PARENT/'source.pt'),split_sha256=core.digest(PARENT/'split_manifest.json'),
        participant=1,source_session=1,target_session=2,seed=42,epochs=20,batch_size=1024,lr=.05,optimizer='Adam',weight_decay=0,
        loss='sum cross entropy',warmup_epochs=10,input_dropout=.5,trainable_names=sorted(ALLOWED),
        frozen='SAL identity and baseline zero throughout',bn='affine and running statistics updated on calibration only',
        checkpoint='last fixed epoch',score='exact same234trials/305152frames; no held-out selection',
        calibration_cost='78s labeled gestures plus90s unlabeled target rest for precomputed segmentation',
        source_basis='The pinned old runner freezes SAL/LBN during source training and retains those flags during its fine-tuning branch',
        code_sha256=core.digest(Path(__file__)))
    core.dump(OUT/'config.json',config)
    (OUT/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    nets,_,_=core.upstream();test_mask(nets)
    model=nets.LogisticRegressor(num_classes=26,input_shape=(7,24),channels=168,p_input=.5,baseline=True)
    base=torch.load(PARENT/'source.pt',map_location='cpu',weights_only=True);model.load_state_dict(base);configure(model)
    x,y,entries=core.load_role(manifest,'calibration')
    assert len(entries)==26 and all(e['session']==2 and e['rep']==0 for e in entries)
    torch.manual_seed(42)
    loader=DataLoader(TensorDataset(x,y),batch_size=1024,shuffle=True,generator=torch.Generator().manual_seed(42))
    opt=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=.05,weight_decay=0)
    schedule=torch.optim.lr_scheduler.LinearLR(opt,start_factor=.01,end_factor=1,total_iters=len(loader)*10)
    losses=[];train_start=time.perf_counter()
    for epoch in range(20):
        total=0
        for signals,labels in loader:
            opt.zero_grad();loss=nn.functional.cross_entropy(model(signals),labels,reduction='sum');assert torch.isfinite(loss)
            loss.backward();opt.step();schedule.step();total+=float(loss.detach())
        losses.append(total/len(x));print('epoch',epoch+1,'loss',losses[-1],flush=True)
    training_seconds=time.perf_counter()-train_start
    unchanged_spatial(base,model.state_dict());model.eval()
    torch.save(model.state_dict(),OUT/'classifier_finetune.pt')
    fitting=core.metrics(core.predict(model,x),y,entries)
    core.dump(OUT/'training.json',dict(epochs=20,steps=20*len(loader),trainable_scalars=sum(p.numel() for p in model.parameters() if p.requires_grad),loss_by_epoch=losses,calibration_fit=fitting,wall_seconds=training_seconds))
    del x,y
    tx,ty,te=core.load_role(manifest,'score');pred=core.predict(model,tx)
    results=core.metrics(pred,ty,te)
    np.savez_compressed(OUT/'predictions.npz',predictions=pred,labels=ty.numpy(),lengths=np.array([e['frames'] for e in te]),trial_ids=np.array([e['id'] for e in te]))
    core.dump(OUT/'results.json',dict(condition='classifier_only_finetune',results=results,wall_seconds=time.perf_counter()-began))
    print(json.dumps(results),flush=True)
    # Separate checkpoint reload and direct scoring/voting implementation verify outputs.
    verify_start=time.perf_counter()
    check=nets.LogisticRegressor(num_classes=26,input_shape=(7,24),channels=168,p_input=.5,baseline=True)
    state=torch.load(OUT/'classifier_finetune.pt',map_location='cpu',weights_only=True)
    unchanged_spatial(base,state);check.load_state_dict(state);check.eval()
    preds=[]
    with torch.no_grad():
        for i in range(0,len(tx),1024):preds.append(check(tx[i:i+1024]).argmax(1).numpy())
    reconstructed=np.concatenate(preds);saved=np.load(OUT/'predictions.npz')
    np.testing.assert_array_equal(reconstructed,saved['predictions'])
    original=np.load(PARENT/'frozen_predictions.npz')
    for key in ('labels','lengths','trial_ids'):np.testing.assert_array_equal(saved[key],original[key])
    offset=0;correct=0;trial_results=[]
    for e in te:
        values=reconstructed[offset:offset+e['frames']];counts=Counter(values.tolist());vote=max(counts,key=counts.get)
        correct+=int(vote==e['label']);offset+=e['frames']
        trial_results.append(dict(trial_id=e['id'],label=e['label'],prediction=vote,correct=vote==e['label']))
    assert offset==305152 and correct==results['correct_trials']
    assert float(np.mean(reconstructed==original['labels']))==results['frame_accuracy']
    assert correct/234==results['trial_accuracy']
    core.dump(OUT/'trial_predictions.json',trial_results)
    core.dump(OUT/'verification.json',dict(passed=True,prior_artifacts_unchanged=True,cache_hashes_checked=len(manifest),predictions_verified=len(pred),score_identity_verified=True,spatial_and_baseline_unchanged=True,correct_trials=correct,wall_seconds=time.perf_counter()-verify_start))
    core.dump(OUT/'artifact_sha256.json',{p.name:core.digest(p) for p in OUT.iterdir() if p.is_file() and p.name!='artifact_sha256.json'})
    print('Saved-checkpoint verification passed',flush=True)

if __name__=='__main__':main()
