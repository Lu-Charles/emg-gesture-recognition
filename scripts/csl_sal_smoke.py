"""Fixed subject1/session1 -> session2 SAL implementation smoke, not cohort replication.

Reuse the hash-pinned author layers, filters, RMS and segmenter. Correct image layout
against the dataset provider's example; retain per-trial identities and variable lengths.
"""
import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import importlib
import json
from pathlib import Path
import sys
import time
import warnings

import numpy as np
import scipy
import scipy.io as sio
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'research/runs/20260907_csl_sal_smoke_v1'
AUDIT=ROOT/'research/runs/20260907_sal_code_audit'
DATA=ROOT/'data/public/csl-hdemg'

def dump(path,value): path.write_text(json.dumps(value,indent=2)+'\n')
def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        while block:=f.read(8*1024*1024): h.update(block)
    return h.hexdigest()

def upstream():
    ledger=json.loads((AUDIT/'paper_era_snapshot_sha256.json').read_text())
    for name,expected in ledger['files'].items(): assert digest(AUDIT/'paper_era_snapshot'/name)==expected
    sys.path.insert(0,str(AUDIT/'paper_era_snapshot'))
    return importlib.import_module('networks'),importlib.import_module('tensorize_emg'),importlib.import_module('emg_processing')

def images(emg):
    # Provider example: remove 0-based channels7,15,...191, reshape Fortran and flip spatial rows.
    emg=np.asarray(emg)
    assert emg.ndim==2 and emg.shape[1]==192
    useful=emg[:,np.arange(192)%8!=7]
    return useful.reshape(len(emg),24,7).transpose(0,2,1)[:,None,::-1,:].copy()

def role(session,gesture,rep):
    if gesture==0: return 'segmentation_rest_source' if session==1 else 'segmentation_rest_target'
    return 'source' if session==1 else 'calibration' if rep==0 else 'score'

def protocol():
    return dict(scope='Fixed implementation smoke; not full paper replication or novel-method evaluation',
        participant=1,source_session=1,target_session=2,development_participants=[1],reserved_participants=[2,3,4,5],
        upstream_revision='cc5b530cca1070994465993f0ce400009f16d6ff',seed=42,device='cpu',threads=1,
        source_repetitions='all 10/class',calibration_repetition_index=0,score_repetitions=list(range(1,10)),
        raw_gestures=list(range(1,27)),label_mapping='raw gesture minus1',input_shape=[7,24],fs=2048,
        geometry='Dataset example: exclude each eighth channel, Fortran reshape7x24, reverse spatial row axis; preserve time',
        filters='unchanged author causal fourth-order Butterworth20-380Hz then45-55Hz stop',
        rms_half_window_samples=153,rms_kernel_samples=307,
        segmentation='Unchanged author73.2ms RMS/median/longest-active segment with dataset-correct image mapping; isolated per trial',
        rest_access='All available gest0 trials in each permitted session for segmentation baseline only; separately disclosed extra unlabeled recording cost',
        normalization='No additional scaling; source-fitted BN in author LogReg; frozen during SAL/LBN adaptation',
        source_epochs=2,adaptation_epochs=20,batch_size=1024,optimizer='Adam',lr=.05,weight_decay=0,
        loss='cross entropy sum',warmup='LinearLR .01 to1; first source epoch and first10 adaptation epochs',input_dropout=.5,
        conditions=['frozen','sal_lbn','lbn_only','all_parameter_finetune'],
        parameter_masks=dict(source='fc and BN affine; SAL/LBN frozen',sal_lbn='seven SAL and168baseline parameters only',lbn_only='168baseline parameters only',all_parameter_finetune='all author model parameters, including SAL/LBN and classifier; BN statistics updated'),
        scoring='eval mode; same score trials; frame accuracy and full-active-trial majority vote; Counter first-occurrence tie rule',
        differences_from_untagged_runner=['Dataset-correct channel map and temporal ordering','Variable trial lengths, no duplication or padded frames','Explicit eval before scoring','Per-target recomputed score','Explicit all-parameter mask for fine-tuning; original source flags not inherited'],
        checkpoint='last fixed epoch; no held-out selection')

def tests():
    nets,tens,proc=upstream()
    arr=np.arange(5*192,dtype=float).reshape(5,192)
    mapped=images(arr)
    np.testing.assert_array_equal(images(torch.from_numpy(arr)),mapped)
    for t in range(5):
        ref=np.flipud(np.delete(arr[t],np.s_[7:192:8]).reshape(24,7).T)
        np.testing.assert_array_equal(mapped[t,0],ref)
    assert np.all(mapped[1]-mapped[0]==192)
    roles={key:set() for key in ('source','calibration','score')}
    for s in (1,2):
        for g in range(1,27):
            for r in range(10): roles[role(s,g,r)].add((s,g,r))
    assert [len(roles[k]) for k in roles]==[260,26,234]
    assert all(not roles[a]&roles[b] for a,b in [('source','calibration'),('source','score'),('calibration','score')])
    # Variable lengths must not mix adjacent trials during voting.
    p=np.array([2,2,1, 1,1,1,2,2]); lengths=[3,5]
    assert votes(p,lengths)==[2,1]
    torch.manual_seed(42)
    m=nets.LogisticRegressor(num_classes=26,input_shape=(7,24),channels=168,p_input=.5)
    before=deepcopy(m.state_dict()); configure(m,'sal_lbn')
    opt=torch.optim.Adam([p for p in m.parameters() if p.requires_grad],lr=.001)
    opt.zero_grad(); nn.functional.cross_entropy(m(torch.rand(12,1,7,24)),torch.arange(12)).backward(); opt.step()
    changed=[n for n,p in m.state_dict().items() if not torch.equal(p,before[n])]
    assert changed and all(n=='baseline' or n.startswith('spatial_adapt.') for n in changed)
    configure(m,'all_parameter_finetune'); assert all(p.requires_grad for p in m.parameters()) and m.bn.training
    checks=['geometry exactly matches provider example','time order preserved','260/26/234 disjoint trial identities','variable-length majority vote boundaries','SAL updates only authorized state','all-parameter fine-tuning mask explicit']
    dump(OUT/'pipeline_tests.json',dict(passed=True,checks=checks)); print('Passed',len(checks),'pipeline tests',flush=True)

def prepare():
    start=time.perf_counter(); nets,tens,proc=upstream()
    archive=json.loads((OUT/'archive_audit.json').read_text()); assert archive['passed']
    extracted={a['name']:a for a in archive['extracted']}
    manifest=[]; objects={}
    # Read and identify every permitted trial before selecting any frame.
    for s in (1,2):
        for g in range(27):
            path=DATA/f'subject1/session{s}/gest{g}.mat'
            assert digest(path)==extracted[str(path.relative_to(DATA))]['sha256']
            cells=sio.loadmat(path)['gestures']; assert cells.ndim==2 and cells.shape[1]==1
            assert len(cells)==(30 if g==0 else 10),(str(path),cells.shape)
            for r in range(len(cells)):
                raw=cells[r,0]; assert raw.ndim==2 and raw.shape[0]==192 and raw.shape[1]>300
                assert np.isfinite(raw).all()
                rawhash=hashlib.sha256(np.ascontiguousarray(raw).tobytes()).hexdigest()
                manifest.append(dict(id=f's1_s{s}_g{g:02}_r{r:02}',session=s,gesture=g,label=g-1,rep=r,role=role(s,g,r),
                    path=str(path),raw_sha256=rawhash,shape=list(raw.shape),raw_seconds=raw.shape[1]/2048))
            print('Validated raw session',s,'gesture',g,flush=True)
    assert len({r['raw_sha256'] for r in manifest})==len(manifest),'Duplicate raw trial detected'
    dump(OUT/'raw_trial_manifest.json',manifest)
    # A __new__ instance avoids the upstream all-session multi-GB zero allocation.
    class ProviderMapped(tens.EMGSegmentData):
        def get_images(self,emg): return images(emg)
    helper=ProviderMapped.__new__(ProviderMapped); helper.dataset='csl';helper.fs=2048;helper.Mrms=153
    cache=OUT/'frames';cache.mkdir(exist_ok=True)
    active=[]; baseline_info=[]
    for s in (1,2):
        baseline=helper.get_baseline(str(DATA/f'subject1/session{s}'))
        assert np.isfinite(baseline).all()
        np.save(OUT/f'session{s}_segmentation_baseline.npy',baseline)
        baseline_info.append(dict(session=s,trials=30,seconds=sum(r['raw_seconds'] for r in manifest if r['session']==s and r['gesture']==0)))
        for g in range(1,27):
            cells=sio.loadmat(DATA/f'subject1/session{s}/gest{g}.mat')['gestures']
            for r in range(10):
                entry=next(a for a in manifest if a['session']==s and a['gesture']==g and a['rep']==r)
                raw=cells[r,0].T
                emg=proc.bandstop(proc.bandpass(raw,fs=2048),fs=2048)
                lo,hi=helper.segment(emg,baseline)
                assert 0<=lo<hi<=len(raw),(entry['id'],lo,hi,len(raw))
                rms=proc.get_rms_signal(emg,Mrms=153)
                x=images(rms)[lo:hi].astype(np.float32)
                assert len(x)==hi-lo and np.isfinite(x).all()
                path=cache/(entry['id']+'.npy');np.save(path,x)
                active.append(dict(**entry,start=int(lo),end=int(hi),frames=len(x),cache_path=str(path),cache_sha256=digest(path)))
            print('Prepared session',s,'gesture',g,flush=True)
    dump(OUT/'split_manifest.json',active)
    report=dict(passed=True,raw_trials=len(manifest),active_trials=len(active),roles={k:dict(trials=sum(e['role']==k for e in active),frames=sum(e['frames'] for e in active if e['role']==k),recording_seconds=sum(e['raw_seconds'] for e in active if e['role']==k)) for k in ('source','calibration','score')},additional_segmentation_rest=baseline_info,raw_sample_lengths=sorted(set(r['shape'][1] for r in manifest)),wall_seconds=time.perf_counter()-start)
    dump(OUT/'preprocessing.json',report);print(json.dumps(report),flush=True)

def configure(model,kind):
    model.eval()
    for name,p in model.named_parameters():
        p.requires_grad = (name=='baseline' or name.startswith('spatial_adapt.')) if kind=='sal_lbn' else name=='baseline' if kind=='lbn_only' else not(name=='baseline' or name.startswith('spatial_adapt.')) if kind=='source' else kind=='all_parameter_finetune'
    if kind in ('source','all_parameter_finetune'): model.train()
    else: model.input_dropout.train()

def load_role(manifest,kind):
    entries=[e for e in manifest if e['role']==kind]
    x=np.concatenate([np.load(e['cache_path']) for e in entries])
    y=np.concatenate([np.full(e['frames'],e['label'],dtype=np.int64) for e in entries])
    return torch.from_numpy(x),torch.from_numpy(y),entries

def train(model,x,y,kind):
    start=time.perf_counter(); torch.manual_seed(42);configure(model,kind)
    batch=1024;epochs=2 if kind=='source' else 20
    loader=DataLoader(TensorDataset(x,y),batch_size=batch,shuffle=True,generator=torch.Generator().manual_seed(42))
    params=[p for p in model.parameters() if p.requires_grad]
    mask=[n for n,p in model.named_parameters() if p.requires_grad]
    opt=torch.optim.Adam(params,lr=.05,weight_decay=0)
    schedule=torch.optim.lr_scheduler.LinearLR(opt,start_factor=.01,end_factor=1,total_iters=len(loader)*(1 if kind=='source' else 10))
    losses=[]
    for epoch in range(epochs):
        total=0
        for signals,labels in loader:
            opt.zero_grad();loss=nn.functional.cross_entropy(model(signals),labels,reduction='sum')
            assert torch.isfinite(loss),kind
            loss.backward();opt.step();schedule.step();total+=float(loss.detach())
        losses.append(total/len(x));print(kind,'epoch',epoch+1,'loss',losses[-1],flush=True)
    model.eval()
    assert all(torch.isfinite(p).all() for p in model.parameters())
    return dict(kind=kind,epochs=epochs,steps=epochs*len(loader),trainable_names=mask,trainable_scalars=sum(p.numel() for p in params),loss_by_epoch=losses,wall_seconds=time.perf_counter()-start)

def predict(model,x):
    model.eval();parts=[]
    with torch.no_grad():
        for a in range(0,len(x),1024):
            logits=model(x[a:a+1024]);assert torch.isfinite(logits).all();parts.append(logits.argmax(1).numpy())
    return np.concatenate(parts)

def votes(pred,lengths):
    assert sum(lengths)==len(pred)
    ends=np.cumsum(lengths);starts=np.r_[0,ends[:-1]]
    return [Counter(pred[a:b].tolist()).most_common(1)[0][0] for a,b in zip(starts,ends)]

def metrics(pred,y,entries):
    vp=votes(pred,[e['frames'] for e in entries]);truth=[e['label'] for e in entries]
    return dict(frame_accuracy=float(np.mean(pred==np.asarray(y))),trial_accuracy=float(np.mean(np.array(vp)==truth)),frames=len(pred),trials=len(entries),correct_trials=int(np.sum(np.array(vp)==truth)))

def run():
    start=time.perf_counter();nets,_,_=upstream();manifest=json.loads((OUT/'split_manifest.json').read_text())
    x,y,se=load_role(manifest,'source');cx,cy,ce=load_role(manifest,'calibration')
    torch.manual_seed(42)
    base=nets.LogisticRegressor(num_classes=26,input_shape=(7,24),channels=168,p_input=.5,baseline=True)
    source_log=train(base,x,y,'source');torch.save(base.state_dict(),OUT/'source.pt')
    source_fit=metrics(predict(base,x),y,se);del x,y
    models={'frozen':base};train_logs={'source':source_log};fits={}
    for kind in ('sal_lbn','lbn_only','all_parameter_finetune'):
        m=deepcopy(base);train_logs[kind]=train(m,cx,cy,kind)
        if kind in ('sal_lbn','lbn_only'):
            for name,v in base.state_dict().items():
                if name=='baseline' or (kind=='sal_lbn' and name.startswith('spatial_adapt.')):continue
                assert torch.equal(v,m.state_dict()[name]),('Unauthorized change',kind,name)
        torch.save(m.state_dict(),OUT/(kind+'.pt'));models[kind]=m
        fits[kind]=metrics(predict(m,cx),cy,ce)
    dump(OUT/'training.json',dict(source_fit=source_fit,calibration_fits=fits,training=train_logs))
    del cx,cy
    # First scoring happens only after all fixed training conditions are complete and saved.
    tx,ty,te=load_role(manifest,'score');results={}
    for kind,m in models.items():
        pred=predict(m,tx);np.savez_compressed(OUT/(kind+'_predictions.npz'),predictions=pred,labels=ty.numpy(),lengths=np.array([e['frames'] for e in te]),trial_ids=np.array([e['id'] for e in te]))
        results[kind]=metrics(pred,ty,te)
        assert np.array_equal(predict(m,tx[:2048]),pred[:2048]),'Nondeterministic scoring'
        print(kind,results[kind],flush=True)
    report=dict(scope='Single development participant/session pair, seed42; preliminary implementation smoke',results=results,wall_seconds=time.perf_counter()-start,
        spatial_parameters={k:{n:float(v.detach()) for n,v in m.spatial_adapt.named_parameters()} for k,m in models.items()},versions=dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__,torch=torch.__version__))
    dump(OUT/'results.json',report)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['freeze','test','prepare','train']);a=ap.parse_args()
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    warnings.filterwarnings('ignore',message='Default grid_sample and affine_grid behavior')
    if a.stage=='freeze':
        assert not (OUT/'config.json').exists();dump(OUT/'config.json',protocol())
    elif a.stage=='test': tests()
    elif a.stage=='prepare': prepare()
    else: run()

if __name__=='__main__': main()
