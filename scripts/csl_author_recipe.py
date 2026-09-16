"""Author June2025 SAL recipe, with provider geometry and explicit component controls."""
import argparse
import ast
from collections import Counter
from copy import deepcopy
from datetime import datetime,timezone
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
from torch.utils.data import DataLoader,TensorDataset

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'research/runs/20260907_csl_author_recipe_v1'
DATA=ROOT/'data/public/csl-hdemg'
RAW_GESTURES=[8,9,12,13,16,21,23,24]
BOUNDS=[[-8/23,8/23],[-8/6,8/6],[-15/180,15/180],[1/1.1,1.1],[1/1.1,1.1],[-.1,.1],[-.1,.1]]

def dump(p,obj):p.write_text(json.dumps(obj,indent=2)+'\n')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        while b:=f.read(8*1024*1024):h.update(b)
    return h.hexdigest()

def modules():
    source=OUT/'source';ledger=json.loads((source/'sha256.json').read_text())
    for name,h in ledger['files'].items():assert sha(source/name)==h,name
    sys.path.insert(0,str(source));nets=importlib.import_module('networks');utils=importlib.import_module('networks_utils')
    tensor=importlib.import_module('tensorize_emg');proc=importlib.import_module('emg_processing')
    return nets,utils,tensor,proc

def provider_images(x):
    a=np.asarray(x);assert a.ndim==2 and a.shape[1]==192
    return a[:,np.arange(192)%8!=7].reshape(len(a),24,7).transpose(0,2,1)[:,None,::-1,:].copy()

def config():
    return dict(timestamp=datetime.now(timezone.utc).isoformat(),scope='Original author June2025eight-class recipe with explicit provider geometry correction; component ablation is development only',
        author_revision='7c5a075a58dff06566b82965f8235ba744377fb8',participant=1,source_session=1,target_session=2,raw_gestures=RAW_GESTURES,
        calibration_raw_gesture=21,calibration_repetition=0,score_repetitions=list(range(1,10)),seed=42,
        preprocessing=dict(fs=2048,rms_half_window_samples=512,channel_centering=True,geometry='provider-correct; original author reverses time/drops first row',
        baseline_flag='mean_square (literal generator spelling; does NOT activate mean-square subtraction branch)',median_filter=True,
        segmentation='unchanged author function with provider get_images; rest-baseline as original mean square'),
        source=dict(epochs=15,batch_size=128,lr=.001,warmup_epochs=3,loss='meanCE',dropout=.5),
        sal=dict(search_points=16384,search='author Latin hypercube plusidentity; seed42 specified',search_mode='bicubic',prototype='sqrt(mean(frame**2)) percalibrationclass',steps=500,lr=.01,dropout_during_adaptation=False,adapt_mode='bilinear',score_mode='bicubic',bounds=BOUNDS,learnable_baseline=False),
        fine=dict(epochs=150,lr=.001,warmup_epochs=30,batch_size=128,loss='meanCE',dropout=.5,trainable='classifier andBN; identityinputtransform'),
        conditions=['frozen','gain_only','spatial_only','spatial_gain','classifier_fine','classifier_fine_gain'],
        access='Only existing participant1sessions1/2;3s labeled plus90s targetrest; identical72scoretrials',
        decisions='All conditions fixed before outcome; no best condition selection; preserved author component and recipe differences disclosed')

def make_model(nets,transform='spatial-adaptation'):
    return nets.LogisticRegressor(num_classes=8,input_shape=(7,24),channels=168,p_input=.5,baseline=False,input_transform_name=transform,circular=False,boundaries=BOUNDS)

def load_initial_functions():
    # Extract only these reviewed pure functions; no external logging or author runner is executed.
    source=(OUT/'source/sal_classification/deep_learning.py').read_text();tree=ast.parse(source)
    names={'get_inv_constrained_params','initial_search'}
    selected=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    assert {n.name for n in selected}==names
    # Seed the existing stochastic algorithm explicitly; no candidate is chosen using scores.
    class SeededQMC:
        @staticmethod
        def LatinHypercube(d):return scipy.stats.qmc.LatinHypercube(d=d,seed=42)
    from types import SimpleNamespace
    scope=dict(torch=torch,scipy=SimpleNamespace(stats=SimpleNamespace(qmc=SeededQMC)),tqdm=lambda x:x)
    exec(compile(ast.Module(body=selected,type_ignores=[]),'reviewed_author_initial_functions','exec'),scope)
    return scope

def prep():
    started=time.perf_counter();_,utils,tensor,proc=modules()
    class Mapped(tensor.EMGData):
        def get_images(self,x):return provider_images(x)
    helper=Mapped.__new__(Mapped);helper.dataset='csl';helper.fs=2048;helper.Mrms=512;helper.remove_baseline='mean_square'
    # Match the author's segmented extractor without allocating all-session zero tensors.
    cache=OUT/'frames';cache.mkdir(exist_ok=True);manifest=[];hashes=set()
    prior=json.loads((ROOT/'research/runs/20260907_csl_sal_smoke_v1/archive_audit.json').read_text())
    filehashes={e['name']:e['sha256'] for e in prior['extracted']}
    for session in (1,2):
        directory=DATA/f'subject1/session{session}'
        assert sha(directory/'gest0.mat')==filehashes[f'subject1/session{session}/gest0.mat']
        baseline=helper.get_baseline(str(directory));np.save(OUT/f's{session}_baseline.npy',baseline)
        for label,gesture in enumerate(RAW_GESTURES):
            path=directory/f'gest{gesture}.mat';assert sha(path)==filehashes[f'subject1/session{session}/gest{gesture}.mat']
            cells=sio.loadmat(path)['gestures'];assert cells.shape==(10,1)
            for rep in range(10):
                raw=cells[rep,0];assert raw.shape==(192,6144) and np.isfinite(raw).all()
                h=hashlib.sha256(raw.tobytes()).hexdigest();assert h not in hashes;hashes.add(h)
                role='source' if session==1 else 'calibration' if rep==0 else 'score'
                e=dict(id=f's1_s{session}_g{gesture:02}_r{rep:02}',session=session,gesture=gesture,label=label,rep=rep,role=role,path=str(path),raw_sha256=h)
                # Role chosen before frame processing. No target score used for any fitted normalization.
                emg=raw.T;emg=emg-emg.mean(axis=0,keepdims=True);emg=proc.bandstop(proc.bandpass(emg,fs=2048),fs=2048)
                lo,hi=helper.segment(emg,baseline);assert 0<=lo<hi<=len(emg),(e['id'],lo,hi)
                rms=proc.get_rms_signal(emg,Mrms=512)
                # Literal `mean_square` flag intentionally leaves subtraction disabled, matching generator.
                x=torch.from_numpy(provider_images(rms)[lo:hi]).float()
                x=utils.median_pool_2d(x).numpy();assert np.isfinite(x).all()
                p=cache/(e['id']+'.npy');np.save(p,x)
                manifest.append(dict(**e,start=int(lo),end=int(hi),frames=len(x),cache_path=str(p),cache_sha256=sha(p)))
            print('Prepared',session,gesture,flush=True)
    dump(OUT/'manifest.json',manifest);dump(OUT/'preprocessing.json',dict(passed=True,trials=len(manifest),wall_seconds=time.perf_counter()-started,
        role_frames={k:sum(e['frames'] for e in manifest if e['role']==k) for k in ('source','calibration','score')}))

def role_data(manifest,role,gesture=None):
    entries=[e for e in manifest if e['role']==role and (gesture is None or e['gesture']==gesture)]
    return torch.from_numpy(np.concatenate([np.load(e['cache_path']) for e in entries])),torch.from_numpy(np.concatenate([np.full(e['frames'],e['label'],dtype=np.int64) for e in entries])),entries

def fit(model,x,y,epochs,lr,warmup_epochs,batch=128):
    torch.manual_seed(42);loader=DataLoader(TensorDataset(x,y),batch_size=batch,shuffle=True,generator=torch.Generator().manual_seed(42))
    opt=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=lr,weight_decay=0)
    warmup=torch.optim.lr_scheduler.LinearLR(opt,start_factor=.01 if warmup_epochs else 1,end_factor=1,total_iters=max(1,len(loader)*warmup_epochs))
    losses=[];started=time.perf_counter()
    for epoch in range(epochs):
        total=0
        for a,b in loader:
            opt.zero_grad();loss=nn.functional.cross_entropy(model(a),b);assert torch.isfinite(loss)
            loss.backward();opt.step();warmup.step();total+=float(loss.detach())*len(a)
        losses.append(total/len(x))
        if epoch%25==0 or epoch==epochs-1:print('Fit epoch',epoch+1,'loss',losses[-1],flush=True)
    return dict(losses=losses,steps=epochs*len(loader),wall_seconds=time.perf_counter()-started)

def pred(model,x):
    model.eval();a=[]
    with torch.no_grad():
        for i in range(0,len(x),1024):
            logits=model(x[i:i+1024]);assert torch.isfinite(logits).all();a.append(logits.argmax(1).numpy())
    return np.concatenate(a)

def metrics(p,y,entries,label=5):
    at=0;votes=[]
    for e in entries:
        votes.append(Counter(p[at:at+e['frames']].tolist()).most_common(1)[0][0]);at+=e['frames']
    assert at==len(p);truth=np.array([e['label'] for e in entries]);v=np.array(votes);unseen=truth!=label
    return dict(frame_accuracy=float(np.mean(p==np.asarray(y))),trial_accuracy=float(np.mean(v==truth)),unseen_trial_accuracy=float(np.mean(v[unseen]==truth[unseen])),trial_predictions=v.tolist(),trials=len(entries))

def save_model(model,name):
    torch.save(model.state_dict(),OUT/(name+'.pt'))
    dump(OUT/(name+'_state.json'),dict(adaptation_phase=model.adaptation_phase,corrective_gain=model.corrective_gain,
        transform=model.input_transform_name,mode=getattr(model.input_transform,'mode',None)))

def run():
    started=time.perf_counter();nets,_,_,_=modules();manifest=json.loads((OUT/'manifest.json').read_text())
    for e in manifest:assert sha(e['cache_path'])==e['cache_sha256']
    sx,sy,se=role_data(manifest,'source');cx,cy,ce=role_data(manifest,'calibration',21)
    assert len(se)==80 and len(ce)==1 and len(set(cy.tolist()))==1
    torch.manual_seed(42);base=make_model(nets);base.train()
    source_log=fit(base,sx,sy,15,.001,3);base.eval();save_model(base,'source')
    source_fit=metrics(pred(base,sx),sy,se);models={'frozen':base};logs={'source':source_log}
    funcs=load_initial_functions();prototype=torch.sqrt((cx**2).mean(dim=0,keepdim=True));plabel=cy[:1]
    for condition in ('gain_only','spatial_only','spatial_gain','classifier_fine','classifier_fine_gain'):
        is_spatial=condition.startswith('spatial');gain=condition in ('gain_only','spatial_gain','classifier_fine_gain')
        m=deepcopy(base) if is_spatial else make_model(nets,transform='fine-tuning')
        if not is_spatial:m.load_state_dict({n:v for n,v in base.state_dict().items() if not n.startswith('input_transform.')})
        m.eval()
        for p in m.parameters():p.requires_grad=False
        if gain:m.get_session_means(sx[sy==int(cy[0])],cx)
        m.adaptation_phase=True
        if is_spatial:
            for p in m.input_transform.parameters():p.requires_grad=True
            loader=DataLoader(TensorDataset(prototype,plabel),batch_size=1,shuffle=False)
            m.input_transform.mode='bicubic';begin=time.perf_counter()
            funcs['initial_search'](m,loader,torch.tensor([4.,4.,15/180,.1,.1,.1,.1]),{n:True for n in ('xshift','yshift','rot_theta','xscale','yscale','xshear','yshear')},H=7,W=24,npoints=16384)
            search_seconds=time.perf_counter()-begin;m.input_transform.mode='bilinear'
            logs[condition]=fit(m,prototype,plabel,500,.01,0,batch=1);logs[condition]['search_seconds']=search_seconds
            m.input_transform.mode='bicubic'
        elif condition.startswith('classifier'):
            for p in m.parameters():p.requires_grad=True
            m.train();logs[condition]=fit(m,cx,cy,150,.001,30)
        else:logs[condition]=dict(steps=0)
        m.eval()
        if is_spatial or condition=='gain_only':
            for n,v in base.state_dict().items():
                if n.startswith('input_transform.'):continue
                assert torch.equal(v,m.state_dict()[n]),(condition,n)
        save_model(m,condition);models[condition]=m
        logs[condition]['calibration_fit']=float(np.mean(pred(m,cx)==np.asarray(cy)))
        print('Completed adaptation',condition,flush=True)
    dump(OUT/'training.json',dict(source_fit=source_fit,logs=logs))
    tx,ty,te=role_data(manifest,'score');assert len(te)==72
    results={}
    for condition,m in models.items():
        p=pred(m,tx);np.savez_compressed(OUT/(condition+'_predictions.npz'),predictions=p,labels=np.asarray(ty),trial_ids=np.array([e['id'] for e in te]),lengths=np.array([e['frames'] for e in te]))
        results[condition]=metrics(p,ty,te);print(condition,results[condition],flush=True)
    dump(OUT/'results.json',dict(results=results,wall_seconds=time.perf_counter()-started))

def checks():
    nets,utils,tensor,proc=modules();torch.manual_seed(42)
    m=make_model(nets);m.eval();x=torch.rand(10,1,7,24);before=deepcopy(m.state_dict())
    # Original transform is bypassed before adaptation; prove identity no-gain control agrees.
    ident=make_model(nets,transform='fine-tuning');ident.load_state_dict({n:v for n,v in before.items() if not n.startswith('input_transform.')});ident.eval();ident.adaptation_phase=True
    with torch.no_grad():torch.testing.assert_close(m(x),ident(x),rtol=0,atol=0)
    src=torch.rand(4,1,7,24)+1;target=src/1.5;ident.get_session_means(src,target)
    with torch.no_grad():
        expected=ident.fc(ident.bn(x*1.5).reshape(10,-1));torch.testing.assert_close(ident(x),expected,rtol=1e-5,atol=1e-6)
    # Bounded transform inverse must recover an actual identity (raw xscale=1 is not identity under tanh).
    f=load_initial_functions();ps=f['get_inv_constrained_params'](*[torch.tensor(v) for v in (0,0,0,1,1,0,0)],boundaries=BOUNDS)
    with torch.no_grad():
        for p,v in zip(m.input_transform.parameters(),ps):p.copy_(v)
        mapped=m.input_transform(x);torch.testing.assert_close(mapped,x,rtol=2e-5,atol=2e-5)
    # Provider spatial mapping and no time reversal.
    a=np.arange(192*2).reshape(2,192);b=provider_images(a)
    np.testing.assert_array_equal(b[0,0],np.flipud(np.delete(a[0],np.s_[7:192:8]).reshape(24,7).T));assert np.all(b[1]-b[0]==192)
    dump(OUT/'component_checks.json',dict(passed=True,checks=['Source bypass and no-gain identity agree','Gain matches known1.5ratio','Bounded inverse recovers identity','Provider channel/time mapping']))

if __name__=='__main__':
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);warnings.filterwarnings('ignore',message='Default grid_sample and affine_grid behavior')
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['freeze','check','prepare','run']);args=ap.parse_args()
    if args.stage=='freeze':assert not (OUT/'config.json').exists();dump(OUT/'config.json',config());(OUT/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    elif args.stage=='check':checks()
    elif args.stage=='prepare':prep()
    else:run()
