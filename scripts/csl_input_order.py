"""Audit control: resample raw amplitudes, then apply the gain in aligned coordinates."""
import argparse,json,time,warnings,types,contextlib,io,multiprocessing
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import numpy as np
import torch
from scripts import csl_author_recipe as r
from scripts import csl_component_study as c
from scripts import csl_search_batch as b
from scripts.verify_csl_author_recipe import restore
PRIMARY=r.ROOT/'research/runs/20260907_csl_components_confirmatory'
OUT=r.ROOT/'research/runs/20260907_csl_input_order'

def aligned_forward(self,x):
    if self.adaptation_phase:
        x=self.input_transform(x)
        if self.corrective_gain:
            gain=torch.clamp(self.mean_session1/(self.input_transform(self.mean_session2)+1e-12),.5,2.)
            x=x*gain
    x=self.bn(x);x=self.input_dropout(x)
    return self.fc(x.reshape(x.shape[0],-1))
def attach(m):
    assert torch.count_nonzero(m.baseline)==0
    m.forward=types.MethodType(aligned_forward,m);return m

def check():
    OUT.mkdir(exist_ok=True);nets,*_=r.modules();m=r.make_model(nets);m.eval();a=attach(deepcopy(m));torch.manual_seed(42);x=torch.rand(4,1,7,24)+1
    torch.testing.assert_close(m(x),a(x),rtol=0,atol=0)
    ident=r.make_model(nets,'fine-tuning');ident.load_state_dict({k:v for k,v in m.state_dict().items() if not k.startswith('input_transform.')});ident.eval();ident.adaptation_phase=True
    corrected=attach(deepcopy(ident));src=x*1.4;ident.get_session_means(src,x);corrected.get_session_means(src,x)
    torch.testing.assert_close(ident(x),corrected(x),rtol=0,atol=0)
    # Synthetic positive amplitudes with an EXACT known resampling operator and spatially varying gain.
    xx=torch.linspace(0,1,24)[None,None,None,:];v=(2+xx).expand(1,1,7,24).clone()
    theta=torch.tensor([[[1.,0.,4/24],[0.,1.,0.]]]);grid=torch.nn.functional.affine_grid(theta,v.shape,align_corners=False)
    warp=lambda z:torch.nn.functional.grid_sample(z,grid,mode='bilinear',align_corners=False)
    desired_gain=(1+.4*torch.sin(2*torch.pi*xx)).expand_as(v);u=warp(v)*desired_gain
    fitted=torch.clamp(u/(warp(v)+1e-12),.5,2.)
    old=warp(v*fitted);new=warp(v)*fitted;interior=(slice(None),slice(None),slice(1,6),slice(2,20))
    old_error=float((old[interior]-u[interior]).abs().mean());new_error=float((new[interior]-u[interior]).abs().max())
    assert old_error>.05 and new_error<1e-6
    r.dump(OUT/'component_checks.json',dict(passed=True,synthetic_only=True,source_bypass_exact=True,identity_gain_exact=True,known_transform_gain_before_mean_absolute_error=old_error,gain_after_max_absolute_error=new_error,interpretation='Coordinate-order algebra check, not an EMG performance result. Also moves resampling beforeBN; the real-data control assesses bothorderingdifferences.'))
    print('Order checks',old_error,new_error,flush=True)

def case(base,sx,sy,cx,cy,method):
    corrected=attach(deepcopy(base));return c.adapt(corrected,sx,sy,cx,cy,method)

def evaluate(person,manifest,dest,source_sessions,target_sessions,base_restore):
    rows=[]
    for source in source_sessions:
        sx,sy,_=r.role_data([dict(e,role='source') for e in manifest if e['session']==source],'source');base=base_restore(source)
        for target in target_sessions:
            if source==target:continue
            tm=[dict(e,role='calibration' if e['rep']==0 else 'score') for e in manifest if e['session']==target];tx,ty,te=r.role_data(tm,'score')
            for gesture in r.RAW_GESTURES:
                folder=dest/f's{source}_to_s{target}/g{gesture:02}';folder.mkdir(parents=True,exist_ok=True)
                if (folder/'results.json').exists():rows.extend(json.loads((folder/'results.json').read_text()));continue
                cx,cy,_=r.role_data(tm,'calibration',gesture);local=[];preds={}
                for method in ('spatial_only','spatial_gain'):
                    m,log=case(base,sx,sy,cx,cy,method);c.save(m,folder/method)
                    flags=json.loads((folder/(method+'_state.json')).read_text());flags['forward_variant']='resample_then_gain_then_bn';r.dump(folder/(method+'_state.json'),flags)
                    p=r.pred(m,tx);preds[method]=p;local.append(dict(participant=person,source_session=source,target_session=target,gesture=gesture,method=method,metrics=r.metrics(p,ty,te,int(cy[0])),**log))
                np.savez_compressed(folder/'predictions.npz',**preds,labels=np.asarray(ty),trial_ids=np.array([e['id'] for e in te]),lengths=np.array([e['frames'] for e in te]));r.dump(folder/'results.json',local);rows.extend(local)
            print('Order control completed pair',person,source,target,flush=True)
    return rows

def development():
    dest=OUT/'development';dest.mkdir(exist_ok=True);r.dump(dest/'config.json',dict(timestamp=r.datetime.now(r.timezone.utc).isoformat(),pair=[1,2],participant=1,gestures=r.RAW_GESTURES,change='resample amplitudes beforeBN; apply gain afterresampling. Allotherrecipecontrolsunchanged',script_sha256=r.sha(__file__)))
    manifest=json.loads((r.OUT/'manifest.json').read_text());t=time.perf_counter();rows=evaluate(1,manifest,dest,[1],[2],lambda _:restore('source'));r.dump(dest/'results.json',dict(results=rows,wall_seconds=time.perf_counter()-t))
    print({m:np.mean([z['metrics']['unseen_trial_accuracy'] for z in rows if z['method']==m]) for m in ('spatial_only','spatial_gain')},flush=True)

def freeze():
    assert json.loads((OUT/'component_checks.json').read_text())['passed'];assert not (OUT/'protocol.json').exists()
    cfg=dict(timestamp=r.datetime.now(r.timezone.utc).isoformat(),participants=[2,3,4,5],source_sessions=[1,2,3,4,5],target_sessions=[1,2,3,4,5],gestures=r.RAW_GESTURES,methods=['spatial_only','spatial_gain'],change='Warp rawamplitudes beforeBN, applygain afterwardin alignedcoordinates. Sameclassifiercheckpoints, splits,bounds,searchseed42,16384candidates,500updates,learningrate andinterpolationmodes asprimary.',purpose='Implementation-order sensitivity, motivated by a synthetic reconstruction check and source audit. Does not establish a newregistrationalgorithm; compares two orderingchanges, notgainorderalone.',frozen_before='No primarycohortperformance summaries inspected. No method/hyperparameter selectedfromfinalscores. All80pairsandall8calchoices retained.',script_sha256=r.sha(__file__),primary_protocol_sha256=r.sha(PRIMARY/'protocol.json'))
    r.dump(OUT/'protocol.json',cfg);(OUT/Path(__file__).name).write_bytes(Path(__file__).read_bytes());print(cfg,flush=True)
def worker(person):
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);warnings.filterwarnings('ignore',message='There is a performance drop');manifest=json.loads((PRIMARY/'manifest.json').read_text());pm=[e for e in manifest if e['participant']==person];dest=OUT/f'subject{person}';dest.mkdir(exist_ok=True)
    t=time.perf_counter();rows=evaluate(person,pm,dest,range(1,6),range(1,6),lambda s:restore('source',PRIMARY/f'subject{person}/source_s{s}'));r.dump(dest/'results.json',dict(results=rows,wall_seconds=time.perf_counter()-t));return rows

def run():
    cfg=json.loads((OUT/'protocol.json').read_text());assert r.sha(__file__)==cfg['script_sha256'];assert r.sha(PRIMARY/'protocol.json')==cfg['primary_protocol_sha256'];t=time.perf_counter()
    with ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('spawn')) as pool:rows=sum(list(pool.map(worker,[2,3,4,5])),[])
    assert len(rows)==1280;r.dump(OUT/'results.json',dict(results=rows,wall_seconds=time.perf_counter()-t));print('All order controls complete',flush=True)
if __name__=='__main__':
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);warnings.filterwarnings('ignore',message='There is a performance drop')
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['check','development','freeze','run']);a=ap.parse_args();{'check':check,'development':development,'freeze':freeze,'run':run}[a.stage]()
