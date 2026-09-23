"""Fixed component study using the audited author recipe; no score-driven selection."""
import argparse,contextlib,io,json,time,warnings
from pathlib import Path
from copy import deepcopy
import numpy as np
import torch
from scripts import csl_author_recipe as r
from scripts import csl_search_batch as b
from scripts.verify_csl_author_recipe import restore
METHODS=['gain_only','spatial_only','spatial_gain','classifier_fine','classifier_fine_gain']

def save(m,path):
    torch.save(m.state_dict(),path.with_suffix('.pt'))
    r.dump(path.parent/(path.name+'_state.json'),dict(adaptation_phase=m.adaptation_phase,corrective_gain=m.corrective_gain,transform=m.input_transform_name,mode=getattr(m.input_transform,'mode',None)))

def adapt(base,sx,sy,cx,cy,method):
    nets,*_=r.modules();spatial=method.startswith('spatial');gain=method in ('gain_only','spatial_gain','classifier_fine_gain')
    m=deepcopy(base) if spatial else r.make_model(nets,'fine-tuning')
    if not spatial:m.load_state_dict({k:v for k,v in base.state_dict().items() if not k.startswith('input_transform.')})
    m.eval();m.adaptation_phase=True
    for p in m.parameters():p.requires_grad=False
    classes=torch.unique(cy)
    if gain:m.get_session_means(sx[torch.isin(sy,classes)],cx)
    log={};started=time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        if spatial:
            for p in m.input_transform.parameters():p.requires_grad=True
            proto=torch.stack([cx[cy==c].square().mean(0).sqrt() for c in classes]);m.input_transform.mode='bicubic'
            log['search']=b.search(m,proto,classes);m.input_transform.mode='bilinear'
            log['fit']=r.fit(m,proto,classes,500,.01,0,batch=len(classes));m.input_transform.mode='bicubic'
        elif method.startswith('classifier'):
            for p in m.parameters():p.requires_grad=True
            m.train();log['fit']=r.fit(m,cx,cy,150,.001,30)
    m.eval();log['wall_seconds']=time.perf_counter()-started
    log['calibration_frame_accuracy']=float(np.mean(r.pred(m,cx)==np.asarray(cy)))
    if spatial or method=='gain_only':
        for k,v in base.state_dict().items():
            if not k.startswith('input_transform.'):assert torch.equal(v,m.state_dict()[k]),(method,k)
    return m,log

def pair(manifest,out,base,sx,sy,source_session,target_session,methods=METHODS):
    target=[dict(e,role='calibration' if e['rep']==0 else 'score') for e in manifest if e['session']==target_session]
    tx,ty,te=r.role_data(target,'score');assert set(ty.tolist())==set(range(8)) and all(e['rep']>0 for e in te)
    fp=r.pred(base,tx);results=[]
    for gesture in r.RAW_GESTURES:
        folder=out/f's{source_session}_to_s{target_session}/g{gesture:02}';folder.mkdir(parents=True,exist_ok=True)
        if (folder/'results.json').exists():results.extend(json.loads((folder/'results.json').read_text()));continue
        cx,cy,ce=r.role_data(target,'calibration',gesture);assert len(ce)==1
        local=[dict(method='frozen',gesture=gesture,metrics=r.metrics(fp,ty,te,int(cy[0])),calibration_frame_accuracy=float(np.mean(r.pred(base,cx)==np.asarray(cy))))]
        predictions={'frozen':fp}
        for method in methods:
            m,log=adapt(base,sx,sy,cx,cy,method);save(m,folder/method)
            p=r.pred(m,tx);predictions[method]=p;local.append(dict(method=method,gesture=gesture,metrics=r.metrics(p,ty,te,int(cy[0])),**log))
        np.savez_compressed(folder/'predictions.npz',**predictions,labels=np.asarray(ty),trial_ids=np.array([e['id'] for e in te]),lengths=np.array([e['frames'] for e in te]))
        r.dump(folder/'results.json',local);results.extend(local);print('Completed',out.name,source_session,target_session,gesture,flush=True)
    return results

def development():
    out=r.ROOT/'research/runs/20260907_csl_components_development';out.mkdir(exist_ok=True)
    config=dict(timestamp=r.datetime.now(r.timezone.utc).isoformat(),participant=1,pairs=[[1,2]],gestures=r.RAW_GESTURES,methods=['frozen']+METHODS,source='reuse author_recipe_v1 checkpoint',calibration_rep=0,score_reps=list(range(1,10)),seed=42,recipe='author_recipe_v1; batch candidate search checked against scalar; no tuning or gesture selection',script_sha256=r.sha(__file__),search_sha256=r.sha(b.__file__))
    if not (out/'config.json').exists():r.dump(out/'config.json',config)
    manifest=json.loads((r.OUT/'manifest.json').read_text());sx,sy,_=r.role_data(manifest,'source');base=restore('source')
    t=time.perf_counter();results=pair(manifest,out,base,sx,sy,1,2);r.dump(out/'results.json',dict(results=results,wall_seconds=time.perf_counter()-t))
    print({m:round(np.mean([z['metrics']['trial_accuracy'] for z in results if z['method']==m])*100,3) for m in ['frozen']+METHODS},flush=True)
if __name__=='__main__':
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);warnings.filterwarnings('ignore',message='There is a performance drop')
    development()
