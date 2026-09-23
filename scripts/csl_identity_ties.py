"""Prefer the existing identity candidate when it has exactly minimum calibration loss."""
import argparse,json,time,warnings,multiprocessing
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import torch
from scripts import csl_author_recipe as r
from scripts import csl_component_study as c
from scripts import csl_search_batch as b
from scripts.verify_csl_author_recipe import restore
PRIMARY=r.ROOT/'research/runs/20260907_csl_components_confirmatory'
OUT=r.ROOT/'research/runs/20260907_csl_identity_ties'

def search(model,prototype,labels):
    a=b.candidates();v=b.losses(model,prototype,labels,a);first=int(v.argmin());identity=len(a)-1
    best=identity if bool(v[identity]==v[first]) else first
    params=r.load_initial_functions()['get_inv_constrained_params'](*a[best],boundaries=r.BOUNDS)
    with torch.no_grad():
        for name,value in zip(b.NAMES,params):getattr(model.input_transform,name)[0].copy_(value)
    return dict(index=best,original_index=first,identity_preferred=best==identity and first!=identity,loss=float(v[best]),identity_loss=float(v[identity]),zero_loss_candidates=int((v==0).sum()),near_min_candidates=int((v<=v.min()+1e-5).sum()))

def check():
    OUT.mkdir(exist_ok=True);nets,*_=r.modules();m=r.make_model(nets);m.eval();m.adaptation_phase=True;m.input_transform.mode='bicubic'
    with torch.no_grad():m.fc.weight.zero_();m.fc.bias.zero_()
    torch.manual_seed(42);x=torch.rand(1,1,7,24);y=torch.tensor([0]);log=search(m,x,y)
    assert log['original_index']==0 and log['index']==16384 and log['identity_preferred'];torch.testing.assert_close(m.input_transform(x),x,rtol=2e-5,atol=2e-5)
    r.dump(OUT/'component_checks.json',dict(passed=True,synthetic_only=True,all_candidate_losses_equal=True,original_selects_random_candidate=0,control_selects_existing_identity=16384,no_threshold_or_new_candidate=True));print('Identity tie check passed',flush=True)

def evaluate(person,manifest,dest,source_sessions,target_sessions,loader):
    b.search=search;rows=[]
    for source in source_sessions:
        sx,sy,_=r.role_data([dict(e,role='source') for e in manifest if e['session']==source],'source');base=loader(source)
        for target in target_sessions:
            if source==target:continue
            local=c.pair(manifest,dest,base,sx,sy,source,target,methods=['spatial_only','spatial_gain'])
            rows.extend([dict(participant=person,source_session=source,target_session=target,**z) for z in local])
    return rows

def development():
    dest=OUT/'development';dest.mkdir(exist_ok=True);r.dump(dest/'config.json',dict(timestamp=r.datetime.now(r.timezone.utc).isoformat(),participant=1,pair=[1,2],all8choices=True,change='PreferidentityonlywhenitscomputedlossEXACTLYequalsminimum; samecandidatepoolandalltrainingupdates',script_sha256=r.sha(__file__)))
    manifest=json.loads((r.OUT/'manifest.json').read_text());t=time.perf_counter();rows=evaluate(1,manifest,dest,[1],[2],lambda _:restore('source'));r.dump(dest/'results.json',dict(results=rows,wall_seconds=time.perf_counter()-t));print({m:np.mean([z['metrics']['unseen_trial_accuracy'] for z in rows if z['method']==m]) for m in ['frozen','spatial_only','spatial_gain']},flush=True)

def freeze():
    assert not (OUT/'protocol.json').exists();assert json.loads((OUT/'component_checks.json').read_text())['passed']
    cfg=dict(timestamp=r.datetime.now(r.timezone.utc).isoformat(),participants=[2,3,4,5],sessions=[1,2,3,4,5],gestures=r.RAW_GESTURES,methods=['frozen','spatial_only','spatial_gain'],change='Onlyinitialcandidate tie-breaking: whenexistingidentitycandidatehasEXACT minimumfloat32calibrationloss, chooseit. Otherwisechooseoriginalfirstargmin. No tolerance, no newcandidate, no earlystop orchangedtraining.',purpose='Test the sourcecode identity-last/firstargmin behavior when numerous transforms tie. Minimal implementation intervention, not anewSALarchitecture orclaimthatidentityalwaysbest.',frozen_before='No primary cohortperformance summaries inspected; settings based onauthorcode anddevelopment only. Reportall80pairs×8choiceswithoutselection.',script_sha256=r.sha(__file__),primary_protocol_sha256=r.sha(PRIMARY/'protocol.json'))
    r.dump(OUT/'protocol.json',cfg);(OUT/Path(__file__).name).write_bytes(Path(__file__).read_bytes());print(cfg,flush=True)

def worker(person):
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);warnings.filterwarnings('ignore',message='There is a performance drop')
    manifest=json.loads((PRIMARY/'manifest.json').read_text());pm=[e for e in manifest if e['participant']==person];dest=OUT/f'subject{person}';dest.mkdir(exist_ok=True);t=time.perf_counter()
    rows=evaluate(person,pm,dest,range(1,6),range(1,6),lambda s:restore('source',PRIMARY/f'subject{person}/source_s{s}'));r.dump(dest/'results.json',dict(results=rows,wall_seconds=time.perf_counter()-t));return rows

def run():
    cfg=json.loads((OUT/'protocol.json').read_text());assert r.sha(__file__)==cfg['script_sha256'];assert r.sha(PRIMARY/'protocol.json')==cfg['primary_protocol_sha256'];t=time.perf_counter()
    with ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('spawn')) as pool:rows=sum(list(pool.map(worker,[2,3,4,5])),[])
    assert len(rows)==1920;r.dump(OUT/'results.json',dict(results=rows,wall_seconds=time.perf_counter()-t));print('All identity controls completed',flush=True)
if __name__=='__main__':
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);warnings.filterwarnings('ignore',message='There is a performance drop');ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['check','development','freeze','run']);a=ap.parse_args();{'check':check,'development':development,'freeze':freeze,'run':run}[a.stage]()
