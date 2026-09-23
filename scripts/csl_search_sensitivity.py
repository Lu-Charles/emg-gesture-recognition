"""Prespecified initialization-search seed sensitivity; primary fits unchanged."""
import argparse,json,time,warnings,multiprocessing
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import scipy.stats
import torch
from scripts import csl_author_recipe as r
from scripts import csl_component_study as c
from scripts import csl_search_batch as b
from scripts.verify_csl_author_recipe import restore
PRIMARY=r.ROOT/'research/runs/20260907_csl_components_confirmatory'
OUT=r.ROOT/'research/runs/20260907_csl_search_seed_sensitivity'

def candidate_seed(n,seed):
    a=2*torch.tensor(scipy.stats.qmc.LatinHypercube(d=7,seed=seed).random(n=n)).float()-1
    a[:,0]=8*a[:,0]/23;a[:,1]=8*a[:,1]/6;a[:,2]=(15/180)*a[:,2];a[:,3:5]=torch.pow(1+torch.abs(a[:,3:5])*.1,torch.sign(a[:,3:5]));a[:,5:]*=.1
    return torch.cat([a,torch.tensor([[0.,0.,0.,1.,1.,0.,0.]])])
def freeze():
    OUT.mkdir(exist_ok=True);assert not (OUT/'protocol.json').exists()
    # Demonstrate exact equality to primary generation when the same seed is requested.
    torch.testing.assert_close(candidate_seed(16384,42),b.candidates(),rtol=0,atol=0)
    cfg=dict(timestamp=r.datetime.now(r.timezone.utc).isoformat(),participants=[2,3,4,5],pair=[1,2],gestures=r.RAW_GESTURES,search_seeds=[43,44],reference_seed=42,methods=['spatial_only','spatial_gain'],unchanged='Reuse each primary source checkpoint and all data/splits/hyperparameters. Only Latin-hypercube seed changes; all seeds retained, no selection.',purpose='Assess whether the component result depends on one arbitrary spatial-search seed. Single predeterminedpair limits scope. Supplementary sensitivity, not independentparticipants.',frozen_before='No primary cohort score summaries inspected; training in progress. No dependence on any finalmetric.',script_sha256=r.sha(__file__),primary_protocol_sha256=r.sha(PRIMARY/'protocol.json'))
    r.dump(OUT/'protocol.json',cfg);(OUT/Path(__file__).name).write_bytes(Path(__file__).read_bytes());print(cfg,flush=True)
def worker(person):
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);warnings.filterwarnings('ignore',message='There is a performance drop')
    start=time.perf_counter();manifest=json.loads((PRIMARY/'manifest.json').read_text());pm=[e for e in manifest if e['participant']==person]
    sm=[dict(e,role='source') for e in pm if e['session']==1];sx,sy,_=r.role_data(sm,'source');base=restore('source',PRIMARY/f'subject{person}/source_s1');allrows=[]
    for seed in (43,44):
        b.candidates=lambda n=16384: candidate_seed(n,seed)
        dest=OUT/f'seed{seed}/subject{person}';dest.mkdir(parents=True,exist_ok=True)
        results=c.pair(pm,dest,base,sx,sy,1,2,methods=['spatial_only','spatial_gain']);allrows.extend([dict(participant=person,seed=seed,source_session=1,target_session=2,**v) for v in results])
    r.dump(OUT/f'subject{person}_results.json',dict(results=allrows,wall_seconds=time.perf_counter()-start));return allrows
def run():
    cfg=json.loads((OUT/'protocol.json').read_text());assert r.sha(__file__)==cfg['script_sha256'];assert r.sha(PRIMARY/'protocol.json')==cfg['primary_protocol_sha256'];start=time.perf_counter()
    with ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('spawn')) as pool:rows=sum(list(pool.map(worker,[2,3,4,5])),[])
    assert len(rows)==192;r.dump(OUT/'results.json',dict(results=rows,wall_seconds=time.perf_counter()-start));print('Search-seed sensitivity completed',flush=True)
if __name__=='__main__':
    torch.set_num_threads(1);ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['freeze','run']);a=ap.parse_args();{'freeze':freeze,'run':run}[a.stage]()
