"""Independent replay of both supplementary spatial-search seeds."""
import json,time,multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import torch
from scripts import csl_author_recipe as r
from scripts.verify_csl_author_recipe import restore
from scripts.verify_csl_components import predictions,scores
PRIMARY=r.ROOT/'research/runs/20260907_csl_components_confirmatory'
OUT=r.ROOT/'research/runs/20260907_csl_search_seed_sensitivity'

def worker(person):
    torch.set_num_threads(1);manifest=json.loads((PRIMARY/'manifest.json').read_text());pm=[e for e in manifest if e['participant']==person];te=[e for e in pm if e['session']==2 and e['rep']>0];x=torch.from_numpy(np.concatenate([np.load(e['cache_path']) for e in te]));base=restore('source',PRIMARY/f'subject{person}/source_s1');n=0
    for seed in (43,44):
        for gesture in r.RAW_GESTURES:
            folder=OUT/f'seed{seed}/subject{person}/s1_to_s2/g{gesture:02}';a=np.load(folder/'predictions.npz');assert a['trial_ids'].tolist()==[e['id'] for e in te];label=r.RAW_GESTURES.index(gesture)
            ce=next(e for e in pm if e['session']==2 and e['rep']==0 and e['gesture']==gesture);cx=torch.from_numpy(np.load(ce['cache_path']))
            for z in json.loads((folder/'results.json').read_text()):
                method=z['method'];m=base if method=='frozen' else restore(method,folder);p=predictions(m,x);np.testing.assert_array_equal(p,a[method]);v,acc,unseen=scores(p,te,label)
                assert v.tolist()==z['metrics']['trial_predictions'] and acc==z['metrics']['trial_accuracy'] and unseen==z['metrics']['unseen_trial_accuracy'];assert float(np.mean(predictions(m,cx)==label))==z['calibration_frame_accuracy'];n+=len(p)
                if method!='frozen':
                    for key,value in base.state_dict().items():
                        if not key.startswith('input_transform.'):assert torch.equal(value,m.state_dict()[key])
    return dict(participant=person,cases=16,conditions=48,predictions=n,passed=True)
def main():
    start=time.perf_counter()
    with ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('spawn')) as pool:reports=list(pool.map(worker,[2,3,4,5]))
    r.dump(OUT/'verification.json',dict(passed=all(z['passed'] for z in reports),cases=64,conditions=192,predictions=sum(z['predictions'] for z in reports),participant_reports=reports,wall_seconds=time.perf_counter()-start));print('Both supplementary seeds verified',flush=True)
if __name__=='__main__':main()
