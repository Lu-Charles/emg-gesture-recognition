"""Independent checkpoint replay and whole-trial verification for later author recipe."""
from scripts import csl_author_recipe as r
import json,time
import numpy as np
import torch

def restore(name,path=None):
    p=path or r.OUT;s=torch.load(p/(name+'.pt'),map_location='cpu',weights_only=True)
    flags=json.loads((p/(name+'_state.json')).read_text());nets,*_=r.modules()
    m=r.make_model(nets,flags['transform'])
    for key in ('mean_session1','mean_session2'):
        if key in s:m.register_buffer(key,torch.zeros_like(s[key]))
    m.load_state_dict(s);m.adaptation_phase=flags['adaptation_phase'];m.corrective_gain=flags['corrective_gain']
    if flags['mode']:m.input_transform.mode=flags['mode']
    return m.eval()

def main():
    t=time.perf_counter();manifest=json.loads((r.OUT/'manifest.json').read_text());assert len({e['id'] for e in manifest})==160
    assert len({e['raw_sha256'] for e in manifest})==160
    for e in manifest:assert r.sha(e['cache_path'])==e['cache_sha256']
    scored=[e for e in manifest if e['role']=='score'];assert len(scored)==72 and all(e['rep']>0 and e['session']==2 for e in scored)
    y=np.concatenate([np.repeat(e['label'],e['frames']) for e in scored]);total=0
    results=json.loads((r.OUT/'results.json').read_text())['results']
    for name,result in results.items():
        m=restore('source' if name=='frozen' else name);saved=np.load(r.OUT/(name+'_predictions.npz'));preds=[];votes=[]
        with torch.no_grad():
            for e in scored:
                x=torch.from_numpy(np.load(e['cache_path']));parts=[]
                for chunk in x.split(257):parts.append(m(chunk).argmax(1).numpy())
                p=np.concatenate(parts);preds.append(p)
                # Explicit first-occurrence tie convention, independent from training's Counter.
                counts=np.bincount(p,minlength=8);tie=set(np.flatnonzero(counts==counts.max()));votes.append(next(int(v) for v in p if v in tie))
        p=np.concatenate(preds);np.testing.assert_array_equal(p,saved['predictions']);np.testing.assert_array_equal(y,saved['labels'])
        assert saved['trial_ids'].tolist()==[e['id'] for e in scored];assert votes==result['trial_predictions'];total+=len(p)
        assert np.mean(np.array(votes)==[e['label'] for e in scored])==result['trial_accuracy']
    r.dump(r.OUT/'verification.json',dict(passed=True,models=6,predictions=total,cache_hashes=160,whole_trials=72,wall_seconds=time.perf_counter()-t))
    print('Verified',total,'predictions',flush=True)
if __name__=='__main__':torch.set_num_threads(1);main()
