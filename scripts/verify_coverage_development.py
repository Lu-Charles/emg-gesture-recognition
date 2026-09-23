"""Verify saved trial metrics, fixed splits, and raw-to-feature spot checks.

This is not an independent replay of all fitted LDA predictions.
"""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'research/runs/20260909_coverage_development_v1'
CACHE=ROOT/'data/public/grabmyo/cache_20260906_v1'


def main():
    result=json.loads((RUN/'results.json').read_text())
    cfg=result['config'];rows=list(csv.DictReader((CACHE/'development_manifest.csv').open()))
    access=json.loads((RUN/'access_manifest.json').read_text())
    signals=np.load(CACHE/'development_signals.npy',mmap_mode='r')
    features=np.load(CACHE/'development_features.npy',mmap_mode='r')
    index={r['record']:i for i,r in enumerate(rows)}
    assert not set(cfg['participants']) & set(cfg['reserved_participants'])
    assert set(int(r['participant']) for r in rows) == set(cfg['participants'])
    checked=[]
    for a in access:
        source=set(a['source_records']);cal=set(a['calibration_records'].values());score=set(a['scoring_records'])
        assert not source&cal and not source&score and not cal&score
        assert len(source)==119 and len(cal)==16 and len(score)==68
        for record in source|cal|score:
            row=rows[index[record]]
            assert int(row['participant'])==a['participant'] and row['group']=='development'
        for record in [a['calibration_records']['0'],a['scoring_records'][0]]:
            i=index[record]
            # Independent formula, cast to float64 as the cache builder does.
            raw=np.asarray(signals[i],dtype=np.float64)
            windows=np.stack([raw[:,s:s+512] for s in range(1024,10240-512+1,256)])
            manual=np.concatenate((np.abs(windows).mean(2),np.sqrt(np.square(windows).mean(2)),np.abs(np.diff(windows,axis=2)).mean(2)),axis=1)
            np.testing.assert_allclose(manual,features[i],rtol=2e-6,atol=1e-9)
            checked.append(record)
    frozen={};metrics=0
    with np.load(RUN/'trial_predictions.npz') as saved:
        assert len(saved.files)==len(result['results'])==5120
        for r in result['results']:
            a=next(a for a in access if a['participant']==r['participant'] and a['session']==r['session'])
            truth=np.array([int(rows[index[t]]['class_index']) for t in a['scoring_records']])
            assert np.all(np.bincount(truth,minlength=17)==4)
            key=f"p{r['participant']}_s{r['session']}_o{r['order']}_k{r['k']}_{r['method']}"
            pred=saved[key];assert pred.shape==(68,) and set(pred)<=set(range(17))
            np.testing.assert_allclose(np.mean(pred==truth),r['accuracy'],atol=1e-14)
            np.testing.assert_allclose(f1_score(truth,pred,labels=range(17),average='macro',zero_division=0),r['macro_f1'],atol=1e-14)
            pair=(r['participant'],r['session'])
            if r['method']=='frozen':
                if pair in frozen:np.testing.assert_array_equal(pred,frozen[pair])
                frozen[pair]=pred.copy()
            metrics+=1
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [CACHE/'development_features.npy',CACHE/'development_manifest.csv',RUN/'protocol.json',RUN/'results.json',RUN/'trial_predictions.npz',Path(__file__)]}
    report=dict(passed=True,metric_records=metrics,trial_predictions=metrics*68,raw_feature_spot_checks=checked,hashes=hashes,
        scope='Saved-vote metrics, trial access, fixed scoring, raw-feature spot checks; not full independent fitted-model replay')
    (RUN/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    (RUN/'verify_coverage_development.py').write_text(Path(__file__).read_text())
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
