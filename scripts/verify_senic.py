"""Independent ledger, normalization, saved prediction and aggregate verification."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import joblib
import numpy as np
from threadpoolctl import threadpool_limits


def main():
    ap=argparse.ArgumentParser();ap.add_argument('run',type=Path);a=ap.parse_args();r=a.run
    start=time.perf_counter()
    read=lambda name:json.loads((r/name).read_text())
    config=read('config.json');rows=read('trial_manifest.json');plans=read('allocations.json')
    metrics=read('metrics.json');summary=read('summary.json');access=read('model_access.json')
    assert len(rows)==1386 and len(plans)==606
    assert {x['subject'] for x in rows}=={0,1,14,18,24,25}
    assert {x['session'] for x in rows}=={0}
    assert {x['windows'] for x in rows}=={15}
    assert config['analysis_samples']==[600,1000]
    hashes={}
    for row in rows:
        sha=hashlib.sha256(Path(row['path']).read_bytes()).hexdigest()
        assert sha==row['sha256']
        assert sha not in hashes, f'Duplicate raw content: {row["path"]}'
        hashes[sha]=row['id']
    for path,sha in config['code_hashes'].items():
        assert hashlib.sha256((r/'code_snapshot'/Path(path).name).read_bytes()).hexdigest()==sha
    features={k:np.load(r/(k+'_features.npy'),mmap_mode='r') for k in ('amplitude','tdar_rms')}
    ytrial=np.array([x['label'] for x in rows]);score_sets={}
    for plan in plans:
        fit,cal,test=map(set,[plan['source'],plan['calibration'],plan['score']])
        assert not(fit&cal or fit&test or cal&test)
        assert len(cal)==plan['trials'] and len(test)==7 and len(fit)==14
        assert {rows[i]['label'] for i in test}==set(range(7))
        assert all(rows[i]['repetition']==2 for i in test)
        assert all(rows[i]['repetition'] in (0,1) for i in fit|cal)
        assert all(rows[i]['position']==0 for i in fit)
        assert all(rows[i]['position']==plan['position'] for i in cal|test)
        assert all(rows[i]['subject']==plan['subject'] for i in fit|cal|test)
        key=(plan['subject'],plan['position'])
        assert key not in score_sets or score_sets[key]==test
        score_sets[key]=test
    # Check raw feature indexing and feature group scale by a separate manual calculation.
    for i in range(0,len(rows),97):
        x=np.loadtxt(rows[i]['path'],delimiter=',')
        ww=np.stack([x[s:s+50] for s in range(600,951,25)])
        expected=np.concatenate([abs(ww).mean(1),np.sqrt((ww**2).mean(1)),abs(np.diff(ww,axis=1)).mean(1)],axis=1)
        np.testing.assert_allclose(features['amplitude'][i],expected,rtol=1e-12,atol=1e-12)
    loaded={};ledger={x['model']:x for x in access}
    for name,item in ledger.items():
        model=joblib.load(r/'models'/(name+'.joblib'))
        f=features[item['feature']][item['fit_ids']].reshape(-1,features[item['feature']].shape[-1])
        np.testing.assert_allclose(model[0].mean_,f.mean(0),rtol=1e-11,atol=1e-11)
        np.testing.assert_allclose(model[0].var_,f.var(0),rtol=1e-10,atol=1e-10)
        np.testing.assert_allclose(model[1].priors_,np.ones(7)/7)
        loaded[name]=model
    for m in metrics:
        p=plans[m['allocation']];kind=m['feature'];item=ledger[m['model']]
        allowed=p['calibration'] if m['method']=='target_only' else p['source']+p['calibration'] if m['method']=='source_plus_target' else p['source']
        assert set(allowed)==set(item['fit_ids']) and kind==item['feature']
        np.testing.assert_allclose(m['recorded_seconds'],sum(rows[i]['samples']/200 for i in p['calibration']),atol=1e-12)
        assert m['active_analysis_seconds']==2*len(p['calibration'])
        pred=np.load(r/'predictions'/m['prediction_file'])
        assert list(pred['score_ids'])==p['score']
        truth=np.repeat(ytrial[p['score']],15);assert np.array_equal(pred['y'],truth)
        x=features[kind][p['score']].reshape(-1,features[kind].shape[-1])
        if m['method']=='spatial_registration':
            # Recompute the template-search result without the implementation helper.
            classes=sorted(set(ytrial[p['calibration']]));sa=[];ta=[]
            for g in classes:
                sa.append(features[kind][[i for i in p['source'] if ytrial[i]==g],:,8:16].mean((0,1)))
                ta.append(features[kind][[i for i in p['calibration'] if ytrial[i]==g],:,8:16].mean((0,1)))
            sa=np.array(sa);ta=np.array(ta);sa/=np.maximum(np.sqrt((sa*sa).sum(1))[:,None],1e-12)
            shifts=sorted(np.arange(-4,4,.125),key=lambda z:(abs(z),z));cost=[]
            def rotate(v,shift):
                z=v.reshape(-1,v.shape[-1]//8,8);idx=(np.arange(8)-shift)%8
                lo=np.floor(idx).astype(int);frac=idx-lo
                return ((1-frac)*z[:,:,lo]+frac*z[:,:,(lo+1)%8]).reshape(v.shape)
            for shift in shifts:
                z=rotate(ta,shift);z/=np.maximum(np.sqrt((z*z).sum(1))[:,None],1e-12)
                cost.append(np.square(sa-z).sum(1).mean())
            shift=shifts[np.argmin(cost)];assert shift==m['shift_channels']
            x=rotate(x,shift)
        model=loaded[m['model']]
        np.testing.assert_array_equal(model.predict(x),pred['pred'])
        np.testing.assert_allclose(model.predict_proba(x),pred['probabilities'],rtol=1e-10,atol=1e-10)
        cm=np.zeros((7,7),int);np.add.at(cm,(truth,pred['pred']),1)
        np.testing.assert_allclose(m['accuracy'],np.trace(cm)/cm.sum())
        f1=np.mean(np.divide(2*np.diag(cm),cm.sum(0)+cm.sum(1),out=np.zeros(7),where=cm.sum(0)+cm.sum(1)>0))
        np.testing.assert_allclose(m['macro_f1'],f1)
        votes=np.array([np.bincount(z,minlength=7).argmax() for z in pred['pred'].reshape(7,15)])
        np.testing.assert_array_equal(votes,pred['trial_pred'])
        np.testing.assert_allclose(m['trial_accuracy'],np.mean(votes==ytrial[p['score']]))
    for group in summary['groups']:
        vals=[]
        for p in config['development']:
            rr=[m for m in metrics if m['position']>0 and m['subject']==p and all(m[k]==group[k] for k in ['feature','method','trials'])]
            vals.append([np.mean([m[k] for m in rr]) for k in ['accuracy','macro_f1','trial_accuracy','recorded_seconds']])
        np.testing.assert_allclose(np.mean(vals,0),[group[k] for k in ['accuracy','macro_f1','trial_accuracy','recorded_seconds']])
    result=dict(passed=True,raw_trials=len(rows),allocations=len(plans),models=len(loaded),
                prediction_files=len(metrics),prediction_rows=sum(len(np.load(r/'predictions'/m['prediction_file'])['pred']) for m in metrics),
                summary_groups=len(summary['groups']),reserved_signal_access=0,wall_seconds=time.perf_counter()-start)
    (r/'independent_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
