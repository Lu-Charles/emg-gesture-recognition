"""Check SeNic neural input provenance, training ledgers and every saved prediction."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from src.senic_neural import SeNicCNN


def main():
    ap=argparse.ArgumentParser();ap.add_argument('run',type=Path);a=ap.parse_args();r=a.run;t=time.perf_counter()
    read=lambda f:json.loads((r/f).read_text())
    c=read('config.json');rows=read('trial_manifest.json');plans=read('allocations.json')
    metrics=read('metrics.json');access=read('model_access.json');training=read('training.json');s=read('summary.json')
    assert {x['subject'] for x in rows}=={0,1,14,18,24,25} and {x['session'] for x in rows}=={0}
    for f,digest in c['upstream_hashes'].items():assert hashlib.sha256((Path(c['classical'])/f).read_bytes()).hexdigest()==digest
    for f,digest in c['code_hashes'].items():assert hashlib.sha256((r/'code_snapshot'/Path(f).name).read_bytes()).hexdigest()==digest
    raw=np.load(r/'raw_active.npy',mmap_mode='r');cached=np.load(r/'windows.npy',mmap_mode='r')
    win=np.ascontiguousarray(cached.transpose(0,1,3,2)).transpose(0,1,3,2)
    assert raw.shape==(1386,400,8) and win.shape==(1386,15,8,50)
    for i,row in enumerate(rows):
        assert hashlib.sha256(Path(row['path']).read_bytes()).hexdigest()==row['sha256']
        x=np.loadtxt(row['path'],delimiter=',')
        np.testing.assert_array_equal(raw[i],x[600:1000])
        expected=np.stack([x[j:j+50].T for j in range(600,951,25)]).astype(np.float32)
        np.testing.assert_array_equal(win[i],expected)
    ledgers={x['model']:x for x in access};histories={x['model']:x for x in training}
    planmap={(p['subject'],p['position'],p['trials']):p for p in plans}
    assert len(access)==738 and len(metrics)==918 and len(plans)==186
    models={};fitacc={}
    for name,ledger in ledgers.items():
        ids=ledger['fit_ids'];norm=ledger['normalization_ids'];source=ledger['source_history']
        assert all(rows[i]['repetition'] in (0,1) and rows[i]['subject']==ledger['subject'] for i in set(ids+norm+source))
        expected=float(np.sqrt(np.square(raw[norm]).sum()/raw[norm].size));expected=max(expected,1e-8)
        np.testing.assert_allclose(ledger['scale'],expected,atol=1e-12)
        method=ledger['method']
        if method=='enrollment':
            assert len(ids)==14 and ids==norm and not source and ledger['parent_model'] is None
            assert all(rows[i]['position']==0 for i in ids)
        else:
            p=planmap[ledger['subject'],ledger['position'],ledger['trials']]
            assert not (set(ids+norm+source)&set(p['score']))
            if method=='scratch_target':
                assert ids==p['calibration'] and norm==ids and not source and ledger['parent_model'] is None
            else:
                assert source==p['source'] and norm==source
                assert ids==(source+p['calibration'] if method=='fine_pooled' else p['calibration'])
                parent=ledgers[ledger['parent_model']]
                assert parent['subject']==ledger['subject'] and parent['family']==ledger['family'] and parent['method']=='enrollment'
        assert ledger['steps']==(300 if method in ('enrollment','scratch_target') else 150)
        losses=histories[name]['losses'];assert len(losses)==ledger['steps'] and np.isfinite(losses).all()
        model=SeNicCNN();model.load_state_dict(torch.load(r/'models'/(name+'.pt'),map_location='cpu',weights_only=True));model.eval();models[name]=model
        x=(win[ids].reshape(-1,8,50)/ledger['scale']).astype(np.float32)
        truth=np.repeat([rows[i]['label'] for i in ids],15)
        with torch.no_grad():pred=model(torch.from_numpy(x)).argmax(1).numpy()
        fitacc[name]=float(np.mean(pred==truth));np.testing.assert_allclose(fitacc[name],histories[name]['training_accuracy'])
    for m in metrics:
        p=planmap[m['subject'],m['position'],m['trials']];ledger=ledgers[m['model']]
        assert ledger['family']==m['family'] and ledger['subject']==m['subject']
        assert (m['method']=='frozen' and ledger['method']=='enrollment') or ledger['method']==m['method']
        pp=np.load(r/'predictions'/m['prediction_file']);assert list(pp['score_ids'])==p['score']
        truth=np.repeat([rows[i]['label'] for i in p['score']],15);np.testing.assert_array_equal(truth,pp['y'])
        x=(win[p['score']].reshape(-1,8,50)/ledger['scale']).astype(np.float32)
        with torch.no_grad():prob=models[m['model']](torch.from_numpy(x)).softmax(1).numpy()
        np.testing.assert_allclose(prob,pp['probabilities'],rtol=1e-6,atol=1e-7)
        np.testing.assert_array_equal(prob.argmax(1),pp['pred'])
        cm=np.zeros((7,7),int);np.add.at(cm,(truth,pp['pred']),1)
        acc=np.trace(cm)/cm.sum();den=cm.sum(0)+cm.sum(1)
        f1=np.mean(np.divide(2*np.diag(cm),den,out=np.zeros(7),where=den>0))
        votes=np.array([np.bincount(z,minlength=7).argmax() for z in pp['pred'].reshape(7,15)])
        np.testing.assert_allclose([m['accuracy'],m['macro_f1'],m['trial_accuracy']],
            [acc,f1,np.mean(votes==[rows[i]['label'] for i in p['score']])])
        np.testing.assert_allclose(m['recorded_seconds'],sum(rows[i]['samples']/200 for i in p['calibration']))
        assert m['active_analysis_seconds']==2*len(p['calibration'])
    for g in s['groups']:
        means=[]
        for p in g['participants']:
            mm=[x for x in metrics if x['position']>0 and x['subject']==p['subject'] and all(x[k]==g[k] for k in ('family','method','trials'))]
            values=[np.mean([x[k] for x in mm]) for k in ('accuracy','macro_f1','trial_accuracy','recorded_seconds')]
            np.testing.assert_allclose(values,[p[k] for k in ('accuracy','macro_f1','trial_accuracy','recorded_seconds')]);means.append(values)
        np.testing.assert_allclose(np.mean(means,axis=0),[g[k] for k in ('accuracy','macro_f1','trial_accuracy','recorded_seconds')])
    result=dict(passed=True,raw_trials=len(rows),allocations=len(plans),models=len(access),prediction_files=len(metrics),
        prediction_rows=len(metrics)*105,groups=len(s['groups']),training_accuracy_rechecked=len(fitacc),reserved_signal_access=0,wall_seconds=time.perf_counter()-t)
    (r/'independent_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):main()
