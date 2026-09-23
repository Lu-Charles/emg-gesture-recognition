"""Independently verify the extended plain-CNN optimization check."""
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
    read=lambda f:json.loads((r/f).read_text());c=read('config.json');s=read('summary.json')
    origin=Path(c['neural']);rows=json.loads((origin/'trial_manifest.json').read_text());raw=np.load(origin/'raw_active.npy')
    cached=np.load(origin/'windows.npy',mmap_mode='r');win=np.ascontiguousarray(cached.transpose(0,1,3,2)).transpose(0,1,3,2)
    old={x['model']:x for x in json.loads((origin/'model_access.json').read_text())};metrics=read('metrics.json');training={x['model']:x for x in read('training.json')}
    assert hashlib.sha256((origin/'config.json').read_bytes()).hexdigest()==c['original_config_sha256']
    assert hashlib.sha256((r/'code_snapshot/check_senic_convergence.py').read_bytes()).hexdigest()==c['code_sha256']
    assert s['models']==360 and s['original_prefixes_equal'] and len(metrics)==360
    original_predictions={x['model']:x['prediction_file'] for x in json.loads((origin/'metrics.json').read_text()) if x['trials']>0}
    for m in metrics:
        name=m['model'];ledger=old[name];ids=ledger['fit_ids'];norm=ledger['normalization_ids'];hist=ledger['source_history']
        assert m['family']=='cnn' and len(training[name]['losses'])==600 and training[name]['prefix_equal']
        pred=np.load(r/'predictions'/m['prediction_file']);oldpred=np.load(origin/'predictions'/original_predictions[name])
        score=pred['score_ids'].tolist();np.testing.assert_array_equal(pred['score_ids'],oldpred['score_ids']);np.testing.assert_array_equal(pred['y'],oldpred['y'])
        assert not (set(ids+norm+hist)&set(score))
        assert all(rows[i]['subject'] in (0,1,14,18,24,25) and rows[i]['session']==0 for i in set(ids+norm+hist+score))
        np.testing.assert_allclose(ledger['scale'],max(float(np.sqrt(np.mean(np.square(raw[norm])))),1e-8))
        model=SeNicCNN();model.load_state_dict(torch.load(r/'models'/(name+'.pt'),map_location='cpu',weights_only=True));model.eval()
        for selected,role in [(ids,'fit'),(score,'score')]:
            x=(win[selected].reshape(-1,8,50)/ledger['scale']).astype(np.float32)
            truth=np.repeat([rows[i]['label'] for i in selected],15)
            with torch.no_grad():p=model(torch.from_numpy(x)).softmax(1).numpy()
            if role=='fit':np.testing.assert_allclose(np.mean(p.argmax(1)==truth),training[name]['training_accuracy']);continue
            np.testing.assert_allclose(p,pred['probabilities'],rtol=1e-6,atol=1e-7);np.testing.assert_array_equal(p.argmax(1),pred['pred'])
            cm=np.zeros((7,7),int);np.add.at(cm,(truth,p.argmax(1)),1);den=cm.sum(0)+cm.sum(1)
            f1=np.mean(np.divide(2*np.diag(cm),den,out=np.zeros(7),where=den>0));acc=np.trace(cm)/cm.sum()
            votes=np.array([np.bincount(v,minlength=7).argmax() for v in p.argmax(1).reshape(7,15)])
            np.testing.assert_allclose([acc,f1,np.mean(votes==[rows[i]['label'] for i in score])],[m['accuracy'],m['macro_f1'],m['trial_accuracy']])
    result=dict(passed=True,models=360,original_prefixes_equal=360,prediction_rows=37800,training_accuracy_rechecked=360,reserved_signal_access=0,wall_seconds=time.perf_counter()-t)
    (r/'independent_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):main()
