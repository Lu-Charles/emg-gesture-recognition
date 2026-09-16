"""Post-hoc plain-CNN optimization check; preserves the original frozen screen.

Rerun from exactly the original source model / scratch initialization to600steps.
Verify that step150 (fine) /300 (scratch) matches the original saved checkpoint.
This separates an implementation change from merely spending more optimizer steps.
"""
import argparse
import hashlib
import json
import time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from src.senic_neural import SeNicCNN,probabilities


def write(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--neural',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=False);tick=time.perf_counter()
    read=lambda f:json.loads((a.neural/f).read_text())
    c=read('config.json');ledger={x['model']:x for x in read('model_access.json')}
    selected=[x for x in ledger.values() if x['family']=='cnn' and x['method']!='enrollment']
    assert len(selected)==360
    oldmetrics={x['model']:x for x in read('metrics.json') if x['family']=='cnn' and x['trials']>0}
    rows=read('trial_manifest.json');cached=np.load(a.neural/'windows.npy',mmap_mode='r')
    # The original transpose/stack pipeline used channel-strided NumPy arrays.
    # Recreate that layout: identical values with a different stride can select
    # a different floating-point convolution kernel and alter an optimizer trace.
    win=np.ascontiguousarray(cached.transpose(0,1,3,2)).transpose(0,1,3,2)
    labels=np.array([r['label'] for r in rows]);out=a.out
    config=dict(created_utc=datetime.now(timezone.utc).isoformat(),scope='post-hoc fitting-driven optimization diagnostic, not new architecture or test-tuned winning run',
        reason='plainCNN fine-target63/120 and fine-pooled78/120 fits below95%clean fitting accuracy at150steps; scratch fits all>=98%',
        neural=str(a.neural.resolve()),original_config_sha256=hashlib.sha256((a.neural/'config.json').read_bytes()).hexdigest(),
        seed=42,steps=600,methods=['fine_target','fine_pooled','scratch_target'],
        rule='all360plainCNN updates rerun from original source checkpoint/scratchinit, regardless of individual scoring results; original prefix checked exactly; final600step retained',
        excluded='no augmentation model changed; original same-step augmentation comparisons remain separately reported',
        constraints='same windows/normalizers/source/calibration/scoring files; no final participants; cannot select between150/300/600 using scoring for a final claim',
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    write(out/'config.json',config);(out/'models').mkdir();(out/'predictions').mkdir();(out/'code_snapshot').mkdir()
    for p in [Path(__file__),Path('src/senic_neural.py')]: (out/'code_snapshot'/p.name).write_bytes(p.read_bytes())
    metrics=[];training=[];access=[]
    for i,old in enumerate(selected):
        name=old['model'];ids=old['fit_ids'];torch.manual_seed(42);model=SeNicCNN()
        if old['parent_model']:model.load_state_dict(torch.load(a.neural/'models'/(old['parent_model']+'.pt'),weights_only=True,map_location='cpu'))
        original=torch.load(a.neural/'models'/(name+'.pt'),weights_only=True,map_location='cpu')
        x=(win[ids].reshape(-1,8,50)/old['scale']).astype(np.float32);y=np.repeat(labels[ids],15)
        xt=torch.from_numpy(x);yt=torch.from_numpy(y);torch.manual_seed(42);rng=np.random.default_rng(42)
        opt=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=.0001);losses=[];prefix_equal=False;t=time.perf_counter();model.train()
        for step in range(600):
            batch=torch.tensor(rng.choice(len(x),size=min(64,len(x)),replace=False))
            opt.zero_grad();loss=torch.nn.functional.cross_entropy(model(xt[batch]),yt[batch])
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite convergence check')
            loss.backward();opt.step();losses.append(float(loss.detach()))
            if step+1==old['steps']:
                prefix_equal=all(torch.equal(v,original[k]) for k,v in model.state_dict().items())
                if not prefix_equal:raise ValueError(f'Original optimization prefix mismatch: {name}')
        torch.save(model.state_dict(),out/'models'/(name+'.pt'))
        acc=float(np.mean(probabilities(model,x).argmax(1)==y))
        training.append(dict(model=name,losses=losses,training_accuracy=acc,prefix_equal=prefix_equal,seconds=time.perf_counter()-t))
        prior=oldmetrics[name];oldpred=np.load(a.neural/'predictions'/prior['prediction_file']);scoreids=oldpred['score_ids'].tolist()
        xx=(win[scoreids].reshape(-1,8,50)/old['scale']).astype(np.float32);prob=probabilities(model,xx);pred=prob.argmax(1);truth=oldpred['y']
        cm=np.zeros((7,7),int);np.add.at(cm,(truth,pred),1);den=cm.sum(0)+cm.sum(1)
        f1=float(np.mean(np.divide(2*np.diag(cm),den,out=np.zeros(7),where=den>0)))
        votes=np.array([np.bincount(p,minlength=7).argmax() for p in pred.reshape(7,15)])
        rec=dict(prior,accuracy=float(np.mean(pred==truth)),macro_f1=f1,trial_accuracy=float(np.mean(votes==labels[scoreids])),prediction_file=name+'.npz')
        metrics.append(rec);np.savez_compressed(out/'predictions'/rec['prediction_file'],score_ids=scoreids,y=truth,pred=pred,probabilities=prob,trial_pred=votes)
        access.append(dict(old,steps=600,original_steps=old['steps'],original_model_path=str((a.neural/'models'/(name+'.pt')).resolve())))
        if (i+1)%60==0:
            write(out/'metrics.json',metrics);write(out/'training.json',training);write(out/'model_access.json',access)
            print(json.dumps(dict(models=i+1,subject=old['subject'],elapsed_seconds=time.perf_counter()-tick)),flush=True)
    write(out/'summary.json',dict(models=len(access),wall_seconds=time.perf_counter()-tick,original_prefixes_equal=all(x['prefix_equal'] for x in training),completed_utc=datetime.now(timezone.utc).isoformat()))


if __name__=='__main__':
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):main()
