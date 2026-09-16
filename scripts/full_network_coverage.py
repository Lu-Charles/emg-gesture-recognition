"""Full-network robustness control, all prespecified 512 development cases."""
import copy,json,time
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import torch
from src.emg_cnn import CompactEMGNet
from src.grabmyo_corpus import WindowCorpus,hash_stream
from scripts.decisive_calibration import schedules,measures,vote
R=Path(__file__).resolve().parents[1];O=R/'research/runs/20260916_full_network_coverage'
def main():
 O.mkdir(exist_ok=False);(O/'models').mkdir();torch.set_num_threads(1);torch.manual_seed(42)
 c=WindowCorpus(R/'data/public/grabmyo/cache_20260906_v1','development');rows=c.rows;prior=R/'research/runs/20260906_shared_pretraining_v2';sc=np.load(prior/'training_scaler.npz');device=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
 cfg=dict(timestamp=datetime.now(timezone.utc).isoformat(),participants=sorted({int(r['participant']) for r in rows}),scope='development robustness, no final access',seed=42,steps=[25,100],primary_steps=100,lr=.0001,optimizer='Adam',target_batch=64,replay_source_batch=64,update='all model parameters',methods=['full_naive','full_replay'],same_cases='20260916_decisive_seed42',recording_seconds=10,device=str(device),code_sha256=hash_stream(__file__),primary_interpretation='sensitivity control; not tune learning rate from scoring')
 (O/'protocol.json').write_text(json.dumps(cfg,indent=2)+'\n');(O/'full_network_coverage.py').write_bytes(Path(__file__).read_bytes());records=[];access=[];preds={};start=time.perf_counter()
 def predict(model,x):
  with torch.no_grad():return vote(np.concatenate([model(x[i:i+256]).argmax(1).cpu().numpy() for i in range(0,len(x),256)]))
 for p in cfg['participants']:
  ids=[i for i,r in enumerate(rows) if int(r['participant'])==p];lookup={i:j for j,i in enumerate(ids)}
  xx,yy=c.batch(c.window_ids(ids),sc['mean'],sc['scale']);xx=torch.from_numpy(xx).to(device);yy=torch.tensor(yy,dtype=torch.long,device=device)
  def subset(tids):
   wi=np.concatenate([np.arange(lookup[i]*35,(lookup[i]+1)*35) for i in tids]);return xx[wi],yy[wi]
  src=[i for i in ids if rows[i]['role']=='enrollment'];sx,sy=subset(src);byclass=[np.flatnonzero(sy.cpu().numpy()==k) for k in range(17)]
  base=CompactEMGNet().to(device);base.load_state_dict(torch.load(prior/f'models/p{p}_s2_shared_none_b0.pt',map_location='cpu',weights_only=True));base.eval()
  for session in [2,3]:
   score=[i for i in ids if int(rows[i]['session'])==session and rows[i]['role']=='scoring'];tx,_=subset(score);truth=np.array([int(rows[i]['class_index']) for i in score])
   for case in schedules(rows,p,session):
    name=f"p{p}_s{session}_o{case['choice']}_k{case['k']}";cal=case['ids'];assert not set(cal)&set(score);cx,cy=subset(cal)
    access.append(dict(name=name,source=src,calibration=cal,scoring=score))
    for method in ['naive','replay']:
     model=copy.deepcopy(base).train();opt=torch.optim.Adam(model.parameters(),lr=cfg['lr']);rng=np.random.default_rng(42+case['choice']);srng=np.random.default_rng(100042+case['choice'])
     for step in range(1,101):
      ix=rng.integers(0,len(cx),64);opt.zero_grad();loss=torch.nn.functional.cross_entropy(model(cx[ix]),cy[ix])
      if method=='replay':
       classes=srng.integers(0,17,64);si=np.array([srng.choice(byclass[k]) for k in classes]);loss=loss+torch.nn.functional.cross_entropy(model(sx[si]),sy[si])
      if not torch.isfinite(loss):raise FloatingPointError(name)
      loss.backward();opt.step()
      if step in cfg['steps']:
       model.eval();v=predict(model,tx);key=f'{name}_full_{method}_t{step}';preds[key]=v
       torch.save({k:z.cpu() for k,z in model.state_dict().items()},O/f'models/{key}.pt')
       records.append(dict(name=key,participant=p,session=session,choice=case['choice'],k=case['k'],gestures=case['gestures'],method='full_'+method,steps=step,**measures(v,truth,case['gestures'])))
       model.train()
   (O/'results.json').write_text(json.dumps(dict(config=cfg,records=records),indent=2)+'\n');(O/'access.json').write_text(json.dumps(access,indent=2)+'\n');np.savez_compressed(O/'predictions.npz',**preds)
   print(json.dumps(dict(participant=p,session=session,records=len(records),seconds=time.perf_counter()-start)),flush=True)
  del xx,yy,sx,sy,tx,cx,cy,base,model
  if device.type=='mps':torch.mps.empty_cache()
 (O/'complete.json').write_text(json.dumps(dict(records=len(records),cases=len(access),seconds=time.perf_counter()-start))+'\n')
if __name__=='__main__':main()
