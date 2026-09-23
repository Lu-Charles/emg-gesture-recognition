"""Separate saved-parameter replay and information-boundary audit."""
import json
from pathlib import Path
import numpy as np
from src.grabmyo_corpus import hash_stream
R=Path(__file__).resolve().parents[1]
O=R/'research/runs/20260916_calibration_audit_development'
C=R/'research/runs/20260907_sensor_shift_pilot_v2'

def main():
 cfg=json.loads((O/'config.json').read_text()); rows=json.loads((C/'trial_manifest.json').read_text())
 assert hash_stream(C/'tdar_rms_features.npy')==cfg['cache_sha256']
 assert hash_stream(C/'trial_manifest.json')==cfg['manifest_sha256']
 x=np.load(C/'tdar_rms_features.npy'); models=np.load(O/'models.npz'); preds=np.load(O/'predictions.npz')
 access=json.loads((O/'access.json').read_text()); states=json.loads((O/'states.json').read_text()); results=json.loads((O/'results.json').read_text())
 lookup={(r['name'],r['policy']):r for r in results}; n=0
 def predict(ids,p,state=None):
  z=x[ids].reshape(-1,9,8).copy()
  if state:
   shift=state['shift']; lo=int(np.floor(shift));f=shift-lo
   z=(1-f)*np.roll(z,lo,axis=2)+f*np.roll(z,lo+1,axis=2)
   z[:,[0,3,8],:]*=np.asarray(state['gain'])[None,None,:]
  z=z.reshape(-1,72)
  return np.argmax(((z-models[f'p{p}_mean'])/models[f'p{p}_scale'])@models[f'p{p}_coef'].T+models[f'p{p}_intercept'],axis=1)
 def vote(a): return np.array([np.bincount(z,minlength=7).argmax() for z in a.reshape(-1,15)])
 for a in access:
  name=a['name']; src=set(a['source']);fit=set(a['fit']); score=set(a['score']);p=rows[a['source'][0]]['subject'];g=a['novel_label']
  assert p in (0,1,14,18,24,25)
  assert not src&fit and not src&score and not fit&score
  assert a['check'] not in src|fit|score and a['same_check'] not in src|fit|score
  assert rows[a['check']]['label'] not in a['fit_labels']
  assert all(rows[i]['repetition']==2 for i in score)
  assert all(rows[i]['repetition']!=2 for i in fit|{a['check'],a['same_check']})
  assert all(rows[i]['subject']==p and rows[i]['session']==0 for i in src|fit|score|{a['check'],a['same_check']})
  s2=states[name]['two'];s3=states[name]['three']
  # Independently reconstruct source templates, selected fit gestures and gains.
  template=np.stack([x[[i for i in src if rows[i]['label']==k],:,64:].mean((0,1)) for k in range(7)])
  assert a['fit_labels']==np.argsort(-template.sum(1),kind='stable')[:2].tolist()
  for state,ids,labels in [(s2,a['fit'],a['fit_labels']),(s3,a['fit']+[a['check']],a['fit_labels']+[g])]:
   target=np.stack([x[i,:,64:].mean(0) for i in ids]); source=template[labels]
   method=lookup[name,'frozen']['method']
   shifts=sorted(np.arange(-4,4,.125),key=lambda v:(abs(v),v)) if method=='cosine_gain' else [0.]
   costs=[]
   for shift in shifts:
    lo=int(np.floor(shift));f=shift-lo
    z=(1-f)*np.roll(target,lo,axis=1)+f*np.roll(target,lo+1,axis=1)
    costs.append(np.mean(np.sum((source/np.linalg.norm(source,axis=1,keepdims=True)-z/np.linalg.norm(z,axis=1,keepdims=True))**2,axis=1)))
   best=shifts[int(np.argmin(costs))]
   assert best==state['shift']
   lo=int(np.floor(best));f=best-lo;z=(1-f)*np.roll(target,lo,axis=1)+f*np.roll(target,lo+1,axis=1)
   gain=np.clip(np.exp(np.mean(np.log(source)-np.log(z),axis=0)),.5,2)
   np.testing.assert_allclose(gain,state['gain'],rtol=1e-12,atol=1e-12)
  cf=predict([a['check']],p); ca=predict([a['check']],p,s2)
  sf=predict([a['same_check']],p);sa=predict([a['same_check']],p,s2)
  ac=int(np.sum(ca==g))>int(np.sum(cf==g));sg=rows[a['same_check']]['label'];same=int(np.sum(sa==sg))>int(np.sum(sf==sg))
  f=vote(predict(a['score'],p));two=vote(predict(a['score'],p,s2));three=vote(predict(a['score'],p,s3))
  choices=dict(frozen=f,fit_two=two,fit_three=three,audit_novel=two if ac else f,audit_same=two if same else f)
  for suffix,arr in dict(check_frozen=cf,check_adapted=ca,same_frozen=sf,same_adapted=sa,**choices).items():
   np.testing.assert_array_equal(arr,preds[name+'_'+suffix]);n+=len(arr)
  omitted=a['omitted_labels']; assert set(omitted)==set(range(7))-set(a['fit_labels'])-{g}
  for policy,v in choices.items():
   r=lookup[name,policy]
   np.testing.assert_allclose(r['accuracy'],np.mean(v==np.arange(7)))
   np.testing.assert_allclose(r['omitted_accuracy'],np.mean(v[omitted]==omitted))
   assert r['harm']==bool(np.mean(v[omitted]==omitted)<np.mean(f[omitted]==omitted))
   if policy.startswith('audit'):
    expected=ac if policy=='audit_novel' else same
    assert r['accepted']==expected
    assert r['false_accept']==bool(expected and np.mean(two[omitted]==omitted)<np.mean(f[omitted]==omitted))
 assert len(access)==600 and len(results)==3000
 summaries=json.loads((O/'summary.json').read_text())
 for s in summaries:
  rr=[r for r in results if r['method']==s['method'] and r['policy']==s['policy'] and (s['selection']=='uniform' or r['selected_'+s['selection']])]
  for metric,value in s['means'].items():
   person=[np.mean([r[metric] for r in rr if r['participant']==p]) for p in cfg['participants']]
   np.testing.assert_allclose(value,np.mean(person))
 out=dict(passed=True,cases=len(access),records=len(results),prediction_entries_replayed=n,participants=6,
          scope='separate numerical replay and boundary checks; no clean-machine refit or independent laboratory replication',
          verifier_sha256=hash_stream(__file__))
 (O/'verification.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
if __name__=='__main__':main()
