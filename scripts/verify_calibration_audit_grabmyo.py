"""Replay cross-day pilot from saved coefficients and assert all trial roles."""
import json
from pathlib import Path
import numpy as np
from src.grabmyo_corpus import hash_stream
R=Path(__file__).resolve().parents[1];O=R/'research/runs/20260916_calibration_audit_grab_development'

def main():
 cfg=json.loads((O/'config.json').read_text());f=R/'research/runs/20260916_tdar_development/features.npy';assert hash_stream(f)==cfg['feature_sha256'];x=np.load(f,mmap_mode='r')
 rows=json.loads((O/'manifest.json').read_text());access=json.loads((O/'access.json').read_text());records=json.loads((O/'results.json').read_text());ss=json.loads((O/'summary.json').read_text());states=np.load(O/'parameters.npz');preds=np.load(O/'predictions.npz');lookup={(r['name'],r['policy']):r for r in records};count=0
 def predict(ids,p,key=None):
  z=np.asarray(x[ids]).reshape(-1,144).copy();mean=states[f'p{p}_mean'];scale=states[f'p{p}_scale']
  if key:
   for sl in (slice(0,16),slice(48,64),slice(128,144)):z[:,sl]*=states[key+'_gain']
   coef=states[key+'_coef'];intercept=states[key+'_intercept']
  else:coef=states[f'p{p}_coef'];intercept=states[f'p{p}_intercept']
  return np.argmax(((z-mean)/scale)@coef.T+intercept,axis=1)
 def vote(z):return np.array([np.bincount(v,minlength=17).argmax() for v in z.reshape(-1,35)])
 for a in access:
  name=a['name'];p=a['participant'];src=set(a['source']);fit=set(a['fit']);score=set(a['score']);check=a['check'];same=a['same_check'];g=a['check_label']
  assert p in [3,5,7,20,21,22,24,38]
  assert len(fit|{check,same})==4 and not src&(fit|{check,same}|score) and not (fit|{check,same})&score
  assert all(int(rows[i]['participant'])==p for i in src|fit|{check,same}|score)
  assert all(rows[i]['role']=='enrollment' and int(rows[i]['session'])==1 for i in src)
  assert all(rows[i]['role']=='calibration' for i in fit|{check,same})
  assert all(rows[i]['role']=='scoring' for i in score)
  assert g not in a['fit_labels'] and int(rows[same]['class_index'])==a['fit_labels'][0]
  assert all(int(rows[i]['session'])==a['session'] for i in fit|{check,same}|score)
  assert [int(rows[i]['calibration_rank']) for i in a['fit']]==[1,2]
  assert int(rows[check]['calibration_rank'])==int(rows[same]['calibration_rank'])==3
  truth=np.array([int(rows[i]['class_index']) for i in a['score']]);np.testing.assert_array_equal(truth,preds[name+'_truth'])
  cf=predict([check],p);ca=predict([check],p,name+'_two');sf=predict([same],p);sa=predict([same],p,name+'_two')
  ac=np.sum(ca==g)>np.sum(cf==g);ac_same=np.sum(sa==a['fit_labels'][0])>np.sum(sf==a['fit_labels'][0])
  frozen=vote(predict(a['score'],p));two=vote(predict(a['score'],p,name+'_two'));three=vote(predict(a['score'],p,name+'_three'))
  choices=dict(frozen=frozen,fit_two=two,fit_three=three,audit_novel=two if ac else frozen,audit_same=two if ac_same else frozen)
  for key,arr in dict(check_frozen=cf,check_adapted=ca,same_frozen=sf,same_adapted=sa,**choices).items():np.testing.assert_array_equal(arr,preds[name+'_'+key]);count+=len(arr)
  mask=(truth<16)&~np.isin(truth,a['fit_labels']+[g]);assert mask.sum()==52
  base=np.mean(frozen[mask]==truth[mask]);other_gain=np.mean(two[mask]==truth[mask])-base
  for policy,v in choices.items():
   rec=lookup[name,policy];oa=np.mean(v[mask]==truth[mask]);np.testing.assert_allclose(rec['accuracy'],np.mean(v==truth));np.testing.assert_allclose(rec['omitted_accuracy'],oa)
   assert rec['harm']==bool(oa<base);np.testing.assert_allclose(rec['positive_loss'],max(0,base-oa))
   if policy.startswith('audit'):
    ac_expected=ac if policy=='audit_novel' else ac_same
    assert rec['accepted']==bool(ac_expected) and rec['false_accept']==bool(ac_expected and other_gain<0)
 for summary in ss:
  rr=[r for r in records if r['method']==summary['method'] and r['policy']==summary['policy']]
  for k,v in summary['means'].items():np.testing.assert_allclose(v,np.mean([np.mean([r[k] for r in rr if r['participant']==p]) for p in cfg['participants']]))
 assert len(access)==448 and len(records)==2240
 report=dict(passed=True,cases=len(access),records=len(records),prediction_entries_replayed=count,participants=8,scope='saved-parameter replay and trial-role audit, not independent refit or external replication',verifier_sha256=hash_stream(__file__))
 (O/'verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':main()
