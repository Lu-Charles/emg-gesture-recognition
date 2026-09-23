"""Independently replay every classical selection prediction from stored parameters."""
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score
from threadpoolctl import threadpool_limits
from src.grabmyo_corpus import hash_stream
from src.final_coverage import FinalCorpus as WindowCorpus
R=Path(__file__).resolve().parents[1];O=R/'research/runs/20260916_classical_final'
def main():
 c=WindowCorpus(R/'data/public/grabmyo/cache_final_20260916','final');rows=c.rows
 d=np.load(R/'research/runs/20260916_tdar_final/features.npy',mmap_mode='r');a=np.load(O/'parameters.npz');v=np.load(O/'predictions.npz');res=json.loads((O/'results.json').read_text());access={x['name']:x for x in json.loads((O/'access.json').read_text())}
 assert json.loads((O/'complete.json').read_text())['records']==30720
 def pred(x,prefix):
  logits=((x-a[prefix+'_mean'])/a[prefix+'_scale'])@a[prefix+'_coef'].T+a[prefix+'_intercept'];return logits.argmax(1)
 def vote(x):return np.array([max(range(17),key=lambda k:(np.count_nonzero(row==k),-k)) for row in x.reshape(-1,35)])
 for s in res['selections']:
  p=s['participant'];src=next(z['source'] for z in access.values() if z['name'].startswith(f'p{p}_'))
  x=np.asarray(c.features[src]);y=np.array([int(rows[i]['class_index']) for i in src]);t=np.array([x[y==g,:,16:32].mean((0,1)) for g in range(16)])
  assert np.allclose(t,s['source_rms_templates']);u=t/np.linalg.norm(t,axis=1)[:,None];dist=1-u@u.T;np.fill_diagonal(dist,-np.inf)
  assert list(np.unravel_index(np.argmax(dist),dist.shape))==s['diverse'];assert np.argsort(-t.sum(1),kind='stable')[:2].tolist()==s['active']
 for j,r in enumerate(res['records']):
  name=r['name'];base=name.removesuffix('_'+r['method']);z=access[base];src,cal,score=z['source'],z['calibration'],z['score'];p=r['participant'];prefix=f'p{p}'
  assert len(src)==119 and len(cal)==2 and len(score)==68 and not(set(src)&set(cal) or set(src)&set(score) or set(cal)&set(score))
  assert all(int(rows[i]['participant'])==p for i in src+cal+score)
  assert all(rows[i]['role']=='enrollment' for i in src) and all(rows[i]['role']=='scoring' for i in score)
  assert [int(rows[i]['calibration_rank']) for i in cal]==[1,2] and [int(rows[i]['class_index']) for i in cal]==[r['g'],r['h']]
  y=np.array([int(rows[i]['class_index']) for i in score]);x=np.asarray(d[score]).reshape(-1,144);m=r['method']
  if m=='tdar_frozen':q=vote(pred(x,prefix+'_tdar'))
  elif m=='tdar_pooled':
   q=vote((((x-a[prefix+'_tdar_mean'])/a[prefix+'_tdar_scale'])@a[base+'_pooled_coef'].T+a[base+'_pooled_intercept']).argmax(1))
  else:
   sx=np.asarray(c.features[src]);sy=np.array([int(rows[i]['class_index']) for i in src]);cx=np.asarray(c.features[cal]);cy=np.array([int(rows[i]['class_index']) for i in cal]);gs=sorted(set(cy))
   gain=np.clip(np.exp(np.mean([np.log(np.maximum(sx[sy==g].mean((0,1)),1e-12))-np.log(np.maximum(cx[cy==g].mean((0,1)),1e-12)) for g in gs],axis=0)),.5,2)
   np.testing.assert_allclose(gain,a[base+'_gain'],rtol=1e-10,atol=1e-10)
   if m=='amplitude_gain48':q=vote(pred(np.asarray(c.features[score]).reshape(-1,48)*gain,prefix+'_amplitude'))
   else:
    x[:,np.r_[0:16,48:64,128:144]]*=np.tile(gain[16:32],3);q=vote(pred(x,prefix+'_tdar'))
  np.testing.assert_array_equal(q,v[name]);other=(y<16)&~np.isin(y,[r['g'],r['h']])
  metrics={'accuracy':np.mean(q==y),'macro_f1':f1_score(y,q,labels=range(17),average='macro',zero_division=0),'other_active_accuracy':np.mean(q[other]==y[other]),'other_into_calibration':np.mean(np.isin(q[other],[r['g'],r['h']]))}
  for k,value in metrics.items():assert abs(value-r[k])<1e-12,(name,k)
  if (j+1)%2048==0:print(j+1,flush=True)
 report=dict(passed=True,records=len(res['records']),trial_votes=len(res['records'])*68,cases=len(access),source_only_selections_verified=15,scope='replay stored fitted coefficients; no independent refit of covariance estimators',code_sha256=hash_stream(__file__))
 (O/'verification.json').write_text(json.dumps(report,indent=2)+'\n');print(report)
if __name__=='__main__':
 with threadpool_limits(limits=1):main()
