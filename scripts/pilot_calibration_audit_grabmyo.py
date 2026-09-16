"""Cross-day exploratory extension; no final participants or gate tuning."""
import json
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from src.grabmyo_corpus import WindowCorpus,hash_stream
from scripts.pilot_calibration_audit import accept_update
R=Path(__file__).resolve().parents[1];O=R/'research/runs/20260916_calibration_audit_grab_development'
FEATURE=R/'research/runs/20260916_tdar_development/features.npy'

def main():
 O.mkdir(exist_ok=False)
 corpus=WindowCorpus(R/'data/public/grabmyo/cache_20260906_v1','development');rows=corpus.rows;x=np.load(FEATURE,mmap_mode='r')
 people=sorted({int(r['participant']) for r in rows});assert people==[3,5,7,20,21,22,24,38]
 cfg=dict(timestamp=datetime.now(timezone.utc).isoformat(),scope='exploratory development only, extension after SeNic screen',participants=people,methods=['gain','pooled'],gate='strict check-window accuracy improvement; frozen on tie',feature_sha256=hash_stream(FEATURE),code_sha256=hash_stream(__file__),protocol_sha256=hash_stream(R/'research/pivot_calibration_audit_20260916/PROTOCOL.md'),seed=None,deterministic=True)
 (O/'config.json').write_text(json.dumps(cfg,indent=2)+'\n');(O/'PROTOCOL.md').write_bytes((R/'research/pivot_calibration_audit_20260916/PROTOCOL.md').read_bytes());(O/'code.py').write_bytes(Path(__file__).read_bytes())
 records=[];access=[];saved={};parameters={}
 def vote(z):return np.array([np.bincount(q,minlength=17).argmax() for q in z.reshape(-1,35)])
 for p in people:
  source=[i for i,r in enumerate(rows) if int(r['participant'])==p and r['role']=='enrollment'];sy=np.repeat([int(rows[i]['class_index']) for i in source],35);sx=np.asarray(x[source]).reshape(-1,144)
  sc=StandardScaler().fit(sx)
  def fit(z,y):return LinearDiscriminantAnalysis(solver='lsqr',shrinkage='auto',priors=np.full(17,1/17)).fit(sc.transform(z),y)
  lda=fit(sx,sy);template=np.stack([sx[sy==g,128:].mean(0) for g in range(16)]);order=np.argsort(-template.sum(1),kind='stable').tolist();gs=order[:2]
  for k,v in dict(mean=sc.mean_,scale=sc.scale_,coef=lda.coef_,intercept=lda.intercept_).items():parameters[f'p{p}_{k}']=v
  for session in (2,3):
   pool={(int(r['class_index']),int(r['calibration_rank'])):i for i,r in enumerate(rows) if int(r['participant'])==p and int(r['session'])==session and r['role']=='calibration'}
   score=[i for i,r in enumerate(rows) if int(r['participant'])==p and int(r['session'])==session and r['role']=='scoring'];truth=np.array([int(rows[i]['class_index']) for i in score]);fitting=[pool[gs[0],1],pool[gs[1],2]];same=pool[gs[0],3]
   def adapter(ids,labels,method):
    if method=='pooled':
     model=fit(np.concatenate([sx,np.asarray(x[ids]).reshape(-1,144)]),np.r_[sy,np.repeat(labels,35)])
     return dict(coef=model.coef_,intercept=model.intercept_,gain=np.ones(16))
    target=np.asarray(x[ids])[:,:,128:].mean(1)
    gain=np.clip(np.exp((np.log(template[labels])-np.log(target)).mean(0)),.5,2)
    return dict(coef=lda.coef_,intercept=lda.intercept_,gain=gain)
   def pred(ids,state=None):
    z=np.asarray(x[ids]).reshape(-1,144).copy()
    if state is None:return lda.predict(sc.transform(z))
    z[:,np.r_[0:16,48:64,128:144]]*=np.tile(state['gain'],3)
    return np.argmax(sc.transform(z)@state['coef'].T+state['intercept'],axis=1)
   frozen=vote(pred(score));sf=pred([same])
   for method in cfg['methods']:
    state2=adapter(fitting,gs,method);two=vote(pred(score,state2));sa=pred([same],state2);acs=accept_update(sf,sa,gs[0])
    for g in order[2:]:
     check=pool[g,3];name=f'p{p}_s{session}_{method}_check{g}';assert not set(source)&set(fitting+[check,same]+score) and not set(fitting+[check,same])&set(score);assert len(set(fitting+[check,same]))==4
     cf=pred([check]);ca=pred([check],state2);ac=accept_update(cf,ca,g);state3=adapter(fitting+[check],gs+[g],method);three=vote(pred(score,state3))
     for label,state in [('two',state2),('three',state3)]:
      for key,val in state.items():parameters[name+'_'+label+'_'+key]=val
     access.append(dict(name=name,source=source,fit=fitting,check=check,same_check=same,score=score,fit_labels=gs,check_label=g,participant=p,session=session))
     choices=dict(frozen=frozen,fit_two=two,fit_three=three,audit_novel=two if ac else frozen,audit_same=two if acs else frozen)
     for key,val in dict(check_frozen=cf,check_adapted=ca,same_frozen=sf,same_adapted=sa,truth=truth,**choices).items():saved[name+'_'+key]=val
     mask=(truth<16)&~np.isin(truth,gs+[g]);base=float(np.mean(frozen[mask]==truth[mask]));proposed=float(np.mean(two[mask]==truth[mask]))-base
     for policy,v in choices.items():
      oa=float(np.mean(v[mask]==truth[mask]));accept=ac if policy=='audit_novel' else acs if policy=='audit_same' else None;cost=0 if policy=='frozen' else 2 if policy=='fit_two' else 3
      records.append(dict(name=name,participant=p,session=session,method=method,policy=policy,check_label=g,accuracy=float(np.mean(v==truth)),omitted_accuracy=oa,harm=oa<base,positive_loss=max(0.,base-oa),accepted=accept,false_accept=bool(accept and proposed<0) if accept is not None else None,recorded_seconds=cost*5,analyzed_seconds=cost*4.5,proposed_omitted_gain=proposed))
  print(json.dumps(dict(participant=p,records=len(records))),flush=True)
 (O/'manifest.json').write_text(json.dumps(rows,indent=2)+'\n');(O/'results.json').write_text(json.dumps(records,indent=2)+'\n');(O/'access.json').write_text(json.dumps(access,indent=2)+'\n');np.savez_compressed(O/'parameters.npz',**parameters);np.savez_compressed(O/'predictions.npz',**saved)
 summaries=[]
 for method in cfg['methods']:
  for policy in ('frozen','fit_two','fit_three','audit_novel','audit_same'):
   per=[]
   for p in people:
    rr=[r for r in records if r['participant']==p and r['method']==method and r['policy']==policy]
    metrics={k:float(np.mean([r[k] for r in rr])) for k in ('accuracy','omitted_accuracy','harm','positive_loss','recorded_seconds','analyzed_seconds')}
    if policy.startswith('audit'):metrics.update({k:float(np.mean([r[k] for r in rr])) for k in ('accepted','false_accept')})
    per.append(dict(participant=p,cases=len(rr),**metrics))
   summary=dict(method=method,policy=policy,participants=per,means={k:float(np.mean([v[k] for v in per])) for k in per[0] if k not in ('participant','cases')});summaries.append(summary);print(json.dumps({k:v for k,v in summary.items() if k!='participants'}),flush=True)
 (O/'summary.json').write_text(json.dumps(summaries,indent=2)+'\n');(O/'complete.json').write_text(json.dumps(dict(records=len(records),cases=len(access),timestamp=datetime.now(timezone.utc).isoformat()))+'\n')
if __name__=='__main__':
 with threadpool_limits(limits=1):main()
