"""Seven-recording pooled and target-only controls frozen before final scoring."""
import json
from pathlib import Path
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from src.final_coverage import validate_senic
from scripts.cache_final_coverage import frozen
from src.grabmyo_corpus import hash_stream
from threadpoolctl import threadpool_limits
R=Path(__file__).resolve().parents[1];O=R/'research/runs/20260916_rotation_final_reference';C=R/'data/public/senic/cache_final_20260916'
def main():
 cfg=frozen();O.mkdir(exist_ok=False);rows=json.loads((C/'trial_manifest.json').read_text());x=np.load(C/'tdar_rms_features.npy');records=[];preds={};params={};access=[]
 (O/'protocol.json').write_text(json.dumps(dict(protocol_sha256=hash_stream(R/'research/runs/20260625_confirmatory_protocol/protocol.json'),scope='higher-cost reference;7wholetrials;sourceonlyscaler',code_sha256=hash_stream(__file__)),indent=2)+'\n')
 for p in cfg['senic']['participants']:
  src=[i for i,r in enumerate(rows) if r['subject']==p and r['position']==0 and r['repetition']<2];sx=x[src].reshape(-1,72);sy=np.repeat([rows[i]['label'] for i in src],15);mean=sx.mean(0);scale=sx.std(0);scale[scale==0]=1
  params[f'p{p}_mean']=mean;params[f'p{p}_scale']=scale
  for pos in range(1,11):
   cal=[i for i,r in enumerate(rows) if r['subject']==p and r['position']==pos and r['repetition']==0];score=[i for i,r in enumerate(rows) if r['subject']==p and r['position']==pos and r['repetition']==2];validate_senic(src,cal,score,rows);cy=np.repeat([rows[i]['label'] for i in cal],15);cx=x[cal].reshape(-1,72);tx=x[score].reshape(-1,72);truth=np.array([rows[i]['label'] for i in score])
   for method in ['pooled7','target7']:
    xx,yy=(np.r_[sx,cx],np.r_[sy,cy]) if method=='pooled7' else (cx,cy);model=LinearDiscriminantAnalysis(solver='lsqr',shrinkage='auto',priors=np.ones(7)/7).fit((xx-mean)/scale,yy);w=model.predict((tx-mean)/scale);v=np.array([np.bincount(z,minlength=7).argmax() for z in w.reshape(-1,15)]);name=f'p{p}_pos{pos}_{method}';preds[name]=v;params[name+'_coef']=model.coef_;params[name+'_intercept']=model.intercept_;access.append(dict(name=name,source=src,calibration=cal,score=score));records.append(dict(name=name,participant=p,position=pos,method=method,accuracy=float(np.mean(v==truth)),window_accuracy=float(np.mean(w==np.repeat(truth,15))),recorded_seconds=sum(rows[i]['recorded_seconds'] for i in cal)))
 np.savez_compressed(O/'predictions.npz',**preds);np.savez_compressed(O/'parameters.npz',**params);(O/'access.json').write_text(json.dumps(access,indent=2)+'\n');(O/'results.json').write_text(json.dumps(records,indent=2)+'\n');(O/'complete.json').write_text(json.dumps(dict(records=len(records)))+'\n')
 # Independent stored-parameter readback, wholetrial predictions and metrics.
 a=np.load(O/'parameters.npz');v=np.load(O/'predictions.npz')
 for z,rr in zip(access,records):
  tx=x[z['score']].reshape(-1,72);p=rr['participant'];name=rr['name'];w=(((tx-a[f'p{p}_mean'])/a[f'p{p}_scale'])@a[name+'_coef'].T+a[name+'_intercept']).argmax(1);pred=np.array([max(range(7),key=lambda g:(sum(t==g),-g)) for t in w.reshape(-1,15)]);assert np.array_equal(pred,v[name]);truth=np.array([rows[i]['label'] for i in z['score']]);assert np.mean(pred==truth)==rr['accuracy']
 (O/'verification.json').write_text(json.dumps(dict(passed=True,records=len(records),trial_votes=len(records)*7,scope='independent parameter readback, same program; no raw-feature rederivation'))+'\n')
if __name__=='__main__':
 with threadpool_limits(limits=1):main()
