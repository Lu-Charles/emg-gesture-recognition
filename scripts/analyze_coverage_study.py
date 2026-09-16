"""Participant-level statistics for the frozen primary contrasts and disclosed controls."""
import json
from pathlib import Path
import numpy as np
from scipy.stats import t,ttest_1samp
R=Path(__file__).resolve().parents[1];O=R/'research/expanded_study_20260916'

def interval(d):
 d=np.asarray(d,float);n=len(d);mean=float(d.mean());se=d.std(ddof=1)/np.sqrt(n);half=t.ppf(.975,n-1)*se;boots=np.random.default_rng(20260916).choice(d,(10000,n),replace=True).mean(1)
 return dict(n=n,mean=mean,ci95=[float(mean-half),float(mean+half)],bootstrap95=np.quantile(boots,[.025,.975]).tolist(),p=float(ttest_1samp(d,0).pvalue) if se else (1. if mean==0 else 0.),positive=int(sum(d>1e-12)),negative=int(sum(d< -1e-12)),per_person=d.tolist())
def paired(a,b,key='accuracy'):
 ps=sorted(set(z['participant'] for z in a));assert ps==sorted(set(z['participant'] for z in b));d=[np.mean([z[key] for z in a if z['participant']==p])-np.mean([z[key] for z in b if z['participant']==p]) for p in ps];return dict(participants=ps,metric=key,**interval(d))
def grouped(records,fields,metrics):
 out=[]
 for keys in sorted({tuple(z[f] for f in fields) for z in records}):
  a=[z for z in records if tuple(z[f] for f in fields)==keys];ps=sorted({z['participant'] for z in a});values={m:[float(np.mean([z[m] for z in a if z['participant']==p])) for p in ps] for m in metrics}
  out.append(dict(zip(fields,keys),participants=ps,per_person=values,**{m:float(np.mean(v)) for m,v in values.items()}))
 return out

def main():
 O.mkdir(exist_ok=True);study={}
 all_neural={}
 for cohort in ['development','final']:
  records=[]
  for seed in [42,0,1]:
   folder=R/'research/runs'/f"20260916_decisive_{'final_' if cohort=='final' else ''}seed{seed}";assert json.loads((folder/'verification.json').read_text())['passed'];a=json.loads((folder/'results.json').read_text())['records'];records.extend([dict(z,seed=seed) for z in a if z['method'].startswith('cnn_')])
  all_neural[cohort]=records;study[cohort+'_neural']=grouped(records,['method','steps','k'],['accuracy','other_active_accuracy','other_into_calibration','calibrated_recall','rest_recall','macro_f1'])
  primary=[z for z in records if z['steps'] in [0,100]]
  for k in [1,2]:
   for m in ['cnn_naive','cnn_replay','cnn_l2']:
    a=[z for z in primary if z['k']==k and z['method']==m];b=[z for z in primary if z['k']==k and z['method']=='cnn_frozen']
    study[f'{cohort}_{m}_k{k}_vs_frozen']={metric:paired(a,b,metric) for metric in ['accuracy','other_active_accuracy','other_into_calibration']}
  study[cohort+'_replay_coverage']=paired([z for z in primary if z['k']==2 and z['method']=='cnn_replay'],[z for z in primary if z['k']==1 and z['method']=='cnn_replay'])
 for cohort in ['development','final']:
  folder=R/'research/runs'/('20260916_rotation_coverage' if cohort=='development' else '20260916_rotation_final');assert json.loads((folder/'verification.json').read_text())['passed'];r=json.loads((folder/'results.json').read_text());a=r['records'];sels={s['participant']:s for s in r['selections']}
  study[cohort+'_rotation']=grouped(a,['method','k'],['accuracy','window_accuracy','other_accuracy','other_into_calibration','calibrated_recall','recorded_seconds'])
  selected=[]
  for z in a:
   if z['k']==2:
    for rule in ['active','diverse','identifiable','fixed']:
     if [z['g'],z['h']]==sels[z['participant']][rule]:selected.append(dict(z,selector=rule))
  study[cohort+'_rotation_selection']=grouped(selected,['method','selector'],['accuracy','other_accuracy','recorded_seconds'])
  active=[z for z in selected if z['method']=='cosine_gain' and z['selector']=='active'];random=[z for z in a if z['method']=='cosine_gain' and z['k']==2]
  study[cohort+'_selection_primary']=paired(active,random)
  study[cohort+'_rotation_coverage_contrast']=paired(random,[z for z in a if z['method']=='cosine_gain' and z['k']==1])
  study[cohort+'_rotation_vs_frozen']=paired(random,[z for z in a if z['method']=='frozen' and z['k']==2])
  study[cohort+'_rotation_gain_vs_cosine']=paired(random,[z for z in a if z['method']=='cosine_rotation' and z['k']==2])
 for cohort in ['development','final']:
  folder=R/'research/runs'/('20260916_classical_selection' if cohort=='development' else '20260916_classical_final');assert json.loads((folder/'verification.json').read_text())['passed'];r=json.loads((folder/'results.json').read_text());a=r['records'];sels={s['participant']:s for s in r['selections']}
  study[cohort+'_classical']=grouped(a,['method','k'],['accuracy','other_active_accuracy','other_into_calibration','macro_f1'])
  selected=[dict(z,selector=rule) for z in a for rule in ['active','diverse','fixed'] if [z['g'],z['h']]==sels[z['participant']][rule]];study[cohort+'_classical_selection']=grouped(selected,['method','selector'],['accuracy','other_active_accuracy'])
 folder=R/'research/runs/20260916_full_network_coverage';assert json.loads((folder/'verification.json').read_text())['passed'];r=json.loads((folder/'results.json').read_text());study['development_full_network']=grouped(r['records'],['method','steps','k'],['accuracy','other_active_accuracy','other_into_calibration','calibrated_recall'])
 ref=json.loads((R/'research/runs/20260916_rotation_final_reference/results.json').read_text());study['final_rotation_reference']=grouped(ref,['method'],['accuracy','window_accuracy','recorded_seconds'])
 # Confirmatory family of two primary hypotheses: Holm step-down adjustment.
 keys=['final_cnn_replay_k2_vs_frozen','final_selection_primary'];p=[study[keys[0]]['accuracy']['p'],study[keys[1]]['p']];order=np.argsort(p);adjusted=np.zeros(2);prev=0.
 for rank,i in enumerate(order):prev=max(prev,min(1.,(2-rank)*p[i]));adjusted[i]=prev
 study['primary_tests']={k:dict(raw_p=p[i],holm_p=float(adjusted[i])) for i,k in enumerate(keys)}
 precision={}
 for rr in all_neural['final']:
  if rr['k']==2 and rr['steps'] in [0,100]:
   cm=np.array(rr['confusion']);gs=rr['gestures'];entry=precision.setdefault(rr['method'],dict(correct=0,predicted=0));entry['correct']+=int(cm.diagonal()[gs].sum());entry['predicted']+=int(cm[:,gs].sum())
 study['posthoc_calibrated_prediction_correctness']={m:dict(**v,fraction=v['correct']/v['predicted'],scope='pooled exact-label correctness conditional on predicting either calibrated gesture; post-hoc descriptive') for m,v in precision.items()}
 (O/'statistics.json').write_text(json.dumps(study,indent=2)+'\n')
 print(json.dumps(dict(grabmyo=study['final_cnn_replay_k2_vs_frozen'],senic=study['final_selection_primary'],primary=study['primary_tests']),indent=2))
if __name__=='__main__':main()
