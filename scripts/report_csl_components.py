"""Produce fixed participant-level summaries and scientific figures after full verification."""
import csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts import csl_author_recipe as r
ROOT=r.ROOT/'research/runs/20260907_csl_components_confirmatory'
METHODS=['frozen','gain_only','spatial_only','spatial_gain','classifier_fine','classifier_fine_gain']
LABELS=['Frozen','Gain','Spatial','Spatial + gain','Fine-tuning','Fine-tuning + gain']

def boot(a):
    a=np.array(a);rng=np.random.default_rng(2026);values=a[rng.integers(0,4,(10000,4))].mean(1)
    return [float(x) for x in np.quantile(values,[.025,.975])]
def main():
    assert json.loads((ROOT/'verification.json').read_text())['passed']
    records=json.loads((ROOT/'results.json').read_text());rows=records['results'];assert len(rows)==3840
    keys=[(z['participant'],z['source_session'],z['target_session'],z['gesture'],z['method']) for z in rows];assert len(set(keys))==len(keys)
    base={k[:-1]:z for k,z in zip(keys,rows) if z['method']=='frozen'}
    manifest=json.loads((ROOT/'manifest.json').read_text());truth={}
    for person in range(2,6):
        for session in range(1,6):truth[(person,session)]=np.array([e['label'] for e in manifest if e['participant']==person and e['session']==session and e['rep']>0])
    for z in rows:
        key=(z['participant'],z['source_session'],z['target_session'],z['gesture']);f=base[key];label=r.RAW_GESTURES.index(z['gesture']);t=truth[(z['participant'],z['target_session'])];v=np.array(z['metrics']['trial_predictions'])
        z['seen_trial_accuracy']=float(np.mean(v[t==label]==t[t==label]));z['negative_transfer']=z['metrics']['trial_accuracy']<f['metrics']['trial_accuracy'];z['unseen_delta']=z['metrics']['unseen_trial_accuracy']-f['metrics']['unseen_trial_accuracy'];z['high_calibration_fit']=z['calibration_frame_accuracy']>=.99
    stats={};participants={};flat=[]
    for method in METHODS:
        a=[z for z in rows if z['method']==method];assert len(a)==640
        pa=[]
        for person in range(2,6):
            p=[z for z in a if z['participant']==person];assert len(p)==160
            high=[z for z in p if z['high_calibration_fit']]
            summary=dict(participant=person,method=method,overall=float(np.mean([z['metrics']['trial_accuracy'] for z in p])),unseen=float(np.mean([z['metrics']['unseen_trial_accuracy'] for z in p])),seen=float(np.mean([z['seen_trial_accuracy'] for z in p])),calibration_fit=float(np.mean([z['calibration_frame_accuracy'] for z in p])),negative_transfer=float(np.mean([z['negative_transfer'] for z in p])),unseen_delta=float(np.mean([z['unseen_delta'] for z in p])),high_fit_cases=len(high),high_fit_negative=sum(z['negative_transfer'] for z in high))
            pa.append(summary);flat.append(summary)
        participants[method]=pa
        stats[method]={field:dict(mean=float(np.mean([p[field] for p in pa])),participant_bootstrap_95=boot([p[field] for p in pa])) for field in ['overall','unseen','seen','calibration_fit','negative_transfer','unseen_delta']}
        high=[z for z in a if z['high_calibration_fit']];stats[method]['high_fit_cases']=len(high);stats[method]['high_fit_negative']=sum(z['negative_transfer'] for z in high);stats[method]['high_fit_negative_rate']=float(np.mean([z['negative_transfer'] for z in high])) if high else None
    contrasts={}
    for name,terms in {'gain_vs_frozen':{'gain_only':1,'frozen':-1},'spatial_added_to_gain':{'spatial_gain':1,'gain_only':-1},'spatial_vs_frozen':{'spatial_only':1,'frozen':-1},'interaction':{'spatial_gain':1,'spatial_only':-1,'gain_only':-1,'frozen':1}}.items():
        vals=[sum(weight*participants[m][i]['unseen'] for m,weight in terms.items()) for i in range(4)];contrasts[name]=dict(participant_effects=vals,mean=float(np.mean(vals)),participant_bootstrap_95=boot(vals))
    refs=[]
    for person in range(2,6):
        files=list((ROOT/f'subject{person}').glob('s*_to_s*/allclass_reference/results.json'));assert len(files)==20
        refs.append(float(np.mean([json.loads(p.read_text())['metrics']['trial_accuracy'] for p in files])))
    summary=dict(methods=stats,participants=participants,contrasts=contrasts,allclass_reference=dict(participant_overall=refs,mean=float(np.mean(refs)),participant_bootstrap_95=boot(refs)),run_wall_seconds=records['wall_seconds'],warning='Four independent people. Repeated pairs/gestures/windows are not independent; bootstrap descriptive. No score-selectedmethod.')
    r.dump(ROOT/'analysis.json',summary)
    with (ROOT/'participant_summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
    casefields=['participant','source_session','target_session','gesture','method','overall','unseen','seen','calibration_fit','negative_transfer','unseen_delta']
    with (ROOT/'case_summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=casefields);w.writeheader()
        for z in rows:w.writerow(dict(participant=z['participant'],source_session=z['source_session'],target_session=z['target_session'],gesture=z['gesture'],method=z['method'],overall=z['metrics']['trial_accuracy'],unseen=z['metrics']['unseen_trial_accuracy'],seen=z['seen_trial_accuracy'],calibration_fit=z['calibration_frame_accuracy'],negative_transfer=z['negative_transfer'],unseen_delta=z['unseen_delta']))
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(13,9),layout='constrained');ax=axes[0,0];xx=np.arange(6)
    for offset,field,color,name in [(-.18,'overall','#346c9d','All gestures'),(.18,'unseen','#de8741','Other seven gestures')]:
        vals=np.array([stats[m][field]['mean'] for m in METHODS])*100;cis=np.array([stats[m][field]['participant_bootstrap_95'] for m in METHODS])*100
        ax.bar(xx+offset,vals,.34,color=color,label=name);ax.errorbar(xx+offset,vals,yerr=np.array([vals-cis[:,0],cis[:,1]-vals]),fmt='none',ecolor='#333',capsize=2,lw=1)
    ax.set_xticks(xx,LABELS,rotation=30,ha='right');ax.set_ylabel('Trial accuracy (%)');ax.set_ylim(0,100);ax.legend(frameon=False);ax.set_title('A. Transfer to gestures absent from calibration',loc='left')
    ax=axes[0,1]
    contrasts_show=['gain_vs_frozen','spatial_vs_frozen','spatial_added_to_gain']
    for i,name in enumerate(contrasts_show):
        vals=np.array(contrasts[name]['participant_effects'])*100;ax.scatter(np.arange(4)*.04+i-.06,vals,s=45,color=['#346c9d','#de8741','#669966','#9d619b']);ax.plot([i-.16,i+.16],[vals.mean()]*2,color='black',lw=2)
    ax.axhline(0,color='gray',ls='--',lw=1);ax.set_xticks(range(3),['Gain − frozen','Spatial − frozen','Spatial + gain\n− gain']);ax.set_ylabel('Other-gesture difference (percentage points)');ax.set_title('B. Each dot is one reserved participant',loc='left')
    ax=axes[1,0];mat=np.array([[np.mean([z['unseen_delta'] for z in rows if z['participant']==person and z['method']=='gain_only' and z['gesture']==g])*100 for g in r.RAW_GESTURES] for person in range(2,6)])
    bound=max(1,float(np.abs(mat).max()));im=ax.imshow(mat,cmap='RdBu',vmin=-bound,vmax=bound,aspect='auto');ax.set_xticks(range(8),r.RAW_GESTURES);ax.set_yticks(range(4),[f'Participant {p}' for p in range(2,6)]);ax.set_xlabel('Calibration gesture (provider ID)');ax.set_title('C. Gain correction: other-gesture change vs frozen',loc='left')
    for i in range(4):
        for j in range(8):ax.text(j,i,f'{mat[i,j]:+.1f}',ha='center',va='center',color='white' if abs(mat[i,j])>.55*bound else 'black',fontsize=9)
    fig.colorbar(im,ax=ax,label='Percentage points',shrink=.8)
    ax=axes[1,1];methods=METHODS[1:];values=[stats[m]['high_fit_negative_rate']*100 for m in methods]
    ax.bar(np.arange(5),values,color='#806583');ax.set_xticks(range(5),LABELS[1:],rotation=30,ha='right');ax.set_ylim(0,100);ax.set_ylabel('Cases below frozen overall accuracy (%)');ax.set_title('D. Negative transfer despite ≥99% calibration fit',loc='left')
    for i,m in enumerate(methods):ax.text(i,values[i]+1,f"{stats[m]['high_fit_negative']}/{stats[m]['high_fit_cases']}",ha='center',fontsize=9)
    fig.suptitle('Single-gesture CSL-HDEMG recalibration | 4 reserved participants · 80 session pairs · 640 choices',fontsize=13)
    fig.savefig(ROOT/'component_results.png',dpi=180);fig.savefig(ROOT/'component_results.svg');plt.close(fig)
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
