"""Matched contrasts for prespecified implementation and search-seed controls."""
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts import csl_author_recipe as r
from scripts.report_csl_components import boot
P=r.ROOT/'research/runs/20260907_csl_components_confirmatory'
O=r.ROOT/'research/runs/20260907_csl_input_order'
I=r.ROOT/'research/runs/20260907_csl_identity_ties'
S=r.ROOT/'research/runs/20260907_csl_search_seed_sensitivity'

def key(z):return (z['participant'],z['source_session'],z['target_session'],z['gesture'],z['method'])
def checkpoint(root,z):return root/f"subject{z['participant']}/s{z['source_session']}_to_s{z['target_session']}/g{z['gesture']:02}/{z['method']}.pt"

def main():
    for path in (P,O,I,S):assert json.loads((path/'verification.json').read_text())['passed']
    primary=json.loads((P/'results.json').read_text())['results'];lookup={key(z):z for z in primary};results={};unchanged=0;identity_checks=0
    for name,path in [('resample_then_gain',O),('identity_tie',I)]:
        rows=json.loads((path/'results.json').read_text())['results'];a=[z for z in rows if z['method']!='frozen'];assert len(a)==1280
        if name=='identity_tie':
            for z in a:
                old=lookup[key(z)];assert z['search']['original_index']==old['search']['index'];assert z['search']['loss']==old['search']['loss'];identity_checks+=1
                if not z['search']['identity_preferred']:
                    left=torch.load(checkpoint(P,z),map_location='cpu',weights_only=True);right=torch.load(checkpoint(I,z),map_location='cpu',weights_only=True);assert left.keys()==right.keys()
                    for k in left:assert torch.equal(left[k],right[k]),(key(z),k)
                    assert z['metrics']['trial_predictions']==old['metrics']['trial_predictions'];unchanged+=1
        results[name]={}
        for method in ['spatial_only','spatial_gain']:
            selected=[z for z in a if z['method']==method];participants=[]
            for person in [2,3,4,5]:
                z=[v for v in selected if v['participant']==person];assert len(z)==160
                old=[lookup[key(v)] for v in z];frozen=[lookup[(*key(v)[:-1],'frozen')] for v in z]
                p=dict(participant=person,overall=float(np.mean([v['metrics']['trial_accuracy'] for v in z])),unseen=float(np.mean([v['metrics']['unseen_trial_accuracy'] for v in z])),delta_from_original=float(np.mean([v['metrics']['unseen_trial_accuracy']-q['metrics']['unseen_trial_accuracy'] for v,q in zip(z,old)])),delta_from_frozen=float(np.mean([v['metrics']['unseen_trial_accuracy']-q['metrics']['unseen_trial_accuracy'] for v,q in zip(z,frozen)])),negative_transfer=float(np.mean([v['metrics']['trial_accuracy']<q['metrics']['trial_accuracy'] for v,q in zip(z,frozen)])))
                if name=='identity_tie':p['identity_preferred']=sum(v['search']['identity_preferred'] for v in z)
                participants.append(p)
            diffs=np.array([v['metrics']['unseen_trial_accuracy']-lookup[key(v)]['metrics']['unseen_trial_accuracy'] for v in selected])
            case_changes=dict(improved=int((diffs>0).sum()),harmed=int((diffs<0).sum()),tied=int((diffs==0).sum()),denominator=len(diffs),endpoint='other-class trial accuracy; descriptive repeated cases')
            results[name][method]=dict(case_changes_vs_original=case_changes,participants=participants,**{field:dict(mean=float(np.mean([p[field] for p in participants])),participant_bootstrap_95=boot([p[field] for p in participants])) for field in ['overall','unseen','delta_from_original','delta_from_frozen','negative_transfer']})
            if name=='identity_tie':results[name][method]['identity_preferred_cases']=sum(p['identity_preferred'] for p in participants)
    seeds=json.loads((S/'results.json').read_text())['results'];seedsummary={};ranges={}
    for method in ['spatial_only','spatial_gain']:
        seedsummary[method]={};bycase={}
        for seed in [42,43,44]:
            rows=[z for z in primary if z['source_session']==1 and z['target_session']==2 and z['method']==method] if seed==42 else [z for z in seeds if z['seed']==seed and z['method']==method]
            assert len(rows)==32
            part=[float(np.mean([z['metrics']['unseen_trial_accuracy'] for z in rows if z['participant']==person])) for person in [2,3,4,5]]
            seedsummary[method][str(seed)]=dict(participant_unseen=part,mean=float(np.mean(part)))
            for z in rows:bycase.setdefault(key(z),[]).append(z['metrics']['unseen_trial_accuracy'])
        spread=np.array([max(v)-min(v) for v in bycase.values()]);ranges[method]=dict(cases=len(spread),mean_range=float(spread.mean()),median_range=float(np.median(spread)),max_range=float(spread.max()),range_over10pp=int((spread>.1).sum()))
    fixed_frozen=[z for z in primary if z['method']=='frozen' and z['source_session']==1 and z['target_session']==2]
    frozen_part=[float(np.mean([z['metrics']['unseen_trial_accuracy'] for z in fixed_frozen if z['participant']==person])) for person in [2,3,4,5]]
    summary=dict(search_pair_frozen=dict(participant_unseen=frozen_part,mean=float(np.mean(frozen_part))),controls=results,search_seed_sensitivity=seedsummary,search_seed_ranges=ranges,identity_intervention_audit=dict(passed=True,initial_loss_equal_cases=identity_checks,nonintervened_checkpoints_identical=unchanged),scope='Four independentparticipants; all80pairs fororderandties, onlypair1→2 forsearchseeds. All controls fixedbeforeinspectingcohortsummary; no winnerselection.')
    r.dump(P/'controls_analysis.json',summary)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,3,figsize=(14,5.4),layout='constrained');colors=['#346c9d','#de8741','#669966','#9d619b']
    for ax,name,title in [(axes[0],'identity_tie','A. Prefer identity in exact loss ties'),(axes[1],'resample_then_gain','B. Change resampling / gain order')]:
        for i,method in enumerate(['spatial_only','spatial_gain']):
            vals=np.array([p['delta_from_original'] for p in results[name][method]['participants']])*100;ax.scatter(i+np.arange(4)*.04-.06,vals,c=colors,s=42);ax.plot([i-.15,i+.15],[vals.mean()]*2,color='black',lw=2)
        ax.axhline(0,color='gray',ls='--',lw=1);ax.set_xticks([0,1],['Spatial','Spatial + gain']);ax.set_title(title,loc='left');ax.set_ylabel('Other-gesture change from original (pp)')
    ax=axes[2]
    for method,color,label in [('spatial_only','#346c9d','Spatial'),('spatial_gain','#de8741','Spatial + gain')]:
        ax.plot([42,43,44],[seedsummary[method][str(s)]['mean']*100 for s in [42,43,44]],'o-',color=color,label=label)
    ax.axhline(np.mean(frozen_part)*100,color='#666666',ls='--',lw=1.4,label='Frozen, same pair')
    ax.set_xticks([42,43,44]);ax.set_xlabel('Spatial-search seed');ax.set_ylabel('Other-gesture accuracy (%)');ax.set_title('C. Fixed session pair 1→2',loc='left');ax.legend(frameon=False,loc='upper center',bbox_to_anchor=(.5,-.18),fontsize=9)
    fig.suptitle('Implementation controls | dots in A–B are participant means over all session pairs',fontsize=12);fig.savefig(P/'implementation_controls.png',dpi=180);fig.savefig(P/'implementation_controls.svg');plt.close(fig)
    print(json.dumps(summary,indent=2))
if __name__=='__main__':torch.set_num_threads(1);main()
