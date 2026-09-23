"""Participant-level descriptive summaries; all prescribed settings retained."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.stats import t


def interval(values):
    x=np.asarray(values,dtype=float)*100
    mean=float(x.mean())
    half=float(t.ppf(.975,len(x)-1)*x.std(ddof=1)/np.sqrt(len(x))) if len(x)>1 else None
    return dict(mean=mean,ci95=None if half is None else [mean-half,mean+half],participants=x.tolist())


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);args=ap.parse_args()
    data=json.loads((args.run/'results.json').read_text());cfg=data['config'];records=data['records']
    assert (args.run/'complete.json').exists()
    people=cfg['participants'];metrics=['accuracy','other_active_accuracy','calibrated_recall']
    conditions=[('frozen',0)]+[(m,s) for m in cfg['methods'] for s in cfg['steps']]
    vectors={};rows=[];contrasts=[]
    for k in [1,2]:
        for method,steps in conditions:
            subset=[r for r in records if r['k']==k and r['method']==method and r['steps']==steps]
            row=dict(k=k,method=method,steps=steps)
            for metric in metrics:
                values=[np.mean([r[metric] for r in subset if r['participant']==p]) for p in people]
                assert all(sum(r['participant']==p for r in subset)==len(cfg['sessions'])*len(cfg['choices']) for p in people)
                vectors[k,method,steps,metric]=np.array(values);row[metric]=interval(values)
            rows.append(row)
        for method,steps in conditions[1:]:
            for control in ['frozen','source_ce','replay']:
                if method==control:continue
                cs=0 if control=='frozen' else steps
                if (k,control,cs,metrics[0]) not in vectors:continue
                contrasts.append(dict(k=k,method=method,steps=steps,control=control,
                    **{metric:interval(vectors[k,method,steps,metric]-vectors[k,control,cs,metric]) for metric in metrics}))
    diagnostics=[]
    for method,steps in conditions[1:]:
        selected=[r['diagnostics'] for r in records if r['method']==method and r['steps']==steps]
        item=dict(method=method,steps=steps)
        for metric in ['source_ce','target_ce','coral_raw','update_l2','seconds_including_prior_checkpoint_scoring']:
            values=[r[metric] for r in selected if r[metric] is not None]
            if values:item[metric]=dict(min=float(np.min(values)),median=float(np.median(values)),max=float(np.max(values)))
        if method.startswith('coral'):
            values=[cfg['methods'][method]*r['coral_raw']/max(r['source_ce'],1e-30) for r in selected]
            item['weighted_alignment_to_source_ce_ratio']=dict(min=float(np.min(values)),median=float(np.median(values)),max=float(np.max(values)))
        diagnostics.append(item)
    result=dict(scope=cfg['scope'],participants=people,unit='percentage points; participant averages over sessions and choices',
                warning='Reused development data, one seed; descriptive intervals, no confirmatory testing or selected winner.',
                conditions=rows,contrasts=contrasts,diagnostics=diagnostics)
    (args.run/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# Published-objective development comparison','',result['warning'],'',
           'Means below first average the sessions and fixed gesture choices within each participant.',
           'This adapts the Yuan/Deep CORAL objective to our backbone and task; it is not a Hyser or author-code reproduction.','',
           '| K | Method | Steps | Overall (%) | Omitted active (%) | Calibrated recall (%) |',
           '|---|---|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['k']} | {r['method']} | {r['steps']} | "+' | '.join(f"{r[m]['mean']:.2f}" for m in metrics)+' |')
    lines += ['', '## Paired descriptive differences', '',
              '| K | Method | Steps | Control | Overall difference, pp (95% interval) | Omitted difference, pp |',
              '|---|---|---:|---|---:|---:|']
    for r in contrasts:
        if not r['method'].startswith('coral'):continue
        ci=r['accuracy']['ci95'];cistr='' if ci is None else f' ({ci[0]:.2f}, {ci[1]:.2f})'
        lines.append(f"| {r['k']} | {r['method']} | {r['steps']} | {r['control']} | {r['accuracy']['mean']:.2f}{cistr} | {r['other_active_accuracy']['mean']:.2f} |")
    lines += ['', 'All tested weights and checkpoints are shown. No final participants were accessed.',
              'Saved checkpoints and window predictions permit independent replay; verification files record its actual scope.','']
    (args.run/'SUMMARY.md').write_text('\n'.join(lines))
    print(json.dumps({'conditions':len(rows),'contrasts':len(contrasts),'participants':people}))


if __name__=='__main__':main()
