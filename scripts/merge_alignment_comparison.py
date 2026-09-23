"""Check CUDA/MPS bridge controls before combining descriptive development tables."""
import argparse
import json
from pathlib import Path
import numpy as np
from scripts.summarize_alignment_comparison import interval


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--original',type=Path,required=True)
    ap.add_argument('--small',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    a=json.loads((args.original/'results.json').read_text())
    b=json.loads((args.small/'results.json').read_text())
    assert (args.original/'complete.json').exists() and (args.small/'complete.json').exists()
    for key in ['seed','participants','sessions','choices','k','steps','inputs','learning_rate','batch_size']:
        assert a['config'][key]==b['config'][key],key
    assert json.loads((args.original/'access.json').read_text())==json.loads((args.small/'access.json').read_text())
    old={r['name']:r for r in a['records']}
    pa=np.load(args.original/'window_predictions.npz');pb=np.load(args.small/'window_predictions.npz')
    differences=[];windows=0;bridge=[]
    for r in b['records']:
        if r['method'] not in ['frozen','source_ce']:continue
        o=old[r['name']];bridge.append(r['name'])
        wd=int(np.count_nonzero(pa[r['name']]!=pb[r['name']]))
        windows+=wd
        if o['confusion']!=r['confusion']:
            differences.append(dict(name=r['name'],window_differences=wd,
                                    accuracy_difference=r['accuracy']-o['accuracy']))
    bridge_report=dict(identical_trial_metrics=not differences,bridge_evaluations=len(bridge),
                       compared_windows=len(bridge)*68*35,window_differences=windows,
                       changed_trial_confusions=differences,
                       original_device=a['config']['device'],original_torch=a['config']['torch'],
                       followup_device=b['config']['device'],followup_torch=b['config']['torch'])
    (args.out/'bridge_verification.json').write_text(json.dumps(bridge_report,indent=2)+'\n')
    if differences:
        raise ValueError('Trial metrics differ across bridge controls; report separately or rerun matched controls')
    records=a['records']+[r for r in b['records'] if r['method'].startswith('coral')]
    metrics=['accuracy','other_active_accuracy','calibrated_recall']
    people=a['config']['participants'];steps=a['config']['steps']
    methods=['target_ce','replay','source_ce']+[f'coral_{w}' for w in ['0.0001','0.001','0.01','0.1','1','10']]
    conditions=[('frozen',0)]+[(m,s) for m in methods for s in steps]
    summary=[];vectors={};contrasts=[]
    for k in [1,2]:
        for method,step in conditions:
            subset=[r for r in records if r['k']==k and r['method']==method and r['steps']==step]
            entry=dict(k=k,method=method,steps=step)
            for metric in metrics:
                vals=np.array([np.mean([r[metric] for r in subset if r['participant']==p]) for p in people])
                assert all(sum(r['participant']==p for r in subset)==8 for p in people)
                vectors[k,method,step,metric]=vals;entry[metric]=interval(vals)
            summary.append(entry)
        for method,step in conditions[1:]:
            for control in ['frozen','source_ce','replay']:
                if method==control:continue
                cs=0 if control=='frozen' else step
                contrasts.append(dict(k=k,method=method,steps=step,control=control,
                    **{metric:interval(vectors[k,method,step,metric]-vectors[k,control,cs,metric]) for metric in metrics}))
    result=dict(participants=people,distinct_evaluations=len(records),conditions=summary,contrasts=contrasts,
                scope='Adaptive development-only comparison; all six weights retained; no confirmatory hypothesis test.',
                bridge=bridge_report)
    (args.out/'combined_summary.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# Complete development comparison','',result['scope'],'',
           'Eight participants; one initialization seed; four fixed gesture choices and two target sessions.',
           'Additional smaller weights were chosen after the original loss diagnostics showed dominance of the alignment term.',
           'Original run: RunPod CUDA/PyTorch '+a['config']['torch']+'. Follow-up: local MPS/PyTorch '+b['config']['torch']+'.',
           f"All {len(bridge)} repeated frozen/source-only trial confusion matrices match; {windows} window labels differ across devices.",'',
           '| K | Method | Steps | Overall (%) | Omitted active (%) | Calibrated recall (%) |',
           '|---|---|---:|---:|---:|---:|']
    for r in summary:
        lines.append(f"| {r['k']} | {r['method']} | {r['steps']} | "+' | '.join(f"{r[m]['mean']:.2f}" for m in metrics)+' |')
    lines+=['','All settings are descriptive. Paired participant differences and unadjusted intervals are in combined_summary.json.',
            'These results adapt a published objective to this project’s backbone and data access; they do not reproduce the original Hyser/VGG accuracy.','']
    (args.out/'COMPARISON.md').write_text('\n'.join(lines))
    print(json.dumps({'bridge':bridge_report,'conditions':len(summary),'distinct_evaluations':len(records)}))


if __name__=='__main__':main()
