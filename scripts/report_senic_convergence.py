"""Append the separately labeled optimization diagnostic to the main report."""
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    ap=argparse.ArgumentParser();ap.add_argument('run',type=Path);a=ap.parse_args();r=a.run
    read=lambda f:json.loads((r/f).read_text());c=read('config.json');m=read('metrics.json');t=read('training.json');s=read('summary.json');audit=read('independent_audit.json')
    assert audit['passed'];origin=Path(c['neural']);old=json.loads((origin/'metrics.json').read_text())
    models={x['model']:x for x in read('model_access.json')};people=[0,1,14,18,24,25]
    groups=[];rng=np.random.default_rng(42);index=rng.integers(0,6,(10000,6))
    lines=['## Additional optimization check (post-hoc)','',
        'The initial plainCNN fine-tuning runs often failed to fit their own calibration data at150steps, while scratch models fit well. After inspecting those fitting diagnostics and initial scores, all360plainCNN fine-target/fine-pooled/scratch-target cases were rerun to600steps. Same initialization/source model, optimizer, minibatch sequence, normalization and trials. Every original150/300step checkpoint was reproduced bit-for-bit before extending training. No case or checkpoint was chosen for its test performance. This is an explicitly post-hoc optimization diagnostic, not a new prespecified winning experiment.', '',
        'The augmented models retain their original fixed-step results; they were not retrained in this check. Therefore,600step plain models must not be used to claim an equal-compute advantage against150/300step augmented models.','',
        '| Strategy | Trials | Initial accuracy | 600step accuracy | Change, points | Mean fit accuracy | Fits below95% |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for method in ['fine_target','fine_pooled','scratch_target']:
        for budget in (7,14):
            new=[x for x in m if x['method']==method and x['trials']==budget];initial=[x for x in old if x['family']=='cnn' and x['method']==method and x['trials']==budget]
            part=[];diff=[]
            for p in people:
                acc=float(np.mean([x['accuracy'] for x in new if x['subject']==p]));before=float(np.mean([x['accuracy'] for x in initial if x['subject']==p]))
                part.append(dict(subject=p,accuracy=acc,initial_accuracy=before));diff.append(acc-before)
            tt=[x for x in t if models[x['model']]['method']==method and models[x['model']]['trials']==budget]
            group=dict(method=method,trials=budget,participants=part,accuracy=float(np.mean([p['accuracy'] for p in part])),
                initial_accuracy=float(np.mean([p['initial_accuracy'] for p in part])),mean_fitting_accuracy=float(np.mean([x['training_accuracy'] for x in tt])),
                fits_below95=sum(x['training_accuracy']<.95 for x in tt),mean_gain=float(np.mean(diff)),gain_ci=np.quantile(np.array(diff)[index].mean(1),[.025,.975]).tolist())
            groups.append(group)
            lines.append(f"| {method} | {budget} | {100*group['initial_accuracy']:.2f}% | {100*group['accuracy']:.2f}% | {100*group['mean_gain']:+.2f} | {100*group['mean_fitting_accuracy']:.2f}% | {group['fits_below95']}/60 |")
    lines+=['',f"Audit passed:360extended checkpoints and training accuracies,360exact original optimization prefixes,37,800saved scoring predictions. Run time {s['wall_seconds']:.2f}seconds ({s['wall_seconds']/60:.2f}minutes); audit {audit['wall_seconds']:.2f}seconds. Part of the run overlapped the main audit/reporting on this machine; timings are observed shared-machine wall time.", '',
        'The classical TDAR+RMS references remain80.19% at7trials and88.00% at14trials. Compare those with the whole table, not only the initially underoptimized CNN. Remaining fitting shortfalls or poor generalization must be disclosed. Longer fitting alone does not establish a distinct paper contribution.','',
        'Diagnostic outputs: '+str(r.resolve())+'.','']
    report='\n'.join(lines);(r/'report.md').write_text(report);(r/'aggregate_results.json').write_text(json.dumps(groups,indent=2)+'\n')
    for p in [origin/'report.md',Path('research/SENIC_NEURAL_RESULTS.md')]:
        base=p.read_text().split('\n## Additional optimization check (post-hoc)')[0]
        marker='## Current interpretation'
        if marker not in base:
            interpretation=('## Current interpretation\n\n'
                'The amplitude-inclusive classical baseline remains strongest among the tested configurations: 80.19% at seven trials and 88.00% at fourteen. After 600 steps, plain CNN target fine-tuning reaches 69.56%/79.81%, and target-only scratch training reaches 70.68%/81.38%. Every extended fit exceeds 95% training accuracy, so the original fine-tuning shortfall was partly an optimization issue, but longer training did not close the held-out gap. These are six-person development results, not a universal ranking.\n\n'
                'At the original matched step counts, nominal ±45-degree augmentation improves aggregate frozen accuracy from 27.27% to 33.83%, but reduces calibrated accuracy and fits its own data poorly in many cases. Its superiority or failure is unresolved until a matched longer-training control; this implementation is an adaptation of the published principle. The initial figure calibration_comparison.png uses 150/300-step results only.\n\n'
                'Next: one bounded matched 600-step augmentation control on these development participants, retaining the current plain-CNN and LDA references. Use fitting diagnostics to assess adequacy and preserve all outcomes. Final cohort signals remain unscored. This is a calibration benchmark in development; a distinct publication contribution is not yet established.\n\n')
            base=base.replace('## Methods and scope',interpretation+'## Methods and scope')
        p.write_text(base.rstrip()+'\n\n'+report)
    print(json.dumps(groups,indent=2))


if __name__=='__main__':main()
