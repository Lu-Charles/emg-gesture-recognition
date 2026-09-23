"""Report every frozen neural comparison, including poor fits and adverse effects."""
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    ap=argparse.ArgumentParser();ap.add_argument('run',type=Path);a=ap.parse_args();r=a.run
    read=lambda f:json.loads((r/f).read_text())
    s=read('summary.json');m=read('metrics.json');c=read('config.json');audit=read('independent_audit.json')
    training=read('training.json');access=read('model_access.json');ledger={x['model']:x for x in access}
    prior=json.loads((Path(c['classical'])/'summary.json').read_text());assert audit['passed']
    groups={(g['family'],g['method'],g['trials']):g for g in s['groups']}
    rng=np.random.default_rng(42);indices=rng.integers(0,6,(10000,6))
    def contrast(left,right,label):
        assert [p['subject'] for p in left['participants']]==[p['subject'] for p in right['participants']]
        difference=np.array([p['accuracy'] for p in left['participants']])-np.array([p['accuracy'] for p in right['participants']])
        return dict(label=label,mean_accuracy_difference=float(difference.mean()),participant_differences=difference.tolist(),
                    participant_bootstrap_ci=np.quantile(difference[indices].mean(1),[.025,.975]).tolist(),
                    participants_improving=int((difference>0).sum()),participants_within_minus_3pp=int((difference>=-.03).sum()))
    contrasts=[]
    for fam in ('cnn','cnn_roll45'):
        for method in ('fine_target','fine_pooled','scratch_target'):
            contrasts.append(contrast(groups[fam,method,7],groups[fam,method,14],f'{fam}/{method}:7 minus14'))
    for method,budget in [('frozen',0)]+[(mm,b) for mm in ('fine_target','fine_pooled','scratch_target') for b in (7,14)]:
        contrasts.append(contrast(groups['cnn_roll45',method,budget],groups['cnn',method,budget],f'roll45 minus plain:{method}/{budget}'))
    (r/'paired_contrasts.json').write_text(json.dumps(contrasts,indent=2)+'\n')
    angles=json.loads(Path('research/runs/20260907_sensor_shift_audit/development_angles.json').read_text())
    strata={}
    for person,positions in angles.items():
        ref=np.array(positions['0'])
        for p,values in positions.items():
            d=np.deg2rad(np.array(values)-ref);deg=float(np.rad2deg(np.arctan2(np.sin(d).mean(),np.cos(d).mean())))
            name='source' if p=='0' else 'random_positions' if int(p)>=9 else 'within45' if abs(deg)<=45 else '45to90' if abs(deg)<=90 else 'over90'
            strata[int(person),int(p)]=(name,deg)
    angle_rows=[]
    for key,g in groups.items():
        for name in ('within45','45to90','over90','random_positions'):
            people=[]
            for person in c['development']:
                mm=[x for x in m if x['subject']==person and (x['family'],x['method'],x['trials'])==key and strata[person,x['position']][0]==name]
                if mm:people.append(dict(subject=person,positions=[x['position'] for x in mm],accuracy=float(np.mean([x['accuracy'] for x in mm]))))
            if people:angle_rows.append(dict(family=key[0],method=key[1],trials=key[2],stratum=name,participants=people,
                                            accuracy=float(np.mean([p['accuracy'] for p in people]))))
    (r/'angle_strata.json').write_text(json.dumps(dict(definition='circular mean of8channel angle changes relative to same-sessionposition0; absolute threshold45/90degrees; random9/10separate; participant equal weighting within stratum',
                    positions=[dict(subject=k[0],position=k[1],stratum=v[0],degrees=v[1]) for k,v in strata.items()],groups=angle_rows),indent=2)+'\n')
    convergence=[]
    for family in c['families']:
        for method in ('enrollment','fine_target','fine_pooled','scratch_target'):
            tt=[t for t in training if ledger[t['model']]['family']==family and ledger[t['model']]['method']==method]
            if not tt:continue
            convergence.append(dict(family=family,method=method,models=len(tt),mean_fitting_accuracy=float(np.mean([t['training_accuracy'] for t in tt])),
                minimum_fitting_accuracy=float(min(t['training_accuracy'] for t in tt)),below95=sum(t['training_accuracy']<.95 for t in tt),
                median_last_loss=float(np.median([t['losses'][-1] for t in tt]))))
    (r/'convergence_summary.json').write_text(json.dumps(convergence,indent=2)+'\n')
    lines=['# SeNic neural and rotation-augmentation comparison','',
        'Development evidence only: six people, session0, seed42. The study remains calibration efficiency after real electrode rotation. This stage tests stronger comparators; it does not introduce a new method or establish final publication readiness.','',
        '## Methods and scope','',
        'Same original enroll/calibration/scoring trials and3–5second signal interval as the classical pilot. The CNN retains amplitude using an explicit log1pMAV path alongside temporal convolutions. Fixed300steps for enrollment/scratch and150steps for fine-tuning, Adam.001, minibatch64. No scoring-based model selection. One scalar RMS is fitted on permitted unique training samples. Fine-tuning keeps source normalization; target-only scratch fits target normalization.', '',
        'Primary families: plainCNN and identicalCNN with circular channel permutations−1,0,+1 during training, approximating±45degrees. A separate zero-calibration sensitivity uses all8cyclic shifts. This follows the published ABSDA permutation principle, but changes the sensor, representation, backbone and training; it is not a reproduction of original HD-EMG scores. Physical ruler angles never enter a model. See SENIC_NEURAL_METHODS.md.', '',
        'This neural stage covers0/7/14trials. One-gesture cases remain classical-only. Fine-target reuses the source model but updates on target data; fine-pooled updates using source+target; scratch-target trains anew using target data. All methods have identical target recording budgets and scoring files.', '',
        '## Initial fixed-step target-position results','',
        'Equal participant weighting, then ten positions per participant. Recording time includes complete calibration files. Window accuracy is not real-time control success.','',
        '| Model | Training strategy | Trials | Recording seconds | Accuracy | Macro F1 |',
        '|---|---|---:|---:|---:|---:|']
    for g in prior['groups']:
        if g['feature']=='tdar_rms' and (g['method']=='frozen' or g['method']=='target_only'):
            lines.append(f"| TDAR+RMS LDA | {g['method']} | {g['trials']} | {g['recorded_seconds']:.1f} | {100*g['accuracy']:.2f}% | {g['macro_f1']:.4f} |")
    for g in s['groups']:lines.append(f"| {g['family']} | {g['method']} | {g['trials']} | {g['recorded_seconds']:.1f} | {100*g['accuracy']:.2f}% | {g['macro_f1']:.4f} |")
    lines+=['','## Paired comparisons','',
        'Exploratory participant bootstrap:10000draws, seed42; intervals condition on this development sample and the fixed scoring repetitions. No correction for multiple comparisons; do not use these as confirmatory tests. All specified comparisons are shown.','',
        '| Comparison | Accuracy difference, points | 95% interval, points | People improving |',
        '|---|---:|---|---:|']
    for x in contrasts:
        lo,hi=100*np.array(x['participant_bootstrap_ci'])
        lines.append(f"| {x['label']} | {100*x['mean_accuracy_difference']:+.2f} | [{lo:+.2f}, {hi:+.2f}] | {x['participants_improving']}/6 |")
    lines+=['','## Designed shifts within45degrees','',
        'Computed from circular mean changes in all8recorded channel angles relative to source position0. Positions9/10 are excluded here as the separate random regime. Near360-degree returns may have small net displacement and remain included. Numbers of positions differ between participants; each person has equal weight within a stratum. Full strata and position mappings are in angle_strata.json.','',
        '| Model | Strategy | Trials | Accuracy | Participants |','|---|---|---:|---:|---:|']
    for g in angle_rows:
        if g['stratum']=='within45':lines.append(f"| {g['family']} | {g['method']} | {g['trials']} | {100*g['accuracy']:.2f}% | {len(g['participants'])} |")
    lines+=['','## Fitting diagnostics','',
        'Accuracy here is on the unaugmented fitting windows, not held-out performance. Low values limit interpretation of that trained baseline; they are not evidence against the original published method. The final training loss for augmented models includes randomly shifted windows and need not agree with clean fitting accuracy.','',
        '| Model | Stage | Fits | Mean fit accuracy | Minimum | Fits below95% |','|---|---|---:|---:|---:|---:|']
    for x in convergence:lines.append(f"| {x['family']} | {x['method']} | {x['models']} | {100*x['mean_fitting_accuracy']:.2f}% | {100*x['minimum_fitting_accuracy']:.2f}% | {x['below95']} |")
    lines+=['','## Validation and runtime','',
        f"Six risk tests passed before training. Independent verification checked all {audit['raw_trials']} raw trials, {audit['allocations']} allocations, {audit['models']} saved checkpoints and fitting accuracies, {audit['prediction_files']} prediction files/{audit['prediction_rows']:,} prediction rows, and {audit['groups']}aggregate groups. Training-only normalization, source-history and target-budget ledgers were checked. No reserved-participant signal values or Charles-device signals were decoded or scored. Reserved archives were separately downloaded and listed for file-structure verification.", '',
        f"Training run wall time: {s['wall_seconds']:.2f}seconds ({s['wall_seconds']/60:.2f}minutes). Audit: {audit['wall_seconds']:.2f}seconds. Excludes implementation/literature review. CPU1thread; source/fine/scratch times are separately logged.", '',
        '## Interpretation limits','',
        'Same-day results only; physical rotation cannot establish cross-day robustness. Six independent development people; only one scoring repetition per position; overlapping windows do not increase participant count. Fixed position order can confound angle with time/fatigue. The ±45degree augmentation has intentionally limited synthetic support; poor results at larger angles do not refute its intended range. All8shift exposure is an explicitly wider sensitivity. Sparse8channel permutation approximates a real rotation and lacks the spatial resolution of the original64channel sensor.', '',
        'Preserve all outcomes, including inferior neural or augmentation results. Existing classical7trial accuracy80.19% versus14trial88.00% remains the reference. No method should be selected using reserved-subject results. The report tables describe the tested configurations; any next-stage selection and reason must be recorded separately before final evaluation.', '',
        '## Provenance','',
        '- Methods: research/SENIC_NEURAL_METHODS.md.',
        '- ABSDA source: https://unbscholar.dspace.lib.unb.ca/server/api/core/bitstreams/54407f52-28ec-4abf-95ef-33f55e98ff41/content.',
        '- SeNic: https://github.com/BoZhuBo/SeNic at a4c12f7daab28a80d557677ae8dbcef0d7871ba2.',
        '- Full configs, source snapshots, inputs, fitted-model ledger, predictions and audit: '+str(r.resolve())+'.',
        '- Source snapshots and hashes preserve the evaluated implementation.','']
    report='\n'.join(lines);(r/'report.md').write_text(report);Path('research/SENIC_NEURAL_RESULTS.md').write_text(report)
    print(json.dumps(dict(groups=len(s['groups']),contrasts=len(contrasts),stratum_groups=len(angle_rows),fitting_groups=len(convergence))))


if __name__=='__main__':main()
