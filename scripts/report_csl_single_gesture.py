"""Summarize verified one-class results without selecting a calibration gesture."""
from pathlib import Path
from datetime import datetime,timezone
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts import csl_sal_smoke as core

OUT=core.ROOT/'research/runs/20260907_csl_single_gesture_v1'
LABELS={'sal_lbn':'SAL + baseline','lbn_only':'Baseline only','classifier_finetune':'Classifier fine-tuning','classifier_frozen_bn':'Fine-tuning, fixed BN stats'}


def main():
    results=json.loads((OUT/'results.json').read_text());verify=json.loads((OUT/'verification.json').read_text());assert verify['passed']
    records=json.loads((OUT/'models.json').read_text());summary=results['summary']
    baseline=records[0]['frozen_trial_accuracy'];now=datetime.now(timezone.utc).isoformat()
    rows=[]
    for method,s in summary.items():
        rows.append(f"| {LABELS[method]} | {100*s['trial_accuracy']:.2f}% | {100*s['unseen_trial_accuracy']:.2f}% | {100*s['seen_trial_accuracy']:.2f}% | {s['beats_frozen']}/26 |")
    fig,ax=plt.subplots(figsize=(8.6,4.6));colors=['#2166ac','#5e7c6e','#b65f3c','#825eaa']
    for i,(method,s) in enumerate(summary.items()):
        values=np.array([r['trial_accuracy'] for r in records if r['method']==method])*100
        ax.scatter(i+np.linspace(-.13,.13,len(values)),values,color=colors[i],alpha=.6,s=23)
        ax.plot([i-.21,i+.21],[values.mean()]*2,color=colors[i],lw=3)
    ax.axhline(baseline*100,color='#333333',ls='--',lw=1.2,label=f'No adaptation: {baseline*100:.1f}%')
    ax.set_xticks(range(4),[label + f'\n(mean {100*summary[method]["trial_accuracy"]:.1f}%)' for label,method in zip(['SAL +\nbaseline','Baseline\nonly','Classifier\nfine-tuning','Fine-tuning\nfixed BN stats'],summary)])
    ax.set_ylim(0,max(r['trial_accuracy']*100 for r in records)+7);ax.set_ylabel('Held-out trial accuracy (%)')
    ax.set_title('One calibration gesture: all 26 choices',loc='left',fontweight='bold',pad=15)
    ax.legend(loc='upper right',frameon=False);ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.18)
    fig.text(.08,.025,'Each dot is a calibration gesture choice on the same participant/session pair—not an independent person.',fontsize=8)
    fig.tight_layout(rect=(0,.06,1,1));fig.savefig(OUT/'one_gesture_results.png',dpi=170);plt.close(fig)
    fitmin=min(r['fit_frame_accuracy'] for r in records)
    text=f'''# One-gesture calibration: original SAL development screen

Updated {now}. Per-run configurations and source hashes identify the evaluated implementation.

This directly tests calibration from a single gesture rather than all 26. It is a fixed development screen of the original audited model, **not execution of the later author's eight-class, gain-corrected, bounded-transform and prototype-search recipe**. See CSL_SINGLE_GESTURE_CODE_AUDIT.md for the source differences.

## Results

Results below average all 26 possible calibration gestures equally on the same participant 1, session 1 → 2 split. Every model scores the exact same 234 held-out trials. For each choice, 225 scoring trials belong to the other 25 gesture classes absent from calibration. No calibration gesture was selected using these results.

| Method | All gestures | Other 25 gestures | Calibration gesture class | Choices beating no adaptation |
|---|---:|---:|---:|---:|
| No adaptation | {100*baseline:.2f}% | {100*np.mean([r['frozen_unseen_accuracy'] for r in records]):.2f}% | {100*baseline:.2f}% | — |
{chr(10).join(rows)}

The no-adaptation calibration-class and other-class averages also equal its overall accuracy because all 26 class choices are included and each class has nine scoring trials. These are class-conditioned held-out results, not scores on the fitted calibration trial.

![Distribution across calibration choices](runs/20260907_csl_single_gesture_v1/one_gesture_results.png)

SAL+baseline's average change from no adaptation is {100*(summary['sal_lbn']['trial_accuracy']-baseline):+.2f} percentage points. Its range across choices is {100*summary['sal_lbn']['trial_min']:.2f}% to {100*summary['sal_lbn']['trial_max']:.2f}%. That range describes sensitivity; the best choice is not a deployable selection rule. Classifier fine-tuning can fit the provided class while degrading predictions on unseen classes. Freezing BN running statistics is included to check that ordinary target-only normalization updates are not the sole explanation.

## What this establishes

This screen evaluates the actual one-class question, unlike the previous all-class result. The tested update must outperform doing nothing on the full class set to be useful; beating a fine-tuning model that forgets other classes is insufficient. Fitting is not the execution blocker: the minimum calibration-frame fitting accuracy across all 104 models is {100*fitmin:.2f}%. These models may still differ in optimization/generalization behavior; this is not proof that every possible one-class method fails.

Results cover one development participant, one session pair, one calibration repetition and one seed. Calibration-choice variability is not participant-level replication. We do not claim statistical generalization, a clinical benefit, a new method, or reproduction of the later author results. Changing the pipeline, constraints, initialization or calibration budget could change the outcome; none were selected from these scores.

## Fixed protocol and validation

Reuse the prior 26-class source checkpoint, provider-correct channel map, amplitude-bearing RMS frames and whole-trial split. Calibrate using repetition 0 from one class; repetitions 0 of all unselected classes remain unused, while repetitions 1–9 score every class. Each method gets the same trial and random minibatch stream. Each takes 500 Adam updates at base learning rate .05, summed cross-entropy, batch limit 1024, dropout .5 and 250-step warm-up, seed42. This fixed-update diagnostic differs from the earlier20epoch full-budget experiment; do not interpret the two as differing only in calibration data.

SAL+LBN updates seven affine parameters and 168 baselines, with classifier and BN statistics fixed. LBNonly updates only those168baselines. Classifier variants update the classifier and BN affine parameters; only the ordinary classifier variant also updates BN running statistics. Mask tests and state comparisons cover these distinctions. All conditions use the last fixed checkpoint, with no held-out early stopping or parameter selection.

Labeled calibration is one3second recording. The inherited segmentation also consumes90seconds of target rest, so this is not a3second total-setup claim. No new raw data or reserved participants were decoded.

The run completed104models and saved {results['predictions']:,} scoring predictions in {results['wall_seconds']:.3f}seconds. Separate verification reloaded every checkpoint, reproduced all saved predictions exactly, recomputed fitting and overall/seen/unseen voting, checked every permitted-state mask and recomputed aggregate means. Verification took {verify['wall_seconds']:.3f}seconds. No failed case was omitted.

## Artifacts and next action

Run: `research/runs/20260907_csl_single_gesture_v1/`.
Configuration/config timestamp:config.json; allocations:calibration_allocations.json; all individual losses/fits/metrics:models.json; aggregate:results.json; independent audit:verification.json; model states and prediction files named per gesture/method; code snapshots and hashes; author source snapshot and sha256.json. Logs are alongside the run directory.

The next evidence-backed step is a separate, faithful implementation check of the later author's bounded/prototype/gain-corrected single-gesture recipe on the same permitted data, with its own matched controls and explicit code/data corrections. Do not treat this simpler negative or positive screen as a verdict on that recipe. Do not start a new architecture search or a reserved-participant evaluation. No author contact is needed for the currently accessible source.
'''
    (core.ROOT/'research/CSL_SINGLE_GESTURE_RESULTS.md').write_text(text)
    print('Report and figure saved.')

if __name__=='__main__':main()
