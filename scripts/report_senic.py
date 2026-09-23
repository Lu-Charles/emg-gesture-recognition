"""Describe the completed development pilot without selecting a winning gesture."""
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    ap=argparse.ArgumentParser();ap.add_argument('run',type=Path);a=ap.parse_args();r=a.run
    read=lambda name:json.loads((r/name).read_text())
    s=read('summary.json');m=read('metrics.json');audit=read('independent_audit.json')
    assert audit['passed']
    g={(x['feature'],x['method'],x['trials']):x for x in s['groups']}
    zero=g['tdar_rms','frozen',0];short=g['tdar_rms','target_only',7];full=g['tdar_rms','target_only',14]
    values=lambda group:np.array([p['accuracy'] for p in group['participants']])
    gain=values(short)-values(zero);difference=values(short)-values(full)
    rng=np.random.default_rng(42);index=rng.integers(0,6,(10000,6))
    ci=lambda x:np.quantile(x[index].mean(1),[.025,.975]).tolist()
    contrasts=dict(short_minus_frozen_accuracy=float(gain.mean()),short_minus_full_accuracy=float(difference.mean()),
        short_minus_frozen_participant_ci=ci(gain),short_minus_full_participant_ci=ci(difference),
        participants_gaining_at_least_10pp=int((gain>=.1).sum()),participants_within_3pp_of_full=int((difference>=-.03).sum()),
        recording_saving_fraction=1-short['recorded_seconds']/full['recorded_seconds'],
        provisional_screen_pass=bool(difference.mean()>=-.03 and (gain>=.1).sum()>=4),
        interval_scope='exploratory six-development-participant bootstrap, 10000 draws seed42; conditional on fixed positions/repetitions/methods')
    (r/'paired_contrasts.json').write_text(json.dumps(contrasts,indent=2)+'\n')
    lines=['# SeNic calibration-efficiency pilot','',
        'Development results only. The selected route is a focused public-data study of brief recalibration after real electrode rotation. No new algorithm, independent final result or publication readiness is established.','',
        '## Protocol','',
        'Six development participants: h0,h1,h14,h18,h24,h25; session0 only. Source position0 repetitions0,1 enroll the classifier; repetition2 scores every condition. Target calibration uses one labeled gesture (all seven choices averaged), one repetition of all seven gestures, or two repetitions. All method/budget comparisons use identical scoring files. Positions1–10 include designed rotations and two random positions. The other rotation participants remain reserved; fatigue participants are excluded.','',
        'Raw eight-channel Myo counts at200Hz; fixed3–5second interval; 250ms windows with125ms step,15windows/trial. Costs below include entire CSV durations, not just the2seconds analyzed. Three recordings were shorter than6seconds, so the initial run stopped before model fitting; all trials received the same shorter interval in v2. No selective trial exclusion.','',
        '## Main results','',
        'Equal participant weighting, averaged across ten target positions. One-gesture results average every gesture choice, not the best-performing gesture. Accuracy is per held-gesture window, not online control success.','',
        '| Features | Method | Calibration trials | Recorded seconds | Accuracy | Macro F1 |',
        '|---|---|---:|---:|---:|---:|']
    names={'frozen':'Frozen LDA','source_plus_target':'Source + target refit','target_only':'Target-only refit','spatial_registration':'RMS spatial registration'}
    for x in s['groups']:
        lines.append(f"| {x['feature']} | {names[x['method']]} | {x['trials']} | {x['recorded_seconds']:.1f} | {100*x['accuracy']:.2f}% | {x['macro_f1']:.4f} |")
    lines+=['','## Participant results, TDAR + RMS','',
        '| Participant | Frozen | 7 trials | 14 trials |','|---|---:|---:|---:|']
    for i,p in enumerate(short['participants']):
        lines.append(f"| h{p['subject']} | {100*values(zero)[i]:.2f}% | {100*values(short)[i]:.2f}% | {100*values(full)[i]:.2f}% |")
    lo,hi=100*np.array(contrasts['short_minus_full_participant_ci'])
    source=[x['accuracy'] for x in m if x['position']==0 and x['feature']=='tdar_rms']
    lines+=['','## Interpretation','',
        f"The unshifted held-out reference averages {100*np.mean(source):.2f}% accuracy. At target positions, seven-trial target-only recalibration improves accuracy by {100*gain.mean():.2f} percentage points over frozen, with at least10points improvement for {int((gain>=.1).sum())}/6people. The effect is large, but ordinary supervised refitting is an established method.", '',
        f"Seven trials save {100*contrasts['recording_saving_fraction']:.1f}% of the fourteen-trial recording duration but lose {abs(100*difference.mean()):.2f} accuracy points. The exploratory paired participant-bootstrap interval for seven minus fourteen is [{lo:.2f}, {hi:.2f}] points. The pre-outcome3point short-budget screen does not pass. No participant is within3points. Do not claim essentially unchanged performance at half the recording cost.", '',
        'Single-gesture spatial registration is substantially better than the frozen amplitude baseline and a one-gesture source-plus-target refit, but remains below all-gesture target-only retraining. This baseline is simple circular interpolation of amplitude features using labeled RMS templates; it is not a literal reproduction of Li/Xu and cannot establish superiority over published shift correction.', '',
        'As a planned descriptive regime check, small-rotation participants h24,h25 at positions1–8 are shown separately below. These are only two people, so their result is not a general small-shift conclusion. No significance test or model choice is based on this subgroup.', '',
        '| Small-rotation subgroup method | Accuracy |','|---|---:|']
    for kind,method,budget in [('amplitude','spatial_registration',1),('tdar_rms','frozen',0),('tdar_rms','target_only',7),('tdar_rms','target_only',14)]:
        mm=[x for x in m if x['subject'] in (24,25) and 1<=x['position']<=8 and (x['feature'],x['method'],x['trials'])==(kind,method,budget)]
        lines.append(f"| {kind}, {names[method]}, {budget} trials | {100*np.mean([x['accuracy'] for x in mm]):.2f}% |")
    lines+=['','## Validation and limitations','',
        f"Three risk tests pass. Independent audit verified {audit['raw_trials']} raw hashes with no duplicate files, {audit['allocations']} whole-trial allocations, training-only scalers for {audit['models']} saved models, and all {audit['prediction_rows']:,} saved prediction rows in {audit['prediction_files']} files. All15aggregate groups were recalculated. Reserved-participant signals and Charles recordings were not accessed.", '',
        f"Successful pilot wall time: {s['wall_seconds']:.2f}seconds; independent audit: {audit['wall_seconds']:.2f}seconds. These exclude implementation, downloads and the failed first extraction attempt. This is a classical feature/LDA experiment, not neural training.", '',
        'Only six development people and one scoring repetition per position. Multiple rotation positions do not create additional independent participants. Position order can confound angle with elapsed time/fatigue; approximately360-degree positions are physically near the original orientation. No cross-day conclusion from this run, no rest/unknown-gesture rejection task, no real-time prosthesis demonstration. Exact final cohort completeness remains to audit. Raw data redistribution rights are not established; provide retrieval instructions rather than copying raw data into a public package.', '',
        '## Next action','',
        'Complete a matched-budget compact CNN and applicable published rotation-correction/augmentation comparison on these same development splits. Preserve the14trial reference and adverse short-budget result. Continue this calibration-efficiency study without claiming a winning new method; freeze the final protocol before reserved-participant evaluation.', '',
        '## Provenance','',
        '- Dataset: https://github.com/BoZhuBo/SeNic at a4c12f7daab28a80d557677ae8dbcef0d7871ba2.',
        '- Dataset paper: https://doi.org/10.1109/TNSRE.2022.3173708.',
        '- Prior adaptive correction: https://doi.org/10.1109/JBHI.2020.3012698; synchronous gesture correction: https://pmc.ncbi.nlm.nih.gov/articles/PMC7070560/.',
        '- Audit metadata and download hashes: research/runs/20260907_sensor_shift_audit/.',
        '- Config, code snapshots, environment, trial/fit ledgers, features, models and predictions: '+str(r.resolve())+'.',
        '- Source snapshots and hashes preserve the evaluated implementation.','']
    report='\n'.join(lines);(r/'report.md').write_text(report)
    Path('research/SENIC_PILOT_RESULTS.md').write_text(report)
    print(json.dumps(contrasts,indent=2))


if __name__=='__main__':main()
