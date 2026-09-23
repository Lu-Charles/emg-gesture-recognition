"""Readable development report and static research figure."""
import argparse
import csv
import json
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.grabmyo_corpus import hash_stream

LABELS={'fixed_85':'Fixed 85 s','fixed_170':'Fixed 170 s','fixed_255':'Fixed 255 s',
        'fixed':'Fixed budget selected in inner CV','pre_error':'Pre-update error rule',
        'post_entropy':'Post-update entropy rule','probability_change':'Prediction-change rule','ridge':'Learned stopping rule'}
COLORS={'fixed':'#576575','pre_error':'#CA6F1E','post_entropy':'#849324','probability_change':'#8B73AC','ridge':'#187E91'}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args(); args.out.mkdir(parents=True,exist_ok=False)
    validation=json.loads((args.run/'independent_validation.json').read_text())
    assert validation['status']=='passed'
    cfg=json.loads((args.run/'config.json').read_text())
    obs=Path(cfg['observations'])
    results=json.loads((args.run/'results.json').read_text())
    curves=json.loads((args.run/'curves.json').read_text())
    decisions=json.loads((args.run/'decisions.json').read_text())
    outcomes={(r['participant'],r['session'],r['budget']):r['macro_f1'] for r in json.loads((obs/'outcomes.json').read_text())}
    metric_rows=[]
    for d in decisions:
        p,s,b=d['participant'],d['session'],d['selected_budget']
        metric_rows.append({'family':d['family'],'participant':p,'session':s,'selected_seconds':85*b,
                            'macro_f1':outcomes[p,s,b],'full255_macro_f1':outcomes[p,s,3],
                            'loss_vs_full':outcomes[p,s,3]-outcomes[p,s,b]})
    for name,rows in (('summary',results),('per_session',metric_rows),('curves',curves)):
        fields=list(dict.fromkeys(k for r in rows for k in r))
        with (args.out/f'{name}.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    ridge=next(r for r in results if r['family']=='ridge')
    simple=next(r for r in results if r['family']=='pre_error')
    pairs=sorted({(r['participant'],r['session']) for r in decisions})
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,(ax,bx)=plt.subplots(1,2,figsize=(13.8,5.4),gridspec_kw={'width_ratios':[1.1,1]})
    fixed=[r for r in results if r['family'].startswith('fixed_')]
    ax.plot([r['mean_seconds'] for r in fixed],[r['mean_macro_f1'] for r in fixed],'o--',color=COLORS['fixed'],label='Fixed routines / expected mixtures')
    for family in ('pre_error','ridge'):
        rr=sorted([r for r in curves if r['family']==family],key=lambda r:r['mean_seconds'])
        ax.plot([r['mean_seconds'] for r in rr],[r['mean_macro_f1'] for r in rr],color=COLORS[family],alpha=.3,lw=1.5)
        r=next(r for r in results if r['family']==family)
        ax.scatter(r['mean_seconds'],r['mean_macro_f1'],s=95,color=COLORS[family],marker='D' if family=='ridge' else 'o',label=LABELS[family]+' (nested CV)',zorder=4)
    ax.set(xlabel='Mean new-day recording (seconds)',ylabel='Mean macro-F1',title='Calibration cost and recognition quality',xlim=(75,265),ylim=(.82,.87))
    ax.grid(alpha=.2);ax.legend(fontsize=8,loc='lower right')
    for offset,family in ((-.12,'pre_error'),(.12,'ridge')):
        losses=[next(r['loss_vs_full'] for r in metric_rows if r['family']==family and (r['participant'],r['session'])==pair)*100 for pair in pairs]
        bx.scatter(np.arange(len(pairs))+offset,losses,color=COLORS[family],label=LABELS[family],s=32)
    bx.axhline(0,color='#576575',lw=1);bx.axhline(2,color='#A54848',lw=1,ls='--',label='2-point loss reference')
    bx.set_xticks(range(len(pairs)),[f'{p}/{s}' for p,s in pairs],rotation=60,ha='right')
    bx.set(xlabel='Participant / recording session',ylabel='Macro-F1 loss vs full calibration (points)',title='Every development session, including harm')
    bx.legend(fontsize=8,loc='upper right');bx.grid(axis='y',alpha=.2)
    fig.suptitle('Adaptive calibration duration — first development pilot',fontsize=15,x=.04,ha='left')
    fig.text(.04,.01,'8 development participants; original trial order; one encoder seed. Faint curves show unselected operating points. Final evaluation remains untouched.',fontsize=9,color='#46515D')
    fig.tight_layout(rect=[0,.055,1,.94])
    for extension in ('png','svg'):
        fig.savefig(args.out/f'calibration_stopping.{extension}',dpi=180)
    plt.close(fig)
    differences=[]
    for p,s in pairs:
        choices={f:next(r['selected_seconds'] for r in metric_rows if r['family']==f and r['participant']==p and r['session']==s) for f in ('ridge','pre_error')}
        if choices['ridge']!=choices['pre_error']:
            differences.append({'participant':p,'session':s,**choices})
    lines=['# Adaptive calibration duration: first development pilot','',
           'Completed '+json.loads((args.run/'summary.json').read_text())['completed_utc']+'. Public GRABMyo only. No final evaluation participants.','',
           f'The learned stopping rule uses **{ridge["mean_seconds"]:.1f} recorded seconds**, versus 255 seconds for the full routine (**{ridge["recording_reduction_fraction"]*100:.1f}% less new-day recording**). Mean macro-F1 is **{ridge["mean_macro_f1"]:.3f} versus {fixed[-1]["mean_macro_f1"]:.3f}**. This is a promising development result, not a confirmed population effect or a guarantee for each person.','',
           f'The simple pre-update error rule is almost as effective: **{simple["mean_seconds"]:.1f} seconds, macro-F1 {simple["mean_macro_f1"]:.3f}**. The learned and simple rules differ in only **{len(differences)} of 16 decisions**. A distinctive advantage for the learned predictor has not been established. Retain the simpler rule as a serious candidate.','',
           '## Complete comparisons','',
           '| Method | Mean recording, s | Macro-F1 | Mean loss vs 255 s | Worst loss | Sessions losing >0.02 |',
           '| --- | ---: | ---: | ---: | ---: | ---: |']
    for r in results:
        lines.append(f'| {LABELS[r["family"]]} | {r["mean_seconds"]:.1f} | {r["mean_macro_f1"]:.4f} | {r["mean_loss_vs_full"]:.4f} | {r["worst_loss_vs_full"]:.4f} | {round(r["fraction_loss_over_002"]*16)}/16 |')
    lines += ['',f'![Calibration cost and per-session outcomes]({(args.out/"calibration_stopping.png").resolve()})','',
              '## What was tested','',
              'The previous shared encoder and day-1 enrollment were reused. Each cumulative target budget starts from the enrolled model and receives the same ten-epoch full update. Saved checkpoints exactly implement this protocol; they are not a continuously trained optimizer trajectory. No recognizer retraining was necessary.','',
              'A policy must acquire the first complete round: one five-second trial for each of 17 labels. It may stop at 85 seconds, acquire another round and stop at 170, or continue to 255. It never receives observations from later rounds after stopping. Models and observations for all possible paths are computed offline for comparison; the decision trace records only the reached path.','',
              'Signals come only from the just-acquired round, evaluated before and after its update. The before-update model has not fitted those trials. Post-update error, entropy and changes are explicitly in-sample diagnostics. All queried labels and observations are included in the recording budget. No scoring windows or future calibration rounds enter policy inputs.','',
              'The learned rule is ridge regression on eight fixed features, predicting average remaining improvement per extra round through the full budget. Standardization and benefit-label fitting exclude the outer participant. Inner participant cross-validation chooses regularization and stopping thresholds. Simple rules use training-participant quantiles of pre-update error, post-update entropy or probability change. All methods use the same held-out person and scoring trials.','',
              'Each outer fold withholds both later sessions of one person; threshold/model selection runs on the other seven people with another participant split internally. Repeated states are not independent people. The inner selection target is mean macro-F1 loss <=0.01, then minimum recording cost. It is an empirical criterion, not an assurance of performance on an unseen person.','',
              '## Equal-cost comparison and failure cases','',
              f'At {ridge["mean_seconds"]:.1f} seconds, a label-independent mixture of adjacent fixed routines has expected macro-F1 {ridge["same_cost_fixed_mixture_f1"]:.4f}; the learned rule scores {ridge["mean_macro_f1"]:.4f}, a difference of {ridge["delta_vs_same_cost_mixture"]:.4f}. The descriptive participant bootstrap interval for this difference is [{ridge["descriptive_participant_bootstrap_delta_ci95"][0]:.4f}, {ridge["descriptive_participant_bootstrap_delta_ci95"][1]:.4f}]. Cost matching is retrospective; the mixture expectation is computed analytically. It is not an independently tuned deployed policy.','',
              'Bootstrap intervals resample eight people with their two sessions together. They condition on the already fitted cross-validation policies and do not capture all training, model-selection or research-design uncertainty. Earlier development analyses influenced the project choice. These remain development findings.','',
              'Both the learned rule and the pre-update error rule stop too early for participant 20/session 2: macro-F1 falls by 0.0252 relative to the full routine. The simple rule additionally loses 0.0224 for participant 3/session 3; the learned rule requests one more round there. Other rule families have worse individual losses despite favorable averages. No universal no-harm claim is supported.','',
              'All routines also rely on the existing 595 seconds of day-1 enrollment and population pretraining. The 35.4% saving applies only to new-day recorded calibration. Including one enrollment plus one later calibration, recorded time is about 759.7 versus 850 seconds, a 10.6% reduction. Rest, instructions and interaction overhead are unmeasured.','',
              '## Verification and runtime','',
              'Five tests passed: three new stopping tests and two existing participant/access tests. Readback verification checked 32 observation states using 64 CPU checkpoint evaluations, feature reproduction, input and model hashes, 40 outer policy fits/scalers/thresholds, 80 decision traces and all eight summary rows. Maximum saved/MPS versus CPU probability difference was '+f'{validation["max_cpu_vs_saved_probability_difference"]:.2g}'+'. The verifier does not repeat every inner hyperparameter fit.','',
              'Observation extraction including initial artifact checks took '+f'{json.loads((obs/"summary.json").read_text())["total_seconds"]:.2f}'+ ' seconds. Nested policy evaluation and bootstrap took '+f'{json.loads((args.run/"summary.json").read_text())["total_seconds"]:.2f}'+ ' seconds. These are measured command runtimes and exclude implementation, review and report preparation. Existing encoder training was reused.','',
              '## Decision and next experiment','',
              'The first pilot meets the declared aggregate feasibility screen. Continue studying adaptive duration, retaining the learned rule and simple pre-update error rule. Do not claim the learned component is necessary: their decisions are almost identical.','',
              'Next, test sensitivity to calibration-trial order using all six permutations of the existing three calibration ranks. Keep the same four scoring trials/class and participant allocation. Fit/tune with participant grouping across every order, not separate random splits of episodes. Include every permutation and keep the original result. This is a proposed robustness run, not yet performed. Check seed sensitivity and appropriate recognition baselines before a locked final evaluation.','',
              '## Exact artifacts','',f'- Observations, probabilities and access: `{obs.resolve()}`',f'- Policies, inner-CV candidate tables, decisions and validation: `{args.run.resolve()}`',f'- CSVs and figure: `{args.out.resolve()}`','- Source snapshots, input hashes and environment records identify the experiment.','']
    Path('research/CALIBRATION_STOPPING_RESULTS.md').write_text('\n'.join(lines))
    (args.out/'decision_differences.json').write_text(json.dumps(differences,indent=2)+'\n')
    (args.out/'report_calibration_stopping.py').write_bytes(Path(__file__).read_bytes())
    (args.out/'provenance.json').write_text(json.dumps({'created_utc':datetime.now(timezone.utc).isoformat(),'run':str(args.run.resolve()),'input_results_sha256':hash_stream(args.run/'results.json'),'input_validation_sha256':hash_stream(args.run/'independent_validation.json'),'report_sha256':hash_stream('research/CALIBRATION_STOPPING_RESULTS.md')},indent=2)+'\n')
    (args.out/'artifact_sha256.json').write_text(json.dumps({str(p.relative_to(args.out)):hash_stream(p) for p in args.out.rglob('*') if p.is_file()},indent=2)+'\n')
    print(json.dumps({'report':str(Path('research/CALIBRATION_STOPPING_RESULTS.md').resolve()),'figure':str((args.out/'calibration_stopping.png').resolve()),'decision_differences':differences}))


if __name__=='__main__':
    main()
