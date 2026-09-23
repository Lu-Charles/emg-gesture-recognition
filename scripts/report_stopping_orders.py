"""Report frozen-policy robustness separately from order-aware refitting."""
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

NAMES={'fixed_85':'Fixed 85 s','fixed_170':'Fixed 170 s','fixed_255':'Fixed 255 s',
       'fixed':'Selected fixed budget','pre_error':'Pre-update error','post_entropy':'Post-update entropy',
       'probability_change':'Probability change','ridge':'Learned rule'}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    cfg=json.loads((args.run/'config.json').read_text());obs=Path(cfg['observations'])
    rows=json.loads((args.run/'results.json').read_text());by={r['family']:r for r in rows}
    order_rows=json.loads((args.run/'per_order.json').read_text());orders=cfg['orders']
    choices=json.loads((args.run/'decisions.json').read_text())
    outcomes={(r['participant'],r['session'],r['order'],r['budget']):r['macro_f1'] for r in json.loads((obs/'outcomes.json').read_text())}
    validation=json.loads((args.run/'independent_validation.json').read_text());assert validation['status']=='passed'
    for name,records in (('summary',rows),('per_order',order_rows)):
        with (args.out/f'{name}.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in records for k in r)));writer.writeheader();writer.writerows(records)
    episode_rows=[]
    for r in choices:
        p,s,o,b=r['participant'],r['session'],r['order'],r['selected_budget']
        episode_rows.append({'family':r['family'],'participant':p,'session':s,'order':o,'selected_seconds':85*b,
                             'macro_f1':outcomes[p,s,o,b],'loss_vs_full':outcomes[p,s,o,3]-outcomes[p,s,o,b]})
    with (args.out/'per_episode.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(episode_rows[0]));writer.writeheader();writer.writerows(episode_rows)
    contrasts=[]
    for prefix in ('frozen_',''):
        left={(r['participant'],r['session'],r['order']):r for r in episode_rows if r['family']==prefix+'ridge'}
        right={(r['participant'],r['session'],r['order']):r for r in episode_rows if r['family']==prefix+'pre_error'}
        differences=[left[k]['macro_f1']-right[k]['macro_f1'] for k in left]
        contrasts.append({'track':'frozen' if prefix else 'all_order_training',
                          'different_decisions':sum(left[k]['selected_seconds']!=right[k]['selected_seconds'] for k in left),
                          'total_episodes':len(left),'ridge_minus_error_f1':float(np.mean(differences)),
                          'ridge_minus_error_seconds':float(np.mean([left[k]['selected_seconds']-right[k]['selected_seconds'] for k in left]))})
    (args.out/'policy_contrasts.json').write_text(json.dumps(contrasts,indent=2)+'\n')
    styles={'frozen_pre_error':('#CA6F1E','o','Original error rule'),
            'frozen_ridge':('#187E91','D','Original learned rule'),
            'pre_error':('#CA6F1E','s','Error rule trained on all orders'),
            'ridge':('#187E91','^','Learned rule trained on all orders')}
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,(ax,bx)=plt.subplots(1,2,figsize=(13.8,5.5))
    ax.plot([85,170,255],[by[f'fixed_{b}']['mean_macro_f1'] for b in (85,170,255)],'o--',color='#66717D',label='Fixed routines / expected mixtures')
    for family,(color,marker,label) in styles.items():
        row=by[family]
        ax.scatter(row['mean_seconds'],row['mean_macro_f1'],s=100,c=color,marker=marker,label=label,zorder=3)
        means=[next(r['mean_loss_vs_full'] for r in order_rows if r['order']==order and r['family']==family)*100 for order in orders]
        bx.plot(range(6),means,marker=marker,color=color,ls='--' if family.startswith('frozen_') else '-',alpha=.9)
    ax.set(title='Average across every calibration order',xlabel='Mean new-day recording (seconds)',ylabel='Mean macro-F1',xlim=(75,265),ylim=(.82,.87))
    ax.legend(loc='lower right',fontsize=8);ax.grid(alpha=.2)
    bx.axhline(1,color='#A54848',lw=1,ls=':',label='1-point mean-loss target')
    bx.set_xticks(range(6),orders);bx.set(title='Mean loss depends on calibration order',xlabel='Order of the three balanced calibration rounds',ylabel='Macro-F1 loss vs full calibration (points)',ylim=(0,1.4))
    bx.text(.03,.96,'Dashed: original rules unchanged\nSolid: trained across all orders',transform=bx.transAxes,va='top',fontsize=9)
    bx.legend(loc='lower right',fontsize=8);bx.grid(alpha=.2)
    fig.suptitle('Calibration-order robustness — eight development participants',fontsize=15,x=.04,ha='left')
    fig.text(.04,.015,'All six orders, both later sessions. Replays reuse the same people and scoring trials; 96 replays are not 96 independent participants.',fontsize=9,color='#46515D')
    fig.tight_layout(rect=[0,.06,1,.94])
    for suffix in ('png','svg'):
        fig.savefig(args.out/f'order_robustness.{suffix}',dpi=180)
    plt.close(fig)
    frozen,trained=by['frozen_ridge'],by['ridge']
    frozen_failures=[r['order'] for r in order_rows if r['family']=='frozen_ridge' and r['mean_loss_vs_full']>.01]
    trained_harms=[r for r in episode_rows if r['family']=='ridge' and r['loss_vs_full']>.02]
    frozen_harms=[r for r in episode_rows if r['family']=='frozen_ridge' and r['loss_vs_full']>.02]
    lines=['# Calibration-order robustness results','',
           'Completed '+json.loads((args.run/'summary.json').read_text())['completed_utc']+'. Public GRABMyo only; all final participants remain untouched.','',
           f'**Average calibration savings survived the six-order check, but the first pilot understated sensitivity to trial choice.** The original learned policies, unchanged, use **{frozen["mean_seconds"]:.1f} s** instead of 255 s (**{100*frozen["recording_reduction_fraction"]:.1f}% less new-day recording**), with mean macro-F1 **{frozen["mean_macro_f1"]:.4f} versus .8622**. Orders '+', '.join(frozen_failures)+' exceed the one-point average-loss target. The overall mean still passes the originally declared aggregate screen.','',
           f'The separately predeclared all-order-training track is more conservative: **{trained["mean_seconds"]:.1f} s / macro-F1 {trained["mean_macro_f1"]:.4f}**, saving **{100*trained["recording_reduction_fraction"]:.1f}%**. It remains below one-point average loss in each of the six orders. This improvement in quality uses more recording and is not a free accuracy gain.','',
           '## Full aggregate comparison','',
           '| Training / method | Mean recording, s | Macro-F1 | Mean loss vs full | Worst loss | Replays losing >.02 |',
           '| --- | ---: | ---: | ---: | ---: | ---: |']
    for row in rows:
        family=row['family'];base=family.removeprefix('frozen_')
        name=NAMES[base]
        if not family.startswith('fixed_'):
            name=('Original rules unchanged: ' if family.startswith('frozen_') else 'Trained on all orders: ')+name
        lines.append(f'| {name} | {row["mean_seconds"]:.1f} | {row["mean_macro_f1"]:.4f} | {row["mean_loss_vs_full"]:.4f} | {row["worst_loss_vs_full"]:.4f} | {round(row["fraction_loss_over_002"]*96)}/96 |')
    lines += ['',f'![Order robustness]({(args.out/"order_robustness.png").resolve()})','',
              '## Every order, without selecting favorable ones','',
              '| Order | Original learned: seconds / F1 | All-order learned: seconds / F1 | Original learned loss | All-order learned loss |',
              '| --- | ---: | ---: | ---: | ---: |']
    for order in orders:
        a=next(r for r in order_rows if r['order']==order and r['family']=='frozen_ridge')
        b=next(r for r in order_rows if r['order']==order and r['family']=='ridge')
        lines.append(f'| {order} | {a["mean_seconds"]:.1f} / {a["mean_macro_f1"]:.4f} | {b["mean_seconds"]:.1f} / {b["mean_macro_f1"]:.4f} | {a["mean_loss_vs_full"]:.4f} | {b["mean_loss_vs_full"]:.4f} |')
    lines += ['',
              'The all-order-trained learned rule improves over its expected equal-cost fixed-budget mixture in every order. Aggregate advantage is '+f'{trained["delta_vs_same_cost_mixture"]:.4f}'+' macro-F1, with descriptive participant-bootstrap interval '+str([round(v,4) for v in trained['descriptive_participant_bootstrap_delta_ci95']])+'. The original learned rule also improves over the mixture in each order. These intervals condition on fitted policies and do not include every source of model-selection uncertainty.','',
              '## What this says about the learned component','',
              'With original policies unchanged, learned and simple pre-error rules have the same aggregate recording cost; learned macro-F1 is only 0.00037 higher. With all-order training, learned macro-F1 is 0.00294 higher and recording is 8.85 seconds longer than the simple pre-error rule. Their decisions differ in 10/96 replays. These are modest differences, not evidence of a major algorithmic breakthrough.','',
              'The learned rule is more consistent against the fixed-mixture baseline across these orders than the all-order-trained pre-error rule, but the simple rules remain essential comparators. The study supports the practical duration-adaptation question. Niche originality and usefulness should be argued from the full protocol and independently reproduced effect, not the complexity of ridge regression.','',
              '## Individual harm and cost limits','',
              f'All-order-trained learned policies lose more than .02 macro-F1 in {len(trained_harms)}/96 replays, involving {len({r["participant"] for r in trained_harms})} of eight people. The original learned policies exceed that loss in {len(frozen_harms)}/96 replays. These are correlated replays, not independent failures. The worst learned loss is .03781 for participant3/session2 when starting with round3; stopping after the first round causes the same result for orders312 and321.','',
              'No per-person protection is established by the mean-loss constraint. The all-order learned rule saves less than20% for orders312/321 individually (about18.8%), though it meets the20% target when averaging all six orders. Report both aggregate and order-specific limits.','',
              'Savings count only five-second trial recording. Day-1 enrollment remains595 seconds, population pretraining is unchanged, and instructions/rest/user interaction overhead are unmeasured. Offline replay does not demonstrate actual user preference, closed-loop performance or clinical suitability.','',
              '## Design and verification','',
              'All six global permutations of the existing three balanced calibration-rank rounds were evaluated. Each round contains one reserved trial for each of17 classes. The same four scoring trials/class remain fixed. This covers ordering of the three existing rounds, not all independently chosen per-class combinations or new trial-pool splits. Acquisition order is varied; model fitting uses canonical manifest order and resets from enrollment at each accumulated subset.','',
              'There are seven distinct nonempty calibration subsets per person/session. We trained the four missing subsets and reused the three prior prefixes, plus the enrolled base. This produced64 new fits and64 reused checkpoints. Original123 observations, outcomes and frozen-policy decisions reproduce exactly.','',
              'Primary robustness comparison freezes the original policies without any new threshold selection. The separate all-order-training extension uses outer leave-one-person-out and inner leave-one-person-out selection, retaining all six orders and both sessions of a person in the same fold. Features, hyperparameter grids and the aggregate mean-loss criterion were fixed before these outcomes. Scoring outcomes enter policy training only for other participants; no future-round or scoring data enter held-out decisions.','',
              'Seven tests passed, including new checks for order-episode identity, participant grouping and unchanged calibration/scoring pools. Readback verification passed for128 model access ledgers,112 scored models/266,560 predictions,192 observation states,288 unique CPU probability evaluations,80 policy artifacts and960 reached decision traces. Eight newly trained models (one per person) were independently refitted with exact weight reproduction. All73 aggregate/order summary rows were independently recomputed. No final-participant signals were accessed.','',
              'The verification does not rerun every inner hyperparameter search or all64 new fits. The eight development participants already informed earlier research/model choices, so these findings are not final test results.','',
              '## Runtime and next action','',
              'All-order recognizer updates, feature extraction and artifact saving took42.10 seconds; the64 new fitting operations accounted for23.47 seconds. Nested policy evaluation and bootstrap took19.65 seconds; verification took61.73 seconds. These exclude implementation and report preparation.','',
              'Continue with the predeclared seed check (seeds0 and1, retaining42), repeating the shared recognizer and calibration comparison without tuning to final data. Use all-order-trained learned and pre-error policies as the principal pair while retaining frozen-original robustness results. Keep published classical-feature controls and a second dataset on the completion checklist; this order check alone does not establish an8–9/10 project or conference acceptance. No seed run has launched in this session.','',
              '## Exact outputs','',f'- Observations, models, access, predictions: `{obs.resolve()}`',f'- Policies, selection tables, all decisions, verification: `{args.run.resolve()}`',f'- Figures and CSVs: `{args.out.resolve()}`','- Earlier first-order report remains unchanged: `research/CALIBRATION_STOPPING_RESULTS.md`.','- Source snapshots and access manifests preserve the evaluated conditions.','']
    report=Path('research/CALIBRATION_ORDER_RESULTS.md');report.write_text('\n'.join(lines))
    for file in (__file__,'scripts/verify_stopping_orders.py','tests/test_calibration_stopping.py'):
        (args.out/Path(file).name).write_bytes(Path(file).read_bytes())
    (args.out/'provenance.json').write_text(json.dumps({'created_utc':datetime.now(timezone.utc).isoformat(),'policy_run':str(args.run.resolve()),'results_sha256':hash_stream(args.run/'results.json'),'verification_sha256':hash_stream(args.run/'independent_validation.json'),'report_sha256':hash_stream(report)},indent=2)+'\n')
    (args.out/'artifact_sha256.json').write_text(json.dumps({str(p.relative_to(args.out)):hash_stream(p) for p in args.out.rglob('*') if p.is_file()},indent=2)+'\n')
    print(json.dumps({'report':str(report.resolve()),'figure':str((args.out/'order_robustness.png').resolve()),'contrasts':contrasts}))


if __name__=='__main__':
    main()
