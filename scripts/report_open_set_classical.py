"""Summarize the audited first classical screen without declaring a research win."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();run=a.run
    audit=json.loads((run/'validation.json').read_text());assert audit['passed']
    df=pd.DataFrame(json.loads((run/'metrics.json').read_text()));primary=df[df.retention==.95].copy()
    assert len(primary)==2688, 'Missing primary-retention rows'
    fields=['known_correct_acceptance','unknown_false_acceptance','known_wrong_command','known_rejection','known_predicted_rest','rest_false_activation','closed_known_accuracy']
    keys=['method','budget','seconds','fit_strategy','rejector']
    # Equal participant weighting; within each participant average sessions/orders.
    per=primary.groupby(keys+['participant'],as_index=False)[fields].mean()
    aggregate=per.groupby(keys,as_index=False)[fields].mean()
    gate=per.assign(residual=(per.known_correct_acceptance>=.80)&(per.unknown_false_acceptance>=.10)).groupby(keys,as_index=False)['residual'].sum()
    aggregate=aggregate.merge(gate,on=keys).rename(columns={'residual':'people_with_residual_at_80pct_correct'})
    per.to_csv(run/'per_participant.csv',index=False);aggregate.to_csv(run/'aggregate.csv',index=False)
    primary.groupby(keys+['session'],as_index=False)[fields].mean().to_csv(run/'per_session.csv',index=False)
    df.groupby(keys+['retention'],as_index=False)[fields].mean().to_csv(run/'retention_sensitivity.csv',index=False)
    source=pd.DataFrame(json.loads((run/'source_metrics.json').read_text()))
    source[source.retention==.95].groupby('rejector',as_index=False)[fields].mean().to_csv(run/'source_aggregate.csv',index=False)
    prediction_cache={};class_rows=[]
    for row in primary.to_dict('records'):
        file=row['score_file']
        if file not in prediction_cache:
            z=np.load(run/'predictions'/file);prediction_cache[file]={k:z[k] for k in z.files}
        z=prediction_cache[file];accepted=z[row['rejector']]>=row['threshold']
        command=accepted&(z['pred']!=16)
        for label in range(17):
            m=z['y']==label
            class_rows.append({**{k:row[k] for k in keys+['participant','session','order']},'class_index':label,
               'accepted_command_rate':float(command[m].mean()),'rejected_rate':float((~accepted[m]).mean()),
               'correct_accepted_rate':float((accepted[m]&(z['pred'][m]==label)).mean()) if label>=10 else None})
    pd.DataFrame(class_rows).to_csv(run/'per_class.csv',index=False)
    # Intervals are descriptive participant bootstrap conditional on these fixed procedures.
    rng=np.random.default_rng(20260906);draws=rng.integers(0,8,size=(10000,8));intervals=[]
    for key,g in per.groupby(keys):
        g=g.sort_values('participant');assert len(g)==8
        for field in fields:
            v=g[field].to_numpy();lo,hi=np.quantile(v[draws].mean(1),[.025,.975])
            intervals.append(dict(zip(keys,key),metric=field,mean=float(v.mean()),lower=float(lo),upper=float(hi)))
    pd.DataFrame(intervals).to_csv(run/'participant_intervals.csv',index=False)
    # Planned full-curve diagnosis, with this exact oracle summary added after
    # viewing operating points. Not a fitted/deployable rejection rule.
    curve_map={(c['score_file'],c['rejector']):c for c in json.loads((run/'curve_index.json').read_text())}
    oracle=[]
    for r in primary[primary.budget==3].to_dict('records'):
        c=np.load(run/'predictions'/curve_map[r['score_file'],r['rejector']]['curve_file'])
        mask=c['known_correct_acceptance']>=.8
        oracle.append({**{k:r[k] for k in keys+['participant','session','order']},
            'feasible':bool(mask.any()),'oracle_unknown_acceptance_at80':float(c['unknown_false_acceptance'][mask].min()) if mask.any() else None})
    odf=pd.DataFrame(oracle);odf.to_csv(run/'posthoc_oracle_thresholds.csv',index=False)
    odf.groupby(keys,as_index=False)[['feasible','oracle_unknown_acceptance_at80']].mean().to_csv(run/'posthoc_oracle_summary.csv',index=False)
    source_probability=source[(source.retention==.95)&(source.rejector=='probability')][fields].mean()
    joint_probability=aggregate[(aggregate.budget==3)&(aggregate.method=='both')&(aggregate.fit_strategy=='source_plus_target')&(aggregate.rejector=='probability')].iloc[0]
    oracle_target=odf[(odf.method=='model_only')&(odf.fit_strategy=='target_only')&(odf.rejector=='probability')]
    oracle_pooled=odf[(odf.method=='model_only')&(odf.fit_strategy=='source_plus_target')&(odf.rejector=='probability')]
    summary=json.loads((run/'summary.json').read_text())
    def table(frame):
        lines=['| Method | Model fit data | Rejection score | Correct commands accepted | Unfamiliar gestures accepted as commands | People meeting residual-problem screen |',
               '| --- | --- | --- | ---: | ---: | ---: |']
        for r in frame.to_dict('records'):
            lines.append(f"| {r['method']} | {r['fit_strategy']} | {r['rejector']} | {100*r['known_correct_acceptance']:.2f}% | {100*r['unknown_false_acceptance']:.2f}% | {int(r['people_with_residual_at_80pct_correct'])}/8 |")
        return '\n'.join(lines)
    report=f'''# First open-set EMG baseline results

Completed classical development screen; no neural open-set model or final evaluation yet. This establishes baseline behavior, not novelty or a successful new method. All results below are pooled over eight development participants with equal person weight; each person contributes two target sessions and six calibration orders. Windows/orders are not independent people.

## No new-day calibration

{table(aggregate[aggregate.budget==0])}

## Equal105-second new-day recording budget

{table(aggregate[aggregate.budget==3])}

`model_only` updates/refits the classifier and retains the original model's numeric threshold. `threshold_only` leaves the source classifier fixed and uses all three new-day rounds to set its threshold. `both` uses two rounds for the classifier and a disjoint third round for threshold setting. `source_plus_target` includes day1 model-fitting data; `target_only` refits from the permitted new-day model-fitting rounds. The source classifier uses five day1 trials per known class, with a separate sixth trial for its threshold.

Probability and distance use the same fitted LDA classifier. Distance is Mahalanobis distance to its predicted class centroid, using its shrinkage covariance; this is a simple distance baseline, not Gao's learned prototype method.95% known calibration-score retention is fixed before outcomes. Actual test acceptance can differ. An unfamiliar gesture predicted as rest does not issue a command and does not count as unknown false activation; rest false activation is reported separately.

## Interpretation limits and decision

The threshold-transfer hypothesis is not established by these results. In the independent day1 reference, probability rejection correctly accepts{100*source_probability.known_correct_acceptance:.2f}% of active known commands but also accepts{100*source_probability.unknown_false_acceptance:.2f}% of unfamiliar gestures. The joint source-plus-target probability baseline on later days reaches{100*joint_probability.known_correct_acceptance:.2f}% correct acceptance with{100*joint_probability.unknown_false_acceptance:.2f}% unknown false acceptance. Thus a major open-set limitation already exists within day1; these figures do not show personalization causing it.

An explicitly post-hoc oracle-threshold summary of the planned full curves further tests whether simply changing the cutoff could suffice. For the105-second target-only model with probability scores, even a test-informed threshold chosen separately for each person/session/order to retain at least80% correct commands still accepts{100*oracle_target.oracle_unknown_acceptance_at80.mean():.2f}% of unfamiliar gestures on average among feasible cases. {int(oracle_target.feasible.sum())}/{len(oracle_target)} such replay cases can reach80%. This is an optimistic diagnostic using test labels, not a deployable method or a new result selected for submission. The corresponding source-plus-target model-only diagnostic is{100*oracle_pooled.oracle_unknown_acceptance_at80.mean():.2f}%; all methods and infeasible cases are saved in posthoc_oracle_summary.csv. This indicates limited score separation in these classical models, beyond a stale cutoff alone.

The last column checks the prespecified first residual-problem condition within each participant's average: at least80% correctly accepted known active commands alongside at least10% unknown false acceptance. It is a descriptive screen for each baseline, not an overall pass. The six-of-eight-person requirement must be assessed against the strongest simple/neural competitors; this first classical stage does not settle it. Per-session results are available separately. The alternative source-to-target deterioration criterion requires comparison at comparable correct acceptance, not a raw change between unrelated operating points.

Do not claim a method contribution from improving a stale threshold alone. Threshold-only, joint and target-only controls are included to prevent that interpretation. No new learned rule has been fitted. The next planned comparison is a freshly pretrained known-only compact CNN and appropriate rejection controls, keeping all held-out gesture/model exposure constraints. If strong simple methods remove the problem, preserve that result and stop this branch.

These are cued unfamiliar-gesture proxies, not activities-of-daily-living recordings, real-time command events or a clinical safety test. No false activations/hour or reaction latency is inferred. The source known/unknown vocabulary was chosen before these results. Existing17-class checkpoints and their normalization were not reused. Final15 participants and Charles's recordings were excluded.

## Verification and runtime

- {summary['models']} saved LDA models; {summary['metric_rows']} operating-point rows, with complete scores and thresholds.
- {audit['allocations']} allocations checked for participant/known-class access, disjoint fit/threshold/score trials and exact recording budgets.
- {audit['feature_windows_independent']} raw-signal windows independently checked using scalar features and a separate Burg implementation; maximum AR difference {audit['ar_max_abs_error']:.3g}.
- {audit['predictions']} saved window predictions independently reproduced from fitted discriminant coefficients; all operating-point metrics/counts recalculated; {audit['curve_spot_checks']} curve points checked.
- Six risk-focused tests passed in0.916 seconds. Earlier v1 stopped before model fitting because of a tuple-indexing error; its artifacts/log are preserved, and a regression test covers the fix.
- Completed v2 runtime {summary['total_seconds']:.2f} seconds, including {summary['feature_seconds']:.2f} seconds feature extraction and {summary['fit_seconds']:.2f} seconds summed fit/save time. Independent audit {audit['seconds']:.2f} seconds. These exclude implementation/setup/report time.

## Exact artifacts

Run: `{run.resolve()}`. `config.json` and `allocations.json` were written before extracting features or inspecting outcomes. `trial_manifest.json`, `model_access.json`, saved models and prediction files preserve access/provenance. `aggregate.csv`, `per_participant.csv`, `per_session.csv`, `per_class.csv`, `retention_sensitivity.csv` and `participant_intervals.csv` contain the full0/35/70/105-second results. Intervals use10000 participant-bootstrap draws, conditional on fixed procedures, from only eight independent people. `curve_index.json` links complete descriptive command tradeoff curves and standard OSCR/AUROC summaries; test-derived curve thresholds are not deployable calibration procedures.

Source snapshots and hashes are saved. Environment is `.venv-open-set`, with pinned librosa0.11.0 and complete dependency versions saved in `environment.txt`; earlier `.venv-public` unchanged.
'''
    Path('research/OPEN_SET_CLASSICAL_RESULTS.md').write_text(report)
    (run/'report.md').write_text(report)
    print(table(aggregate[aggregate.budget==3]))
    print(json.dumps(summary))


if __name__=='__main__':main()
