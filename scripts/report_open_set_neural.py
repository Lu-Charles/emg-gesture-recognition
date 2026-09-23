"""Complete fixed-protocol tables and descriptive, explicitly oracle diagnostics."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from src.grabmyo_corpus import hash_stream

FIELDS=['known_correct_acceptance','unknown_false_acceptance','known_wrong_command','known_rejection','known_predicted_rest','rest_false_activation','closed_known_accuracy']
KEYS=['family','method','budget','seconds','update_strategy','rejector']


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);run=parser.parse_args().run
    load=lambda p:json.loads(p.read_text());audit=load(run/'validation.json');assert audit['passed']
    cfg=load(run/'config.json');summary=load(run/'summary.json');allrows=[];sources=[];oracle=[];classrows=[];ranking=[]
    for family in cfg['families']:
        folder=run/family;metrics=load(folder/'metrics.json');source=load(folder/'source_metrics.json');allrows+=metrics;sources+=source
        predictions={};curve_index={(c['score_file'],c['rejector']):c for c in load(folder/'curve_index.json')}
        for r in metrics+source:
            if r['retention']!=.95:continue
            if r['score_file'] not in predictions:
                with np.load(folder/'predictions'/r['score_file']) as z:predictions[r['score_file']]={k:z[k] for k in z.files}
            z=predictions[r['score_file']];y,pr=z['y'],z['pred'];known=(y>=10)&(y<16);score=z[r['rejector']]
            # Independent order statistic rather than selecting rows of the
            # stored command curve. Test labels make this an oracle diagnostic.
            correct_scores=np.sort(score[known&(pr==y)])[::-1];needed=int(np.ceil(.8*known.sum()));feasible=len(correct_scores)>=needed
            t=correct_scores[needed-1] if feasible else None
            values={k:r[k] for k in ['family','participant','rejector']}
            values.update({k:r.get(k,default) for k,default in [('session',1),('method','source'),('budget',0),('seconds',0),('order','source'),('update_strategy','source')]})
            values.update(feasible=feasible,oracle_threshold=t,oracle_unknown_acceptance_at80=float(np.mean((score[y<10]>=t)&(pr[y<10]!=16))) if feasible else None)
            if feasible:
                assert np.sum(known&(pr==y)&(score>=t))/known.sum()>=.8
                # The immediately higher observed score must fail the target,
                # unless tied correct samples already make it unattainable.
                above=score[score>t]
                if len(above):assert np.sum(known&(pr==y)&(score>=above.min()))/known.sum()<.8
            oracle.append(values)
            if 'session' in r:
                c=curve_index[r['score_file'],r['rejector']]
                ranking.append({**{k:r[k] for k in KEYS+['participant','session','order']},'auroc':c['auroc'],'standard_oscr':c['standard_oscr']})
                for label in range(17):
                    mask=y==label;accepted=score[mask]>=r['threshold']
                    classrows.append({**{k:r[k] for k in KEYS+['participant','session','order']},'class_index':label,'accepted_command_rate':float(np.mean(accepted&(pr[mask]!=16))),'rejected_rate':float(np.mean(~accepted))})
    df=pd.DataFrame(allrows);primary=df[df.retention==.95];source=pd.DataFrame(sources);source=source[source.retention==.95]
    per=primary.groupby(KEYS+['participant'],as_index=False)[FIELDS].mean();agg=per.groupby(KEYS,as_index=False)[FIELDS].mean()
    gate=per.assign(residual=(per.known_correct_acceptance>=.8)&(per.unknown_false_acceptance>=.1)).groupby(KEYS,as_index=False).residual.sum()
    agg=agg.merge(gate,on=KEYS).rename(columns={'residual':'people_with_residual'})
    per.to_csv(run/'per_participant.csv',index=False);agg.to_csv(run/'aggregate.csv',index=False)
    primary.groupby(KEYS+['session'],as_index=False)[FIELDS].mean().to_csv(run/'per_session.csv',index=False)
    df.groupby(KEYS+['retention'],as_index=False)[FIELDS].mean().to_csv(run/'retention_sensitivity.csv',index=False)
    pd.DataFrame(classrows).to_csv(run/'per_class.csv',index=False)
    pd.DataFrame(ranking).groupby(KEYS,as_index=False)[['auroc','standard_oscr']].mean().to_csv(run/'ranking_summary.csv',index=False)
    srcagg=source.groupby(['family','rejector'],as_index=False)[FIELDS].mean();srcagg.to_csv(run/'source_aggregate.csv',index=False)
    odf=pd.DataFrame(oracle);odf.to_csv(run/'oracle_diagnostics.csv',index=False)
    # Average orders inside each person, then people; feasibility reported
    # separately. Infeasible cases are not assigned favorable zero errors.
    op=odf.groupby(KEYS+['participant'],as_index=False)[['feasible','oracle_unknown_acceptance_at80']].mean()
    oa=op.groupby(KEYS,as_index=False)[['feasible','oracle_unknown_acceptance_at80']].mean();oa.to_csv(run/'oracle_summary.csv',index=False)
    rng=np.random.default_rng(20260906);draw=rng.integers(0,8,size=(10000,8));intervals=[]
    for key,g in per.groupby(KEYS):
        g=g.sort_values('participant');assert len(g)==8
        for field in FIELDS:
            v=g[field].to_numpy();lo,hi=np.quantile(v[draw].mean(1),[.025,.975]);intervals.append(dict(zip(KEYS,key),metric=field,mean=v.mean(),lower=lo,upper=hi))
    pd.DataFrame(intervals).to_csv(run/'participant_intervals.csv',index=False)
    def primary_score(frame):return frame[(frame.family.eq('cnn')&frame.rejector.eq('probability'))|(~frame.family.eq('cnn')&frame.rejector.eq('prototype'))]
    def table(frame):
        lines=['| Family | Update | Fit strategy | Correct commands accepted | Unfamiliar commands accepted | Residual screen |','| --- | --- | --- | ---: | ---: | ---: |']
        for r in frame.to_dict('records'):
            lines.append(f"| {r['family']} | {r['method']} | {r['update_strategy']} | {100*r['known_correct_acceptance']:.2f}% | {100*r['unknown_false_acceptance']:.2f}% | {int(r['people_with_residual'])}/8 |")
        return '\n'.join(lines)
    source_lines=['| Family | Correct commands accepted | Unfamiliar commands accepted |','| --- | ---: | ---: |']
    for r in primary_score(srcagg).to_dict('records'):source_lines.append(f"| {r['family']} | {100*r['known_correct_acceptance']:.2f}% | {100*r['unknown_false_acceptance']:.2f}% |")
    oracle_lines=['| Family | Update | Fit strategy | Feasible fraction | Unknown acceptance at ≥80% correct |','| --- | --- | --- | ---: | ---: |']
    for r in primary_score(oa[oa.budget.eq(3)]).to_dict('records'):
        value=r['oracle_unknown_acceptance_at80'];shown=f'{100*value:.2f}%' if np.isfinite(value) else 'infeasible'
        oracle_lines.append(f"| {r['family']} | {r['method']} | {r['update_strategy']} | {100*r['feasible']:.1f}% | {shown} |")
    history=[]
    for family in cfg['families']:
        h=load(run/family/'pretraining_history.json');history.append(f"- {family}: training CE {h[0]['ce']:.4f} → {h[-1]['ce']:.4f}; total loss {h[0]['total']:.4f} → {h[-1]['total']:.4f}. PredIN CE sums two branches. These are training losses, not held-out accuracy or proof of convergence.")
    diagnostics=[]
    for file in run.glob('training_diagnostics_*.json'):diagnostics+=load(file)['results']
    diagnostic_lines=[]
    for family in cfg['families']:
        d=[r for r in diagnostics if r['family']==family]
        if d:
            accuracy=np.asarray([r['accuracy'] for r in d]);poor=sum(min(r['per_class_recall'])<.2 for r in d)
            diagnostic_lines.append(f"- {family}: in-sample enrollment accuracy {100*accuracy.mean():.2f}% mean ({100*accuracy.min():.2f}–{100*accuracy.max():.2f}% across people); {poor}/{len(d)} people have a known class with less than20% training recall.")
    interpretation=(run/'interpretation.md').read_text() if (run/'interpretation.md').exists() else 'Interpretation pending review of this run.'
    report=f'''# Stronger open-set EMG baseline screen

Eight development participants, one training seed, two later sessions and six calibration orders. Twenty separate participants supply shared representation training; the final15 remain untouched. These are matched-backbone adaptations of published ideas, not original-benchmark reproductions or a new proposed method.

{interpretation}

## Independent day1 reference

{chr(10).join(source_lines)}

## No new-day calibration

{table(primary_score(agg[agg.budget.eq(0)]))}

## Equal105-second new-day budget

{table(primary_score(agg[agg.budget.eq(3)]))}

CNN uses maximum softmax probability. PL and PredIN use maximum prototype similarity, with softmax controls retained in the complete tables. All use the preregistered95% known calibration-score retention. Correct command acceptance excludes rest; unfamiliar gestures predicted as rest do not issue an active command. Rest false activation, wrong known commands and rejection are saved separately.

`model_only` fine-tunes using three target rounds but keeps the original numeric threshold. `threshold_only` uses all three rounds for the fixed source model's threshold. `both` uses two fitting rounds and one disjoint threshold round. `target_finetune` starts from the enrolled neural model and updates on target data; it is not training from scratch. `pooled_replay` also reuses day1 fitting data. No extra target recordings are granted to replay. Known fitting/threshold/scoring trials are disjoint throughout. Order permutations are repeated allocations, not independent people.

The residual screen counts people averaging≥80% correct acceptance and≥10% unfamiliar command acceptance at this operating point. This is a baseline limitation screen; six people passing it does not establish novelty or a successful mitigation.

## Optimistic threshold-separation diagnostic

{chr(10).join(oracle_lines)}

These thresholds use scoring labels to attain at least80% correct known commands separately per participant/session/order. They are explicitly test-informed oracle diagnostics, never deployable calibration or training inputs. This diagnostic was introduced after the earlier classical results and retained for this comparison. Infeasible cases cannot reach80% even with no rejection; their unknown rates are missing rather than zero. Feasibility differences limit comparisons of feasible-only means. A score-separation problem already present on day1 is not a distinctive day-shift contribution.

## Fidelity and training limitations

[PredIN author manuscript v2](https://arxiv.org/html/2407.19753v2) supplies the prototype, compactness, triplet and two-branch inconsistency equations. The local implementation interprets the unusual compactness equation literally: half vectorL2 norm below vectorL1 norm1, otherwise vectorL1 minus0.5. No official author code was located. The published work uses different backbones, SGD and100epochs; this screen uses the project's compact encoder plus128-dimensional embedding, Adam,20 shared epochs,20 enrollment epochs and10 update epochs. All settings and final-checkpoint selection were fixed before scoring. Thus unfavorable results do not establish that the original published method fails. See NEURAL_OPEN_SET_METHODS.md for complete choices and departures. PL is the PredIN manuscript's single-prototype baseline, not an exact reproduction of Gao's personalized method.

{chr(10).join(history)}

After viewing PL's poor recognition, an explicitly post-hoc training-fit diagnostic scored all enrollment fitting trials without rejection. It changes no settings or selected checkpoints:

{chr(10).join(diagnostic_lines)}

The single-prototype PL adaptation severely confuses known classes on its own fitting data for most people. It therefore cannot serve as evidence that a well-trained original PL method has been defeated. Compact-backbone/training-schedule and literal-equation fidelity remain limitations. This diagnostic uses training data, not additional held-out or final data. CPU fit diagnostics and artifact auditing overlapped portions of PredIN execution, so wall times reflect the actual shared-machine workload rather than isolated hardware benchmarks. Audit elapsed time includes waiting for the last family to complete.

Only one vocabulary and seed are tested. No unseen-gesture diversity, second-dataset result, final-participant result, daily-life safety claim or8–9/10 research claim is supported yet. Comparisons with LDA also differ in access to20-person representation training; architecture superiority cannot be isolated from that difference.

## Verification and measured runtime

Five focused loss/access/gradient tests passed before the run. Independent audit checked all1344allocation ledgers, recomputed normalization from raw known-only signals with overlap multiplicities, checked every saved prediction's labels and all reported metric counts/cutoffs, and reproduced nine fixed windows per checkpoint using direct functional CPU operations. This is sampled numerical checkpoint verification, not replay of every neural prediction or optimizer update. Complete curves were spot-checked. CPU/MPS tolerances and exact counts are in validation.json.

Total experiment wall time: {summary['seconds']/60:.2f} minutes, excluding implementation and verification. Per-family timing:

{chr(10).join(f"- {r['family']}: {r['total_seconds']/60:.2f} minutes total, {r['pretraining_seconds']/60:.2f} minutes shared training, {r['models']} saved calibrated models." for r in summary['families'])}

Run: `{run.resolve()}`. Full aggregate, participant/session/class, retention-sensitivity, participant-bootstrap intervals, source and oracle tables are saved alongside per-family models/scores/loss histories. Bootstrap intervals resample eight people, conditional on this fixed vocabulary/procedure; overlapping windows are not independent samples. Config, source snapshots, hashes, environment and access manifests preserve provenance. No custom data or final15 signal access.
'''
    (run/'report.md').write_text(report);Path('research/OPEN_SET_NEURAL_RESULTS.md').write_text(report)
    (run/'code_snapshot'/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    files=[f for f in [run/'interpretation.md',run/'paired_contrasts.json'] if f.exists()]+list(run.glob('*.csv'))+list(run.glob('training_diagnostics_*.json'))+[run/'report.md',run/'validation.json',run/'code_snapshot'/Path(__file__).name]
    (run/'report_artifact_sha256.json').write_text(json.dumps({str(f.relative_to(run)):hash_stream(f) for f in files},indent=2)+'\n')
    print(table(primary_score(agg[agg.budget.eq(3)])))
    print(chr(10).join(source_lines));print(chr(10).join(oracle_lines))


if __name__=='__main__':main()
