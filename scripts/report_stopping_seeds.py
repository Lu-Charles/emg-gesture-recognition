"""Read verified seed runs; report all seeds without treating replays as people."""
import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from src.grabmyo import LABELS
from src.grabmyo_corpus import hash_stream


FAMILIES=('fixed_85','fixed_170','fixed_255','pre_error','ridge')
NAMES={'fixed_85':'Fixed 85 s','fixed_170':'Fixed 170 s','fixed_255':'Fixed 255 s',
       'pre_error':'Simple error rule','ridge':'Learned rule'}


def read(path):
    return json.loads(path.read_text())


def save_csv(path,rows):
    with path.open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
        w.writeheader();w.writerows(rows)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--runs',type=Path,nargs=3,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    all_rows=[];episodes=[];sources=[];order_rows=[];baseline=None;people=None;seed_order=[]
    for run in args.runs:
        cfg=read(run/'config.json');obs=Path(cfg['observations']);ocfg=read(obs/'config.json')
        prior=Path(ocfg['prior']);ncfg=read(prior/'config.json');seed=ncfg['seed'];seed_order.append(seed)
        assert ocfg['seed']==seed
        for folder in (run,obs,prior):
            for name,digest in read(folder/'artifact_sha256.json').items():
                if hash_stream(folder/name)!=digest: raise ValueError('Changed artifact '+str(folder/name))
        assert read(run/'independent_validation.json')['status']=='passed'
        neural_validation=read(prior/'independent_validation.json')
        assert neural_validation['models']==224 and neural_validation['fit_access_verified']
        assert neural_validation['final_participants_accessed']==0
        signature={k:ncfg[k] for k in ('cache_summary_sha256','training_participants','development_participants',
                    'pretraining_epochs','pretraining_lr','enrollment_epochs','enrollment_lr','update_epochs',
                    'update_lr','optimizer','batch_size','initializations','update_modes','calibration_budgets','target_sessions')}
        signature.update(orders=ocfg['orders'],features=ocfg['features'],
            model_hash=ncfg['code_hashes']['src/emg_cnn.py'],fit_hash=ncfg['code_hashes']['scripts/pilot_emg_cnn.py'],
            policy_candidates=cfg['candidates'],policy_selection=cfg['selection'],policy_target=cfg['ridge_target'],
            policy_code_hashes=cfg['code_hashes'])
        scaler=np.load(prior/'training_scaler.npz')
        if baseline is None:
            baseline=signature;people=ncfg['development_participants'];normalizer={k:scaler[k] for k in scaler.files}
        else:
            assert signature==baseline, 'Protocol changed between seeds'
            for k,v in normalizer.items(): np.testing.assert_array_equal(scaler[k],v)
        rows=read(run/'results.json');by={r['family']:r for r in rows}
        outcomes={(r['participant'],r['session'],r['order'],r['budget']):r['macro_f1'] for r in read(obs/'outcomes.json')}
        decisions=read(run/'decisions.json')
        for family in FAMILIES:
            chosen=([dict(participant=p,session=s,order=o,selected_budget=int(family.split('_')[1])//85)
                     for p in people for s in (2,3) for o in ocfg['orders']]
                    if family.startswith('fixed_') else [r for r in decisions if r['family']==family])
            assert len(chosen)==96 and len({(r['participant'],r['session'],r['order']) for r in chosen})==96
            these=[]
            for r in chosen:
                key=r['participant'],r['session'],r['order'];b=r['selected_budget']
                these.append(dict(seed=seed,family=family,participant=key[0],session=key[1],order=key[2],
                                  seconds=85*b,macro_f1=outcomes[key+(b,)],loss_vs_full=outcomes[key+(3,)]-outcomes[key+(b,)]))
            np.testing.assert_allclose(np.mean([r['macro_f1'] for r in these]),by[family]['mean_macro_f1'],rtol=0,atol=1e-12)
            np.testing.assert_allclose(np.mean([r['seconds'] for r in these]),by[family]['mean_seconds'],rtol=0,atol=1e-12)
            episodes.extend(these)
        all_rows.extend(dict(seed=seed,**r) for r in rows)
        order_rows.extend(dict(seed=seed,**r) for r in read(run/'per_order.json'))
        sources.append(dict(seed=seed,recognizer=str(prior.resolve()),observations=str(obs.resolve()),policy=str(run.resolve()),
                       training_summary=read(prior/'summary.json'),order_summary=read(obs/'summary.json'),
                       policy_summary=read(run/'summary.json'),validation_summary=read(run/'independent_validation.json'),
                       manifests={str(folder/'artifact_sha256.json'):hash_stream(folder/'artifact_sha256.json') for folder in (run,obs,prior)},
                       validation_sha256=hash_stream(run/'independent_validation.json')))
    assert set(seed_order)=={0,1,42} and len(people)==8
    contrasts=[]
    for seed in seed_order:
        by={r['family']:r for r in all_rows if r['seed']==seed}
        left={(r['participant'],r['session'],r['order']):r for r in episodes if r['seed']==seed and r['family']=='ridge'}
        right={(r['participant'],r['session'],r['order']):r for r in episodes if r['seed']==seed and r['family']=='pre_error'}
        contrasts.append(dict(seed=seed,ridge_minus_error_f1=by['ridge']['mean_macro_f1']-by['pre_error']['mean_macro_f1'],
                              ridge_minus_error_seconds=by['ridge']['mean_seconds']-by['pre_error']['mean_seconds'],
                              different_decisions=sum(left[k]['seconds']!=right[k]['seconds'] for k in left),episodes=96))
    person_rows=[]
    for seed in seed_order:
        for p in people:
            for family in FAMILIES:
                these=[r for r in episodes if r['seed']==seed and r['participant']==p and r['family']==family]
                assert len(these)==12
                person_rows.append(dict(seed=seed,participant=p,family=family,
                    mean_macro_f1=float(np.mean([r['macro_f1'] for r in these])),
                    mean_seconds=float(np.mean([r['seconds'] for r in these])),
                    mean_loss_vs_full=float(np.mean([r['loss_vs_full'] for r in these]))))
    failures=[]
    for seed in seed_order:
        worst=max((r for r in episodes if r['seed']==seed and r['family']=='ridge'),key=lambda r:r['loss_vs_full'])
        source=next(r for r in sources if r['seed']==seed);folder=Path(source['observations'])
        outcomes={(r['participant'],r['session'],r['order'],r['budget']):r for r in read(folder/'outcomes.json')}
        key=worst['participant'],worst['session'],worst['order'];budget=worst['seconds']//85
        with np.load(folder/'scoring_predictions.npz') as arrays:
            truth=arrays[f'p{key[0]}_s{key[1]}_labels']
            stopped=arrays[outcomes[key+(budget,)]['model_name']+'_predictions']
            full_predictions=arrays[outcomes[key+(3,)]['model_name']+'_predictions']
        early_f1=f1_score(truth,stopped,average=None,labels=list(range(17)),zero_division=0)
        full_f1=f1_score(truth,full_predictions,average=None,labels=list(range(17)),zero_division=0)
        np.testing.assert_allclose(np.mean(full_f1-early_f1),worst['loss_vs_full'],rtol=0,atol=1e-12)
        per_class=[dict(class_index=c,label=LABELS[c],stopped_f1=float(early_f1[c]),full_f1=float(full_f1[c]),
                       loss=float(full_f1[c]-early_f1[c])) for c in range(17)]
        failures.append(dict(**worst,classes=per_class,
            interpretation='Post-hoc illustration selected as the largest learned-policy loss within this seed; not a causal explanation or a tuning input.'))
    (args.out/'worst_learned_examples.json').write_text(json.dumps(failures,indent=2)+'\n')
    aggregated=[]
    for family in FAMILIES:
        these=[r for r in episodes if r['family']==family]
        row=dict(family=family,mean_seconds=float(np.mean([r['seconds'] for r in these])),
                 mean_macro_f1=float(np.mean([r['macro_f1'] for r in these])),
                 mean_loss_vs_full=float(np.mean([r['loss_vs_full'] for r in these])),
                 worst_loss_vs_full=max(r['loss_vs_full'] for r in these),
                 replays_loss_over_002=sum(r['loss_vs_full']>.02 for r in these),
                 people_with_loss_over_002=len({r['participant'] for r in these if r['loss_vs_full']>.02}))
        row['recording_reduction_fraction']=1-row['mean_seconds']/255
        if family in ('ridge','pre_error'):
            row['delta_vs_seedwise_cost_mixture']=float(np.mean([r['delta_vs_same_cost_mixture'] for r in all_rows if r['family']==family]))
            row['seeds_passing_development_screen']=sum(r['development_screen_pass'] for r in all_rows if r['family']==family)
        aggregated.append(row)
    # Resample people, retaining every seed/session/order for each sampled person.
    # Intervals condition on already fitted recognizers and nested policies.
    indexed={(r['seed'],r['participant'],r['family']):r for r in person_rows}
    metric_arrays={family:np.array([[indexed[seed,p,family]['mean_macro_f1'] for p in people] for seed in seed_order]) for family in FAMILIES}
    time_arrays={family:np.array([[indexed[seed,p,family]['mean_seconds'] for p in people] for seed in seed_order]) for family in FAMILIES}
    rng=np.random.default_rng(20260906);draws=rng.integers(0,len(people),size=(10000,len(people)))
    boot_f1=(metric_arrays['ridge']-metric_arrays['pre_error'])[:,draws].mean(axis=(0,2))
    boot_seconds=(time_arrays['ridge']-time_arrays['pre_error'])[:,draws].mean(axis=(0,2))
    contrast_ci=dict(ridge_minus_error_f1=float(np.mean(boot_f1)),
        f1_ci95=np.quantile(boot_f1,[.025,.975]).tolist(),seconds_ci95=np.quantile(boot_seconds,[.025,.975]).tolist(),
        interpretation='Conditional descriptive participant bootstrap, all three seeds retained; not 24 independent people or full model-selection uncertainty.')
    # Point estimate must be the observed average, not bootstrap Monte Carlo mean.
    contrast_ci['ridge_minus_error_f1']=float(np.mean(metric_arrays['ridge']-metric_arrays['pre_error']))
    contrast_ci['ridge_minus_error_seconds']=float(np.mean(time_arrays['ridge']-time_arrays['pre_error']))
    # Post-hoc diagnostic: label-independent mixture of simple and fixed routines.
    # All point estimates interpolate simple -> full. Bootstrap draws may need
    # the fixed85 -> simple branch; never select a favorable accuracy envelope.
    cost_diagnostic=[];boot_deltas=[]
    for si,seed in enumerate(seed_order):
        def reference(m):
            c=time_arrays['ridge'][si,m].mean(axis=-1)
            sc=time_arrays['pre_error'][si,m].mean(axis=-1)
            sf=metric_arrays['pre_error'][si,m].mean(axis=-1)
            low=metric_arrays['fixed_85'][si,m].mean(axis=-1)
            high=metric_arrays['fixed_255'][si,m].mean(axis=-1)
            upper=sf+(c-sc)/np.maximum(255-sc,1e-12)*(high-sf)
            lower=low+(c-85)/np.maximum(sc-85,1e-12)*(sf-low)
            return np.where(c>=sc,upper,lower)
        expected=float(reference(np.arange(len(people))))
        point=float(metric_arrays['ridge'][si].mean()-expected)
        bd=metric_arrays['ridge'][si,draws].mean(axis=-1)-reference(draws)
        boot_deltas.append(bd)
        cost_diagnostic.append(dict(seed=seed,expected_simple_fixed_mixture_f1=expected,
            learned_minus_mixture_f1=point,conditional_participant_ci95=np.quantile(bd,[.025,.975]).tolist()))
    pooled_boot=np.mean(boot_deltas,axis=0)
    diagnostic_summary=dict(scope='Post-hoc descriptive diagnostic added after the three seed outcomes; no policy changes or new primary screen.',
        per_seed=cost_diagnostic,pooled_delta=float(np.mean([r['learned_minus_mixture_f1'] for r in cost_diagnostic])),
        pooled_conditional_participant_ci95=np.quantile(pooled_boot,[.025,.975]).tolist())
    (args.out/'posthoc_simple_cost_match.json').write_text(json.dumps(diagnostic_summary,indent=2)+'\n')
    for name,records in (('all_seed_results',all_rows),('all_seed_per_order',order_rows),('per_episode',episodes),
                         ('per_participant',person_rows),('aggregate',aggregated),('learned_simple_contrasts',contrasts)):
        save_csv(args.out/f'{name}.csv',records)
    (args.out/'summary.json').write_text(json.dumps(dict(aggregate=aggregated,contrasts=contrasts,
        participant_bootstrap=contrast_ci,sources=sources),indent=2)+'\n')
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,3,figsize=(13,4.8),sharey=True)
    for ax,seed in zip(axes,seed_order):
        by={r['family']:r for r in all_rows if r['seed']==seed}
        ax.plot([85,170,255],[by[f'fixed_{b}']['mean_macro_f1'] for b in (85,170,255)],'o--',color='#6C7780',label='Fixed budgets / expected mixtures')
        for family,color,marker in (('pre_error','#BE7524','s'),('ridge','#137E8E','D')):
            r=by[family];ax.scatter(r['mean_seconds'],r['mean_macro_f1'],s=85,color=color,marker=marker,label=NAMES[family],zorder=3)
        ax.set(title=f'Training seed {seed}',xlabel='New-day recording (seconds)',xticks=[85,170,255]);ax.grid(alpha=.2)
    axes[0].set_ylabel('Mean macro-F1');axes[0].legend(fontsize=7,loc='lower right')
    fig.suptitle('Calibration stopping across three training seeds',x=.04,ha='left',fontsize=15)
    fig.text(.04,.02,'Eight development participants, two target days, six orders. All-order policy training; participants stay grouped in every fold.',fontsize=9)
    fig.tight_layout(rect=[0,.06,1,.93])
    for suffix in ('png','svg'): fig.savefig(args.out/f'seed_robustness.{suffix}',dpi=180)
    plt.close(fig)
    byagg={r['family']:r for r in aggregated};ridge=byagg['ridge'];simple=byagg['pre_error'];full=byagg['fixed_255']
    lines=['# Training-seed robustness results','', 'Generated '+datetime.now(timezone.utc).isoformat()+'. Public GRABMyo development evidence only.','',
           f'The learned controller passes the predeclared aggregate development screen in **{ridge["seeds_passing_development_screen"]}/3 seeds**; the simple pre-error controller in **{simple["seeds_passing_development_screen"]}/3**. Each screen requires at least 20% recording savings, mean F1 loss no greater than .01, and positive advantage over the expected fixed-budget mixture at the same mean cost. All seeds are retained.','',
           '## Primary comparison, every seed','',
           '| Seed | Method | Mean seconds | Macro-F1 | Loss vs full | Saving | Same-cost mixture advantage | Screen |',
           '| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |']
    for seed in seed_order:
        by={r['family']:r for r in all_rows if r['seed']==seed}
        for family in FAMILIES:
            r=by[family];adaptive=family in ('ridge','pre_error')
            lines.append(f'| {seed} | {NAMES[family]} | {r["mean_seconds"]:.1f} | {r["mean_macro_f1"]:.4f} | {r["mean_loss_vs_full"]:.4f} | {100*(1-r["mean_seconds"]/255):.1f}% | '+(f'{r["delta_vs_same_cost_mixture"]:.4f} | {"Pass" if r["development_screen_pass"] else "Fail"}' if adaptive else '— | —')+' |')
    lines += ['',f'![Training seed comparison]({(args.out/"seed_robustness.png").resolve()})','',
        '## Pooled descriptive result','',
        f'Averaging equally over seeds, people, sessions and orders: learned **{ridge["mean_seconds"]:.1f}s / F1 {ridge["mean_macro_f1"]:.4f}**, simple **{simple["mean_seconds"]:.1f}s / {simple["mean_macro_f1"]:.4f}**, and full calibration **255s / {full["mean_macro_f1"]:.4f}**. Learned savings are **{100*ridge["recording_reduction_fraction"]:.1f}%** with mean loss **{ridge["mean_loss_vs_full"]:.4f}**. A pooled result does not replace the separate seed checks.','',
        f'Learned minus simple: {contrast_ci["ridge_minus_error_f1"]:+.4f} macro-F1 and {contrast_ci["ridge_minus_error_seconds"]:+.1f}s. The conditional participant-bootstrap F1 interval is '+str([round(v,5) for v in contrast_ci['f1_ci95']])+', and the recording-time interval is '+str([round(v,2) for v in contrast_ci['seconds_ci95']])+'. This is a paired quality/cost difference, not an equal-cost claim.','',
        '| Seed | Learned − simple F1 | Learned − simple seconds | Different decisions |',
        '| --- | ---: | ---: | ---: |']
    for r in contrasts: lines.append(f'| {r["seed"]} | {r["ridge_minus_error_f1"]:+.5f} | {r["ridge_minus_error_seconds"]:+.2f} | {r["different_decisions"]}/96 |')
    lines += ['', '## Harms and uncertainty','',
        f'The worst learned loss versus full calibration is **{ridge["worst_loss_vs_full"]:.4f}**. Loss exceeds .02 in **{ridge["replays_loss_over_002"]}/288** learned-policy replays, involving **{ridge["people_with_loss_over_002"]}/8 people**. The simple rule exceeds .02 in {simple["replays_loss_over_002"]}/288. These are repeated evaluations of the same people and scoring trials; neither 288 replays nor 24 person-seed combinations are independent participants.','',
        'Per-seed and per-order tables include frozen-original policy controls and negative results. The five primary methods also have per-participant and per-episode tables. Bootstrap intervals resample the eight participants with all seeds/sessions/orders retained and condition on the fitted models/policies. They do not account for all development-driven project choices or retrain every model in every resample.','',
        'Recording savings apply to target-day five-second trials only. Day-1 enrollment remains 595 seconds; interaction/rest overhead and actual user benefit have not been measured. No closed-loop, clinical, controlled electrode-displacement or custom-device claim follows.','',
        '## What changed and what stayed fixed','',
        'Seeds 0 and 1 were specified before training; seed 42 is the preserved original. The seed changes neural initialization plus pretraining, enrollment and adaptation minibatch orders together. This is whole-pipeline sensitivity, not a decomposition of each source of randomness. Architecture, epochs, optimizer, learning rates, training-only normalization, population allocation, trial splits, all six round permutations and scoring sets are unchanged. No seed is selected for performance.','',
        'Each new seed repeats the original 224 shared/random, no/head/full recognizer variants, then 64 missing subset fits. Original-order observations are recreated from that same seed; all-order nested policies fit and tune only on other participants. The main families are the learned eight-feature ridge gain predictor and the simple pre-update error rule. Policy grids and selection criteria are unchanged. Original-order frozen controls remain in the saved results.','',
        'Input hashes and saved predictions/access were independently checked for every new recognizer and order run. The order audit recomputes 192 observation states, 80 policy artifacts and 960 decision traces per seed and independently refits one new subset model per person. The report also checks identical protocol/normalization and recomputes primary means directly from the selected episode outcomes. It does not rerun all pretraining or every nested search as a second independent implementation.','',
        '## Next decision','',
        'Use these seed-specific tradeoffs to decide whether the learned component earns its added complexity; retain the simple rule if it provides the better practical tradeoff. Complete the published TDAR plus retained-RMS classical recognition control, then freeze the final procedure before opening the 15 reserved participants. Independent evaluation and a second public dataset remain necessary for the stronger research/portfolio claim. September 9 submission readiness remains the target, not a guarantee.','',
        '## Output provenance','']
    lines[-2:-2]=['## Illustrative largest losses, selected after evaluation','',
        '| Seed | Participant / session / order | Stop budget | Macro-F1 loss | Largest class F1 loss |',
        '| --- | --- | ---: | ---: | --- |'] + [
        f'| {r["seed"]} | {r["participant"]} / {r["session"]} / {r["order"]} | {r["seconds"]}s | {r["loss_vs_full"]:.4f} | '+
        max(r['classes'],key=lambda c:c['loss'])['label']+' |' for r in failures] + [
        '', 'These examples use the same fixed scoring windows. They identify where early stopping lost quality, not why the signals changed. Equivalent orders can share the same stopped model. Complete 17-class values are saved in worst_learned_examples.json; these diagnostics do not change the policy.', '']
    lines[-2:-2]=['## Post-hoc comparison with the simple rule at equal cost','',
        'This diagnostic was added after inspecting the three seed outcomes. It does not replace the predeclared screens or tune any policy. At each observed seed mean, randomly mix the simple pre-error rule with full calibration to match the learned policy mean recording cost. The mixture probability is label-independent; cost matching is retrospective. Compare expected F1, not a favorable realized random draw.', '',
        '| Seed | Learned − equal-cost simple/full mixture F1 | Conditional participant interval |',
        '| --- | ---: | --- |']+[
        f'| {r["seed"]} | {r["learned_minus_mixture_f1"]:+.5f} | '+str([round(v,5) for v in r['conditional_participant_ci95']])+' |'
        for r in cost_diagnostic]+['',
        f'Pooled difference: {diagnostic_summary["pooled_delta"]:+.5f}; conditional participant interval '+str([round(v,5) for v in diagnostic_summary['pooled_conditional_participant_ci95']])+'. Bootstrap draws retain all seeds; if a draw puts learned cost below simple cost, the reference instead mixes fixed85 with simple. This estimates additional accuracy at matched mean cost, not a causal benefit, universal harm protection or independent confirmation.', '']
    new_sources=[r for r in sources if r['seed'] in (0,1)]
    lines[-2:-2]=['## Measured runtime for the two new seeds','',
        f'The two recognizer runs together took {sum(r["training_summary"]["total_seconds"] for r in new_sources)/60:.2f} minutes, including {sum(r["training_summary"]["pretraining_seconds"] for r in new_sources)/60:.2f} minutes of population pretraining. Additional all-order fitting/extraction took {sum(r["order_summary"]["total_seconds"] for r in new_sources):.2f}s; all-order nested policy analysis took {sum(r["policy_summary"]["total_seconds"] for r in new_sources):.2f}s. Independent all-order audits took {sum(r["validation_summary"]["total_seconds"] for r in new_sources):.2f}s. These figures exclude implementation/report preparation, original-order policy stages and recognizer audits; they are not total task wall time.', '']
    for source in sources: lines.append(f'- Seed {source["seed"]}: `{source["policy"]}`; recognizer `{source["recognizer"]}`.')
    lines += [f'- Tables, chart, summary and generator snapshot: `{args.out.resolve()}`.',
              '- No final-participant signals or custom recordings were used in this experiment.','']
    report=Path('research/TRAINING_SEED_RESULTS.md');report.write_text('\n'.join(lines))
    for file in (__file__,'scripts/run_seed_stopping.py','tests/test_training_seed.py','src/grabmyo.py'):
        (args.out/Path(file).name).write_bytes(Path(file).read_bytes())
    (args.out/'provenance.json').write_text(json.dumps(dict(created_utc=datetime.now(timezone.utc).isoformat(),report=str(report.resolve()),
        report_sha256=hash_stream(report),protocol=baseline,sources=sources),indent=2)+'\n')
    (args.out/'artifact_sha256.json').write_text(json.dumps({str(p.relative_to(args.out)):hash_stream(p) for p in args.out.rglob('*') if p.is_file()},indent=2)+'\n')
    print(json.dumps(dict(report=str(report.resolve()),aggregate=aggregated,contrasts=contrasts,bootstrap=contrast_ci)))


if __name__=='__main__':
    main()
