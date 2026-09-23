"""Check report arithmetic and bootstrap with independent scalar interpolation."""
import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from src.grabmyo_corpus import hash_stream


def read(path):
    return json.loads(path.read_text())


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);args=p.parse_args()
    summary=read(args.run/'summary.json')
    with (args.run/'per_episode.csv').open() as f: rows=list(csv.DictReader(f))
    for r in rows:
        for k in ('seed','participant','session','seconds'):r[k]=int(r[k])
        for k in ('macro_f1','loss_vs_full'):r[k]=float(r[k])
    seeds=[r['seed'] for r in summary['sources']];people=sorted({r['participant'] for r in rows})
    assert len(people)==8 and set(seeds)=={0,1,42}
    for source in summary['sources']:
        outcomes={(r['participant'],r['session'],r['order'],r['budget']):r['macro_f1']
                  for r in read(Path(source['observations'])/'outcomes.json')}
        choices={(r['participant'],r['session'],r['order'],r['family']):r['selected_budget']
                 for r in read(Path(source['policy'])/'decisions.json')}
        for row in [r for r in rows if r['seed']==source['seed']]:
            key=row['participant'],row['session'],row['order'];budget=row['seconds']//85
            if not row['family'].startswith('fixed_'):assert choices[key+(row['family'],)]==budget
            assert row['macro_f1']==outcomes[key+(budget,)]
            np.testing.assert_allclose(row['loss_vs_full'],outcomes[key+(3,)]-outcomes[key+(budget,)],atol=1e-12,rtol=0)
    for target in summary['aggregate']:
        these=[r for r in rows if r['family']==target['family']];assert len(these)==288
        np.testing.assert_allclose(target['mean_macro_f1'],np.mean([r['macro_f1'] for r in these]),atol=1e-12,rtol=0)
        np.testing.assert_allclose(target['mean_seconds'],np.mean([r['seconds'] for r in these]),atol=1e-12,rtol=0)
        assert target['replays_loss_over_002']==sum(r['loss_vs_full']>.02 for r in these)
        assert target['people_with_loss_over_002']==len({r['participant'] for r in these if r['loss_vs_full']>.02})
    families=('fixed_85','pre_error','ridge','fixed_255');stats={}
    for seed in seeds:
        for person in people:
            for family in families:
                these=[r for r in rows if (r['seed'],r['participant'],r['family'])==(seed,person,family)]
                assert len(these)==12
                stats[seed,person,family]=np.mean([[r['seconds'],r['macro_f1']] for r in these],axis=0)
    diagnostic=read(args.run/'posthoc_simple_cost_match.json')
    draws=np.random.default_rng(20260906).choice(people,(10000,len(people)),replace=True)
    all_deltas=[];raw=[];durations=[]
    for seed in seeds:
        saved=next(r for r in diagnostic['per_seed'] if r['seed']==seed);values=[]
        for draw in [np.array(people),*draws]:
            means={f:np.mean([stats[seed,int(person),f] for person in draw],axis=0) for f in families}
            expected=np.interp(means['ridge'][0], [85,means['pre_error'][0],255],
                               [means['fixed_85'][1],means['pre_error'][1],means['fixed_255'][1]])
            values.append([means['ridge'][1]-expected,means['ridge'][1]-means['pre_error'][1],
                           means['ridge'][0]-means['pre_error'][0]])
        values=np.array(values)
        np.testing.assert_allclose(saved['learned_minus_mixture_f1'],values[0,0],atol=1e-12,rtol=0)
        np.testing.assert_allclose(saved['conditional_participant_ci95'],np.quantile(values[1:,0],[.025,.975]),atol=1e-12,rtol=0)
        all_deltas.append(values[1:,0]);raw.append(values[1:,1]);durations.append(values[1:,2])
    for expected,observed in ((diagnostic['pooled_conditional_participant_ci95'],all_deltas),
                             (summary['participant_bootstrap']['f1_ci95'],raw),
                             (summary['participant_bootstrap']['seconds_ci95'],durations)):
        np.testing.assert_allclose(expected,np.quantile(np.mean(observed,axis=0),[.025,.975]),atol=1e-12,rtol=0)
    result=dict(verified_utc=datetime.now(timezone.utc).isoformat(),status='passed',episode_rows_checked=len(rows),
                participants=8,seeds=seeds,bootstrap_draws=10000,scalar_mixture_checks=30003,
                checks='Primary episode choices/outcomes, aggregate means and harm counts; participant-group bootstrap and independent np.interp cost diagnostic.',
                verifier_sha256=hash_stream(__file__))
    (args.run/'report_validation.json').write_text(json.dumps(result,indent=2)+'\n')
    (args.run/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    print(json.dumps(result))


if __name__=='__main__':main()
