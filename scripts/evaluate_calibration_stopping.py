"""Nested participant-held-out pilot. Hyperparameters never use outer outcomes."""
import argparse
import json
import time
from pathlib import Path
from datetime import datetime, timezone
import joblib
import numpy as np
from src.calibration_stopping import Policy, tune_policy, decisions, assess, candidates, fit_policy, outcome_key, episode_key
from src.grabmyo_corpus import hash_stream

FAMILIES=('fixed','pre_error','post_entropy','probability_change','ridge')


def cost_matched_f1(seconds, fixed):
    budgets=np.array([85.,170.,255.])
    return float(np.interp(seconds,budgets,[fixed[b] for b in (1,2,3)]))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--features',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--frozen-policies',type=Path)
    args=parser.parse_args(); start=time.perf_counter()
    args.out.mkdir(parents=True,exist_ok=False)
    for relative,expected in json.loads((args.features/'artifact_sha256.json').read_text()).items():
        if hash_stream(args.features/relative)!=expected:
            raise ValueError('Changed observation artifact')
    rows=json.loads((args.features/'features.json').read_text())
    outcomes={outcome_key(r,r['budget']):r['macro_f1'] for r in json.loads((args.features/'outcomes.json').read_text())}
    people=sorted({r['participant'] for r in rows})
    episodes=sorted({episode_key(r) for r in rows})
    orders=sorted({r.get('order','123') for r in rows})
    families=list(FAMILIES)
    frozen_folds={}
    if args.frozen_policies:
        if not (args.frozen_policies/'independent_validation.json').exists():
            raise ValueError('Frozen policies require prior verification')
        for relative,expected in json.loads((args.frozen_policies/'artifact_sha256.json').read_text()).items():
            if hash_stream(args.frozen_policies/relative)!=expected:
                raise ValueError('Frozen policy artifact changed')
        frozen_folds={(r['held_participant'],r['family']):r for r in json.loads((args.frozen_policies/'folds.json').read_text())}
        families += ['frozen_'+f for f in FAMILIES]
    config={'created_utc':datetime.now(timezone.utc).isoformat(),'observations':str(args.features.resolve()),
            'observation_hashes':{f:hash_stream(args.features/f) for f in ('features.json','outcomes.json','config.json')},
            'participants':people,'orders':orders,'families':families,'candidates':{f:candidates(f) for f in FAMILIES},
            'selection':'Nested leave-one-participant-out; inner mean F1 loss <=.01 then minimize time, tie smallest loss; fallback full255.',
            'ridge_target':'(F1 at budget3 - F1 at current budget)/(3-current budget); average remaining gain per extra round.',
            'primary_family':'ridge','scope':'Exploratory development; original model/scope decisions already informed by these participants.',
            'code_hashes':{f:hash_stream(f) for f in ('src/calibration_stopping.py','scripts/evaluate_calibration_stopping.py')}}
    if args.frozen_policies:
        config['frozen_policies']=str(args.frozen_policies.resolve())
        config['frozen_artifact_manifest_sha256']=hash_stream(args.frozen_policies/'artifact_sha256.json')
        config['primary_robustness_comparison']='Previously frozen original-order policies on all orders; all-order nested refitting is a separate predeclared extension.'
    (args.out/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    models=args.out/'policies'; models.mkdir()
    all_decisions=[]; folds=[]; curve_decisions=[]
    # Only one outer participant's features are passed to decisions; held-out
    # scoring outcomes are first used by assess after all policy decisions.
    for held in people:
        train=[r for r in rows if r['participant']!=held]
        train_y={k:v for k,v in outcomes.items() if k[0]!=held}
        test=[r for r in rows if r['participant']==held]
        for family in FAMILIES:
            policy,selection=tune_policy(train,train_y,family)
            choice=decisions(policy,test)
            all_decisions.extend([{**r,'family':family} for r in choice])
            folds.append({'held_participant':held,'family':family,'fit_participants':sorted({r['participant'] for r in train}),
                          'inner_participants':sorted({k[0] for k in train_y}),'fit_orders':orders,'selection':selection})
            joblib.dump(policy,models/f'p{held}_{family}.joblib')
            # Descriptive operating curves, not a second primary selection.
            for parameter,alpha in candidates(family):
                if family=='ridge' and alpha!=10.:
                    continue
                curve_policy=fit_policy(train,train_y,family,parameter,alpha)
                curve_decisions.extend([{**r,'family':family,'parameter':parameter,'alpha':alpha}
                                        for r in decisions(curve_policy,test)])
            if args.frozen_policies:
                frozen=joblib.load(args.frozen_policies/'policies'/f'p{held}_{family}.joblib')
                prior_fold=frozen_folds[held,family]
                if set(prior_fold['fit_participants'])!=set(people)-{held}:
                    raise ValueError('Frozen policy trained on outer participant')
                frozen_name='frozen_'+family
                all_decisions.extend([{**r,'family':frozen_name} for r in decisions(frozen,test)])
                folds.append({**prior_fold,'family':frozen_name,'fit_orders':['123'],'reused_from':str(args.frozen_policies/'policies'/f'p{held}_{family}.joblib')})
                joblib.dump(frozen,models/f'p{held}_{frozen_name}.joblib')
        print(json.dumps({'phase':'outer_fold','held_participant':held,'completed_folds':len(folds)}),flush=True)
    fixed={}
    results=[]
    for budget in (1,2,3):
        chosen=decisions(Policy('fixed',budget),rows)
        result=assess(chosen,outcomes); fixed[budget]=result['mean_macro_f1']
        results.append({'family':f'fixed_{85*budget}',**result})
    for family in families:
        chosen=[r for r in all_decisions if r['family']==family]
        result=assess(chosen,outcomes)
        mixture=cost_matched_f1(result['mean_seconds'],fixed)
        result['same_cost_fixed_mixture_f1']=mixture
        result['delta_vs_same_cost_mixture']=result['mean_macro_f1']-mixture
        result['recording_reduction_fraction']=1-result['mean_seconds']/255
        result['development_screen_pass']=(result['recording_reduction_fraction']>=.2 and result['mean_loss_vs_full']<=.01 and result['delta_vs_same_cost_mixture']>0)
        # Paired participant bootstrap: recompute both controller and mixture.
        rng=np.random.default_rng(20260906); boot=[]
        for _ in range(10000):
            draw=rng.choice(people,len(people),replace=True)
            repeated=[r for p in draw for r in chosen if r['participant']==p]
            stats=assess(repeated,outcomes)
            mix_fixed={b:float(np.mean([outcomes[key+(b,)] for p in draw for key in episodes if key[0]==p])) for b in (1,2,3)}
            boot.append(stats['mean_macro_f1']-cost_matched_f1(stats['mean_seconds'],mix_fixed))
        result['descriptive_participant_bootstrap_delta_ci95']=np.quantile(boot,[.025,.975]).tolist()
        results.append({'family':family,**result})
    curves=[]
    for family in FAMILIES:
        for parameter,alpha in candidates(family):
            selected=[r for r in curve_decisions if r['family']==family and r['parameter']==parameter and r['alpha']==alpha]
            if selected:
                curves.append({'family':family,'parameter':parameter,'alpha':alpha,**assess(selected,outcomes)})
    per_order=[]
    for order in orders:
        selected_rows=[r for r in rows if r.get('order','123')==order]
        order_fixed={b:assess(decisions(Policy('fixed',b),selected_rows),outcomes)['mean_macro_f1'] for b in (1,2,3)}
        for family in families:
            chosen=[r for r in all_decisions if r['family']==family and r.get('order','123')==order]
            result=assess(chosen,outcomes)
            result['delta_vs_same_cost_mixture']=result['mean_macro_f1']-cost_matched_f1(result['mean_seconds'],order_fixed)
            per_order.append({'order':order,'family':family,**result})
    for name,value in (('decisions',all_decisions),('folds',folds),('results',results),('curves',curves),('curve_decisions',curve_decisions),('per_order',per_order)):
        (args.out/f'{name}.json').write_text(json.dumps(value,indent=2)+'\n')
    summary={'completed_utc':datetime.now(timezone.utc).isoformat(),'total_seconds':time.perf_counter()-start,
             'outer_people':len(people),'participant_session_pairs':len({(r['participant'],r['session']) for r in rows}),
             'order_episodes':len(episodes),'orders':orders,'outer_policy_fits':len(people)*len(FAMILIES),
             'frozen_policy_reuses':len(folds)-len(people)*len(FAMILIES),
             'final_participants_accessed':0,'result_scope':'Development pilot, one seed, all declared orders; cluster by participant'}
    (args.out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    snapshot=args.out/'code_snapshot'; snapshot.mkdir()
    for f in config['code_hashes']:
        (snapshot/Path(f).name).write_bytes(Path(f).read_bytes())
    (args.out/'artifact_sha256.json').write_text(json.dumps({str(p.relative_to(args.out)):hash_stream(p) for p in args.out.rglob('*') if p.is_file()},indent=2)+'\n')
    print(json.dumps({'summary':summary,'results':results}),flush=True)


if __name__=='__main__':
    main()
