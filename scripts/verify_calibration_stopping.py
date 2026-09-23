"""Readback audit: CPU probabilities, independent summaries, folds and costs."""
import argparse
import json
from pathlib import Path
from datetime import datetime, timezone
import joblib
import numpy as np
import torch
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from src.calibration_stopping import FEATURES, observed_features
from src.emg_cnn import CompactEMGNet
from src.grabmyo_corpus import WindowCorpus, hash_stream


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--observations',type=Path,required=True)
    parser.add_argument('--policies',type=Path,required=True)
    args=parser.parse_args()
    for root in (args.observations,args.policies):
        for relative,expected in json.loads((root/'artifact_sha256.json').read_text()).items():
            if hash_stream(root/relative)!=expected:
                raise ValueError('Changed artifact '+relative)
        cfg=json.loads((root/'config.json').read_text())
        for source,expected in cfg['code_hashes'].items():
            if hash_stream(root/'code_snapshot'/Path(source).name)!=expected:
                raise ValueError('Changed code snapshot')
    cfg=json.loads((args.observations/'config.json').read_text())
    prior=Path(cfg['prior']); cache=Path(cfg['cache'])
    for p,expected in cfg['hashes'].items():
        if hash_stream(p)!=expected:
            raise ValueError('Changed upstream run')
    prior_hashes=json.loads((prior/'artifact_sha256.json').read_text())
    pcfg=json.loads((args.policies/'config.json').read_text())
    for relative,expected in pcfg['observation_hashes'].items():
        if hash_stream(args.observations/relative)!=expected:
            raise ValueError('Policy observation mismatch')
    dev=WindowCorpus(cache,'development'); torch.set_num_threads(4)
    scale_data=np.load(prior/'training_scaler.npz')
    probabilities=np.load(args.observations/'observation_probabilities.npz')
    ledger=json.loads((args.observations/'access.json').read_text())
    rows=json.loads((args.observations/'features.json').read_text())
    indexed={(r['participant'],r['session'],int(r['budget'])):r for r in rows}
    record_index={r['record']:i for i,r in enumerate(dev.rows)}
    upstream_access={r['name']:r for r in json.loads((prior/'data_access.json').read_text())}
    max_difference=0.
    for entry in ledger:
        p,s,b=entry['participant'],entry['session'],entry['budget']
        wanted=[r['record'] for r in dev.rows if int(r['participant'])==p and int(r['session'])==s and r['role']=='calibration' and int(r['calibration_rank'])==b]
        assert entry['observation_records']==wanted and len(wanted)==17
        expected_acquired={r['record'] for r in dev.rows if int(r['participant'])==p and int(r['session'])==s and r['role']=='calibration' and int(r['calibration_rank'])<=b}
        assert set(entry['acquired_records'])==expected_acquired
        assert not expected_acquired & set(entry['scoring_records_excluded'])
        ids=[record_index[r] for r in wanted]
        x,y=dev.batch(dev.window_ids(ids),scale_data['mean'],scale_data['scale'])
        np.testing.assert_array_equal(y,probabilities[entry['key']+'_labels'])
        cpu_probs=[]
        for position,name in enumerate(entry['checkpoint_names']):
            checkpoint=prior/'models'/f'{name}.pt'
            assert hash_stream(checkpoint)==prior_hashes[f'models/{name}.pt']
            fitted=set(upstream_access[name]['calibration'])
            assert fitted<=expected_acquired
            if position==0:
                assert not fitted & set(wanted)
            else:
                assert fitted==expected_acquired
            model=CompactEMGNet()
            model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True));model.eval()
            with torch.no_grad():
                logits=model(torch.from_numpy(x)).numpy().astype(np.float64)
            exp=np.exp(logits-logits.max(1,keepdims=True)); prob=exp/exp.sum(1,keepdims=True)
            cpu_probs.append(prob)
            saved=probabilities[entry['key']+('_before' if position==0 else '_after')]
            max_difference=max(max_difference,float(np.abs(saved-prob).max()))
            np.testing.assert_allclose(saved,prob,rtol=3e-4,atol=3e-6)
        # Exact feature extraction from stored observations; CPU tolerance tested above.
        regenerated=observed_features(probabilities[entry['key']+'_before'],probabilities[entry['key']+'_after'],y,b)
        for feature,value in regenerated.items():
            np.testing.assert_allclose(value,indexed[p,s,b][feature],rtol=0,atol=1e-12)
    outcomes={(r['participant'],r['session'],r['budget']):r['macro_f1'] for r in json.loads((args.observations/'outcomes.json').read_text())}
    source_outcomes={(r['participant'],r['session'],r['budget']):r['macro_f1'] for r in json.loads((prior/'metrics.json').read_text()) if r['initialization']=='shared' and r['mode']=='full'}
    assert outcomes==source_outcomes
    choices=json.loads((args.policies/'decisions.json').read_text())
    people=set(cfg['participants'])
    for fold in json.loads((args.policies/'folds.json').read_text()):
        held,family=fold['held_participant'],fold['family']
        assert set(fold['fit_participants'])==set(fold['inner_participants'])==people-{held}
        train=[r for r in rows if r['participant']!=held]
        policy=joblib.load(args.policies/'policies'/f'p{held}_{family}.joblib')
        selection=fold['selection']
        if selection['fallback']:
            assert policy.family=='fixed' and policy.parameter==3
        else:
            acceptable=[r for r in selection['candidates'] if r['mean_loss_vs_full']<=.01+1e-12]
            best=min(acceptable,key=lambda r:(r['mean_seconds'],r['mean_loss_vs_full']))
            assert best==selection['selected']
            assert policy.parameter==best['parameter'] and policy.alpha==best['alpha']
            if family=='ridge':
                x=np.array([[r[f] for f in FEATURES] for r in train])
                target=np.array([(outcomes[r['participant'],r['session'],3]-outcomes[r['participant'],r['session'],int(r['budget'])])/(3-r['budget']) for r in train])
                independent=make_pipeline(StandardScaler(),Ridge(alpha=best['alpha'])).fit(x,target)
                np.testing.assert_allclose(policy.model.predict(x),independent.predict(x),atol=1e-12,rtol=0)
                assert policy.model.named_steps['standardscaler'].n_samples_seen_==len(train)
            elif family!='fixed':
                for b in (1,2):
                    q=best['parameter']; values=[r[family] for r in train if r['budget']==b]
                    expected=-np.inf if q<0 else np.inf if q>1 else float(np.quantile(values,q))
                    assert policy.thresholds[b]==expected
        for choice in [r for r in choices if r['participant']==held and r['family']==family]:
            observed=[]; selected=3
            for b in (1,2):
                record=indexed[held,choice['session'],b]
                stop,value=policy.decide(record)
                observed.append({'budget':b,'stop':bool(stop),'value':float(value)})
                if stop:
                    selected=b;break
            assert observed==choice['trace'] and selected==choice['selected_budget']
            assert [r['budget'] for r in observed]==list(range(1,min(selected,2)+1))
    results=json.loads((args.policies/'results.json').read_text())
    fixed={b:np.mean([v for (p,s,k),v in outcomes.items() if k==b]) for b in (1,2,3)}
    for result in results:
        family=result['family']
        if family.startswith('fixed_'):
            b=int(family.split('_')[1])//85
            selected=[(p,s,b) for p in people for s in (2,3)]
        else:
            selected=[(r['participant'],r['session'],r['selected_budget']) for r in choices if r['family']==family]
        assert len(selected)==16
        f1=np.array([outcomes[k] for k in selected]); loss=np.array([outcomes[p,s,3]-outcomes[p,s,b] for p,s,b in selected]);seconds=np.mean([b*85 for p,s,b in selected])
        for k,v in {'mean_macro_f1':f1.mean(),'mean_seconds':seconds,'mean_loss_vs_full':loss.mean(),'worst_loss_vs_full':loss.max(),'fraction_loss_over_002':np.mean(loss>.02)}.items():
            np.testing.assert_allclose(result[k],v,rtol=0,atol=1e-12)
        if not family.startswith('fixed_'):
            expected=float(np.interp(seconds,[85,170,255],[fixed[b] for b in (1,2,3)]))
            np.testing.assert_allclose(result['same_cost_fixed_mixture_f1'],expected,atol=1e-12,rtol=0)
    report={'completed_utc':datetime.now(timezone.utc).isoformat(),'status':'passed','states_verified':len(ledger),
            'cpu_checkpoint_probability_evaluations':len(ledger)*2,'max_cpu_vs_saved_probability_difference':max_difference,
            'outer_policies_verified':40,'decisions_verified':len(choices),'summary_rows_verified':len(results),
            'checks':'Artifact/source hashes, calibration ranks/access, unseen-round predictions, CPU probabilities, exact features, upstream outcomes, selected inner constraints, training-only ridge/scalers/quantiles, reachable decision traces and independent metric/cost arithmetic.',
            'limits':'Does not establish policy generalization, inspect new calibration orders, or repeat full inner-CV candidate training. No final participants or scoring signals read.',
            'verifier_sha256':hash_stream(__file__)}
    (args.policies/'independent_validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':
    main()
