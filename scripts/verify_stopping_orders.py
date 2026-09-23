"""Verify all calibration subsets, order paths, policy grouping and metrics."""
import argparse
import copy
import json
import time
from pathlib import Path
from datetime import datetime, timezone
import joblib
import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score
from src.emg_cnn import CompactEMGNet
from src.grabmyo_corpus import WindowCorpus, hash_stream
from src.calibration_orders import ORDERS
from src.calibration_stopping import FEATURES, observed_features
from scripts.pilot_emg_cnn import predict, fit


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--observations',type=Path,required=True)
    parser.add_argument('--policies',type=Path,required=True)
    args=parser.parse_args();start=time.perf_counter()
    for root in (args.observations,args.policies):
        for file,expected in json.loads((root/'artifact_sha256.json').read_text()).items():
            assert hash_stream(root/file)==expected, file
        cfg=json.loads((root/'config.json').read_text())
        for file,expected in cfg['code_hashes'].items():
            assert hash_stream(root/'code_snapshot'/Path(file).name)==expected,file
    ocfg=json.loads((args.observations/'config.json').read_text())
    pcfg=json.loads((args.policies/'config.json').read_text())
    prior=Path(ocfg['prior']);cache=Path(ocfg['cache'])
    assert hash_stream(cache/'summary.json')==ocfg['cache_summary_sha256']
    for file,expected in ocfg['input_hashes'].items():
        assert hash_stream(file)==expected,file
    for file,expected in pcfg['observation_hashes'].items():
        assert hash_stream(args.observations/file)==expected,file
    old_policies=Path(pcfg['frozen_policies'])
    assert hash_stream(old_policies/'artifact_sha256.json')==pcfg['frozen_artifact_manifest_sha256']
    for file,expected in json.loads((old_policies/'artifact_sha256.json').read_text()).items():
        assert hash_stream(old_policies/file)==expected
    dev=WindowCorpus(cache,'development');torch.set_num_threads(4)
    assert set(ocfg['orders'])==set(ORDERS)
    scale_data=np.load(prior/'training_scaler.npz'); mean,scale=scale_data['mean'],scale_data['scale']
    saved=np.load(args.observations/'observation_probabilities.npz')
    scoring=np.load(args.observations/'scoring_predictions.npz')
    features=json.loads((args.observations/'features.json').read_text())
    lookup={(r['participant'],r['session'],r['order'],int(r['budget'])):r for r in features}
    assert len(lookup)==192
    outcomes={(r['participant'],r['session'],r['order'],r['budget']):r for r in json.loads((args.observations/'outcomes.json').read_text())}
    ledgers=json.loads((args.observations/'model_access.json').read_text());models={};scores={}
    index={r['record']:i for i,r in enumerate(dev.rows)}
    def ids(p,s,role,ranks=None):
        return [i for i,r in enumerate(dev.rows) if int(r['participant'])==p and int(r['session'])==s and r['role']==role and (ranks is None or int(r['calibration_rank']) in ranks)]
    def records(indices):
        return [dev.rows[i]['record'] for i in indices]
    def load(name,device='cpu'):
        model=CompactEMGNet().to(device)
        model.load_state_dict(torch.load(args.observations/'models'/f'{name}.pt',map_location=device,weights_only=True));model.eval()
        return model
    scored=0;replicated=[]
    for entry in ledgers:
        name,p,s=entry['name'],entry['participant'],entry['session']
        assert entry['calibration']==records(ids(p,s,'calibration',entry['ranks']))
        assert entry['source']==records(ids(p,1,'enrollment'))
        assert entry['score']==records(ids(p,s,'scoring'))
        assert not set(entry['calibration']+entry['source'])&set(entry['score'])
        models[name]=load(name)
        if entry['reused_from']:
            old=torch.load(prior/'models'/(entry['reused_from']+'.pt'),map_location='cpu',weights_only=True)
            assert all(torch.equal(value,old[k]) for k,value in models[name].state_dict().items())
        if entry['ranks']:
            if (p,s) not in scores:
                scores[p,s]=dev.batch(dev.window_ids(ids(p,s,'scoring')),mean,scale)
            x,y=scores[p,s]
            np.testing.assert_array_equal(y,scoring[f'p{p}_s{s}_labels'])
            # Reproduce saved scoring predictions on the training device.
            model=models[name].to(ocfg['device']); pred=predict(model,x,ocfg['device']);model.cpu()
            np.testing.assert_array_equal(pred,scoring[name+'_predictions'])
            f1=float(f1_score(y,pred,average='macro',labels=list(range(17)),zero_division=0))
            for row in outcomes.values():
                if row['model_name']==name:
                    assert row['macro_f1']==f1
            scored+=1
        # Refit one declared non-prefix subset per participant, checks the new
        # training path without redundantly repeating all 64 new fits.
        if s==2 and entry['ranks']==[2]:
            base=load(f'p{p}_s{s}_r0',ocfg['device']);base.set_update_mode('full')
            x,y=dev.batch(dev.window_ids(ids(p,s,'calibration',[2])),mean,scale)
            fit(base,x,y,10,.0001,ocfg['seed'],ocfg['device'])
            expected=models[name].state_dict()
            assert all(torch.equal(v.cpu(),expected[k]) for k,v in base.state_dict().items())
            replicated.append(name)
    observations=json.loads((args.observations/'access.json').read_text())
    model_access={r['name']:r for r in ledgers}
    cpu_cache={};max_difference=0.
    for entry in observations:
        p,s,order,b=entry['participant'],entry['session'],entry['order'],entry['budget']
        ranks=list(map(int,order));new_rank=ranks[b-1]
        assert entry['observation_records']==records(ids(p,s,'calibration',[new_rank]))
        assert set(entry['acquired_records'])==set(records(ids(p,s,'calibration',ranks[:b])))
        assert entry['scoring_records_excluded']==records(ids(p,s,'scoring'))
        x,y=dev.batch(dev.window_ids(ids(p,s,'calibration',[new_rank])),mean,scale)
        np.testing.assert_array_equal(y,saved[entry['key']+'_labels'])
        for position,name in enumerate(entry['checkpoint_names']):
            actual=set(model_access[name]['calibration'])
            expected=set(records(ids(p,s,'calibration',ranks[:b-1] if position==0 else ranks[:b])))
            assert actual==expected
            if position==0:
                assert not actual&set(entry['observation_records'])
            key=(name,new_rank)
            if key not in cpu_cache:
                with torch.no_grad():
                    logits=models[name](torch.from_numpy(x)).numpy().astype(np.float64)
                exp=np.exp(logits-logits.max(1,keepdims=True));cpu_cache[key]=exp/exp.sum(1,keepdims=True)
            prob=cpu_cache[key];stored=saved[entry['key']+('_before' if position==0 else '_after')]
            max_difference=max(max_difference,float(np.max(np.abs(prob-stored))))
            np.testing.assert_allclose(prob,stored,rtol=4e-4,atol=4e-6)
        fresh=observed_features(saved[entry['key']+'_before'],saved[entry['key']+'_after'],y,b)
        for key,value in fresh.items():
            assert value==lookup[p,s,order,b][key]
    # Original feature values and checkpoint outcomes must still reproduce.
    old_features=json.loads((Path(ocfg['original_observations'])/'features.json').read_text())
    for r in old_features:
        new=lookup[r['participant'],r['session'],'123',int(r['budget'])]
        for k in FEATURES:
            assert r[k]==new[k]
    old_outcomes=json.loads((Path(ocfg['original_observations'])/'outcomes.json').read_text())
    for r in old_outcomes:
        assert r['macro_f1']==outcomes[r['participant'],r['session'],'123',r['budget']]['macro_f1']
    old_decisions={(r['participant'],r['session'],r['family']):r for r in json.loads((old_policies/'decisions.json').read_text())}
    choices=json.loads((args.policies/'decisions.json').read_text())
    people=set(ocfg['participants'])
    assert len(choices)==960
    for fold in json.loads((args.policies/'folds.json').read_text()):
        held,family=fold['held_participant'],fold['family'];base_family=family.removeprefix('frozen_')
        is_frozen=family.startswith('frozen_');orders=['123'] if is_frozen else list(ORDERS)
        assert set(fold['fit_participants'])==set(fold['inner_participants'])==people-{held}
        assert set(fold['fit_orders'])==set(orders)
        train=[r for r in features if r['participant']!=held and r['order'] in orders]
        assert len(train)==(28 if is_frozen else 168)
        for p in people-{held}:
            assert {r['order'] for r in train if r['participant']==p}==set(orders)
        policy=joblib.load(args.policies/'policies'/f'p{held}_{family}.joblib')
        if is_frozen:
            old=joblib.load(old_policies/'policies'/f'p{held}_{base_family}.joblib')
            assert policy.parameter==old.parameter and policy.alpha==old.alpha and policy.family==old.family
        selection=fold['selection']
        if selection['fallback']:
            assert policy.family=='fixed' and policy.parameter==3
        else:
            acceptable=[r for r in selection['candidates'] if r['mean_loss_vs_full']<=.01+1e-12]
            best=min(acceptable,key=lambda r:(r['mean_seconds'],r['mean_loss_vs_full']))
            assert best==selection['selected']
            assert policy.parameter==best['parameter'] and policy.alpha==best['alpha']
            if base_family=='ridge':
                x=np.array([[r[k] for k in FEATURES] for r in train])
                y=np.array([(outcomes[r['participant'],r['session'],r['order'],3]['macro_f1']-outcomes[r['participant'],r['session'],r['order'],int(r['budget'])]['macro_f1'])/(3-r['budget']) for r in train])
                independent=make_pipeline(StandardScaler(),Ridge(alpha=best['alpha'])).fit(x,y)
                np.testing.assert_allclose(independent.predict(x),policy.model.predict(x),atol=1e-12,rtol=0)
                assert policy.model.named_steps['standardscaler'].n_samples_seen_==len(train)
            elif base_family!='fixed':
                for b in (1,2):
                    q=best['parameter'];values=[r[base_family] for r in train if r['budget']==b]
                    threshold=-np.inf if q<0 else np.inf if q>1 else float(np.quantile(values,q))
                    assert policy.thresholds[b]==threshold
        for choice in [r for r in choices if r['participant']==held and r['family']==family]:
            trace=[];selected=3
            for b in (1,2):
                stop,value=policy.decide(lookup[held,choice['session'],choice['order'],b])
                trace.append({'budget':b,'stop':bool(stop),'value':float(value)})
                if stop:
                    selected=b;break
            assert selected==choice['selected_budget'] and trace==choice['trace']
            if is_frozen and choice['order']=='123':
                old=old_decisions[held,choice['session'],base_family]
                assert old['selected_budget']==selected and old['trace']==trace
    # Independent arithmetic on every overall and order-specific summary.
    results=json.loads((args.policies/'results.json').read_text())+json.loads((args.policies/'per_order.json').read_text())
    for result in results:
        family=result['family'];orders=[result['order']] if 'order' in result else list(ORDERS)
        if family.startswith('fixed_') and family not in ('fixed',):
            b=int(family.split('_')[1])//85
            selected=[(p,s,o,b) for p in people for s in (2,3) for o in orders]
        else:
            selected=[(r['participant'],r['session'],r['order'],r['selected_budget']) for r in choices if r['family']==family and r['order'] in orders]
        assert len(selected)==16*len(orders)
        f1=np.array([outcomes[k]['macro_f1'] for k in selected]);loss=np.array([outcomes[p,s,o,3]['macro_f1']-outcomes[p,s,o,b]['macro_f1'] for p,s,o,b in selected])
        seconds=np.mean([k[-1]*85 for k in selected])
        expected={'mean_macro_f1':f1.mean(),'mean_seconds':seconds,'mean_loss_vs_full':loss.mean(),'worst_loss_vs_full':loss.max(),'fraction_loss_over_002':(loss>.02).mean()}
        for k,v in expected.items():
            np.testing.assert_allclose(result[k],v,atol=1e-12,rtol=0)
        if 'delta_vs_same_cost_mixture' in result:
            fixed=[np.mean([r['macro_f1'] for (p,s,o,k),r in outcomes.items() if o in orders and k==b]) for b in (1,2,3)]
            delta=f1.mean()-np.interp(seconds,[85,170,255],fixed)
            np.testing.assert_allclose(result['delta_vs_same_cost_mixture'],delta,rtol=0,atol=1e-12)
    report={'completed_utc':datetime.now(timezone.utc).isoformat(),'status':'passed','total_seconds':time.perf_counter()-start,
            'model_access_ledgers':len(ledgers),'scored_models_reproduced':scored,'scoring_predictions_reproduced':scored*2380,
            'independent_refits_exact':replicated,'observation_states':len(observations),'unique_cpu_probability_evaluations':len(cpu_cache),
            'max_cpu_probability_difference':max_difference,'policies_verified':80,'decisions_verified':len(choices),
            'summary_rows_verified':len(results),'original_features_exact':32,'original_outcomes_exact':48,'original_frozen_decisions_exact':80,
            'final_participants_accessed':0,'limits':'Development only. Does not repeat all inner hyperparameter searches or all64 new refits. Bootstrap remains descriptive conditional on fitted policies.',
            'verifier_sha256':hash_stream(__file__)}
    (args.policies/'independent_validation.json').write_text(json.dumps(report,indent=2)+'\n')
    (args.policies/'code_snapshot'/'verify_stopping_orders.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    main()
