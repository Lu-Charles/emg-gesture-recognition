"""All six permutations of three fixed balanced calibration rounds.

Fit each unique observed subset once, in canonical manifest order, resetting
from enrollment. Acquisition order never changes the optimizer's input order.
"""
import argparse
import copy
import itertools
import json
import resource
import time
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import torch
from sklearn.metrics import f1_score, balanced_accuracy_score
from src.calibration_orders import ORDERS, calibration_ids, subset_code
from src.calibration_stopping import FEATURES, observed_features
from src.emg_cnn import CompactEMGNet
from src.grabmyo_corpus import WindowCorpus, hash_stream
from scripts.pilot_calibration_stopping import probabilities
from scripts.pilot_emg_cnn import fit, predict
from scripts.train_shared_emg import roles


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--prior',type=Path,required=True)
    parser.add_argument('--original-observations',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();start=time.perf_counter()
    args.out.mkdir(parents=True,exist_ok=False)
    prior_cfg=json.loads((args.prior/'config.json').read_text())
    for root in (args.prior,args.original_observations):
        for relative,expected in json.loads((root/'artifact_sha256.json').read_text()).items():
            if hash_stream(root/relative)!=expected:
                raise ValueError('Changed upstream artifact: '+str(root/relative))
    if not (args.prior/'independent_validation.json').exists():
        raise ValueError('Prior recognizer must be verified')
    original_cfg=json.loads((args.original_observations/'config.json').read_text())
    if Path(original_cfg['prior']).resolve()!=args.prior.resolve() or original_cfg['seed']!=prior_cfg['seed']:
        raise ValueError('Original observations must use the same recognizer run and seed')
    cache=Path(prior_cfg['cache'])
    if hash_stream(cache/'summary.json')!=prior_cfg['cache_summary_sha256']:
        raise ValueError('Changed cache summary')
    summary=json.loads((cache/'summary.json').read_text())
    for kind in ('signals','features','manifest'):
        file=f'development_{kind}.'+('csv' if kind=='manifest' else 'npy')
        if hash_stream(cache/file)!=summary['groups']['development'][kind+'_sha256']:
            raise ValueError('Changed cache')
    dev=WindowCorpus(cache,'development')
    people=sorted({int(r['participant']) for r in dev.rows})
    if people!=prior_cfg['development_participants']:
        raise ValueError('Development allocation changed')
    for f in ('src/emg_cnn.py','scripts/pilot_emg_cnn.py'):
        if hash_stream(f)!=prior_cfg['code_hashes'][f]:
            raise ValueError('Recognizer/optimizer definition changed')
    cfg={'created_utc':datetime.now(timezone.utc).isoformat(),'prior':str(args.prior.resolve()),
         'original_observations':str(args.original_observations.resolve()),'cache':str(cache.resolve()),
         'participants':people,'orders':ORDERS,'sessions':[2,3],'features':FEATURES,
         'seed':prior_cfg['seed'],'update_epochs':10,'update_lr':.0001,'batch_size':128,'optimizer':'Adam',
         'round_seconds':85,'device':'mps' if torch.backends.mps.is_available() else 'cpu',
         'procedure':'Reinitialize from enrolled checkpoint per observed subset. Canonical manifest input order. All 6 global permutations of fixed calibration-rank rounds; no new scoring split or pool.',
         'input_hashes':{str(root/f):hash_stream(root/f) for root in (args.prior,args.original_observations) for f in ('config.json','artifact_sha256.json')},
         'cache_summary_sha256':hash_stream(cache/'summary.json'),
         'code_hashes':{f:hash_stream(f) for f in ('scripts/calibration_order_observations.py','src/calibration_orders.py','src/calibration_stopping.py','src/grabmyo_corpus.py','src/emg_cnn.py','scripts/pilot_emg_cnn.py','scripts/pilot_calibration_stopping.py','scripts/train_shared_emg.py')}}
    (args.out/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
    snapshots=args.out/'code_snapshot';snapshots.mkdir()
    for f in cfg['code_hashes']:
        (snapshots/Path(f).name).write_bytes(Path(f).read_bytes())
    model_dir=args.out/'models';model_dir.mkdir()
    scaler=np.load(args.prior/'training_scaler.npz');mean,scale=scaler['mean'],scaler['scale']
    prior_access={r['name']:r for r in json.loads((args.prior/'data_access.json').read_text())}
    prior_metrics={r['name']:r for r in json.loads((args.prior/'metrics.json').read_text())}
    original_features={(r['participant'],r['session'],int(r['budget'])):r for r in json.loads((args.original_observations/'features.json').read_text())}
    torch.set_num_threads(4)
    features=[];outcomes=[];models_ledger=[];access=[];histories=[];arrays={};scoring_arrays={}
    subsets=[()] + [x for size in (1,2,3) for x in itertools.combinations((1,2,3),size)]
    new_fits=0; reused=0;fit_seconds=0.;checks=[]
    compute_start=time.perf_counter()
    for p in people:
        for session in (2,3):
            score_ids=roles(dev.rows,p,session,'scoring')
            score_records=[dev.rows[i]['record'] for i in score_ids]
            enrolled_records=[dev.rows[i]['record'] for i in roles(dev.rows,p,1,'enrollment')]
            score_x,score_y=dev.batch(dev.window_ids(score_ids),mean,scale)
            scoring_arrays[f'p{p}_s{session}_labels']=score_y
            base_name=f'p{p}_s{session}_shared_none_b0'
            base=CompactEMGNet().to(cfg['device'])
            base.load_state_dict(torch.load(args.prior/'models'/f'{base_name}.pt',map_location=cfg['device'],weights_only=True))
            available={};metrics_by_subset={}
            for subset in subsets:
                code=subset_code(subset);name=f'p{p}_s{session}_r{code}'
                ids=calibration_ids(dev.rows,p,session,subset)
                records=[dev.rows[i]['record'] for i in ids]
                assert not set(records+enrolled_records)&set(score_records)
                original_name=None
                if subset in ((),(1,),(1,2),(1,2,3)):
                    original_name=base_name if not subset else f'p{p}_s{session}_shared_full_b{len(subset)}'
                    a=prior_access[original_name]
                    if a['calibration']!=records or a['source']!=enrolled_records or a['score']!=score_records:
                        raise ValueError('Reused model trial access does not match subset')
                    model=copy.deepcopy(base)
                    model.load_state_dict(torch.load(args.prior/'models'/f'{original_name}.pt',map_location=cfg['device'],weights_only=True))
                    reused+=1
                else:
                    model=copy.deepcopy(base);model.set_update_mode('full')
                    x,y=dev.batch(dev.window_ids(ids),mean,scale)
                    history=fit(model,x,y,10,.0001,cfg['seed'],cfg['device'])
                    histories.append({'name':name,**history});fit_seconds+=history['fit_seconds'];new_fits+=1
                state={k:v.detach().cpu() for k,v in model.state_dict().items()}
                torch.save(state,model_dir/f'{name}.pt')
                available[code]=model
                ledger={'name':name,'participant':p,'session':session,'ranks':list(subset),'calibration':records,
                        'source':enrolled_records,'score':score_records,'reused_from':original_name}
                models_ledger.append(ledger)
                if subset:
                    pred=predict(model,score_x,cfg['device'])
                    scoring_arrays[name+'_predictions']=pred
                    metric={'macro_f1':float(f1_score(score_y,pred,labels=list(range(17)),average='macro',zero_division=0)),
                            'balanced_accuracy':float(balanced_accuracy_score(score_y,pred))}
                    metrics_by_subset[code]=metric
                    if original_name:
                        assert metric['macro_f1']==prior_metrics[original_name]['macro_f1']
            # Each state only sees a newly acquired round and the models fitted
            # before/after that round. Scoring outputs stay in a separate file.
            for order in ORDERS:
                ranks=list(map(int,order))
                for b in (1,2,3):
                    code=subset_code(ranks[:b])
                    outcomes.append({'participant':p,'session':session,'order':order,'budget':b,
                                     'model_name':f'p{p}_s{session}_r{code}',**metrics_by_subset[code]})
                    if b==3:
                        continue
                    before_code=subset_code(ranks[:b-1]);new_ids=calibration_ids(dev.rows,p,session,[ranks[b-1]])
                    x,y=dev.batch(dev.window_ids(new_ids),mean,scale)
                    before=probabilities(available[before_code],x,cfg['device'])
                    after=probabilities(available[code],x,cfg['device'])
                    row={'participant':p,'session':session,'order':order,**observed_features(before,after,y,b)}
                    features.append(row);key=f'p{p}_s{session}_o{order}_b{b}'
                    arrays[key+'_before']=before;arrays[key+'_after']=after;arrays[key+'_labels']=y
                    access.append({'key':key,'participant':p,'session':session,'order':order,'budget':b,
                                   'observation_records':[dev.rows[i]['record'] for i in new_ids],
                                   'acquired_records':[dev.rows[i]['record'] for i in calibration_ids(dev.rows,p,session,ranks[:b])],
                                   'checkpoint_names':[f'p{p}_s{session}_r{before_code}',f'p{p}_s{session}_r{code}'],
                                   'scoring_records_excluded':score_records})
                    if order=='123':
                        for feature in FEATURES:
                            np.testing.assert_allclose(row[feature],original_features[p,session,b][feature],rtol=0,atol=1e-12)
                        checks.append(key)
            print(json.dumps({'participant':p,'session':session,'new_fits':new_fits,'states':len(features),'elapsed_seconds':time.perf_counter()-compute_start}),flush=True)
            for name,value in (('features',features),('outcomes',outcomes),('access',access),('model_access',models_ledger),('training_history',histories)):
                (args.out/f'{name}.json').write_text(json.dumps(value,indent=2)+'\n')
    assert len(features)==192 and len(outcomes)==288 and new_fits==64 and reused==64
    np.savez_compressed(args.out/'observation_probabilities.npz',**arrays)
    np.savez_compressed(args.out/'scoring_predictions.npz',**scoring_arrays)
    (args.out/'original_consistency.json').write_text(json.dumps({'feature_states_exact':checks,'prefix_metric_comparisons_exact':48},indent=2)+'\n')
    result={'completed_utc':datetime.now(timezone.utc).isoformat(),'total_seconds':time.perf_counter()-start,
            'new_fit_seconds':fit_seconds,'new_fits':new_fits,'reused_checkpoints':reused,'decision_states':len(features),
            'independent_participants':8,'person_session_order_episodes':96,'unique_scored_models':112,
            'peak_process_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'final_participants_accessed':0}
    (args.out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    (args.out/'artifact_sha256.json').write_text(json.dumps({str(p.relative_to(args.out)):hash_stream(p) for p in args.out.rglob('*') if p.is_file()},indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    main()
