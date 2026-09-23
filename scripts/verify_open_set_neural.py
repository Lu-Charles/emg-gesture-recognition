"""Independent access/count/threshold audit plus direct functional checkpoint checks."""
import argparse
import csv
import json
import time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from src.grabmyo_corpus import hash_stream


def direct_scores(state,x,family):
    outputs=[]
    with torch.no_grad():
        for b in range(2 if family=='predin' else 1):
            prefix=f'branches.{b}.';z=torch.from_numpy(x)
            for layer,stride,padding in [(0,4,4),(2,4,3),(4,2,2)]:
                key=prefix+f'encoder.{layer}.'
                z=F.relu(F.conv1d(z,state[key+'weight'],state[key+'bias'],stride=stride,padding=padding))
            z=z.mean(2);z=F.linear(z,state[prefix+'embedding.weight'],state[prefix+'embedding.bias'])
            outputs.append(z@state[prefix+'prototypes'].T if family!='cnn' else F.linear(z,state[prefix+'head.weight'],state[prefix+'head.bias']))
        logits=sum(outputs)/len(outputs)
        scores={'probability':logits.softmax(1).max(1).values.numpy()}
        if family!='cnn':scores['prototype']=logits.max(1).values.numpy()
        return logits.argmax(1).numpy()+10,scores


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);ap.add_argument('--wait',action='store_true');args=ap.parse_args();run=args.run
    start=time.perf_counter();torch.set_num_threads(4)
    def ready(path):
        if args.wait:
            while not path.exists():time.sleep(1)
        assert path.exists(),path
    load=lambda p:json.loads(p.read_text())
    cfg=load(run/'config.json');rows=load(run/'trial_manifest.json');plans=load(run/'allocations.json')
    for f,digest in cfg['source_hashes'].items():assert hash_stream(run/'code_snapshot'/Path(f).name)==digest
    cache=Path(cfg['cache']);assert hash_stream(cache/'summary.json')==cfg['cache_summary_sha256']
    cache_summary=load(cache/'summary.json')
    for group in ('train','development'):
        for kind,ext in [('signals','npy'),('manifest','csv')]:
            assert hash_stream(cache/f'{group}_{kind}.{ext}')==cache_summary['groups'][group][kind+'_sha256']
    assert not set(cfg['training_people'])&set(cfg['development_people'])
    training=list(csv.DictReader((cache/'train_manifest.csv').open()))
    ids=[i for i,r in enumerate(training) if 10<=int(r['class_index'])<=16]
    assert ids==cfg['train_trial_ids'] and len(ids)==2940
    assert [training[i] for i in ids]==load(run/'training_access.json')
    assert all(r['group']=='train' for r in load(run/'training_access.json'))
    # Recompute moments from raw cache with explicit overlap multiplicities,
    # independently of WindowCorpus.batch and known_moments.
    raw=np.load(cache/'train_signals.npy',mmap_mode='r');weight=np.zeros(10240)
    for w in range(35):weight[1024+256*w:1024+256*w+512]+=1
    total=np.zeros(16);squares=np.zeros(16)
    for i in ids:
        signal=raw[i].astype(float);total+=(signal*weight).sum(1);squares+=(signal*signal*weight).sum(1)
    n=len(ids)*35*512;mean=total/n;scale=np.sqrt(squares/n-mean*mean)
    norm=np.load(run/'normalization.npz');assert int(norm['count'])==n
    np.testing.assert_allclose(mean,norm['mean'].ravel(),rtol=1e-10,atol=1e-14)
    np.testing.assert_allclose(scale,norm['scale'].ravel(),rtol=1e-10,atol=1e-14)
    dev=np.load(cache/'development_signals.npy',mmap_mode='r');planmap={}
    assert all(r['group']=='development' and int(r['participant']) in cfg['development_people'] for r in rows)
    for a in plans:
        f,q,s=[set(a[k]) for k in ['fit','threshold','score']]
        assert not(f&q or f&s or q&s)
        assert all(int(rows[i]['class_index'])>=10 for i in f|q)
        assert all(int(rows[i]['participant'])==a['participant'] for i in f|q|s)
        target={i for i in f|q if int(rows[i]['session'])>1}
        assert len(target)==7*a['budget'] and all(rows[i]['role']=='calibration' for i in target)
        expected=[i for i,r in enumerate(rows) if int(r['participant'])==a['participant'] and int(r['session'])==a['session'] and r['role']=='scoring']
        assert a['score']==expected and len(expected)==68
        key=(a['participant'],a['session'],a['method'],a['budget'],''.join(map(str,a['order'])),a['update_strategy'])
        assert key not in planmap;planmap[key]=a
    results=[]
    for family in cfg['families']:
        ready(run/family/'summary.json')
        folder=run/family;access={x['model']:x for x in load(folder/'model_access.json')}
        predictions={};checked_models=set();checked_windows=0;score_error=0.;curve_checks=0
        for name,a in access.items():
            assert all(int(rows[i]['class_index'])>=10 for i in a['fit_ids'])
            assert set(a['fit_ids'])==set(a['source_ids'])|set(a['target_ids'])
            expected=a['source_ids']+a['target_ids'] if a['target_ids'] and a['update_strategy']=='pooled_replay' else a['target_ids']
            assert a['update_ids']==expected
        for file in sorted((folder/'predictions').glob('*.npz')):
            if file.name.endswith('_curve.npz'):continue
            with np.load(file) as z: saved={k:z[k] for k in z.files}
            name=file.name.rsplit('_e',1)[0];ids=saved['trial_ids']
            np.testing.assert_array_equal(saved['y'],np.repeat([int(rows[i]['class_index']) for i in ids],35))
            assert np.all((saved['pred']>=10)&(saved['pred']<=16))
            assert all(np.isfinite(v).all() for v in saved.values())
            # Nine fixed positions from one file per saved checkpoint, direct
            # functional forward on CPU, without calling model/infer helpers.
            if name not in checked_models:
                positions=np.unique(np.linspace(0,len(saved['y'])-1,9,dtype=int));windows=[]
                for j in positions:
                    w=j%35;i=ids[j//35];windows.append(dev[i,:,1024+256*w:1024+256*w+512])
                x=((np.asarray(windows)-norm['mean'])/norm['scale']).astype(np.float32)
                state=torch.load(folder/'models'/(name+'.pt'),map_location='cpu',weights_only=True)
                pr,sc=direct_scores(state,x,family)
                np.testing.assert_array_equal(pr,saved['pred'][positions])
                for k,v in sc.items():
                    score_error=max(score_error,float(np.max(np.abs(v-saved[k][positions]))))
                    np.testing.assert_allclose(v,saved[k][positions],atol=2e-4,rtol=2e-4)
                checked_models.add(name);checked_windows+=len(positions)
            predictions[file.name]=saved
        assert checked_models==set(access)
        metrics=load(folder/'metrics.json')+load(folder/'source_metrics.json')
        for r in metrics:
            q=predictions[r['threshold_file']];z=predictions[r['score_file']];rej=r['rejector']
            threshold=np.sort(q[rej])[int(np.floor((len(q[rej])-1)*(1-r['retention'])))];assert threshold==r['threshold']
            qm=r['threshold_file'].rsplit('_e',1)[0]
            assert not set(access[qm]['fit_ids'])&set(q['trial_ids'])
            if 'session' in r:
                key=tuple(r[k] for k in ['participant','session','method','budget','order','update_strategy']);a=planmap[key]
                assert set(access[r['model']]['fit_ids'])==set(a['fit'])
                np.testing.assert_array_equal(z['trial_ids'],a['score']);np.testing.assert_array_equal(q['trial_ids'],a['threshold'])
                if r['method'] in ('none','model_only'):assert not access[qm]['target_ids']
                assert r['seconds']==a['budget']*35
            y,pr=z['y'],z['pred'];ok=z[rej]>=threshold;known=(y>=10)&(y<16);unknown=y<10;rest=y==16
            checks=dict(known_correct_acceptance=(known,ok&(pr==y)),unknown_false_acceptance=(unknown,ok&(pr!=16)),known_wrong_command=(known,ok&(pr!=16)&(pr!=y)),known_rejection=(known,~ok),known_predicted_rest=(known,ok&(pr==16)),rest_false_activation=(rest,ok&(pr!=16)),closed_known_accuracy=(known,pr==y))
            for field,(mask,event) in checks.items():assert abs(r[field]-np.sum(mask&event)/np.sum(mask))<1e-12
        for item in load(folder/'curve_index.json'):
            z=predictions[item['score_file']];curve=np.load(folder/'predictions'/item['curve_file']);y,pr=z['y'],z['pred']
            values=z[item['rejector']];unknown_scores=np.sort(values[y<10]);known_scores=values[y>=10]
            # Pairwise concordance with half credit for ties, independent of
            # sklearn AUROC and the pilot's trapezoid integration.
            concordance=lambda v:.5*np.sum(np.searchsorted(unknown_scores,v,side='left')+np.searchsorted(unknown_scores,v,side='right'))
            denominator=np.sum(y>=10)*np.sum(y<10)
            assert abs(item['auroc']-concordance(known_scores)/denominator)<1e-12
            assert abs(item['standard_oscr']-concordance(values[(y>=10)&(pr==y)])/denominator)<1e-12
            for j in np.unique(np.linspace(0,len(curve['threshold'])-1,7,dtype=int)):
                ok=z[item['rejector']]>=curve['threshold'][j];known=(y>=10)&(y<16)
                values=[np.sum(ok&(pr==y)&known)/sum(known),np.sum(ok&(pr!=16)&(y<10))/sum(y<10)]
                np.testing.assert_allclose(values,[curve[k][j] for k in ['known_correct_acceptance','unknown_false_acceptance']],atol=1e-12);curve_checks+=1
        result=dict(family=family,models=len(access),checkpoint_windows_checked=checked_windows,maximum_cpu_mps_score_difference=score_error,prediction_rows_checked=sum(len(z['y']) for z in predictions.values()),metric_rows_checked=len(metrics),curve_points_checked=curve_checks)
        results.append(result);print(json.dumps(result),flush=True)
    ready(run/'artifact_sha256.json')
    for f,digest in load(run/'artifact_sha256.json').items():assert hash_stream(run/f)==digest,f
    result=dict(passed=True,completed_utc=datetime.now(timezone.utc).isoformat(),seconds=time.perf_counter()-start,wait_for_training=args.wait,allocations=len(plans),normalization_trials=2940,families=results,final_participants_accessed=0,audit_code_sha256=hash_stream(__file__))
    (run/'validation.json').write_text(json.dumps(result,indent=2)+'\n');(run/'code_snapshot'/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
