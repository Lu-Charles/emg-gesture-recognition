"""Independent split, feature, fitted-scaler, prediction and metric audit."""
import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
import joblib
import numpy as np
from scipy.special import softmax
from threadpoolctl import threadpool_limits
from src.grabmyo_corpus import hash_stream


def scalar_burg(x, order=4):
    forward=x[1:].copy(); backward=x[:-1].copy(); a=np.ones(1)
    for _ in range(order):
        denominator=np.dot(forward,forward)+np.dot(backward,backward)
        if denominator==0: return np.pad(a,(0,order+1-len(a)))[1:]
        reflection=-2*np.dot(forward,backward)/denominator
        a=np.r_[a,0]+reflection*np.r_[0,a[::-1]]
        new_forward=forward+reflection*backward
        new_backward=backward+reflection*forward
        forward=new_forward[1:];backward=new_backward[:-1]
    return a[1:]


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();run=a.run
    start=time.perf_counter()
    load=lambda name:json.loads((run/name).read_text())
    config=load('config.json');rows=load('trial_manifest.json');plans=load('allocations.json');metrics=load('metrics.json')
    for file,digest in load('artifact_sha256.json').items():
        assert hash_stream(run/file)==digest,file
    for original,digest in config['code_hashes'].items():
        assert hash_stream(run/'code_snapshot'/Path(original).name)==digest
    cache=Path(config['cache']);assert hash_stream(cache/'summary.json')==config['cache_summary_sha256']
    expected=json.loads((cache/'summary.json').read_text())['groups']['development']
    assert config['participants']==expected['participants']
    assert all(r['group']=='development' and int(r['participant']) in config['participants'] for r in rows)
    assert hash_stream(cache/'development_signals.npy')==expected['signals_sha256']
    assert hash_stream(cache/'development_manifest.csv')==expected['manifest_sha256']
    feats=np.load(run/'features.npy',mmap_mode='r');signal=np.load(cache/'development_signals.npy',mmap_mode='r')
    feature_checks=0;max_ar_error=0.
    # Fixed identity-based sample: every person/session, known gesture11 trial7.
    for i,r in enumerate(rows):
        if r['gesture']!='11' or r['trial']!='7': continue
        for window in (0,17,34):
            x=signal[i,:,1024+256*window:1024+256*window+512].astype(float)
            direct=[]
            for channel in x:
                mav=sum(abs(channel))/512
                zc=sum((v>0 and w<0) or (v<0 and w>0) for v,w in zip(channel[:-1],channel[1:]))
                ssc=sum((channel[j]-channel[j-1])*(channel[j]-channel[j+1])>=0 for j in range(1,511))
                wl=sum(abs(channel[j+1]-channel[j]) for j in range(511))
                ar=scalar_burg(channel)
                direct.append([mav,zc,ssc,wl,*ar,np.sqrt(sum(channel*channel)/512)])
            direct=np.asarray(direct).T.reshape(-1)
            max_ar_error=max(max_ar_error,float(np.max(np.abs(direct[64:128]-feats[i,window,64:128]))))
            np.testing.assert_allclose(feats[i,window],direct,atol=1e-9,rtol=1e-10)
            feature_checks+=1
    assert feature_checks==72
    planmap={}
    for plan in plans:
        f,q,s=[set(plan[k]) for k in ['fit','threshold','score']]
        assert not(f&q or f&s or q&s)
        assert all(int(rows[i]['class_index'])>=10 for i in f|q)
        target={i for i in f|q if int(rows[i]['session'])>1}
        assert len(target)==7*plan['budget']
        assert all(rows[i]['role']=='calibration' for i in target)
        expected_score=[i for i,r in enumerate(rows) if int(r['participant'])==plan['participant'] and int(r['session'])==plan['session'] and r['role']=='scoring']
        assert plan['score']==expected_score and len(expected_score)==68
        key=(plan['participant'],plan['session'],plan['method'],plan['budget'],''.join(map(str,plan['order'])),plan['fit_strategy'])
        assert key not in planmap;planmap[key]=plan
    access={x['model']:x for x in load('model_access.json')};models={}
    for name,entry in access.items():
        ids=entry['fit_ids'];x=np.asarray(feats[ids]).reshape(-1,144)
        assert all(int(rows[i]['class_index'])>=10 for i in ids)
        m=joblib.load(run/'models'/(name+'.joblib'));models[name]=m
        np.testing.assert_allclose(m.scaler.mean_,x.mean(0),atol=1e-12)
        np.testing.assert_allclose(m.scaler.var_,x.var(0),atol=1e-12)
        assert m.scaler.n_samples_seen_==len(ids)*35
    predictions={};n_predictions=0
    for file in (run/'predictions').glob('*.npz'):
        if file.name.endswith('_curve.npz'):continue
        saved=np.load(file);name=file.name.split('_')[0];model=models[name];ids=saved['trial_ids']
        x=np.asarray(feats[ids]).reshape(-1,144)
        labels=np.repeat([int(rows[i]['class_index']) for i in ids],35)
        np.testing.assert_array_equal(labels,saved['y'])
        z=(x-model.scaler.mean_)/model.scaler.scale_
        probability=softmax(z@model.lda.coef_.T+model.lda.intercept_,axis=1)
        index=probability.argmax(1);pred=model.lda.classes_[index]
        np.testing.assert_array_equal(pred,saved['pred'])
        np.testing.assert_allclose(probability.max(1),saved['probability'],atol=1e-12)
        delta=z-model.lda.means_[index]
        distance=-np.sqrt(np.maximum(np.sum((delta@model.precision)*delta,axis=1),0))
        np.testing.assert_allclose(distance,saved['distance'],atol=1e-9)
        predictions[file.name]={k:saved[k] for k in saved.files};n_predictions+=len(labels)
    for metric in metrics+load('source_metrics.json'):
        qs=predictions[metric['threshold_file']];scores=predictions[metric['score_file']]
        cal=np.sort(qs[metric['rejector']]);idx=int(np.floor((len(cal)-1)*(1-metric['retention'])))
        threshold=float(cal[idx]);assert abs(threshold-metric['threshold'])<1e-12
        if 'session' in metric:
            key=tuple(metric[k] for k in ['participant','session','method','budget','order','fit_strategy'])
            plan=planmap[key]
            assert set(access[metric['model']]['fit_ids'])==set(plan['fit'])
            np.testing.assert_array_equal(scores['trial_ids'],plan['score'])
            np.testing.assert_array_equal(qs['trial_ids'],plan['threshold'])
            qmodel=metric['threshold_file'].split('_')[0]
            assert not(set(access[qmodel]['fit_ids']) & set(qs['trial_ids']))
            if metric['method']=='model_only':
                assert all(rows[i]['session']=='1' for i in access[qmodel]['fit_ids'])
        y,pr=scores['y'],scores['pred'];accept=scores[metric['rejector']]>=threshold
        active=(y>=10)&(y<16);unknown=y<10;rest=y==16
        events=dict(known_correct_acceptance=(active,accept&(pr==y)),unknown_false_acceptance=(unknown,accept&(pr!=16)),
                    known_wrong_command=(active,accept&(pr!=16)&(pr!=y)),known_rejection=(active,~accept),
                    known_predicted_rest=(active,accept&(pr==16)),rest_false_activation=(rest,accept&(pr!=16)),
                    closed_known_accuracy=(active,pr==y))
        for field,(mask,event) in events.items():
            expected_value=np.count_nonzero(mask&event)/np.count_nonzero(mask)
            assert abs(metric[field]-expected_value)<1e-12,field
        assert abs(sum(metric[k] for k in ['known_correct_acceptance','known_wrong_command','known_rejection','known_predicted_rest'])-1)<1e-12
    curve_checks=0
    for item in load('curve_index.json'):
        saved=predictions[item['score_file']];curve=np.load(run/'predictions'/item['curve_file'])
        y,pr=saved['y'],saved['pred'];score=saved[item['rejector']]
        for j in np.linspace(0,len(curve['threshold'])-1,7,dtype=int):
            ok=score>=curve['threshold'][j]
            c=np.sum(ok&(pr==y)&(y>=10)&(y<16))/np.sum((y>=10)&(y<16))
            u=np.sum(ok&(pr!=16)&(y<10))/np.sum(y<10)
            np.testing.assert_allclose([c,u],[curve['known_correct_acceptance'][j],curve['unknown_false_acceptance'][j]],atol=1e-12)
            curve_checks+=1
    result=dict(passed=True,completed_utc=datetime.now(timezone.utc).isoformat(),seconds=time.perf_counter()-start,
                feature_windows_independent=feature_checks,ar_max_abs_error=max_ar_error,
                allocations=len(plans),models=len(models),predictions=n_predictions,metric_rows=len(metrics),
                curve_spot_checks=curve_checks,final_participants_accessed=0)
    result['audit_code_sha256']=hash_stream(__file__)
    (run/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
    (run/'code_snapshot'/'verify_open_set_classical.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(result))


if __name__=='__main__':
    with threadpool_limits(limits=4):main()
