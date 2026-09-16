"""Exploratory fit-versus-check allocation, exclusively SeNic development data."""
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from src.senic import DEVELOPMENT, estimate_rotation, rotate_features, validate_split
from src.grabmyo_corpus import hash_stream

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / 'research/runs/20260907_sensor_shift_pilot_v2'
OUT = ROOT / 'research/runs/20260916_calibration_audit_development'
AMP = np.r_[0:8, 24:32, 64:72]


def accept_update(frozen_predictions, adapted_predictions, known_label):
    """Only the separately acquired check trial is admissible here."""
    a, b = np.asarray(frozen_predictions), np.asarray(adapted_predictions)
    if a.shape != b.shape or a.ndim != 1 or not len(a):
        raise ValueError('Matching nonempty check-window arrays required')
    return bool(np.sum(b == known_label) > np.sum(a == known_label))


def validate_roles(source, fitting, check, scoring, rows, novel):
    validate_split(source, fitting + [check], scoring, rows)
    if len(fitting) != len(set(fitting)) or check in fitting:
        raise ValueError('Fit/check trial reuse')
    if novel and rows[check]['label'] in {rows[i]['label'] for i in fitting}:
        raise ValueError('Novel-gesture check overlaps fitted vocabulary')


def fit_adapter(source_templates, features, labels, method):
    gestures = sorted(set(labels))
    a = source_templates[gestures]
    b = np.stack([features[np.asarray(labels) == g, :, 64:72].mean((0, 1))
                  for g in gestures])
    shift = estimate_rotation(a, b)[0] if method == 'cosine_gain' else 0.0
    gain = np.clip(np.exp((np.log(np.maximum(a, 1e-12)) -
                         np.log(np.maximum(rotate_features(b, shift), 1e-12))).mean(0)), .5, 2)
    return {'shift': shift, 'gain': gain.tolist()}


def predict(features, model, state=None):
    x = np.asarray(features).reshape(-1, 72)
    if state is not None:
        x = rotate_features(x, state['shift'])
        x[:, AMP] *= np.tile(state['gain'], 3)
    scores = ((x-model['mean'])/model['scale']) @ model['coef'].T + model['intercept']
    return np.argmax(scores, axis=1)


def votes(pred):
    return np.array([np.bincount(row, minlength=7).argmax()
                     for row in np.asarray(pred).reshape(-1, 15)])


def main():
    OUT.mkdir(exist_ok=False)
    rows = json.loads((CACHE/'trial_manifest.json').read_text())
    # This cache contains only the six established development participants.
    if set(r['subject'] for r in rows) != set(DEVELOPMENT):
        raise ValueError('Unexpected cache cohort')
    x = np.load(CACHE/'tdar_rms_features.npy')
    assert x.shape == (1386, 15, 72)
    cfg = dict(timestamp=datetime.now(timezone.utc).isoformat(), scope='exploratory development only',
               participants=list(DEVELOPMENT), methods=['cosine_gain', 'gain_only'],
               code_sha256=hash_stream(__file__), cache_sha256=hash_stream(CACHE/'tdar_rms_features.npy'),
               manifest_sha256=hash_stream(CACHE/'trial_manifest.json'),
               protocol_sha256=hash_stream(ROOT/'research/pivot_calibration_audit_20260916/PROTOCOL.md'),
               seed=None, deterministic=True, gain_bounds=[.5,2], analysis_seconds_per_trial=2,
               decision='strict check-window accuracy improvement over frozen; frozen on tie')
    (OUT/'config.json').write_text(json.dumps(cfg, indent=2)+'\n')
    (OUT/'pilot_calibration_audit.py').write_bytes(Path(__file__).read_bytes())
    (OUT/'PROTOCOL.md').write_bytes((ROOT/'research/pivot_calibration_audit_20260916/PROTOCOL.md').read_bytes())
    records, access, models, predictions, states = [], [], {}, {}, {}
    for p in DEVELOPMENT:
        source = [i for i,r in enumerate(rows) if r['subject']==p and r['position']==0
                  and r['repetition'] in (0,1)]
        sy = np.repeat([rows[i]['label'] for i in source], 15)
        sx = x[source].reshape(-1,72)
        sc = StandardScaler().fit(sx)
        lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto', priors=np.ones(7)/7)
        lda.fit(sc.transform(sx),sy)
        model = dict(mean=sc.mean_,scale=sc.scale_,coef=lda.coef_,intercept=lda.intercept_)
        for k,v in model.items(): models[f'p{p}_{k}']=v
        templates=np.stack([sx[sy==g,64:72].mean(0) for g in range(7)])
        order=np.argsort(-templates.sum(1),kind='stable').tolist()
        fit_labels=order[:2]
        remaining=order[2:]
        unit=templates/np.maximum(np.linalg.norm(templates,axis=1,keepdims=True),1e-12)
        diversity=1-np.max(unit[remaining] @ unit[fit_labels].T,axis=1)
        diverse=remaining[int(np.argmax(diversity))]
        for pos in range(1,11):
            pool={(r['label'],r['repetition']):i for i,r in enumerate(rows)
                  if r['subject']==p and r['position']==pos}
            fit=[pool[g,0] for g in fit_labels]
            same=pool[fit_labels[0],1]
            score=[pool[g,2] for g in range(7)]
            validate_roles(source,fit,same,score,rows,False)
            frozen=votes(predict(x[score],model))
            same_frozen=predict(x[same],model)
            for method in cfg['methods']:
                state2=fit_adapter(templates,x[fit],fit_labels,method)
                adapted=votes(predict(x[score],model,state2))
                same_adapted=predict(x[same],model,state2)
                accepts_same=accept_update(same_frozen,same_adapted,fit_labels[0])
                for g in remaining:
                    check=pool[g,1]
                    validate_roles(source,fit,check,score,rows,True)
                    name=f'p{p}_pos{pos}_{method}_check{g}'
                    cf=predict(x[check],model)
                    ca=predict(x[check],model,state2)
                    accepted=accept_update(cf,ca,g)
                    state3=fit_adapter(templates,x[fit+[check]],fit_labels+[g],method)
                    fitted3=votes(predict(x[score],model,state3))
                    choices=dict(frozen=frozen,fit_two=adapted,fit_three=fitted3,
                                 audit_novel=adapted if accepted else frozen,
                                 audit_same=adapted if accepts_same else frozen)
                    mask=~np.isin(np.arange(7),fit_labels+[g])
                    base_omitted=float(np.mean(frozen[mask]==np.arange(7)[mask]))
                    other_gain=float(np.mean(adapted[mask]==np.arange(7)[mask]))-base_omitted
                    states[name]=dict(two=state2,three=state3)
                    access.append(dict(name=name,source=source,fit=fit,check=check,
                                       same_check=same,score=score,novel_label=g,
                                       fit_labels=fit_labels,omitted_labels=np.where(mask)[0].tolist()))
                    predictions[name+'_check_frozen']=cf
                    predictions[name+'_check_adapted']=ca
                    predictions[name+'_same_frozen']=same_frozen
                    predictions[name+'_same_adapted']=same_adapted
                    for policy,pred in choices.items():
                        predictions[name+'_'+policy]=pred
                        acc=float(np.mean(pred==np.arange(7)))
                        oa=float(np.mean(pred[mask]==np.arange(7)[mask]))
                        budget=fit+([same] if policy=='audit_same' else [check])
                        if policy=='frozen': budget=[]
                        if policy=='fit_two': budget=fit
                        records.append(dict(name=name,participant=p,position=pos,method=method,
                          policy=policy,check_label=g,selected_active=g==remaining[0],
                          selected_diverse=g==diverse,accuracy=acc,omitted_accuracy=oa,
                          harm=oa<base_omitted,positive_loss=max(0.,base_omitted-oa),
                          accepted=accepted if policy=='audit_novel' else accepts_same if policy=='audit_same' else None,
                          proposed_omitted_gain=other_gain,
                          false_accept=bool((accepted if policy=='audit_novel' else accepts_same) and other_gain<0)
                              if policy in ('audit_novel','audit_same') else None,
                          recorded_seconds=sum(rows[i]['recorded_seconds'] for i in budget),
                          analyzed_seconds=2*len(budget),
                          check_improvement=float(np.mean(ca==g)-np.mean(cf==g))))
        print(json.dumps(dict(participant=p,records=len(records))),flush=True)
    (OUT/'results.json').write_text(json.dumps(records,indent=2)+'\n')
    (OUT/'access.json').write_text(json.dumps(access,indent=2)+'\n')
    (OUT/'states.json').write_text(json.dumps(states,indent=2)+'\n')
    np.savez_compressed(OUT/'models.npz',**models)
    np.savez_compressed(OUT/'predictions.npz',**predictions)
    summaries=[]
    for method in cfg['methods']:
        for selection in ('uniform','active','diverse'):
            for policy in ('frozen','fit_two','fit_three','audit_novel','audit_same'):
                per=[]
                for p in DEVELOPMENT:
                    rr=[r for r in records if r['participant']==p and r['method']==method and r['policy']==policy
                        and (selection=='uniform' or r['selected_'+selection])]
                    metrics={k:float(np.mean([r[k] for r in rr])) for k in
                             ('accuracy','omitted_accuracy','harm','positive_loss','recorded_seconds','analyzed_seconds')}
                    if policy.startswith('audit'):
                        metrics.update({k:float(np.mean([r[k] for r in rr])) for k in ('accepted','false_accept')})
                    per.append(dict(participant=p,cases=len(rr),**metrics))
                summaries.append(dict(method=method,selection=selection,policy=policy,participants=per,
                                      means={k:float(np.mean([v[k] for v in per]))
                                             for k in per[0] if k not in ('participant','cases')}))
    (OUT/'summary.json').write_text(json.dumps(summaries,indent=2)+'\n')
    (OUT/'complete.json').write_text(json.dumps(dict(cases=len(access),records=len(records),
                timestamp=datetime.now(timezone.utc).isoformat()))+'\n')
    for r in summaries:
        if r['method']=='cosine_gain' and r['selection']=='uniform':
            print(json.dumps({k:v for k,v in r.items() if k!='participants'}),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=1): main()
