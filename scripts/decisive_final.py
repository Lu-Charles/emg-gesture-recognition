"""Matched two-recording GRABMyo development experiment; immutable run outputs.

Reuses independently verified enrolled CNNs. Only heads change here; full-network
adaptation is a separate robustness experiment. Every scored trial stays fixed.
"""
import argparse, copy, csv, hashlib, json, platform, time
from datetime import datetime, timezone
from pathlib import Path
import joblib
import numpy as np
import torch
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from src.emg_cnn import CompactEMGNet
from src.grabmyo import allocations
from src.grabmyo_corpus import hash_stream
from src.final_coverage import FinalCorpus as WindowCorpus
from src.final_coverage import validate_roles
from scripts.cache_final_coverage import frozen

ROOT=Path(__file__).resolve().parents[1]
CACHE=ROOT/'data/public/grabmyo/cache_final_20260916'
PRIORS={seed:f'20260916_final_enrollment_seed{seed}' for seed in [42,0,1]}


def schedules(rows,person,session,seed=20260916):
    order=np.random.default_rng(seed).permutation(16).tolist()
    pool={(int(r['class_index']),int(r['calibration_rank'])):i for i,r in enumerate(rows)
          if int(r['participant'])==person and int(r['session'])==session and r['role']=='calibration'}
    for i,g in enumerate(order):
        h=order[(i+1)%16]
        for k in [1,2]:
            yield dict(choice=i,k=k,gestures=[g] if k==1 else [g,h],
                       ids=[pool[g,1],pool[g if k==1 else h,2]])


def measures(votes,y,gestures):
    selected=np.isin(y,gestures);other=(y<16)&~selected
    cm=np.zeros((17,17),int);np.add.at(cm,(y,votes),1)
    return dict(accuracy=float(np.mean(votes==y)),macro_f1=float(f1_score(y,votes,labels=range(17),average='macro',zero_division=0)),
        calibrated_recall=float(np.mean(votes[selected]==y[selected])),
        other_active_accuracy=float(np.mean(votes[other]==y[other])),
        other_into_calibration=float(np.mean(np.isin(votes[other],gestures))),
        rest_recall=float(np.mean(votes[y==16]==16)),confusion=cm.tolist())


def vote(pred):
    return np.array([np.bincount(a,minlength=17).argmax() for a in np.asarray(pred).reshape(-1,35)])


def fit_head(base,cx,cy,sx,sy,method,seed,steps=(25,100,300)):
    """Equal target batches/steps; replay adds one source batch per target batch."""
    model=copy.deepcopy(base);initial=[p.detach().clone() for p in base.parameters()]
    opt=torch.optim.Adam(model.parameters(),lr=.001)
    rng=np.random.default_rng(seed);srng=np.random.default_rng(seed+100000)
    source_by_class=[np.flatnonzero(sy.numpy()==c) for c in range(17)]
    states={};history=[];model.train()
    for step in range(1,max(steps)+1):
        ids=rng.integers(0,len(cx),64)
        opt.zero_grad();target=torch.nn.functional.cross_entropy(model(cx[ids]),cy[ids]);loss=target
        if method=='replay':
            # Uniform class choice, then uniform window in that source class.
            classes=srng.integers(0,17,64)
            si=np.array([srng.choice(source_by_class[c]) for c in classes])
            loss=loss+torch.nn.functional.cross_entropy(model(sx[si]),sy[si])
        elif method=='l2':
            loss=loss+sum(((p-p0)**2).sum() for p,p0 in zip(model.parameters(),initial))
        elif method!='naive':raise ValueError(method)
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite adaptation loss')
        loss.backward();opt.step()
        if step in steps:
            states[step]={k:v.detach().clone() for k,v in model.state_dict().items()}
            with torch.no_grad():
                history.append(dict(step=step,target_loss=float(torch.nn.functional.cross_entropy(model(cx),cy)),
                                    source_accuracy=float((model(sx).argmax(1)==sy).float().mean())))
    return states,history


def embeddings(model,corpus,ids,mean,scale):
    wid=corpus.window_ids(ids);values=[]
    with torch.no_grad():
        for offset in range(0,len(wid),512):
            x,_=corpus.batch(wid[offset:offset+512],mean,scale)
            values.append(model.encoder(torch.from_numpy(x)).numpy())
    return np.concatenate(values).reshape(len(ids),35,64)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--seed',type=int,default=42)
    args=ap.parse_args();out=args.out;out.mkdir(parents=True,exist_ok=False);(out/'models').mkdir()
    torch.set_num_threads(1);torch.manual_seed(args.seed)
    corpus=WindowCorpus(CACHE,'final');rows=corpus.rows
    typed=[{**r,**{k:int(r[k]) for k in ['participant','session','class_index','calibration_rank']}} for r in rows]
    people=sorted(allocations()[0]['final'])
    assert sorted({int(r['participant']) for r in rows})==people
    prior=ROOT/'research/runs'/PRIORS[args.seed]
    assert (prior/'enrollment_complete.json').exists()
    scaler=np.load(prior/'training_scaler.npz');mean,scale=scaler['mean'],scaler['scale']
    cfg=dict(timestamp=datetime.now(timezone.utc).isoformat(),scope='frozen independent evaluation',frozen_protocol_sha256=hash_stream(ROOT/'research/runs/20260625_confirmatory_protocol/protocol.json'),participants=people,
        seed=args.seed,prior=str(prior),source_session=1,target_sessions=[2,3],recording_seconds=10,
        calibration='K1: gesture g ranks1,2. K2: g rank1 and next gesture h rank2. All16 g; one seeded cyclic order.',
        order_seed=20260916,source_network='existing verified shared-pretrained then participant-enrolled CompactEMGNet',
        adaptation='head only; encoder frozen; fixed representation-training scaler',steps=[25,100,300],primary_steps=100,
        target_batch=64,replay_source_batch=64,lr=.001,optimizer='Adam',l2_penalty='sum squared parameter difference, coefficient1',
        methods=['cnn_frozen','cnn_naive','cnn_replay','cnn_l2','lda_frozen','lda_gain','lda_pooled'],
        features='MAV RMS meanWL, amplitude retained',scoring='fixed existing four whole trials per17classes;35windows each; lowest-label tie',
        numpy=np.__version__,torch=torch.__version__,platform=platform.platform(),
        code_sha256=hash_stream(__file__),manifest_sha256=hash_stream(CACHE/'final_manifest.csv'),
        independent_units='15people; session/calibration choices/steps are correlated',final_access=15)
    (out/'protocol.json').write_text(json.dumps(cfg,indent=2)+'\n');(out/'decisive_calibration.py').write_bytes(Path(__file__).read_bytes())
    records=[];access=[];saved={};start=time.perf_counter()
    for person in people:
        allids=[i for i,r in enumerate(rows) if int(r['participant'])==person]
        lookup={i:j for j,i in enumerate(allids)}
        model=CompactEMGNet();sourcefile=prior/f'models/p{person}_s2_shared_none_b0.pt'
        model.load_state_dict(torch.load(sourcefile,weights_only=True,map_location='cpu'));model.eval()
        z=embeddings(model,corpus,allids,mean,scale)
        np.save(out/f'p{person}_embeddings.npy',z)
        torch.save(model.state_dict(),out/f'models/p{person}_source.pt')
        source=[i for i in allids if rows[i]['role']=='enrollment']
        sx=torch.from_numpy(z[[lookup[i] for i in source]].reshape(-1,64));sy=torch.tensor(np.repeat([int(rows[i]['class_index']) for i in source],35))
        fx=np.asarray(corpus.features[source]).reshape(-1,48);fy=sy.numpy()
        standard=StandardScaler().fit(fx);lda=LinearDiscriminantAnalysis(solver='lsqr',shrinkage='auto',priors=np.full(17,1/17)).fit(standard.transform(fx),fy)
        joblib.dump((standard,lda),out/f'models/p{person}_lda.joblib')
        source_means=np.array([fx[fy==g].mean(0) for g in range(16)])
        for session in [2,3]:
            score=[i for i in allids if int(rows[i]['session'])==session and rows[i]['role']=='scoring']
            truth=np.array([int(rows[i]['class_index']) for i in score]);assert np.all(np.bincount(truth,minlength=17)==4)
            tx=torch.from_numpy(z[[lookup[i] for i in score]].reshape(-1,64));tfx=np.asarray(corpus.features[score]).reshape(-1,48)
            with torch.no_grad():frozen=vote(model.head(tx).argmax(1).numpy())
            lda_frozen=vote(lda.predict(standard.transform(tfx)))
            for case in schedules(rows,person,session):
                cal=case['ids'];validate_roles(typed,source,cal,score)
                name=f"p{person}_s{session}_o{case['choice']}_k{case['k']}"
                cx=torch.from_numpy(z[[lookup[i] for i in cal]].reshape(-1,64));cy=torch.tensor(np.repeat([int(rows[i]['class_index']) for i in cal],35))
                base=dict(participant=person,session=session,choice=case['choice'],k=case['k'],gestures=case['gestures'])
                access.append(dict(name=name,source=source,calibration=cal,scoring=score,source_checkpoint_sha256=hash_stream(sourcefile)))
                def save(method,step,pred,extra=None):
                    key=f'{name}_{method}_t{step}';saved[key]=pred
                    records.append(dict(name=key,**base,method=method,steps=step,**measures(pred,truth,case['gestures']),**(extra or {})))
                save('cnn_frozen',0,frozen);save('lda_frozen',0,lda_frozen)
                calfx=np.asarray(corpus.features[cal]).reshape(-1,48)
                logratios=[]
                for g in case['gestures']:
                    logratios.append(np.log(np.maximum(source_means[g],1e-12))-np.log(np.maximum(calfx[cy.numpy()==g].mean(0),1e-12)))
                gain=np.clip(np.exp(np.mean(logratios,0)),.5,2.)
                save('lda_gain',0,vote(lda.predict(standard.transform(tfx*gain))),dict(gain=gain.tolist()))
                pooled=LinearDiscriminantAnalysis(solver='lsqr',shrinkage='auto',priors=np.full(17,1/17)).fit(standard.transform(np.concatenate([fx,calfx])),np.r_[fy,cy.numpy()])
                save('lda_pooled',0,vote(pooled.predict(standard.transform(tfx))))
                joblib.dump(pooled,out/f'models/{name}_lda_pooled.joblib')
                for method in ['naive','replay','l2']:
                    states,history=fit_head(model.head,cx,cy,sx,sy,method,args.seed+case['choice'],steps=cfg['steps'])
                    for step,state in states.items():
                        head=copy.deepcopy(model.head);head.load_state_dict(state)
                        with torch.no_grad():pred=vote(head(tx).argmax(1).numpy())
                        save('cnn_'+method,step,pred,dict(training=next(h for h in history if h['step']==step)))
                        torch.save(state,out/f'models/{name}_cnn_{method}_t{step}.pt')
            (out/'results.json').write_text(json.dumps(dict(config=cfg,records=records,seconds=time.perf_counter()-start),indent=2)+'\n')
            (out/'access.json').write_text(json.dumps(access,indent=2)+'\n')
            np.savez_compressed(out/'trial_predictions.npz',**saved)
            print(json.dumps(dict(participant=person,session=session,records=len(records),seconds=time.perf_counter()-start)),flush=True)
    (out/'complete.json').write_text(json.dumps(dict(records=len(records),cases=len(access),seconds=time.perf_counter()-start,finished=datetime.now(timezone.utc).isoformat()))+'\n')


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
