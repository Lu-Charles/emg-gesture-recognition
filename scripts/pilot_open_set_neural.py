"""Matched compact-backbone CNN/PL/PredIN development experiment."""
import argparse
import copy
import itertools
import json
import platform
import subprocess
import sys
import time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from src.grabmyo_corpus import WindowCorpus,hash_stream,epoch_blocks
from src.open_set_emg import allocations,validate_allocation,select,cutoff,measures,command_curve
from src.open_set_neural import OpenSetNet,known_moments,remap_known,infer


def dump(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def sync(device):
    if device=='mps':torch.mps.synchronize()


def fit_arrays(model,x,y,epochs,lr,seed,device):
    labels=remap_known(y);xt=torch.from_numpy(x).to(device);yt=torch.from_numpy(labels).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=lr);rng=np.random.default_rng(seed);history=[]
    model.train();start=time.perf_counter()
    for epoch in range(epochs):
        order=rng.permutation(len(labels));totals={}
        for offset in range(0,len(order),128):
            index=torch.tensor(order[offset:offset+128],device=device)
            optimizer.zero_grad();loss,terms=model.loss(xt[index],yt[index]);loss.backward();optimizer.step()
            for key,value in dict(terms,total=loss).items():totals[key]=totals.get(key,0.)+float(value.detach().cpu())*len(index)
        history.append({k:v/len(labels) for k,v in totals.items()})
    sync(device)
    if not all(np.isfinite(list(r.values())).all() for r in history):raise FloatingPointError('Nonfinite training')
    return dict(seconds=time.perf_counter()-start,epoch_losses=history)


def pretrain(model,corpus,ids,mean,scale,device,out,seed):
    optimizer=torch.optim.Adam(model.parameters(),lr=.001);history=[];start=time.perf_counter()
    ids=np.asarray(ids);model.train()
    for epoch in range(20):
        tick=time.perf_counter();totals={};seen=0
        for relative,order in epoch_blocks(len(ids),seed,epoch):
            selected=ids[relative];x,y=corpus.batch(corpus.window_ids(selected),mean,scale);y=remap_known(y)
            xt=torch.from_numpy(x).to(device);yt=torch.from_numpy(y).to(device)
            for offset in range(0,len(order),128):
                ind=torch.tensor(order[offset:offset+128],device=device)
                optimizer.zero_grad();loss,terms=model.loss(xt[ind],yt[ind]);loss.backward();optimizer.step()
                seen+=len(ind)
                for key,value in dict(terms,total=loss).items():totals[key]=totals.get(key,0.)+float(value.detach().cpu())*len(ind)
            del xt,yt,x,y
        sync(device);assert seen==len(ids)*35
        row=dict(epoch=epoch+1,seconds=time.perf_counter()-tick,windows=seen,**{k:v/seen for k,v in totals.items()})
        if not np.isfinite(list(row.values())).all():raise FloatingPointError('Nonfinite pretraining')
        history.append(row);dump(out/'pretraining_history.json',history)
        torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()},out/'pretrain_last.pt')
        print(json.dumps(dict(family=model.family,phase='pretrain',**row)),flush=True)
    return time.perf_counter()-start


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--classical',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=False);start=time.perf_counter()
    old=json.loads((args.classical/'config.json').read_text());cache=Path(old['cache'])
    assert json.loads((args.classical/'validation.json').read_text())['passed']
    train,dev=WindowCorpus(cache,'train'),WindowCorpus(cache,'development')
    rows=json.loads((args.classical/'trial_manifest.json').read_text())
    assert all(all(r[k]==v for k,v in original.items()) for r,original in zip(rows,dev.rows,strict=True))
    summary=json.loads((cache/'summary.json').read_text())
    for group in ('train','development'):
        for kind,ext in [('signals','npy'),('manifest','csv')]:
            assert hash_stream(cache/f'{group}_{kind}.{ext}')==summary['groups'][group][kind+'_sha256']
    assert not set(summary['groups']['train']['participants']) & set(old['participants'])
    known_ids=[i for i,r in enumerate(train.rows) if 10<=int(r['class_index'])<=16]
    assert len(known_ids)==2940
    torch.set_num_threads(4);device='mps' if torch.backends.mps.is_available() else 'cpu'
    config=dict(created_utc=datetime.now(timezone.utc).isoformat(),scope='matched-backbone method adaptation, not original-paper numerical reproduction',
        classical=str(args.classical.resolve()),classical_config_sha256=hash_stream(args.classical/'config.json'),
        trial_manifest_sha256=hash_stream(args.classical/'trial_manifest.json'),cache=str(cache),
        cache_summary_sha256=hash_stream(cache/'summary.json'),seed=42,device=device,torch=torch.__version__,platform=platform.platform(),
        families=['cnn','pl','predin'],training_people=summary['groups']['train']['participants'],development_people=old['participants'],
        known_labels=list(range(10,17)),unknown_labels=list(range(10)),train_trial_ids=known_ids,
        embedding_dim=128,backbone='existing compact3conv16channel encoder + linear64to128; no embedding/prototype normalization',
        shared_epochs=20,shared_lr=.001,enrollment_epochs=20,enrollment_lr=.001,update_epochs=10,update_lr=.0001,
        optimizer='Adam; fresh optimizer per enrollment/update; reset enrolled model for every subset',batch_size=128,
        update_strategies=['target_finetune','pooled_replay'],known_retention=[.90,.95,.99],primary_retention=.95,
        literal_predin='arXiv2407.19753v2 Eqs2–16; compactness Eq5 literal vector-norm piecewise, beta=gamma=alpha=1,m1=.5,m2=1; sum per-branch losses and inconsistency once',
        method_limits='paper uses different backbones,SGD100epochs and same-session trial splits; our adaptation uses project Adam20pretrain/20enroll/10update and fixed GRABMyo protocol; no official code located',
        rejection='CNN maxsoftmax;PL/PredIN maximum raw dot-product similarity (primary) plus maxsoftmax control; PredIN averages raw branch similarities',
        normalization='new mean/std over known-only20person training windows; frozen for all methods',git_revision=None,
        source_hashes={f:hash_stream(f) for f in ['scripts/pilot_open_set_neural.py','src/open_set_neural.py','src/open_set_emg.py','src/emg_cnn.py','src/grabmyo_corpus.py','src/grabmyo.py','tests/test_open_set_neural.py']})
    dump(args.out/'config.json',config);dump(args.out/'training_access.json',[train.rows[i] for i in known_ids]);dump(args.out/'trial_manifest.json',rows)
    (args.out/'environment.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
    snap=args.out/'code_snapshot';snap.mkdir()
    for f in config['source_hashes']:(snap/Path(f).name).write_bytes(Path(f).read_bytes())
    plans=[]
    for p in old['participants']:
        for session in (2,3):
            for a in allocations(rows,p,session):
                for strategy in (config['update_strategies'] if a['target_fit'] else ['target_finetune']):
                    b=dict(a,participant=p,session=session,update_strategy=strategy);validate_allocation(b,rows);plans.append(b)
    dump(args.out/'allocations.json',plans)
    print(json.dumps(dict(phase='config_frozen',allocations=len(plans),training_trials=len(known_ids))),flush=True)
    mean,scale,count=known_moments(train,known_ids);np.savez(args.out/'normalization.npz',mean=mean,scale=scale,count=count)
    def arrays(ids):return dev.batch(dev.window_ids(ids),mean,scale)
    family_results=[]
    for family in config['families']:
        out=args.out/family;out.mkdir();(out/'models').mkdir();(out/'predictions').mkdir()
        fstart=time.perf_counter();model=OpenSetNet(family,42).to(device)
        training_seconds=pretrain(model,train,known_ids,mean,scale,device,out,42)
        shared=copy.deepcopy(model).cpu();del model
        metrics=[];source_metrics=[];access=[];histories=[];curves=[];prediction_count=0
        for p in old['participants']:
            source=select(rows,p,1,'enroll_fit');source_q=select(rows,p,1,'source_threshold');source_test=select(rows,p,1,'source_score')
            base=copy.deepcopy(shared).to(device);x,y=arrays(source);histories.append(dict(participant=p,stage='enroll',**fit_arrays(base,x,y,20,.001,42,device)))
            states={};scored={};base_key=((), 'target_finetune')
            def state_for(cal,strategy):
                key=(tuple(sorted(cal)),strategy if cal else 'target_finetune')
                if key not in states:
                    m=copy.deepcopy(base)
                    update_ids=(source+list(cal)) if strategy=='pooled_replay' and cal else list(cal)
                    if cal:
                        xx,yy=arrays(update_ids);h=fit_arrays(m,xx,yy,10,.0001,42,device)
                        histories.append(dict(participant=p,stage='update',target_ids=list(cal),update_strategy=strategy,**h))
                    name=f'p{p}_m{len(states):03d}'
                    torch.save({k:v.detach().cpu() for k,v in m.state_dict().items()},out/'models'/(name+'.pt'))
                    states[key]=(name,m.cpu())
                    access.append(dict(model=name,participant=p,source_ids=source,target_ids=list(cal),update_ids=update_ids,
                                       fit_ids=source+list(cal),update_strategy=strategy,pretraining_access='shared known-only training_access.json'))
                return key,states[key]
            def score_for(cal,strategy,eval_ids):
                nonlocal prediction_count
                key,(name,m)=state_for(cal,strategy);k=(key,tuple(eval_ids))
                if k not in scored:
                    xx,yy=arrays(eval_ids);m.to(device);pr,sc=infer(m,xx,device);m.cpu()
                    fname=f'{name}_e{len(scored):03d}.npz';np.savez_compressed(out/'predictions'/fname,trial_ids=eval_ids,y=yy,pred=pr,**sc)
                    scored[k]=(fname,yy,pr,sc);prediction_count+=len(yy)
                    if np.any(yy<10):
                        for rej,values in sc.items():
                            t,c,u=command_curve(yy,pr,values);cf=fname.replace('.npz','_'+rej+'_curve.npz')
                            np.savez_compressed(out/'predictions'/cf,threshold=t,known_correct_acceptance=c,unknown_false_acceptance=u)
                            order=np.argsort(-values,kind='stable');sv=values[order];ends=np.r_[np.flatnonzero(np.diff(sv)!=0),len(yy)-1]
                            ccr=np.r_[0.,np.cumsum(((yy>=10)&(pr==yy))[order])[ends]/np.sum(yy>=10)]
                            fpr=np.r_[0.,np.cumsum((yy<10)[order])[ends]/np.sum(yy<10)]
                            curves.append(dict(score_file=fname,rejector=rej,curve_file=cf,auroc=float(roc_auc_score(yy>=10,values)),standard_oscr=float(np.trapezoid(ccr,fpr))))
                return scored[k]
            qfile,_,_,source_scores=score_for([], 'target_finetune',source_q)
            sfile,sy,sp,ss=score_for([], 'target_finetune',source_test)
            for rej in ss:
                for retention in config['known_retention']:
                    t=cutoff(source_scores[rej],retention);source_metrics.append(dict(family=family,participant=p,rejector=rej,retention=retention,threshold=t,threshold_file=qfile,score_file=sfile,**measures(sy,sp,ss[rej],t)))
            for a in [a for a in plans if a['participant']==p]:
                cal=a['target_fit'];strategy=a['update_strategy'];_,(name,_)=state_for(cal,strategy)
                qcal=[] if a['method'] in ('none','model_only') else cal
                qstrategy='target_finetune' if not qcal else strategy
                qf,_,_,qs=score_for(qcal,qstrategy,a['threshold']);sf,yy,pr,sc=score_for(cal,strategy,a['score'])
                for rej in sc:
                    for retention in config['known_retention']:
                        t=cutoff(qs[rej],retention)
                        metrics.append(dict(family=family,participant=p,session=a['session'],method=a['method'],budget=a['budget'],seconds=a['budget']*35,
                            order=''.join(map(str,a['order'])),update_strategy=strategy,rejector=rej,retention=retention,threshold=t,model=name,
                            threshold_file=qf,score_file=sf,**measures(yy,pr,sc[rej],t)))
            for filename,content in [('metrics',metrics),('source_metrics',source_metrics),('model_access',access),('adaptation_history',histories),('curve_index',curves)]:dump(out/(filename+'.json'),content)
            print(json.dumps(dict(family=family,phase='development',participant=p,models=len(access),metric_rows=len(metrics))),flush=True)
            del base,states,scored
        result=dict(family=family,completed_utc=datetime.now(timezone.utc).isoformat(),pretraining_seconds=training_seconds,total_seconds=time.perf_counter()-fstart,models=len(access),metric_rows=len(metrics),predictions=prediction_count)
        dump(out/'summary.json',result);family_results.append(result);print(json.dumps(result),flush=True)
    dump(args.out/'summary.json',dict(completed_utc=datetime.now(timezone.utc).isoformat(),seconds=time.perf_counter()-start,families=family_results,final_participants_accessed=0))
    dump(args.out/'artifact_sha256.json',{str(p.relative_to(args.out)):hash_stream(p) for p in args.out.rglob('*') if p.is_file()})


if __name__=='__main__':main()
