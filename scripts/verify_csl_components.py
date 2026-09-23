"""Independent model replay, state masks, splits and sample reconstruction."""
import argparse,json,time
from pathlib import Path
from collections import Counter
import numpy as np
import scipy.io as sio
from scipy.ndimage import median_filter
import torch
from scripts import csl_author_recipe as r
from scripts.verify_csl_author_recipe import restore

def predictions(m,x):
    a=[]
    with torch.no_grad():
        for t in x.split(2048):
            if getattr(m,'_verification_variant',None)=='resample_then_gain_then_bn':
                z=m.input_transform(t)
                if m.corrective_gain:z=z*torch.clamp(m.mean_session1/(m.input_transform(m.mean_session2)+1e-12),.5,2.)
                logits=m.fc(m.bn(z).flatten(1))
            else:logits=m(t)
            a.append(logits.argmax(1).numpy())
    return np.concatenate(a)
def scores(p,entries,label):
    at=0;v=[]
    for e in entries:
        block=p[at:at+e['frames']];counts=np.bincount(block,minlength=8);ties=set(np.flatnonzero(counts==counts.max()));v.append(next(int(n) for n in block if n in ties));at+=e['frames']
    assert at==len(p);y=np.array([e['label'] for e in entries]);v=np.array(v)
    return v,float(np.mean(v==y)),float(np.mean(v[y!=label]==y[y!=label]))
def verify(root,manifest,development=False,participant=None,source_root=None,references_per_person=20):
    torch.set_num_threads(1)
    start=time.perf_counter();assert len({e['id'] for e in manifest})==len(manifest)
    assert len({e['raw_sha256'] for e in manifest})==len(manifest)
    for e in manifest:assert r.sha(e['cache_path'])==e['cache_sha256']
    nets,utils,tensor,proc=r.modules();rawchecks=[]
    people=[1] if development else [participant] if participant else [2,3,4,5]
    for person in people:
        # Verify mapping/median pooling by an independent spatial construction on two raw trials.
        for session,rep in [(1,0),(2,1)]:
            e=next(e for e in manifest if e.get('participant',1)==person and e['session']==session and e['gesture']==8 and e['rep']==rep)
            raw=sio.loadmat(e['path'])['gestures'][rep,0];assert __import__('hashlib').sha256(raw.tobytes()).hexdigest()==e['raw_sha256']
            emg=raw.T-raw.T.mean(0,keepdims=True);emg=proc.bandstop(proc.bandpass(emg,fs=2048),fs=2048);rms=np.asarray(proc.get_rms_signal(emg,Mrms=512))
            frame=rms[e['start']+min(10,e['frames']-1)];grid=np.flipud(np.delete(frame,np.s_[7:192:8]).reshape((7,24),order='F')).astype(np.float32)
            expected=median_filter(grid,size=3,mode='constant',cval=0);actual=np.load(e['cache_path'])[min(10,e['frames']-1),0]
            np.testing.assert_array_equal(expected,actual);rawchecks.append(e['id'])
    files=list(root.glob('s*_to_s*/g*/results.json')) if development else list(root.glob('subject*/s*_to_s*/g*/results.json'))
    if participant:files=[f for f in files if f.parent.parent.parent.name==f'subject{participant}']
    expected=8 if development else 160 if participant else 640;assert len(files)==expected,(len(files),expected)
    modelcount=0;predictioncount=0;paircache={};sourcecache={};calcount=0;gainmean_checks=0;source_means={}
    for i,file in enumerate(sorted(files)):
        folder=file.parent;parts=folder.parent.name.split('_');src=int(parts[0][1:]);tar=int(parts[-1][1:]);person=1 if development else int(folder.parent.parent.name.removeprefix('subject'));key=(person,src,tar)
        if key not in paircache:
            # Keep one pair's frame tensors to bound memory.
            pm=[e for e in manifest if e.get('participant',1)==person and e['session']==tar]
            te=[e for e in pm if e['rep']>0];assert set(e['label'] for e in te)==set(range(8)) and len({e['id'] for e in te})==len(te)
            x=torch.from_numpy(np.concatenate([np.load(e['cache_path']) for e in te]));paircache={key:(te,x,pm)}
        te,x,pm=paircache[key];gesture=int(folder.name[1:]);label=r.RAW_GESTURES.index(gesture)
        ce=[e for e in pm if e['gesture']==gesture and e['rep']==0];assert len(ce)==1 and not set(e['id'] for e in ce)&set(e['id'] for e in te)
        cx=torch.from_numpy(np.load(ce[0]['cache_path']));saved=np.load(folder/'predictions.npz')
        assert saved['trial_ids'].tolist()==[e['id'] for e in te];np.testing.assert_array_equal(saved['labels'],np.concatenate([np.repeat(e['label'],e['frames']) for e in te]))
        sk=(person,src)
        if sk not in sourcecache:
            base=restore('source') if development else restore('source',(source_root or root)/f'subject{person}/source_s{src}');sourcecache={sk:base}
            source_means={}
            for g in r.RAW_GESTURES:
                source_entries=[e for e in manifest if e.get('participant',1)==person and e['session']==src and e['gesture']==g]
                source_means[g]=torch.from_numpy(np.concatenate([np.load(e['cache_path']) for e in source_entries])).mean(0,keepdim=True)
        base=sourcecache[sk]
        for result in json.loads(file.read_text()):
            method=result['method'];m=base if method=='frozen' else restore(method,folder)
            if method!='frozen':
                flags=json.loads((folder/(method+'_state.json')).read_text())
                if 'forward_variant' in flags:
                    assert flags['forward_variant']=='resample_then_gain_then_bn';m._verification_variant=flags['forward_variant']
            uses_gain=method in ('gain_only','spatial_gain','classifier_fine_gain')
            assert m.corrective_gain==uses_gain
            if uses_gain:
                torch.testing.assert_close(m.mean_session1,source_means[gesture],rtol=0,atol=0)
                torch.testing.assert_close(m.mean_session2,cx.mean(0,keepdim=True),rtol=0,atol=0);gainmean_checks+=2
            p=predictions(m,x);np.testing.assert_array_equal(p,saved[method]);predictioncount+=len(p);modelcount+=1
            votes,accuracy,unseen=scores(p,te,label);assert votes.tolist()==result['metrics']['trial_predictions'];assert accuracy==result['metrics']['trial_accuracy'];assert unseen==result['metrics']['unseen_trial_accuracy']
            cp=predictions(m,cx);assert float(np.mean(cp==label))==result['calibration_frame_accuracy'];calcount+=len(cp)
            if method.startswith('spatial') or method=='gain_only':
                for name,value in base.state_dict().items():
                    if not name.startswith('input_transform.'):assert torch.equal(value,m.state_dict()[name])
        if i%16==0:print('Verified cases',i+1,'/',expected,flush=True)
    references=0
    if not development:
        for file in sorted(root.glob('subject*/s*_to_s*/allclass_reference/results.json')):
            folder=file.parent;person=int(folder.parent.parent.name.removeprefix('subject'));target=int(folder.parent.name.split('_')[-1][1:])
            if participant and person!=participant:continue
            entries=[e for e in manifest if e['participant']==person and e['session']==target and e['rep']>0];x=torch.from_numpy(np.concatenate([np.load(e['cache_path']) for e in entries]));m=restore('classifier_fine',folder)
            p=predictions(m,x);saved=np.load(folder/'predictions.npz');np.testing.assert_array_equal(p,saved['predictions']);v,acc,_=scores(p,entries,0);j=json.loads(file.read_text());assert v.tolist()==j['metrics']['trial_predictions'];assert acc==j['metrics']['trial_accuracy'];predictioncount+=len(p);references+=1
        assert references==(references_per_person if participant else 4*references_per_person)
    report=dict(passed=True,calibration_cases=expected,replayed_conditions=modelcount,allclass_references=references,predictions=predictioncount,calibration_predictions=calcount,cache_hashes=len(manifest),gain_mean_checks=gainmean_checks,raw_reconstructions=rawchecks,wall_seconds=time.perf_counter()-start)
    r.dump((root/f'subject{participant}' if participant else root)/'verification.json',report);print(report,flush=True)
    return report

def worker(args):
    person,run_name,references=args
    root=r.ROOT/'research/runs'/run_name;source=r.ROOT/'research/runs/20260907_csl_components_confirmatory';manifest=json.loads((source/'manifest.json').read_text())
    return verify(root,[e for e in manifest if e['participant']==person],participant=person,source_root=source,references_per_person=references)

def final_parallel(run_name='20260907_csl_components_confirmatory',references=20):
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing
    start=time.perf_counter();root=r.ROOT/'research/runs'/run_name
    manifest=json.loads((r.ROOT/'research/runs/20260907_csl_components_confirmatory/manifest.json').read_text());assert len({e['id'] for e in manifest})==len(manifest);assert len({e['raw_sha256'] for e in manifest})==len(manifest)
    with ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('spawn')) as pool:reports=list(pool.map(worker,[(p,run_name,references) for p in [2,3,4,5]]))
    result={k:sum(p[k] for p in reports) for k in ['calibration_cases','replayed_conditions','allclass_references','predictions','calibration_predictions','cache_hashes','gain_mean_checks']}
    result.update(passed=all(p['passed'] for p in reports),raw_reconstructions=sum([p['raw_reconstructions'] for p in reports],[]),wall_seconds=time.perf_counter()-start,participant_reports=reports)
    r.dump(root/'verification.json',result);print(result,flush=True)
if __name__=='__main__':
    torch.set_num_threads(1);ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['development','final','input-order','identity']);a=ap.parse_args();dev=a.mode=='development'
    if dev:
        root=r.ROOT/'research/runs/20260907_csl_components_development';manifest=json.loads((r.OUT/'manifest.json').read_text());verify(root,manifest,True)
    elif a.mode=='input-order':final_parallel('20260907_csl_input_order',0)
    elif a.mode=='identity':final_parallel('20260907_csl_identity_ties',0)
    else:final_parallel()
