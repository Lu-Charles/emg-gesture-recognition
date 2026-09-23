"""Participant-separated confirmation of the frozen CSL component study."""
import argparse,contextlib,io,json,time,warnings,zipfile,hashlib
from pathlib import Path
import numpy as np
import scipy.io as sio
import torch
from scripts import csl_author_recipe as r
from scripts import csl_component_study as c
from scripts.prepare_csl_sal import SplitReader,PARTS
from scripts.verify_csl_author_recipe import restore
OUT=r.ROOT/'research/runs/20260907_csl_components_confirmatory'

def freeze():
    OUT.mkdir(exist_ok=True);assert not (OUT/'protocol.json').exists()
    scripts=['csl_author_recipe.py','csl_search_batch.py','csl_component_study.py','csl_confirmatory.py','verify_csl_author_recipe.py']
    config=dict(timestamp=r.datetime.now(r.timezone.utc).isoformat(),scope='Independent participant evaluation; exploratory development subject1 excluded',participants=[2,3,4,5],sessions=[1,2,3,4,5],pairs=[[a,b] for a in range(1,6) for b in range(1,6) if a!=b],gestures=r.RAW_GESTURES,
       seed=42,source='Each participant/session separately; all10reps each selectedclass; author_recipe_v1 source15epochs',calibration='rep0 exactlyoneclass; all8choices equally evaluated',scoring='reps1..9 all8classes, same72trials for every budget/method/gesture choice',
       methods=['frozen']+c.METHODS,reference='classifier_fine from all8 rep0 trials; same150epochs, moreupdates; contextual reference not data-only causal ablation',
       endpoints=['primary: accuracy on the7 classes absent from calibration','secondary: overall majority-vote trial accuracy','negative transfer: trialaccuracy strictly below frozen','conditional negative transfer when calibrationframeaccuracy>=.99','all calibration-gesture choices, no test-oracle selection'],
       comparisons=['gain_only minus frozen','spatial_gain minus gain_only','spatial_only minus frozen','spatial-gain factorial interaction','singleclass fine vs allclass reference; no newarchitectureclaim'],
       aggregation='Average all8gesturechoices within20pairs withinparticipant; four independent participant summaries, all disclosed. Paired effect and participant bootstrap95%interval (10000resamples seed2026) descriptive withn4; no window-level significance, no population/clinicalclaim',
       quality='Stop on malformed/missing/duplicate/nonfinite rawtrials before training; record exact issue and outcome-blind resolution. Do not fill duplicates or omit failed models.',
       limits='One public dataset,8authorselectedclasses,4held-outparticipants,single seed,offline natural session variation (not controlled electrode displacement). 3s labeled plus90s targetrest; fullclassreference24s labeled plus90srest.',
       hypothesis='Separate gain correction from spatial alignment and test transfer beyond calibratedclass. No method required towin; no parameter changes afterfinaloutcomes.',
       reuse='Unchanged June2025authorcomponents, correctedprovidergeometry and batched-search numericalvalidation, fixedsettings; oldnegativeexperimentsretained',
       scripts={name:r.sha(r.ROOT/'scripts'/name) for name in scripts},author_source=json.loads((r.OUT/'source/sha256.json').read_text()))
    r.dump(OUT/'protocol.json',config)
    for name in scripts:(OUT/name).write_bytes((r.ROOT/'scripts'/name).read_bytes())
    print('Frozen',config['timestamp'],flush=True)

def prepare():
    cfg=json.loads((OUT/'protocol.json').read_text());start=time.perf_counter();_,utils,tensor,proc=r.modules()
    prior=json.loads((r.ROOT/'research/runs/20260907_csl_sal_smoke_v1/archive_audit.json').read_text());ledger={e['name']:e['sha256'] for e in prior['members']}
    class Mapped(tensor.EMGData):
        def get_images(self,x):return r.provider_images(x)
    h=Mapped.__new__(Mapped);h.dataset='csl';h.fs=2048;h.Mrms=512;h.remove_baseline='mean_square'
    manifest=[];seen=set();rawfiles=[]
    with SplitReader(PARTS) as stream,zipfile.ZipFile(stream) as z:
        for person in cfg['participants']:
            for session in cfg['sessions']:
                directory=r.DATA/f'subject{person}/session{session}';directory.mkdir(parents=True,exist_ok=True)
                for gesture in [0]+r.RAW_GESTURES:
                    name=f'subject{person}/session{session}/gest{gesture}.mat';p=directory/f'gest{gesture}.mat'
                    if not p.exists():p.write_bytes(z.read(name))
                    assert r.sha(p)==ledger[name],name;rawfiles.append(dict(path=str(p),sha256=ledger[name]))
                rest=sio.loadmat(directory/'gest0.mat')['gestures'];assert rest.shape==(30,1),(str(directory),rest.shape)
                for rep in range(30):
                    raw=rest[rep,0];assert raw.shape==(192,6144) and np.isfinite(raw).all()
                    digest=hashlib.sha256(raw.tobytes()).hexdigest();assert digest not in seen,('duplicate rest',str(directory),rep);seen.add(digest)
                baseline=h.get_baseline(str(directory));assert np.isfinite(baseline).all()
                folder=OUT/f'subject{person}/frames';folder.mkdir(parents=True,exist_ok=True);np.save(folder/f'session{session}_baseline.npy',baseline)
                for label,gesture in enumerate(r.RAW_GESTURES):
                    path=directory/f'gest{gesture}.mat';cells=sio.loadmat(path)['gestures'];assert cells.ndim==2 and cells.shape[1]==1 and 2<=cells.shape[0]<=10,(str(path),cells.shape)
                    for rep in range(cells.shape[0]):
                        raw=cells[rep,0];assert raw.shape==(192,6144) and np.isfinite(raw).all(),(str(path),rep,raw.shape)
                        digest=hashlib.sha256(raw.tobytes()).hexdigest();assert digest not in seen,('duplicate',str(path),rep);seen.add(digest)
                        # Splits are whole trials: any source session uses all trials; in target rep0 is calibration only.
                        emg=raw.T;emg-=emg.mean(axis=0,keepdims=True);emg=proc.bandstop(proc.bandpass(emg,fs=2048),fs=2048)
                        lo,hi=h.segment(emg,baseline);assert 0<=lo<hi<=6144,(str(path),rep,lo,hi)
                        x=utils.median_pool_2d(torch.from_numpy(r.provider_images(proc.get_rms_signal(emg,Mrms=512))[lo:hi]).float()).numpy()
                        assert np.isfinite(x).all();tid=f'p{person}_s{session}_g{gesture:02}_r{rep:02}';cache=folder/(tid+'.npy');np.save(cache,x)
                        manifest.append(dict(id=tid,participant=person,session=session,gesture=gesture,label=label,rep=rep,path=str(path),raw_sha256=digest,cache_path=str(cache),cache_sha256=r.sha(cache),frames=len(x),start=int(lo),end=int(hi)))
                r.dump(OUT/'manifest_partial.json',manifest);print('Prepared',person,session,flush=True)
    counts={f'p{p}_s{s}_g{g}':sum(e['participant']==p and e['session']==s and e['gesture']==g for e in manifest) for p in cfg['participants'] for s in cfg['sessions'] for g in r.RAW_GESTURES}
    assert len(counts)==160 and all(2<=n<=10 for n in counts.values())
    r.dump(OUT/'manifest.json',manifest);r.dump(OUT/'preprocessing.json',dict(passed=True,trials=len(manifest),rawfiles=rawfiles,trial_counts=counts,missing_repetitions={k:10-v for k,v in counts.items() if v<10},wall_seconds=time.perf_counter()-start))

def run(person_filter=None):
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    warnings.filterwarnings('ignore',message='There is a performance drop')
    cfg=json.loads((OUT/'protocol.json').read_text());manifest=json.loads((OUT/'manifest.json').read_text());start=time.perf_counter()
    if person_filter is not None:cfg['participants']=[person_filter]
    targetout=OUT if person_filter is None else OUT/f'subject{person_filter}'
    for name,h in cfg['scripts'].items():assert r.sha(r.ROOT/'scripts'/name)==h,('codechanged',name)
    for e in manifest:assert r.sha(e['cache_path'])==e['cache_sha256']
    nets,*_=r.modules();combined=[]
    for person in cfg['participants']:
        dest=OUT/f'subject{person}';pm=[e for e in manifest if e['participant']==person]
        for source in cfg['sessions']:
            sm=[dict(e,role='source') for e in pm if e['session']==source];sx,sy,se=r.role_data(sm,'source');assert set(sy.tolist())==set(range(8)) and len(se)==len(sm)
            srcdir=dest/f'source_s{source}';srcdir.mkdir(exist_ok=True)
            if (srcdir/'source.pt').exists():base=restore('source',srcdir)
            else:
                torch.manual_seed(42);base=r.make_model(nets);base.train()
                with contextlib.redirect_stdout(io.StringIO()):log=r.fit(base,sx,sy,15,.001,3)
                base.eval();c.save(base,srcdir/'source');r.dump(srcdir/'training.json',dict(**log,source_fit=r.metrics(r.pred(base,sx),sy,se)))
            for target in cfg['sessions']:
                if source==target:continue
                pairdir=dest/f's{source}_to_s{target}'
                results=c.pair(pm,dest,base,sx,sy,source,target)
                combined.extend([dict(participant=person,source_session=source,target_session=target,**v) for v in results])
                reference=pairdir/'allclass_reference';reference.mkdir(exist_ok=True)
                if not (reference/'results.json').exists():
                    tm=[dict(e,role='calibration' if e['rep']==0 else 'score') for e in pm if e['session']==target];cx,cy,ce=r.role_data(tm,'calibration');tx,ty,te=r.role_data(tm,'score')
                    m,log=c.adapt(base,sx,sy,cx,cy,'classifier_fine');c.save(m,reference/'classifier_fine');p=r.pred(m,tx)
                    np.savez_compressed(reference/'predictions.npz',predictions=p,labels=np.asarray(ty),trial_ids=np.array([e['id'] for e in te]),lengths=np.array([e['frames'] for e in te]))
                    metric=r.metrics(p,ty,te);metric.pop('unseen_trial_accuracy');r.dump(reference/'results.json',dict(metrics=metric,**log))
                r.dump(targetout/'results_partial.json',combined);print('Finished pair',person,source,target,flush=True)
        print('Finished participant',person,flush=True)
    r.dump(targetout/'results.json',dict(results=combined,wall_seconds=time.perf_counter()-start));print('Participant worker completed',person_filter,flush=True)

def parallel_run():
    # Computational workers only: read frozen code/data, write disjoint participant artifacts.
    # The coordinator alone writes the combined result. No concurrent repository editing.
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing
    start=time.perf_counter();cfg=json.loads((OUT/'protocol.json').read_text())
    with ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('spawn')) as pool:
        list(pool.map(run,cfg['participants']))
    results=[];timings={}
    for person in cfg['participants']:
        record=json.loads((OUT/f'subject{person}/results.json').read_text());results.extend(record['results']);timings[str(person)]=record['wall_seconds']
    assert len(results)==4*20*8*6
    r.dump(OUT/'results.json',dict(results=results,wall_seconds=time.perf_counter()-start,participant_worker_seconds=timings,computational_workers=4))
    print('All confirmatory cases completed',flush=True)
if __name__=='__main__':
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);warnings.filterwarnings('ignore',message='There is a performance drop')
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['freeze','prepare','run']);a=ap.parse_args()
    {'freeze':freeze,'prepare':prepare,'run':parallel_run}[a.stage]()
