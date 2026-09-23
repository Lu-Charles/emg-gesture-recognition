"""Development-only matched comparison of source CE + Deep CORAL and controls."""
from __future__ import annotations

import argparse
import copy
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from src.alignment_comparator import source_alignment_objective
from src.emg_cnn import CompactEMGNet
from src.grabmyo import allocations
from src.grabmyo_corpus import WindowCorpus, hash_stream
from scripts.decisive_calibration import PRIORS, measures, schedules, vote
from scripts.pilot_gesture_coverage import validate_roles

ROOT = Path(__file__).resolve().parents[1]
METHODS = {'target_ce': None, 'replay': None, 'source_ce': 0.,
           'coral_0.1': .1, 'coral_1': 1., 'coral_10': 10.}
CHOICES = (0, 4, 8, 12)


def dump(path, value):
    temp = path.with_suffix(path.suffix + '.partial')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def predict(model, x):
    model.eval()
    with torch.no_grad():
        return np.concatenate([model(x[i:i+256]).argmax(1).cpu().numpy()
                               for i in range(0, len(x), 256)])


def fit(base, sx, sy, cx, cy, method, seed, checkpoints):
    """No scoring data accepted by this training function."""
    model = copy.deepcopy(base).train()
    opt = torch.optim.Adam(model.parameters(), lr=.0001)
    trng, srng = np.random.default_rng(seed), np.random.default_rng(seed + 100000)
    classes = sy.cpu().numpy()
    byclass = [np.flatnonzero(classes == c) for c in range(17)]
    if any(len(ix) == 0 for ix in byclass):
        raise ValueError('Source must cover all 17 classes')
    start = time.perf_counter()
    for step in range(1, max(checkpoints) + 1):
        ti = trng.integers(0, len(cx), 64)
        si = np.array([srng.choice(byclass[c]) for c in srng.integers(0, 17, 64)])
        opt.zero_grad()
        source_ce = domain = target_ce = None
        if method in ('target_ce', 'replay'):
            target_ce = F.cross_entropy(model(cx[ti]), cy[ti])
            loss = target_ce
            if method == 'replay':
                source_ce = F.cross_entropy(model(sx[si]), sy[si])
                loss = loss + source_ce
        else:
            loss, source_ce, domain = source_alignment_objective(
                model, sx[si], sy[si], cx[ti], METHODS[method])
        if not torch.isfinite(loss):
            raise FloatingPointError(f'{method}: nonfinite objective at step {step}')
        loss.backward()
        finite_gradients = torch.stack([torch.isfinite(p.grad).all()
                                        for p in model.parameters() if p.grad is not None]).all()
        if not finite_gradients:
            raise FloatingPointError(f'{method}: nonfinite gradient at step {step}')
        opt.step()
        if step in checkpoints:
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            diagnostics = {'loss': float(loss.detach().cpu()),
                           'source_ce': None if source_ce is None else float(source_ce.detach().cpu()),
                           'target_ce': None if target_ce is None else float(target_ce.detach().cpu()),
                           'coral_raw': None if domain is None else float(domain.detach().cpu()),
                           'seconds_including_prior_checkpoint_scoring': time.perf_counter()-start,
                           'update_l2': float(sum((p.detach()-p0.detach()).square().sum()
                                                  for p,p0 in zip(model.parameters(),base.parameters())).sqrt().cpu())}
            yield step, state, diagnostics


def main():
    global METHODS
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--device', choices=['cpu', 'cuda', 'mps'], required=True)
    ap.add_argument('--seed', type=int, choices=[42, 0, 1], default=42)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--weight-grid', choices=['original', 'small'], default='original')
    args = ap.parse_args()
    if args.weight_grid == 'small':
        METHODS = {'source_ce': 0., 'coral_0.0001': .0001,
                   'coral_0.001': .001, 'coral_0.01': .01}
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out/'models').mkdir()
    (out/'code').mkdir()
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if args.device == 'cuda':
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    groups, _ = allocations()
    people = sorted(groups['development'])[:1] if args.smoke else sorted(groups['development'])
    sessions = [2] if args.smoke else [2, 3]
    choices = [0] if args.smoke else list(CHOICES)
    steps = [2] if args.smoke else [25, 100]
    prior = ROOT/'research/runs'/PRIORS[args.seed]
    cache = ROOT/'data/public/grabmyo/cache_20260906_v1'
    codepaths = ['scripts/run_alignment_comparison.py', 'src/alignment_comparator.py',
                 'src/emg_cnn.py', 'src/grabmyo.py', 'src/grabmyo_corpus.py',
                 'scripts/decisive_calibration.py', 'scripts/pilot_gesture_coverage.py']
    inputpaths = [cache/'development_manifest.csv', cache/'development_signals.npy',
                  cache/'development_features.npy', cache/'config.json', cache/'summary.json',
                  prior/'training_scaler.npz', prior/'independent_validation.json']
    inputpaths += [prior/f'models/p{p}_s2_shared_none_b0.pt' for p in people]
    cfg = dict(created_utc=datetime.now(timezone.utc).isoformat(),
               scope='SMOKE ONLY' if args.smoke else 'development feasibility; no final access',
               seed=args.seed,participants=people,sessions=sessions,choices=choices,k=[1,2],
               weight_grid=args.weight_grid,
               steps=steps,methods=METHODS,learning_rate=.0001,batch_size=64,
               initialization='shared pretrained and personally enrolled CompactEMGNet',
               interpretation='published Deep CORAL objective adapted to this backbone/task; not exact Hyser replication',
               optimizer='Adam defaults',source_sampling='uniform class then uniform within-class window',
               target_sampling='uniform with replacement from two permitted whole trials',
               scoring='same 68 whole trials/session;35 windows/trial;lowest-index vote tie',
               preprocessing='existing physical mV cache and representation-training scaler; amplitude preserved',
               recorded_seconds=10,device=str(device),torch=torch.__version__,numpy=np.__version__,
               python=platform.python_version(),platform=platform.platform(),
               device_name=torch.cuda.get_device_name(0) if args.device=='cuda' else args.device,
               final_access=0,code={p:hash_stream(ROOT/p) for p in codepaths},
               inputs={str(p.relative_to(ROOT)):hash_stream(p) for p in inputpaths})
    dump(out/'protocol.json', cfg)
    for p in codepaths:
        dest=out/'code'/p;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((ROOT/p).read_bytes())
    corpus = WindowCorpus(cache, 'development')
    rows = [{**r, **{k:int(r[k]) for k in ['participant','session','class_index','calibration_rank']}}
            for r in corpus.rows]
    assert {r['participant'] for r in rows} == set(groups['development'])
    sc=np.load(prior/'training_scaler.npz')
    records, access, predictions = [], [], {}
    started=time.perf_counter()
    for person in people:
        ids=[i for i,r in enumerate(rows) if r['participant']==person]
        lookup={i:j for j,i in enumerate(ids)}
        xx,yy=corpus.batch(corpus.window_ids(ids),sc['mean'],sc['scale'])
        xx=torch.from_numpy(xx).to(device); yy=torch.as_tensor(yy,dtype=torch.long,device=device)
        def subset(tids):
            wi=np.concatenate([np.arange(lookup[i]*35,(lookup[i]+1)*35) for i in tids])
            return xx[wi], yy[wi]
        source=[i for i in ids if rows[i]['role']=='enrollment']
        sx,sy=subset(source)
        base=CompactEMGNet().to(device)
        base.load_state_dict(torch.load(prior/f'models/p{person}_s2_shared_none_b0.pt',map_location='cpu',weights_only=True))
        base.eval()
        torch.save({k:v.cpu() for k,v in base.state_dict().items()},out/f'models/p{person}_source.pt')
        for session in sessions:
            score=[i for i in ids if rows[i]['session']==session and rows[i]['role']=='scoring']
            truth=np.array([rows[i]['class_index'] for i in score])
            assert len(score)==68 and np.all(np.bincount(truth,minlength=17)==4)
            tx,_=subset(score)
            frozen_windows=predict(base,tx)
            for case in schedules(rows,person,session):
                if case['choice'] not in choices:continue
                cal=case['ids'];validate_roles(rows,source,cal,score)
                assert len(set(cal))==2 and len({rows[i]['class_index'] for i in cal})==case['k']
                name=f"p{person}_s{session}_o{case['choice']}_k{case['k']}"
                access.append(dict(name=name,participant=person,session=session,choice=case['choice'],
                                   k=case['k'],gestures=case['gestures'],source=source,calibration=cal,scoring=score))
                dump(out/'access.json',access)
                cx,cy=subset(cal)
                def save(method,step,windows,diagnostics=None):
                    key=f'{name}_{method}_t{step}'
                    predictions[key]=windows.reshape(68,35)
                    records.append(dict(name=key,participant=person,session=session,choice=case['choice'],
                                        k=case['k'],gestures=case['gestures'],method=method,steps=step,
                                        **measures(vote(windows),truth,case['gestures']),diagnostics=diagnostics))
                save('frozen',0,frozen_windows)
                for method in METHODS:
                    evaluator=copy.deepcopy(base)
                    for step,state,diagnostics in fit(base,sx,sy,cx,cy,method,args.seed+case['choice'],steps):
                        key=f'{name}_{method}_t{step}'
                        torch.save(state,out/f'models/{key}.pt')
                        evaluator.load_state_dict(state)
                        save(method,step,predict(evaluator,tx),diagnostics)
                    del evaluator
                dump(out/'results.json',dict(config=cfg,records=records))
                np.savez_compressed(out/'window_predictions.npz',**predictions)
                print(json.dumps(dict(case=name,cases_done=len(access),records=len(records),
                                      seconds=time.perf_counter()-started)),flush=True)
        del xx,yy,sx,sy,tx,cx,cy,base
        if args.device=='mps':torch.mps.empty_cache()
        if args.device=='cuda':torch.cuda.empty_cache()
    expected=len(people)*len(sessions)*len(choices)*2
    assert len(access)==expected and len(records)==expected*(1+len(METHODS)*len(steps))
    dump(out/'complete.json',dict(cases=len(access),records=len(records),seconds=time.perf_counter()-started,
                                 finished_utc=datetime.now(timezone.utc).isoformat()))


if __name__=='__main__':
    main()
