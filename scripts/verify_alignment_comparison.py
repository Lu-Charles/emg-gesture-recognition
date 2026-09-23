"""Independently reconstruct metrics and replay saved alignment checkpoints."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import torch
from src.emg_cnn import CompactEMGNet
from src.grabmyo_corpus import WindowCorpus, hash_stream

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--device', default='cpu', choices=['cpu', 'cuda', 'mps'])
    ap.add_argument('--raw', default='all', choices=['all', 'sample', 'none'])
    args = ap.parse_args()
    run = args.run.resolve()
    cfg = json.loads((run/'protocol.json').read_text())
    complete = json.loads((run/'complete.json').read_text())
    records = json.loads((run/'results.json').read_text())['records']
    access = json.loads((run/'access.json').read_text())
    pred = np.load(run/'window_predictions.npz')
    for p, digest in cfg['inputs'].items():
        assert hash_stream(ROOT/p) == digest, p
    for p, digest in cfg['code'].items():
        assert hash_stream(run/'code'/p) == digest, p
    assert set(cfg['participants']) <= {3,5,7,20,21,22,24,38}
    expected = len(cfg['participants'])*len(cfg['sessions'])*len(cfg['choices'])*2
    assert len(access) == complete['cases'] == expected
    assert len(records) == complete['records'] == expected*(1+len(cfg['methods'])*len(cfg['steps']))
    assert len({r['name'] for r in records}) == len(records)
    assert set(pred.files) == {r['name'] for r in records}
    cache = ROOT/'data/public/grabmyo/cache_20260906_v1'
    rows = list(csv.DictReader((cache/'development_manifest.csv').open()))
    for row in rows:
        for k in ['participant','session','class_index','calibration_rank']:
            row[k] = int(row[k])
    scoring, cases = {}, {}
    for a in access:
        cases[a['name']] = a
        s,c,t = [set(a[k]) for k in ['source','calibration','scoring']]
        assert not (s&c or s&t or c&t)
        for role, indices in [('enrollment',s),('calibration',c),('scoring',t)]:
            assert indices and all(rows[i]['role']==role and rows[i]['group']=='development'
                                   and rows[i]['participant']==a['participant'] for i in indices)
        assert {rows[i]['session'] for i in s} == {1}
        assert {rows[i]['session'] for i in c|t} == {a['session']}
        assert len(c)==2 and sorted(rows[i]['calibration_rank'] for i in c)==[1,2]
        assert {rows[i]['class_index'] for i in c} == set(a['gestures'])
        assert len(set(a['gestures'])) == a['k']
        key = (a['participant'],a['session'])
        if key in scoring: assert scoring[key] == a['scoring']
        scoring[key] = a['scoring']
    for a in access:
        other = cases[a['name'].rsplit('_k',1)[0]+'_k'+str(3-a['k'])]
        assert a['calibration'][0] == other['calibration'][0]
        names={r['name'] for r in records if r['name'].startswith(a['name']+'_')}
        expected_names={a['name']+'_frozen_t0'}|{f"{a['name']}_{m}_t{s}" for m in cfg['methods'] for s in cfg['steps']}
        assert names==expected_names
    for person in cfg['participants']:
        prior=next(p for p in cfg['inputs'] if p.endswith(f'/p{person}_s2_shared_none_b0.pt'))
        original=torch.load(ROOT/prior,map_location='cpu',weights_only=True)
        saved=torch.load(run/f'models/p{person}_source.pt',map_location='cpu',weights_only=True)
        assert original.keys()==saved.keys()
        assert all(torch.equal(original[k],saved[k]) for k in original)
    raw_count = 0
    torch.set_num_threads(1)
    if args.device == 'cuda':
        torch.backends.cudnn.benchmark=False
        torch.backends.cudnn.deterministic=True
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
    corpus = WindowCorpus(cache,'development') if args.raw!='none' else None
    scaler_path = next(p for p in cfg['inputs'] if p.endswith('training_scaler.npz'))
    scaler = np.load(ROOT/scaler_path)
    model = CompactEMGNet().to(args.device).eval()
    for key,tids in scoring.items():
        truth = np.array([rows[i]['class_index'] for i in tids])
        assert len(tids)==68 and np.all(np.bincount(truth,minlength=17)==4)
        if args.raw!='none':
            xx,_ = corpus.batch(corpus.window_ids(tids),scaler['mean'],scaler['scale'])
            xx = torch.from_numpy(xx).to(args.device)
        subset = [r for r in records if (r['participant'],r['session'])==key]
        for r in subset:
            a = cases[r['name'].split('_'+r['method']+'_t')[0]]
            w = pred[r['name']]
            assert w.shape==(68,35) and np.all((w>=0)&(w<17))
            votes=np.array([max(range(17),key=lambda g:np.count_nonzero(v==g)) for v in w])
            cm=np.array([[sum((truth==i)&(votes==j)) for j in range(17)] for i in range(17)])
            assert np.array_equal(cm,r['confusion'])
            selected=np.isin(truth,a['gestures']); omitted=(truth<16)&~selected
            denom=cm.sum(0)+cm.sum(1)
            metrics=dict(accuracy=np.mean(votes==truth),calibrated_recall=np.mean(votes[selected]==truth[selected]),
                         other_active_accuracy=np.mean(votes[omitted]==truth[omitted]),
                         other_into_calibration=np.mean(np.isin(votes[omitted],a['gestures'])),
                         rest_recall=np.mean(votes[truth==16]==16),
                         macro_f1=np.mean(np.divide(2*cm.diagonal(),denom,out=np.zeros(17),where=denom!=0)))
            for metric,value in metrics.items(): assert abs(value-r[metric])<1e-12,(r['name'],metric)
            replay = args.raw=='all' or (args.raw=='sample' and r['choice']==cfg['choices'][0] and r['k']==1)
            if replay:
                path=run/'models'/(f"p{key[0]}_source.pt" if r['method']=='frozen' else r['name']+'.pt')
                state=torch.load(path,map_location='cpu',weights_only=True)
                assert all(torch.isfinite(v).all() for v in state.values())
                model.load_state_dict(state)
                with torch.inference_mode():
                    labels=np.concatenate([model(xx[i:i+256]).argmax(1).cpu().numpy() for i in range(0,len(xx),256)])
                assert np.array_equal(labels.reshape(68,35),w),(r['name'],'checkpoint mismatch')
                raw_count+=1
        print(json.dumps({'participant':key[0],'session':key[1],'verified_records':len(subset)}),flush=True)
    report=dict(passed=True,records=len(records),cases=len(access),trial_votes=len(records)*68,
                stored_window_predictions=len(records)*68*35,raw_replayed_checkpoints=raw_count,
                raw_replayed_windows=raw_count*68*35,device=args.device,raw_mode=args.raw,
                final_access=0,verifier_sha256=hash_stream(Path(__file__)))
    (run/f'verification_{args.device}_{args.raw}.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__': main()
