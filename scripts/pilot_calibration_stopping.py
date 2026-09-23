"""Extract causal calibration observations from verified saved checkpoints.

The saved procedure refits from enrollment for each accumulated budget. Reuse
those checkpoints exactly; no new encoder fitting or scoring-signal reads.
"""
import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
from src.emg_cnn import CompactEMGNet
from src.calibration_stopping import FEATURES, observed_features
from src.grabmyo_corpus import WindowCorpus, hash_stream
from scripts.train_shared_emg import roles


def probabilities(model, x, device):
    model.eval()
    with torch.no_grad():
        result = np.concatenate([torch.softmax(model(torch.from_numpy(x[i:i+128]).to(device)), 1).cpu().numpy()
                                 for i in range(0, len(x), 128)])
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite model output')
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prior', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    start = time.perf_counter()
    args.out.mkdir(parents=True, exist_ok=False)
    cfg = json.loads((args.prior/'config.json').read_text())
    if not (args.prior/'independent_validation.json').exists():
        raise ValueError('Prior run must have independent validation')
    for relative, expected in json.loads((args.prior/'artifact_sha256.json').read_text()).items():
        if hash_stream(args.prior/relative) != expected:
            raise ValueError('Changed prior artifact: ' + relative)
    cache = Path(cfg['cache'])
    summary = json.loads((cache/'summary.json').read_text())
    for kind in ('signals', 'features', 'manifest'):
        filename = f'development_{kind}.' + ('csv' if kind == 'manifest' else 'npy')
        if hash_stream(cache/filename) != summary['groups']['development'][kind+'_sha256']:
            raise ValueError('Changed development corpus')
    if hash_stream('src/emg_cnn.py') != cfg['code_hashes']['src/emg_cnn.py']:
        raise ValueError('Encoder definition changed')
    dev = WindowCorpus(cache, 'development')
    participants = sorted({int(r['participant']) for r in dev.rows})
    if participants != cfg['development_participants']:
        raise ValueError('Participant allocation mismatch')
    scaler = np.load(args.prior/'training_scaler.npz')
    mean, scale = scaler['mean'], scaler['scale']
    access = {r['name']: r for r in json.loads((args.prior/'data_access.json').read_text())}
    config = {'created_utc': datetime.now(timezone.utc).isoformat(), 'prior': str(args.prior.resolve()),
              'cache': str(cache.resolve()), 'participants': participants, 'sessions': [2,3],
              'features': FEATURES, 'seed': cfg['seed'], 'round_seconds': 85,
              'protocol': 'Saved shared full-update checkpoints; reset to enrolled model for each cumulative budget, 10 epochs Adam .0001 with the prior run seed. Fixed original calibration ranks.',
              'feature_observations': 'Only newly acquired 17-trial round, predicted before and after its update; post-update values are in-sample diagnostics.',
              'hashes': {str(args.prior/f): hash_stream(args.prior/f) for f in ('config.json','metrics.json','data_access.json','artifact_sha256.json','independent_validation.json','training_scaler.npz')},
              'code_hashes': {f: hash_stream(f) for f in ('src/calibration_stopping.py','scripts/pilot_calibration_stopping.py','src/emg_cnn.py','src/grabmyo_corpus.py')},
              'prior_cache_summary_sha256': cfg['cache_summary_sha256'], 'device': 'mps' if torch.backends.mps.is_available() else 'cpu'}
    if hash_stream(cache/'summary.json') != config['prior_cache_summary_sha256']:
        raise ValueError('Cache summary changed')
    (args.out/'config.json').write_text(json.dumps(config, indent=2)+'\n')
    snapshot = args.out/'code_snapshot'; snapshot.mkdir()
    for f in config['code_hashes']:
        (snapshot/Path(f).name).write_bytes(Path(f).read_bytes())
    torch.set_num_threads(4)
    rows, ledger, arrays = [], [], {}
    extraction_start = time.perf_counter()
    for p in participants:
        for session in (2,3):
            for budget in (1,2):
                ids = [i for i in roles(dev.rows,p,session,'calibration') if int(dev.rows[i]['calibration_rank']) == budget]
                records = [dev.rows[i]['record'] for i in ids]
                cumulative = [dev.rows[i]['record'] for i in roles(dev.rows,p,session,'calibration',budget)]
                score = {dev.rows[i]['record'] for i in roles(dev.rows,p,session,'scoring')}
                if len(ids) != 17 or set(cumulative) & score:
                    raise ValueError('Invalid whole-trial calibration allocation')
                x,y = dev.batch(dev.window_ids(ids),mean,scale)
                pred = []
                names = []
                for b in (budget-1,budget):
                    name = f'p{p}_s{session}_shared_' + ('none_b0' if b == 0 else f'full_b{b}')
                    names.append(name)
                    allowed = {dev.rows[i]['record'] for i in roles(dev.rows,p,session,'calibration',b)} if b else set()
                    entry = access[name]
                    enrollment = {dev.rows[i]['record'] for i in roles(dev.rows,p,1,'enrollment')}
                    if set(entry['calibration']) != allowed or set(entry['source']) != enrollment or set(entry['score']) != score:
                        raise ValueError('Checkpoint access mismatch')
                    if b < budget and set(records) & (set(entry['calibration']) | set(entry['source'])):
                        raise ValueError('Pre-update checkpoint has seen new round')
                    model = CompactEMGNet().to(config['device'])
                    model.load_state_dict(torch.load(args.prior/'models'/f'{name}.pt',map_location=config['device'],weights_only=True))
                    pred.append(probabilities(model,x,config['device']))
                key=f'p{p}_s{session}_b{budget}'
                arrays[key+'_before'], arrays[key+'_after'], arrays[key+'_labels'] = *pred, y
                rows.append({'participant':p,'session':session,**observed_features(*pred,y,budget)})
                ledger.append({'key':key,'participant':p,'session':session,'budget':budget,'observation_records':records,
                               'acquired_records':cumulative,'checkpoint_names':names,'scoring_records_excluded':sorted(score)})
            print(json.dumps({'phase':'features','participant':p,'session':session,'states':len(rows)}),flush=True)
    metrics=json.loads((args.prior/'metrics.json').read_text())
    outcomes=[r for r in metrics if r['initialization']=='shared' and r['mode']=='full']
    assert len(rows)==32 and len(outcomes)==48
    for name,value in (('features',rows),('access',ledger),('outcomes',outcomes)):
        (args.out/f'{name}.json').write_text(json.dumps(value,indent=2)+'\n')
    np.savez_compressed(args.out/'observation_probabilities.npz',**arrays)
    result={'completed_utc':datetime.now(timezone.utc).isoformat(),'total_seconds':time.perf_counter()-start,
            'extraction_seconds':time.perf_counter()-extraction_start,'decision_states':32,'participants':8,
            'new_encoder_training':False,'final_participants_accessed':0,'scoring_signal_reads':0}
    (args.out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    (args.out/'artifact_sha256.json').write_text(json.dumps({str(p.relative_to(args.out)):hash_stream(p) for p in args.out.rglob('*') if p.is_file()},indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    main()
