"""Explicit final-cohort cache; requires a hash-verified evaluation protocol."""
import argparse,csv,json,itertools,subprocess,time,hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import numpy as np
from src.grabmyo_corpus import hash_stream,fetch_s3
from src.grabmyo import allocations,checksum_index,read_forearm,short_windows,amplitude_features
R=Path(__file__).resolve().parents[1];P=R/'research/runs/20260625_confirmatory_protocol'
def frozen():
 assert hash_stream(P/'protocol.json')==(P/'protocol.sha256').read_text().strip();return json.loads((P/'protocol.json').read_text())
def main():
 ap=argparse.ArgumentParser();ap.add_argument('dataset',choices=['grabmyo','senic']);args=ap.parse_args();cfg=frozen();O=R/f'data/public/{args.dataset}/cache_final_20260916';O.mkdir(parents=True,exist_ok=False);start=time.perf_counter()
 (O/'authorization.json').write_text(json.dumps(dict(protocol_sha256=hash_stream(P/'protocol.json'),code_sha256=hash_stream(__file__),dataset=args.dataset),indent=2)+'\n')
 if args.dataset=='grabmyo':
  root=R/'data/public/grabmyo/1.1.0';checks=checksum_index(root);rows=[r for r in allocations()[1] if r['group']=='final'];assert sorted({r['participant'] for r in rows})==cfg['grabmyo']['participants']
  files=[r['record']+s for r in rows for s in ['.hea','.dat']]
  with (O/'retrieval.jsonl').open('w') as log,ThreadPoolExecutor(max_workers=12) as ex:
   futures=[ex.submit(fetch_s3,root,f,checks) for f in files]
   for j,f in enumerate(as_completed(futures)):
    item=f.result();log.write(json.dumps(item)+'\n')
    if (j+1)%1000==0:print('download',j+1,len(files),flush=True);log.flush()
  with (O/'final_manifest.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
  sig=np.lib.format.open_memmap(O/'final_signals.npy',mode='w+',dtype='float32',shape=(len(rows),16,10240));features=np.lib.format.open_memmap(O/'final_features.npy',mode='w+',dtype='float64',shape=(len(rows),35,48))
  with (O/'decode_checks.jsonl').open('w') as log:
   for i,r in enumerate(rows):
    x,info=read_forearm(root,r['record'],checks);sig[i]=x.T.astype(np.float32);windows,_=short_windows(x);features[i]=amplitude_features(windows);log.write(json.dumps(dict(record=r['record'],**info))+'\n')
    if (i+1)%357==0:print('decoded',i+1,flush=True)
  sig.flush();features.flush();summary=dict(trials=len(rows),participants=cfg['grabmyo']['participants'],hashes={f:hash_stream(O/f) for f in ['final_manifest.csv','final_signals.npy','final_features.npy']})
 else:
  from src.senic import parse_trial,trial_windows
  from src.open_set_emg import tdar_rms
  root=R/'data/public/SeNic';exroot=root/'extracted_final_20260916';exroot.mkdir(exist_ok=False);rows=[]
  # Audit completeness/durations for every reserved person before any fitting.
  for p in cfg['senic']['participants']:
   archive=root/f'h{p}.rar'
   if not archive.exists():
    candidates=list((root/f'h{p}').glob('*.rar'));assert len(candidates)==1;candidates[0];archive=candidates[0]
   ledger=json.loads((R/'research/runs/20260907_sensor_shift_audit/download_tree_verification.json').read_text());entry=next(z for z in ledger['checked_files'] if z['path']==str(archive.relative_to(root)));blob=archive.read_bytes();assert hashlib.sha1(b'blob '+str(len(blob)).encode()+b'\x00'+blob).hexdigest()==entry['git_blob_sha1']
   listing=subprocess.check_output(['/usr/bin/tar','-tf',str(archive)],text=True).splitlines();nested=not any(v.startswith(f'h{p}/0/') for v in listing);prefix='0/' if nested else f'h{p}/0/';members=[v for v in listing if v.startswith(prefix) and v.endswith('.csv')];assert len(members)==231,(p,len(members))
   destination=exroot/f'h{p}' if nested else exroot;destination.mkdir(exist_ok=True)
   subprocess.run(['/usr/bin/tar','-xf',str(archive),'-C',str(destination),*members],check=True)
   rr=[parse_trial(v) for v in sorted((exroot/f'h{p}/0').glob('*.csv'))];assert len(rr)==231 and {(r['position'],r['repetition'],r['label']) for r in rr}==set(itertools.product(range(11),range(3),range(7)));rows.extend(rr)
  amp=[];tdar=[]
  for i,r in enumerate(rows):
   x=np.loadtxt(r['path'],delimiter=',',dtype=np.float64);w,_=trial_windows(x);assert np.isfinite(x).all();r.update(id=i,samples=len(x),recorded_seconds=len(x)/200,sha256=hash_stream(r['path']),windows=15,analysis_start=600,analysis_stop=1000)
   amp.append(amplitude_features(w));tdar.append(tdar_rms(w.transpose(0,2,1)))
   if (i+1)%231==0:print('features',r['subject'],i+1,flush=True)
  (O/'trial_manifest.json').write_text(json.dumps(rows,indent=2)+'\n');np.save(O/'amplitude_features.npy',np.array(amp));np.save(O/'tdar_rms_features.npy',np.array(tdar));assert np.isfinite(tdar).all()
  summary=dict(trials=len(rows),participants=cfg['senic']['participants'],hashes={f:hash_stream(O/f) for f in ['trial_manifest.json','amplitude_features.npy','tdar_rms_features.npy']})
 summary.update(seconds=time.perf_counter()-start,protocol_sha256=hash_stream(P/'protocol.json'));(O/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(summary)
if __name__=='__main__':main()
