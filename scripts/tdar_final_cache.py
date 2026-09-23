"""Exact TDAR feature extraction for the frozen final cohort, no model scoring."""
import json,csv
from pathlib import Path
import numpy as np
from src.open_set_emg import tdar_rms
from src.grabmyo_corpus import hash_stream
from scripts.cache_final_coverage import frozen
R=Path(__file__).resolve().parents[1];C=R/'data/public/grabmyo/cache_final_20260916';O=R/'research/runs/20260916_tdar_final'
def main():
 cfg=frozen();O.mkdir(exist_ok=False);rows=list(csv.DictReader((C/'final_manifest.csv').open()));assert sorted({int(r['participant']) for r in rows})==cfg['grabmyo']['participants'];x=np.load(C/'final_signals.npy',mmap_mode='r');out=np.lib.format.open_memmap(O/'features.npy',mode='w+',dtype='float64',shape=(len(rows),35,144));starts=1024+np.arange(35)*256
 (O/'protocol.json').write_text(json.dumps(dict(protocol_sha256=hash_stream(R/'research/runs/20260625_confirmatory_protocol/protocol.json'),code_sha256=hash_stream(__file__),features='existing librosa Burg TDAR_RMS, no changes'),indent=2)+'\n')
 for i in range(len(rows)):
  w=np.stack([x[i,:,s:s+512] for s in starts]).astype(np.float64);out[i]=tdar_rms(w)
  if (i+1)%357==0:print(i+1,flush=True)
 out.flush();assert np.isfinite(out).all();(O/'complete.json').write_text(json.dumps(dict(trials=len(rows),sha256=hash_stream(O/'features.npy')))+'\n')
if __name__=='__main__':main()
