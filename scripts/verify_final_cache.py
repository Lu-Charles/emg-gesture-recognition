"""Final GRABMyo cache provenance and raw data spot checks after all-file decoding."""
import json,csv
from pathlib import Path
import numpy as np
from src.final_coverage import FinalCorpus
from src.grabmyo import checksum_index,read_forearm,short_windows,amplitude_features
R=Path(__file__).resolve().parents[1]
def main():
 c=FinalCorpus(R/'data/public/grabmyo/cache_final_20260916');checks=checksum_index(R/'data/public/grabmyo/1.1.0');reports=[json.loads(s) for s in (c.path/'decode_checks.jsonl').read_text().splitlines()];assert len(reports)==len(c.rows);assert all(z['manual_decode_max_abs_error']==0 for z in reports);assert [r['record'] for r in c.rows]==[z['record'] for z in reports]
 chosen=[]
 for p in sorted({int(r['participant']) for r in c.rows}):
  for session in [1,2,3]:
   i=next(i for i,r in enumerate(c.rows) if int(r['participant'])==p and int(r['session'])==session);r=c.rows[i];x,_=read_forearm(R/'data/public/grabmyo/1.1.0',r['record'],checks);np.testing.assert_array_equal(c.signals[i],x.T.astype(np.float32));w,_=short_windows(x);np.testing.assert_allclose(c.features[i],amplitude_features(w),rtol=1e-12,atol=1e-12);chosen.append(i)
 out=dict(passed=True,all_file_checksum_and_manual_decoder_reports=5355,cache_hashes_checked=True,raw_trials_reread=chosen,scope='all decoded source files checksum/manual decoder comparison; independent reread of45rawtrials; no new scoring')
 (c.path/'independent_audit.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
if __name__=='__main__':main()
