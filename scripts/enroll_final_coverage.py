"""Enroll all reserved participants using only day1 under the frozen protocol."""
import json,copy,shutil,time
from pathlib import Path
import numpy as np
import torch
from src.final_coverage import FinalCorpus
from src.emg_cnn import CompactEMGNet
from scripts.decisive_calibration import PRIORS
from scripts.pilot_emg_cnn import fit
from scripts.cache_final_coverage import frozen
from src.grabmyo_corpus import hash_stream
R=Path(__file__).resolve().parents[1]
def main():
 cfg=frozen();c=FinalCorpus(R/'data/public/grabmyo/cache_final_20260916');torch.set_num_threads(1);device='mps';start=time.perf_counter()
 for seed in cfg['grabmyo']['seeds']:
  out=R/f'research/runs/20260916_final_enrollment_seed{seed}';out.mkdir(exist_ok=False);(out/'models').mkdir();prior=R/'research/runs'/PRIORS[seed];sc=np.load(prior/'training_scaler.npz');shutil.copy2(prior/'training_scaler.npz',out/'training_scaler.npz');shared=CompactEMGNet().to(device);shared.load_state_dict(torch.load(prior/'pretrain_epoch20.pt',weights_only=True,map_location='cpu'))
  config=dict(protocol_sha256=hash_stream(R/'research/runs/20260625_confirmatory_protocol/protocol.json'),pretrain_checkpoint_sha256=hash_stream(prior/'pretrain_epoch20.pt'),code_sha256=hash_stream(__file__),seed=seed,scope='source enrollment only, target signals not used by this script',participants=cfg['grabmyo']['participants']);(out/'protocol.json').write_text(json.dumps(config,indent=2)+'\n');(out/'enroll_final_coverage.py').write_bytes(Path(__file__).read_bytes());history=[]
  for p in cfg['grabmyo']['participants']:
   ids=[i for i,r in enumerate(c.rows) if int(r['participant'])==p and r['role']=='enrollment'];assert len(ids)==119 and all(int(c.rows[i]['session'])==1 for i in ids)
   xx,yy=c.batch(c.window_ids(ids),sc['mean'],sc['scale']);torch.manual_seed(seed);m=copy.deepcopy(shared);m.set_update_mode('full');h=fit(m,xx,yy,20,.001,seed,device)
   state={k:z.cpu() for k,z in m.state_dict().items()};assert all(torch.isfinite(z).all() for z in state.values());file=out/f'models/p{p}_s2_shared_none_b0.pt';torch.save(state,file);history.append(dict(participant=p,source=ids,checkpoint_sha256=hash_stream(file),**h));(out/'history.json').write_text(json.dumps(history,indent=2)+'\n');print(seed,p,'enrolled',time.perf_counter()-start,flush=True)
  (out/'enrollment_complete.json').write_text(json.dumps(dict(people=len(history),seed=seed,seconds=time.perf_counter()-start))+'\n')
if __name__=='__main__':main()
