"""Training-fit diagnostic, added after PL outcomes; does not select checkpoints."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from src.grabmyo_corpus import WindowCorpus,hash_stream
from src.open_set_neural import OpenSetNet,infer


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);ap.add_argument('--families',nargs='+',required=True);args=ap.parse_args();run=args.run
    start=time.perf_counter();cfg=json.loads((run/'config.json').read_text());corpus=WindowCorpus(Path(cfg['cache']),'development');norm=np.load(run/'normalization.npz');torch.set_num_threads(4);results=[]
    for family in args.families:
        access=json.loads((run/family/'model_access.json').read_text())
        for a in access:
            if a['target_ids']:continue
            model=OpenSetNet(family,42);model.load_state_dict(torch.load(run/family/'models'/(a['model']+'.pt'),map_location='cpu',weights_only=True))
            x,y=corpus.batch(corpus.window_ids(a['source_ids']),norm['mean'],norm['scale']);pred,scores=infer(model,x,'cpu')
            confusion=np.zeros((7,7),dtype=int);np.add.at(confusion,(y-10,pred-10),1)
            r=dict(family=family,participant=a['participant'],model=a['model'],scope='in-sample known enrollment trials, not generalization',windows=len(y),accuracy=float(np.mean(y==pred)),per_class_recall=(confusion.diagonal()/confusion.sum(1)).tolist(),confusion=confusion.tolist())
            if family!='cnn':
                r['prototype_pair_distances']=[]
                for branch in model.branches:
                    p=branch.prototypes.detach().numpy();dist=np.linalg.norm(p[:,None]-p[None],axis=2);np.fill_diagonal(dist,np.inf);r['prototype_pair_distances'].append(float(dist.min()))
            results.append(r)
        print(family,[(r['participant'],round(r['accuracy'],4),np.round(r['per_class_recall'],3).tolist()) for r in results if r['family']==family],flush=True)
    result=dict(scope='post-hoc fit/convergence diagnostic; no setting/checkpoint changes',seconds=time.perf_counter()-start,results=results,source_sha256=hash_stream(__file__))
    (run/('training_diagnostics_'+'_'.join(args.families)+'.json')).write_text(json.dumps(result,indent=2)+'\n')
    (run/'code_snapshot'/Path(__file__).name).write_bytes(Path(__file__).read_bytes())


if __name__=='__main__':main()
