"""Read-back verification of report aggregates and oracle claims."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from src.grabmyo_corpus import hash_stream


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);run=ap.parse_args().run
    keys=['family','method','budget','seconds','update_strategy','rejector'];fields=['known_correct_acceptance','unknown_false_acceptance','known_wrong_command','known_rejection','known_predicted_rest','rest_false_activation','closed_known_accuracy']
    groups=defaultdict(lambda:defaultdict(list));lookup={};files={}
    for family in ['cnn','pl','predin']:
        for r in json.loads((run/family/'metrics.json').read_text()):
            if r['retention']!=.95:continue
            key=tuple(str(r[k]) for k in keys);groups[key][r['participant']].append([r[k] for k in fields])
            lookup[(family,str(r['participant']),str(r['session']),r['method'],str(r['budget']),r['order'],r['update_strategy'],r['rejector'])]=r
        for r in json.loads((run/family/'source_metrics.json').read_text()):
            if r['retention']==.95:lookup[(family,str(r['participant']),'1','source','0','source','source',r['rejector'])]=r
    aggregate=list(csv.DictReader((run/'aggregate.csv').open()))
    assert len(aggregate)==len(groups)
    for r in aggregate:
        key=tuple(r[k] for k in keys);person=np.asarray([np.mean(v,axis=0) for v in groups[key].values()]);assert person.shape==(8,7)
        np.testing.assert_allclose([float(r[k]) for k in fields],person.mean(0),atol=1e-12)
        assert int(r['people_with_residual'])==int(np.sum((person[:,0]>=.8)&(person[:,1]>=.1)))
    oracle=list(csv.DictReader((run/'oracle_diagnostics.csv').open()))
    for r in oracle:
        key=tuple(r[k] for k in ['family','participant','session','method','budget','order','update_strategy','rejector']);original=lookup[key]
        file=run/r['family']/'predictions'/original['score_file']
        if file not in files:
            with np.load(file) as z:files[file]={k:z[k] for k in z.files}
        z=files[file];known=(z['y']>=10)&(z['y']<16);correct=known&(z['pred']==z['y']);score=z[r['rejector']]
        # Independently find first descending tied-score threshold that reaches
        # requested correct count, using cumulative event counts.
        order=np.argsort(-score,kind='stable');ends=np.r_[np.flatnonzero(np.diff(score[order])!=0),len(score)-1]
        c=np.cumsum(correct[order])[ends]/known.sum();valid=np.flatnonzero(c>=.8)
        assert (r['feasible']=='True')==bool(len(valid))
        if len(valid):
            t=score[order][ends[valid[0]]];assert float(r['oracle_threshold'])==t
            u=np.mean((score[z['y']<10]>=t)&(z['pred'][z['y']<10]!=16))
            assert abs(float(r['oracle_unknown_acceptance_at80'])-u)<1e-12
        else:assert r['oracle_threshold']=='' and r['oracle_unknown_acceptance_at80']==''
    for f,digest in json.loads((run/'report_artifact_sha256.json').read_text()).items():assert hash_stream(run/f)==digest
    result=dict(passed=True,aggregate_groups=len(aggregate),oracle_cases=len(oracle),audit_source_sha256=hash_stream(__file__))
    (run/'report_validation.json').write_text(json.dumps(result,indent=2)+'\n');(run/'code_snapshot'/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    print(json.dumps(result))


if __name__=='__main__':main()
