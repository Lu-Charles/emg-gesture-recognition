"""Check reported means and oracle cutoffs against original saved predictions."""
import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
import numpy as np
from src.grabmyo_corpus import hash_stream


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);r=parser.parse_args().run
    rows=json.loads((r/'metrics.json').read_text());groups=defaultdict(list);lookup={}
    for x in rows:
        if x['retention']!=.95:continue
        groups[(x['method'],x['budget'],x['seconds'],x['fit_strategy'],x['rejector'])].append(x)
        lookup[(x['participant'],x['session'],x['order'],x['method'],x['fit_strategy'],x['rejector'],x['budget'])]=x
    with (r/'aggregate.csv').open() as f:table=list(csv.DictReader(f))
    for x in table:
        raw=groups[(x['method'],int(x['budget']),int(x['seconds']),x['fit_strategy'],x['rejector'])]
        assert len(raw)==96
        for metric in ['known_correct_acceptance','unknown_false_acceptance','closed_known_accuracy']:
            assert abs(float(x[metric])-math.fsum(z[metric] for z in raw)/96)<1e-12
    assert len(groups)==len(table)==28
    with (r/'posthoc_oracle_thresholds.csv').open() as f:oracle=list(csv.DictReader(f))
    checked=0
    for o in oracle:
        m=lookup[(int(o['participant']),int(o['session']),o['order'],o['method'],o['fit_strategy'],o['rejector'],3)]
        z=np.load(r/'predictions'/m['score_file']);known=(z['y']>=10)&(z['y']<16)
        correct_scores=z[o['rejector']][known&(z['pred']==z['y'])]
        needed=math.ceil(.8*known.sum())
        if len(correct_scores)<needed:
            assert o['feasible']=='False' and o['oracle_unknown_acceptance_at80']==''
        else:
            assert o['feasible']=='True'
            # Largest cutoff retaining the required number of correct predictions.
            cutoff=np.sort(correct_scores)[-needed]
            unknown=z['y']<10
            rate=np.sum(unknown&(z['pred']!=16)&(z[o['rejector']]>=cutoff))/unknown.sum()
            assert abs(rate-float(o['oracle_unknown_acceptance_at80']))<1e-12
        checked+=1
    target=[o for o in oracle if o['method']=='model_only' and o['fit_strategy']=='target_only' and o['rejector']=='probability']
    assert len(target)==96 and all(o['feasible']=='True' for o in target)
    mean=math.fsum(float(o['oracle_unknown_acceptance_at80']) for o in target)/96
    report=(r/'report.md').read_text();assert f'{100*mean:.2f}%' in report
    result=dict(passed=True,aggregate_groups=len(table),primary_rows=2688,oracle_cases=checked,
                oracle_target_cases=96,oracle_target_mean=mean,code_sha256=hash_stream(__file__))
    (r/'report_validation.json').write_text(json.dumps(result,indent=2)+'\n')
    for name in ['report_open_set_classical.py','verify_open_set_report.py']:
        (r/'code_snapshot'/name).write_bytes((Path('scripts')/name).read_bytes())
    files=[*r.glob('*.csv'),r/'report.md',r/'validation.json',r/'report_validation.json',
           *[r/'code_snapshot'/n for n in ['report_open_set_classical.py','verify_open_set_report.py','verify_open_set_classical.py']]]
    (r/'report_artifact_sha256.json').write_text(json.dumps({str(p.relative_to(r)):hash_stream(p) for p in files},indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
