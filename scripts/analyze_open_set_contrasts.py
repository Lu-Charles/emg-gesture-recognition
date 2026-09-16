"""Post-hoc paired development contrasts, not a trained rule or confirmatory test."""
import argparse
import json
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import pandas as pd
from src.grabmyo_corpus import hash_stream


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);run=ap.parse_args().run
    df=pd.concat([pd.DataFrame(json.loads((run/f/'metrics.json').read_text())) for f in ['cnn','pl','predin']])
    df=df[(df.retention==.95)&(df.budget==3)]
    comparisons=[('predin_vs_cnn',('predin','both','prototype'),('cnn','both','probability')),('predin_joint_vs_model_only_prototype',('predin','both','prototype'),('predin','model_only','prototype')),('predin_joint_vs_model_only_probability',('predin','both','probability'),('predin','model_only','probability'))]
    fields=['known_correct_acceptance','unknown_false_acceptance'];rng=np.random.default_rng(20260906);draw=rng.integers(0,8,size=(10000,8));records=[];summary=[]
    for strategy in ['pooled_replay','target_finetune']:
        for session in ['both',2,3]:
            q=df[df.update_strategy.eq(strategy)]
            if session!='both':q=q[q.session.eq(session)]
            per=q.groupby(['family','method','rejector','participant'])[fields].mean()
            for name,a,b in comparisons:
                difference=(per.loc[a]-per.loc[b]).sort_index();assert difference.shape==(8,2)
                lo,hi=np.quantile(difference.to_numpy()[draw].mean(1),[.025,.975],axis=0)
                for p,r in difference.iterrows():records.append(dict(comparison=name,update_strategy=strategy,session=session,participant=p,**r.to_dict()))
                row=dict(comparison=name,update_strategy=strategy,session=session,scope='post-hoc development contrast; one seed; conditional participant bootstrap',unknown_rate_improves_people=int((difference.unknown_false_acceptance<0).sum()),people_with_5pp_ufa_reduction_and_at_most_2pp_ccr_loss=int(((difference.unknown_false_acceptance<=-.05)&(difference.known_correct_acceptance>=-.02)).sum()))
                for j,field in enumerate(fields):row.update({field+'_difference':float(difference[field].mean()),field+'_lower':float(lo[j]),field+'_upper':float(hi[j])})
                summary.append(row)
    pd.DataFrame(records).to_csv(run/'paired_contrasts_people.csv',index=False)
    result=dict(created_utc=datetime.now(timezone.utc).isoformat(),selection='Three contrasts chosen after inspecting complete primary operating points; no models/settings changed',source_sha256=hash_stream(__file__),contrasts=summary)
    (run/'paired_contrasts.json').write_text(json.dumps(result,indent=2)+'\n');(run/'code_snapshot'/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    (run/'contrast_artifact_sha256.json').write_text(json.dumps({f:hash_stream(run/f) for f in ['paired_contrasts_people.csv','paired_contrasts.json','code_snapshot/analyze_open_set_contrasts.py']},indent=2)+'\n')
    print(json.dumps([r for r in summary if r['update_strategy']=='pooled_replay' and r['session']=='both'],indent=2))


if __name__=='__main__':main()
