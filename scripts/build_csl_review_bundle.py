"""Build a portable, local analysis-only review bundle after study verification."""
import csv,hashlib,json,shutil,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
RESEARCH=ROOT/'research'
P=RESEARCH/'runs/20260907_csl_components_confirmatory'
OUT=RESEARCH/'artifacts/csl_review_bundle'
RUNS={'order':'20260907_csl_input_order','identity':'20260907_csl_identity_ties','search_seeds':'20260907_csl_search_seed_sensitivity'}

REPRODUCE=r'''"""Recompute reported means and contrasts from included per-case records.
This audits the analysis only. It does not retrain models or replay raw signals.
"""
import csv,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent

def read(name):
    with (ROOT/name).open(newline='') as f:return list(csv.DictReader(f))
def mean(rows,field):return float(np.mean([float(r[field]) for r in rows]))
def check(value,expected):np.testing.assert_allclose(value,expected,rtol=0,atol=1e-12)
def boot(a):
    a=np.array(a);rng=np.random.default_rng(2026)
    return np.quantile(a[rng.integers(0,4,(10000,4))].mean(1),[.025,.975])

def main():
    rows=read('primary_cases.csv');assert len(rows)==3840
    a=json.loads((ROOT/'analysis.json').read_text());c=json.loads((ROOT/'controls_analysis.json').read_text())
    participant={}
    for method,expected in a['methods'].items():
        selected=[r for r in rows if r['method']==method];assert len(selected)==640
        participant[method]={}
        for field in ['overall','unseen','seen','calibration_fit','unseen_delta']:
            part=[mean([r for r in selected if r['participant']==str(p)],field) for p in [2,3,4,5]]
            participant[method][field]=part
            check(np.mean(part),expected[field]['mean']);check(boot(part),expected[field]['participant_bootstrap_95'])
        high=[r for r in selected if float(r['calibration_fit'])>=.99]
        assert len(high)==expected['high_fit_cases']
        assert sum(r['negative_transfer']=='True' for r in high)==expected['high_fit_negative']
    for name,terms in {'gain_vs_frozen':{'gain_only':1,'frozen':-1},'spatial_added_to_gain':{'spatial_gain':1,'gain_only':-1},'spatial_vs_frozen':{'spatial_only':1,'frozen':-1},'interaction':{'spatial_gain':1,'spatial_only':-1,'gain_only':-1,'frozen':1}}.items():
        part=sum(weight*np.array(participant[m]['unseen']) for m,weight in terms.items())
        check(part,a['contrasts'][name]['participant_effects']);check(boot(part),a['contrasts'][name]['participant_bootstrap_95'])
    frozen=[r for r in rows if r['method']=='frozen' and r['source_session']=='1' and r['target_session']=='2']
    check([mean([r for r in frozen if r['participant']==str(p)],'unseen') for p in [2,3,4,5]],c['search_pair_frozen']['participant_unseen'])
    controls=read('control_cases.csv')
    for dataset,label in [('order','resample_then_gain'),('identity','identity_tie')]:
        for method in ['spatial_only','spatial_gain']:
            selected=[r for r in controls if r['dataset']==dataset and r['method']==method];assert len(selected)==640
            for field in ['overall','unseen']:
                part=[mean([r for r in selected if r['participant']==str(p)],field) for p in [2,3,4,5]]
                check(np.mean(part),c['controls'][label][method][field]['mean']);check(boot(part),c['controls'][label][method][field]['participant_bootstrap_95'])
    for seed in ['42','43','44']:
        pool=rows if seed=='42' else [r for r in controls if r['dataset']=='search_seeds' and r['seed']==seed]
        for method in ['spatial_only','spatial_gain']:
            selected=[r for r in pool if r['method']==method and r['source_session']=='1' and r['target_session']=='2'];assert len(selected)==32
            part=[mean([r for r in selected if r['participant']==str(p)],'unseen') for p in [2,3,4,5]]
            check(part,c['search_seed_sensitivity'][method][seed]['participant_unseen'])
    report={'passed':True,'primary_records':len(rows),'control_records':len(controls),'independent_participants':4,'scope':'Analysis reproduction from included case records; no raw-data or checkpoint replay.'}
    (ROOT/'analysis_reproduction.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
'''

def main():
    for run in [P,*[RESEARCH/'runs'/v for v in RUNS.values()]]:
        assert json.loads((run/'verification.json').read_text())['passed']
    assert 'PENDING' not in (RESEARCH/'CSL_PAPER_DRAFT.md').read_text()
    assert not OUT.exists(),'Preserve the existing review bundle; use a new version for changes.'
    OUT.mkdir(parents=True);(OUT/'figures').mkdir();(OUT/'verification').mkdir()
    for source,dest in [('case_summary.csv','primary_cases.csv'),('participant_summary.csv','participant_summary.csv'),('analysis.json','analysis.json'),('controls_analysis.json','controls_analysis.json'),('protocol.json','primary_protocol.json'),('environment.json','environment.json')]:shutil.copy2(P/source,OUT/dest)
    for name in ['transfer_gap','component_results','implementation_controls']:
        for ext in ['png','svg']:shutil.copy2(P/f'{name}.{ext}',OUT/'figures'/f'{name}.{ext}')
    for name in ['CSL_PAPER_DRAFT.md','CSL_SUBMISSION_ABSTRACT.md','CSL_COMPONENT_CONTRIBUTION.md','CSL_COMPONENT_METHODS.md','CSL_CONTROL_FINDINGS.md','CSL_CALIBRATION_AMBIGUITY.md','CSL_REVIEW_NOTES.md','REPRODUCE_CSL_COMPONENT_STUDY.md']:
        text=(RESEARCH/name).read_text().replace(str(P)+'/', 'figures/')
        # Manuscript figure references become relative paths in this portable copy.
        (OUT/name).write_text(text)
    shutil.copy2(P/'verification.json',OUT/'verification/primary.json')
    allrows=[]
    for dataset,name in RUNS.items():
        run=RESEARCH/'runs'/name
        shutil.copy2(run/'verification.json',OUT/'verification'/f'{dataset}.json');shutil.copy2(run/'protocol.json',OUT/f'{dataset}_protocol.json')
        rows=json.loads((run/'results.json').read_text())['results']
        for r in rows:
            allrows.append(dict(dataset=dataset,participant=r['participant'],source_session=r['source_session'],target_session=r['target_session'],gesture=r['gesture'],method=r['method'],seed=r.get('seed',42),overall=r['metrics']['trial_accuracy'],unseen=r['metrics']['unseen_trial_accuracy'],calibration_fit=r['calibration_frame_accuracy']))
    with (OUT/'control_cases.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(allrows[0]));w.writeheader();w.writerows(allrows)
    (OUT/'reproduce_analysis.py').write_text(REPRODUCE)
    version=json.loads((P/'environment.json').read_text())['packages']['numpy'];(OUT/'requirements.txt').write_text(f'numpy=={version}\n')
    (OUT/'README.md').write_text('''# CSL single-gesture calibration: analysis review bundle

Start with `CSL_PAPER_DRAFT.md`. Figures, participant summaries, per-case records, frozen protocols and original verification reports are included.

To reproduce the reported primary means, participant contrasts, descriptive bootstrap intervals, high-fit negative-transfer counts, control means and search-seed means:

```sh
python -m pip install -r requirements.txt
python reproduce_analysis.py
```

The script checks included case records against the reported analysis to absolute tolerance 1e-12 and writes `analysis_reproduction.json`. It does not retrain models, reconstruct raw signals or independently validate the original prediction reports. Those checks were executed in the full local project; their results are included for inspection.

Four people are the independent observations. Thousands of repeated records are not additional participants. All gesture classes were seen during source training; "other" refers only to target calibration coverage. This is an offline natural-session study, not controlled displacement or a clinical evaluation.

Raw recordings, processed signal caches, model checkpoints and third-party source are excluded. The complete local workspace retains these separately. The source revision and data-manifest integrity procedure are described in the methods. This bundle is prepared locally for review and has not been submitted or published. Authors, affiliations and venue formatting are not supplied.
''')
    # The analysis output is added only after its actual reproduction check.
    print('Built',OUT)

if __name__=='__main__':main()
