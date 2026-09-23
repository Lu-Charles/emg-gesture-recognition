"""Reproduce selected means and Holm correction from saved participant summaries.

Python standard library only. This is not raw-data inference or model fitting.
Run from anywhere: python reproduce_summary.py
"""
import json
import statistics
from pathlib import Path


def main():
    root=Path(__file__).resolve().parent
    data=json.loads((root/'statistics.json').read_text())
    rows=[]
    for r in data['final_neural']:
        if r['k']==2 and (r['method']=='cnn_frozen' or (r['steps']==100 and r['method'] in ['cnn_naive','cnn_replay'])):
            mean=statistics.mean(r['per_person']['accuracy'])
            assert abs(mean-r['accuracy'])<1e-12
            rows.append({'method':r['method'],'participants':len(r['participants']),'accuracy_percent':100*mean})
    for key in ['final_selection_primary','final_rotation_coverage_contrast']:
        r=data[key]
        mean=statistics.mean(r['per_person'])
        assert abs(mean-r['mean'])<1e-12
        rows.append({'contrast':key,'participants':r['n'],'difference_percentage_points':100*mean})
    ordered=sorted(data['primary_tests'].items(),key=lambda p:p[1]['raw_p'])
    prev=0
    for i,(name,p) in enumerate(ordered):
        adjusted=max(prev,min(1,(len(ordered)-i)*p['raw_p']))
        assert abs(adjusted-p['holm_p'])<1e-12
        prev=adjusted
    print(json.dumps({'passed':True,'results':rows,'primary_tests':data['primary_tests'],'scope':'Saved-summary arithmetic only; confidence intervals and raw predictions are not reconstructed by this script.'},indent=2))


if __name__=='__main__':main()
