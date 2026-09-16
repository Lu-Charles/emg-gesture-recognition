"""Sequential stage coordinator for the already frozen study and its controls."""
from pathlib import Path
from datetime import datetime,timezone
import json,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'research/runs/20260907_csl_components_confirmatory'
STAGES=[
 ('verify_primary',['scripts.verify_csl_components','final'],'20260907_csl_components_confirmatory/verification.log'),
 ('input_order',['scripts.csl_input_order','run'],'20260907_csl_input_order/training.log'),
 ('verify_input_order',['scripts.verify_csl_components','input-order'],'20260907_csl_input_order/verification.log'),
 ('identity_ties',['scripts.csl_identity_ties','run'],'20260907_csl_identity_ties/training.log'),
 ('verify_identity',['scripts.verify_csl_components','identity'],'20260907_csl_identity_ties/verification.log'),
 ('search_seeds',['scripts.csl_search_sensitivity','run'],'20260907_csl_search_seed_sensitivity/training.log'),
 ('verify_search_seeds',['scripts.verify_csl_search_sensitivity'],'20260907_csl_search_seed_sensitivity/verification.log'),
 ('report_primary',['scripts.report_csl_components'],'20260907_csl_components_confirmatory/analysis.log')]

def main():
    assert (OUT/'results.json').exists();records=[]
    for name,args,logfile in STAGES:
        entry=dict(stage=name,status='running',started=datetime.now(timezone.utc).isoformat());records.append(entry);(OUT/'stage_status.json').write_text(json.dumps(records,indent=2)+'\n');print('Started',name,flush=True);t=time.perf_counter()
        with (ROOT/'research/runs'/logfile).open('w') as f:
            completed=subprocess.run([sys.executable,'-m',*args],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
        entry.update(status='completed' if completed.returncode==0 else 'failed',wall_seconds=time.perf_counter()-t,returncode=completed.returncode);(OUT/'stage_status.json').write_text(json.dumps(records,indent=2)+'\n')
        if completed.returncode:raise RuntimeError(f'{name} failed; inspect {logfile}')
        print('Completed',name,flush=True)
if __name__=='__main__':main()
