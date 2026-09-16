"""Create a local content-hash index for the completed CSL study and manuscript."""
from datetime import datetime,timezone
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PRIMARY=ROOT/'research/runs/20260907_csl_components_confirmatory'
RUN_NAMES=['20260907_csl_components_confirmatory','20260907_csl_input_order','20260907_csl_identity_ties','20260907_csl_search_seed_sensitivity']
DOC_NAMES=['CSL_SUBMISSION_ABSTRACT.md','CSL_PAPER_DRAFT.md','CSL_COMPONENT_CONTRIBUTION.md','CSL_PRIMARY_FINDINGS.md','CSL_CONTROL_FINDINGS.md','CSL_COMPONENT_METHODS.md','CSL_CALIBRATION_AMBIGUITY.md','REPRODUCE_CSL_COMPONENT_STUDY.md','CSL_REVIEW_NOTES.md']

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
    return h.hexdigest()

def main():
    files=set()
    for name in RUN_NAMES:
        run=ROOT/'research/runs'/name
        assert json.loads((run/'verification.json').read_text())['passed']
        # Numeric feature caches and raw recordings remain separate; the data
        # manifest records their existing integrity hashes.
        for path in run.rglob('*'):
            if path.is_file() and path.suffix in {'.pt','.npz','.json','.csv','.png','.svg','.py'} and path.name!='evidence_ledger.json' and 'development' not in path.relative_to(run).parts:
                files.add(path)
    for name in DOC_NAMES:
        p=ROOT/'research'/name;assert p.is_file();files.add(p)
    files.update((ROOT/'scripts').glob('*csl*.py'))
    records=[dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(files)]
    result=dict(timestamp=datetime.now(timezone.utc).isoformat(),canonical_workspace=str(ROOT),project_git_revision=None,scope='Local completed study evidence. Hash index, not a portable distribution or public release. Raw data and numeric caches excluded; data-manifest hashes retained.',files=records,file_count=len(records),total_bytes=sum(r['bytes'] for r in records))
    (PRIMARY/'evidence_ledger.json').write_text(json.dumps(result,indent=2)+'\n')
    print('Indexed',len(records),'files;',result['total_bytes'],'bytes')
if __name__=='__main__':main()
