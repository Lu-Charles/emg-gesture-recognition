"""Create a preserved publication-review version from the audited Sensors draft."""
from pathlib import Path
import hashlib, json, re, shutil

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT/'research/submissions/sensors2026_revised'
NEW = ROOT/'research/submissions/sensors2026_submission_20260923'
CMP = ROOT/'research/comparator_development_20260923'

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    NEW.mkdir(exist_ok=False)
    before = {str(p.relative_to(OLD)):sha(p) for p in OLD.rglob('*') if p.is_file()}
    for name in ['document_format.py', 'Cover_Letter.md', 'CONFERENCE_EXTENSION.md']:
        shutil.copy2(OLD/name, NEW/name)
    for folder in ['figures','analysis_supplement']:
        shutil.copytree(OLD/folder,NEW/folder,dirs_exist_ok=True)
    for ext in ['png','svg','pdf']:
        shutil.copy2(CMP/f'figures/published_objective_comparison.{ext}',NEW/f'figures/published_objective_comparison.{ext}')
    evidence = NEW/'analysis_supplement/comparator'; evidence.mkdir(exist_ok=True)
    for name in ['combined_summary.json','bridge_verification.json','PROTOCOL.md','COMPARISON.md','finite_check_equivalence.json']:
        shutil.copy2(CMP/name,evidence/name)
    text = (OLD/'Sensors_Manuscript.md').read_text()
    text = text.replace('Related conference work: This manuscript extends the earlier CSL-HDEMG study [10] with independent GRABMyo and SeNic evaluations. Exact conference citation and acceptance category await author confirmation.\n\n','')
    text = text.replace('## Highlights\n\n','## Highlights\n\n### Main findings\n\n')
    text = text.replace('- Neither primary superiority test passed correction, and update strength affected the observed damage.', '- Neither primary superiority test passed correction, and update strength affected the observed damage.\n\n### Implication\n\n- Brief calibration should be assessed using the full vocabulary, including gestures absent from calibration.')
    text = text.replace('### 2.6 Outcomes and statistical analysis', '''### 2.6 Published objective comparison on development participants
We added a development-only comparison of source-supervised covariance alignment, adapting Yuan et al.'s objective [12] with the Deep CORAL penalty [13]. All network parameters were updated using source cross-entropy plus a weighted covariance difference between 64-dimensional source and unlabeled target features. Controls comprised frozen recognition, target-only cross-entropy, equal-weight source/target replay, and source-only continued training. Adam used learning rate 0.0001, 64 windows per domain and 25/100 steps.

The comparison used the eight existing development participants, seed 42, both later sessions and four cyclic choice indices (0, 4, 8, 12), yielding 128 calibration cases. All conditions retained the two-recording budget and identical scoring trials. Initial alignment coefficients were 0.1, 1 and 10; after inspecting their results and loss scales, we added 0.0001, 0.001 and 0.01. All six are reported. This adaptive analysis accessed no final participants and did not change the primary methods. It adapts a published objective to our personal enrollment and compact network, rather than reproducing Yuan et al.'s Hyser/VGG inter-user experiment. Supplementary Methods S1 provides the complete implementation and verification details.

### 2.7 Outcomes and statistical analysis''')
    text = text.replace('## 4. Discussion', '''### 3.6 Development comparison of supervised updates and covariance alignment
In the four-choice comparison, two-gesture full-network replay reached 80.95% and 81.41% overall accuracy at 25 and 100 steps, versus 77.48% frozen. Omitted-active accuracy was 77.99% and 78.40%, versus 76.73% frozen. The strongest observed alignment mean across the tested settings was 78.35% (coefficient 0.001, 100 steps), compared with 78.17% for source-only training. The paired alignment-minus-source difference was +0.18 percentage points (descriptive 95% interval −1.32 to +1.69), providing no clear evidence of an alignment-specific benefit. Tables S1–S2 and Figure S1 report all settings. These exploratory results use fewer gesture choices than Table 6 and do not replace the final-cohort comparisons.

## 4. Discussion''',1)
    text = text.replace('Neither observation makes the empirical paper invalid. The scientific weakness would be claiming broad recognition improvement from calibrated recall alone, or portraying an established forgetting mechanism as a new discovery.', 'These observations explain why calibrated recall should be paired with full-vocabulary and omitted-class outcomes.')
    text = text.replace('It is not a new replay algorithm, a new general identifiability theorem, or a state-of-the-art leaderboard claim. ', '')
    text = text.replace('Our configurations have not been matched to every published method, so numerical superiority over those methods is not claimed.', 'The supplementary comparison implements a published covariance-alignment objective under matched data access within our protocol. Its development-only design and changes from the original architecture limit comparisons with published accuracies.')
    text = text.replace('Independent replication and comparisons with additional published adaptation methods would strengthen the generality of these findings.', 'Independent replication, further architectures and additional published adaptation objectives would strengthen generality. The supplementary alignment comparison reuses development participants and includes a loss-guided coefficient extension; its intervals are descriptive and its best observed setting is not independently validated.')
    text = text.replace('The accompanying analysis supplement contains the frozen evaluation protocol, participant summaries, environment records, numerical verification reports, and a script that reconstructs the summary means and primary Holm adjustment.', 'Supplementary Methods S1, Tables S1–S2 and Figure S1 report the development comparison. The accompanying analysis supplement contains the documented evaluation protocol, participant summaries, environment records, numerical verification reports, and a script that reconstructs the original summary means and primary Holm adjustment. Comparator participant means and all six alignment coefficients are supplied separately.')
    text = text.replace('OpenAI Codex assisted with literature checking, experimental software, execution, computational verification, analysis, and drafting. Measurements were calculated from dataset labels and saved model predictions; no synthetic observations were presented as human recordings. Synthetic mechanism experiments are identified separately. AUTHOR CONFIRMATION REQUIRED: Review this disclosure and verify the manuscript\'s interpretation and references before submission.', 'OpenAI Codex assisted with literature review, code development and execution, analysis, figure preparation, and manuscript drafting and editing. AUTHOR CONFIRMATION REQUIRED: Confirm the tools/model versions used, review the final content and references, and approve the author-responsibility statement before submission.')
    text += '\n\n[13] Sun, B.; Saenko, K. Deep CORAL: Correlation Alignment for Deep Domain Adaptation. arXiv:1607.01719, 2016. https://arxiv.org/abs/1607.01719\n'
    body, refs = text.split('## References\n',1)
    reference_map = {int(n):v.strip() for n,v in re.findall(r'^\[(\d+)\] (.*?)(?=^\[\d+\] |\Z)',refs,re.M|re.S)}
    order=[]
    def renumber(m):
        values=[int(v.strip()) for v in m[0][1:-1].split(',')]
        for v in values:
            if v not in order: order.append(v)
        return '['+', '.join(str(order.index(v)+1) for v in values)+']'
    # Only numbered citations, never numeric ranges, IDs or literal fixed gesture pair.
    body = re.sub(r'\[(?:[1-9]\d*)(?:,\s*[1-9]\d*)*\]',renumber,body)
    assert set(order)==set(reference_map),(order,reference_map.keys())
    text = body+'## References\n\n'+'\n\n'.join(f'[{i+1}] {reference_map[v]}' for i,v in enumerate(order))+'\n'
    (NEW/'Sensors_Manuscript.md').write_text(text)
    (NEW/'reference_renumbering.json').write_text(json.dumps({str(v):i+1 for i,v in enumerate(order)},indent=2))
    draft=(CMP/'MANUSCRIPT_ADDITION_DRAFT.md').read_text()
    methods=draft.split('## Methods: published-objective development comparison\n\n')[1].split('\n## Results')[0]
    methods=methods.replace('[12]', '[S1]').replace('[additional reference below]', '[S2]')
    methods=methods.replace('after cloud access became unreliable','')
    methods=methods.replace('PyTorch 2.14.0 .','PyTorch 2.14.0.')
    result=draft.split('## Results\n\n')[1].split('\n## Placement')[0]
    comparison=(CMP/'COMPARISON.md').read_text().splitlines()
    header='| Method | Steps | Overall (%) | Omitted active (%) | Calibrated recall (%) |\n|---|---:|---:|---:|---:|\n'
    tables=[]
    names={'frozen':'Frozen','target_ce':'Target CE','source_ce':'Source CE','replay':'Replay'}
    for k in (1,2):
        rows=[]
        for line in comparison:
            if line.startswith(f'| {k} |'):
                vals=[v.strip() for v in line.strip('|').split('|')][1:]
                vals[0]=names.get(vals[0],vals[0].replace('coral_','CORAL '))
                rows.append('| '+' | '.join(vals)+' |')
        tables.append(f'## Table S{k}\n\n'+header+'\n'.join(rows)+f'\n\nTable S{k}. '+('One-gesture' if k==1 else 'Two-gesture')+' calibration using two recordings. Participant means across eight development participants, two sessions and four cyclic choices; all settings are exploratory. Omitted active excludes rest and prompted gestures. CE denotes categorical cross-entropy. CORAL labels give the alignment coefficient.\n')
    caption=(CMP/'figures/CAPTION.md').read_text().strip()
    supp='# Supplementary Development Comparison of EMG Adaptation Objectives\n\nSupplement to Gesture Coverage and Vocabulary Retention During Brief EMG Calibration\n\nCharles Lu\n\n## Supplementary Methods S1\n\n'+methods+'\n\n## Supplementary Results\n\n'+result+'\n\n'+'\n'.join(tables)+'\n## Figure S1\n\n![Development comparison](figures/published_objective_comparison.png)\n\nFigure S1. '+caption+'\n\n## Supplementary References\n\n[S1] '+reference_map[12]+'\n\n[S2] '+reference_map[13]+'\n'
    (NEW/'Supplementary_Comparison.md').write_text(supp)
    cover=(NEW/'Cover_Letter.md').read_text().replace('I am preparing the research Article','Please consider the research Article').replace('for consideration in Sensors.','for publication in Sensors.')
    cover=cover.replace('These findings are relevant to evaluating reusable EMG sensing interfaces.', 'A supplementary development analysis adds a published covariance-alignment objective, source-only controls and full-network supervised updates, retaining all tested coefficients. These findings are relevant to evaluating reusable EMG sensing interfaces.')
    (NEW/'Cover_Letter.md').write_text(cover)
    (NEW/'build_packet.py').write_text('from pathlib import Path\nfrom document_format import make_doc\nHERE=Path(__file__).resolve().parent\nif __name__=="__main__":\n    for source,name in [("Sensors_Manuscript.md","Manuscript"),("Cover_Letter.md","Cover_Letter"),("Supplementary_Comparison.md","Supplementary_Comparison")]:\n        make_doc((HERE/source).read_text(),HERE/f"Charles_Lu_Sensors_{name}.docx")\n')
    after={str(p.relative_to(OLD)):sha(p) for p in OLD.rglob('*') if p.is_file()}
    assert before==after
    (NEW/'preservation_check.json').write_text(json.dumps({'source':str(OLD),'files':len(before),'unchanged':before==after,'sha256':before},indent=2))
    print(NEW)

if __name__=='__main__':main()
