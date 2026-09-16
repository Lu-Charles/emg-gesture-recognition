"""Fill the manuscript's fixed-control section from fully verified study outputs."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
R=ROOT/'research'
P=R/'runs/20260907_csl_components_confirmatory'

def pc(x):return f'{100*x:.2f}'
def signed(x):return f'{100*x:+.2f}'

def main():
    roots=[P,R/'runs/20260907_csl_input_order',R/'runs/20260907_csl_identity_ties',R/'runs/20260907_csl_search_seed_sensitivity']
    checks=[json.loads((p/'verification.json').read_text()) for p in roots]
    assert all(z['passed'] for z in checks)
    a=json.loads((P/'analysis.json').read_text());c=json.loads((P/'controls_analysis.json').read_text())
    audit=c['identity_intervention_audit'];assert audit['passed'] and audit['initial_loss_equal_cases']==1280
    order=c['controls']['resample_then_gain'];identity=c['controls']['identity_tie']
    lines=['| Spatial procedure | Spatial: other classes (%) | Spatial + gain: other classes (%) |', '|---|---:|---:|',f"| Original | {pc(a['methods']['spatial_only']['unseen']['mean'])} | {pc(a['methods']['spatial_gain']['unseen']['mean'])} |"]
    for label,record in [('Prefer identity in exact loss ties',identity),('Resample, apply gain, then normalize',order)]:
        lines.append(f"| {label} | {pc(record['spatial_only']['unseen']['mean'])} | {pc(record['spatial_gain']['unseen']['mean'])} |")
    lines.extend(['','Table 3. The two controls reuse all 80 session pairs and 640 calibration choices. Frozen other-class accuracy is 60.71%. All control settings were fixed before inspecting primary cohort performance summaries.',''])
    lines.append(f"Preferring identity in exact loss ties changes other-class accuracy by {signed(identity['spatial_only']['delta_from_original']['mean'])} percentage points for spatial adaptation and {signed(identity['spatial_gain']['delta_from_original']['mean'])} points for spatial adaptation with gain. Identity replaces the original selected candidate in {identity['spatial_only']['identity_preferred_cases']}/640 and {identity['spatial_gain']['identity_preferred_cases']}/640 cases, respectively. The selected initial calibration loss is exactly unchanged in all {audit['initial_loss_equal_cases']:,} paired comparisons. All {audit['nonintervened_checkpoints_identical']:,} non-intervened checkpoints are tensor-identical to their primary counterparts and reproduce the same trial predictions.")
    lines.append('')
    for method,label in [('spatial_only','Spatial'),('spatial_gain','Spatial + gain')]:
        changes=identity[method]['case_changes_vs_original']
        lines.append(f"{label} with identity preference improves other-class accuracy in {changes['improved']}/640 cases, harms it in {changes['harmed']}/640 and leaves it tied in {changes['tied']}/640. These descriptive counts concern repeated cases. Preserving the initial calibration loss therefore does not guarantee improved final transfer.")
    lines.extend(['',f"Changing the processing order produces other-class changes of {signed(order['spatial_only']['delta_from_original']['mean'])} and {signed(order['spatial_gain']['delta_from_original']['mean'])} points relative to the respective original procedures. This intervention changes both normalization placement and gain placement. Its results cannot be assigned solely to gain/resampling noncommutativity.",''])
    lines.extend(['| Search seed, fixed session 1→2 | Spatial: other classes (%) | Spatial + gain: other classes (%) |','|---|---:|---:|'])
    for seed in ['42','43','44']:
        lines.append(f"| {seed} | {pc(c['search_seed_sensitivity']['spatial_only'][seed]['mean'])} | {pc(c['search_seed_sensitivity']['spatial_gain'][seed]['mean'])} |")
    lines.extend(['','Table 4. Search-seed sensitivity uses four participants and all eight calibration choices on one fixed session pair. It changes candidate sampling only; the source checkpoint and all optimization settings remain fixed. These values should not be compared as if they were means over all 80 session pairs.',''])
    for method,label in [('spatial_only','Spatial'),('spatial_gain','Spatial + gain')]:
        spread=c['search_seed_ranges'][method]
        lines.append(f"{label} has a mean per-case range across the three search seeds of {pc(spread['mean_range'])} percentage points, with a maximum of {pc(spread['max_range'])}; {spread['range_over10pp']}/{spread['cases']} cases span more than ten points. All seeds are reported without selecting the best result.")
    lines.extend(['','![Implementation controls](research/runs/20260907_csl_components_confirmatory/implementation_controls.png)','','Figure 2. Matched implementation contrasts and the supplementary search-seed results. Points in panels A–B are participant means; panel C is restricted to session pair 1→2.',''])
    section='\n'.join(lines)
    draft=R/'CSL_PAPER_DRAFT.md';text=draft.read_text();assert 'CONTROL_RESULTS_PENDING_VERIFIED_COMPLETION' in text
    text=text.replace('CONTROL_RESULTS_PENDING_VERIFIED_COMPLETION',section)
    text=text.replace('Research manuscript draft. Primary results independently verified; supplementary control sections are being completed.', 'Research manuscript draft. The primary study and all fixed supplementary controls are completed and independently verified.')
    text=text.replace('Independent checkpoint replay verifies the primary predictions and split integrity.',f"An exact-loss identity tie control changes other-class accuracy by {signed(identity['spatial_only']['delta_from_original']['mean'])} and {signed(identity['spatial_gain']['delta_from_original']['mean'])} percentage points for the spatial conditions without changing the selected initial calibration loss. Processing-order and search-seed controls are also completed. Independent checkpoint replay verifies predictions and split integrity.")
    assert 'PENDING' not in text and 'SEE_FINAL' not in text
    draft.write_text(text)
    (R/'CSL_CONTROL_FINDINGS.md').write_text('# Verified implementation controls\n\n'+section+'\nThese results are supplementary controls within the same four participants, not independent cohorts. Full analysis: `runs/20260907_csl_components_confirmatory/controls_analysis.json`.\n')
    print('Filled manuscript control tables and created CSL_CONTROL_FINDINGS.md')
if __name__=='__main__':main()
