"""Build a local reproducibility bundle without raw signals or per-case model banks."""

import ast
import json
import shutil
import hashlib
import zipfile
from pathlib import Path
from datetime import datetime, timezone

R = Path(__file__).resolve().parents[1]
O = R / "research/expanded_study_20260916"
B = O / "reproducibility"
B.mkdir(exist_ok=True)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def copy(source, relative):
    target = B / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


new = [
    "retrieve_senic_final_archives",
    "decisive_calibration",
    "verify_decisive_calibration",
    "complete_tdar_development",
    "coverage_classical_selection",
    "verify_classical_selection",
    "full_network_coverage",
    "verify_full_network_coverage",
    "rotation_coverage",
    "verify_rotation_coverage",
    "cache_final_coverage",
    "enroll_final_coverage",
    "decisive_final",
    "verify_decisive_final",
    "tdar_final_cache",
    "classical_final_selection",
    "verify_classical_final",
    "rotation_final",
    "rotation_final_reference",
    "coverage_mechanism_stress",
    "verify_final_cache",
    "analyze_coverage_study",
    "verify_coverage_statistics",
    "write_coverage_manuscript",
    "render_coverage_manuscript",
    "package_coverage_study",
]
queue = [R / "scripts" / f"{name}.py" for name in new] + [
    R / "tests" / f"{name}.py"
    for name in [
        "test_final_coverage",
        "test_rotation_coverage",
        "test_decisive_calibration",
        "test_coverage_diagnostics",
    ]
]
seen = set()
while queue:
    p = queue.pop()
    if p in seen:
        continue
    seen.add(p)
    copy(p, p.relative_to(R))
    for node in ast.walk(ast.parse(p.read_text())):
        if (
            isinstance(node, ast.ImportFrom)
            and node.module
            and node.module.split(".")[0] in ["src", "scripts", "tests"]
        ):
            q = R / Path(*node.module.split(".")).with_suffix(".py")
            if q.exists():
                queue.append(q)
for folder in ["src", "scripts", "tests"]:
    p = R / folder / "__init__.py"
    if p.exists():
        copy(p, p.relative_to(R))
# Preserve pinned archive ledger used by the public-data retrieval helper.
for name in ["download_tree_verification.json", "senic_commit.json"]:
    p = R / "research/runs/20260907_sensor_shift_audit" / name
    copy(p, p.relative_to(R))
# Preserve immutable protocol and small verified shared-pretraining checkpoints.
P = R / "research/runs/20260625_confirmatory_protocol"
for p in P.rglob("*"):
    if p.is_file():
        copy(p, p.relative_to(R))
for name in [
    "20260906_shared_pretraining_v2",
    "20260906_shared_seed0_v1",
    "20260906_shared_seed1_v1",
]:
    for filename in [
        "pretrain_epoch20.pt",
        "training_scaler.npz",
        "config.json",
        "summary.json",
        "independent_validation.json",
    ]:
        p = R / "research/runs" / name / filename
        copy(p, p.relative_to(R))
run_names = [
    "20260916_decisive_seed42",
    "20260916_decisive_seed0",
    "20260916_decisive_seed1",
    "20260916_classical_selection",
    "20260916_rotation_coverage",
    "20260916_full_network_coverage",
    "20260916_decisive_final_seed42",
    "20260916_decisive_final_seed0",
    "20260916_decisive_final_seed1",
    "20260916_classical_final",
    "20260916_rotation_final",
    "20260916_rotation_final_reference",
    "20260916_mechanism_stress",
    "20260916_tdar_development",
    "20260916_tdar_final",
    "20260916_final_enrollment_seed42",
    "20260916_final_enrollment_seed0",
    "20260916_final_enrollment_seed1",
]
for name in run_names:
    folder = R / "research/runs" / name
    for filename in [
        "protocol.json",
        "complete.json",
        "verification.json",
        "enrollment_complete.json",
    ]:
        p = folder / filename
        if p.exists():
            copy(p, Path("evidence/runs") / name / filename)
for dataset in ["grabmyo", "senic"]:
    for filename in [
        "summary.json",
        "authorization.json",
        "independent_audit.json",
        "cache_final_coverage.py",
    ]:
        p = R / f"data/public/{dataset}/cache_final_20260916" / filename
        if p.exists():
            copy(p, Path("evidence/caches") / dataset / filename)
for name in [
    "READ_ME_FIRST.md",
    "pdf_verification.json",
    "senic_archive_verification.json",
    "statistics.json",
    "statistics_verification.json",
    "tables.json",
    "environment_public.json",
    "environment_features.json",
    "Expanded_EMG_Manuscript.md",
]:
    copy(O / name, Path("research/expanded_study_20260916") / name)
for p in (O / "figures").glob("*"):
    copy(p, Path("research/expanded_study_20260916/figures") / p.name)
copy(
    R / "output/pdf/Charles_Lu_Expanded_EMG_Study.pdf",
    Path("output/pdf/Charles_Lu_Expanded_EMG_Study.pdf"),
)
copy(
    R / "research/RELATED_WORK_AND_PUBLICATION_GATE_20260916.md",
    Path("research/RELATED_WORK_AND_PUBLICATION_GATE_20260916.md"),
)
# Local run archive index; no claim that large artifacts are contained in the ZIP.
index = {
    name: dict(
        path=str(R / "research/runs" / name),
        files={
            p.name: sha(p)
            for p in (R / "research/runs" / name).iterdir()
            if p.is_file()
            and p.name
            in [
                "protocol.json",
                "results.json",
                "access.json",
                "complete.json",
                "verification.json",
                "trial_predictions.npz",
                "predictions.npz",
                "parameters.npz",
            ]
        },
    )
    for name in run_names
}
(O / "artifact_index.json").write_text(json.dumps(index, indent=2) + "\n")
copy(O / "artifact_index.json", Path("evidence/artifact_index.json"))
readme = """# EMG coverage study: local reproducibility bundle

This bundle accompanies the expanded manuscript. It contains the frozen protocol, code and dependencies from this workflow, audited participant statistics, figures, local artifact index, verification reports, and three small shared-pretraining checkpoints. Raw datasets, per-case adaptation model banks and exhaustive result JSON/prediction archives remain in the canonical project and are not included here. Nothing has been published or submitted.

## Rebuild the document from audited summaries

Use the recorded Python environments in environment_public.json and environment_features.json. The public environment supplies PyTorch/scikit-learn/matplotlib; the feature environment supplies librosa0.11.0 for the pinned Burg implementation. From this bundle's root, run `python -m scripts.write_coverage_manuscript`. PDF rendering additionally needs reportlab/Pillow, Liberation Serif and DejaVu Sans fonts; the renderer's default font directory points to the bundled runtime on the author's Mac and must be adjusted on another machine. The PDF is already included for immediate reading.

## Repeat final evaluation from permitted public downloads

This is an outline of the dependency order, not a claim that this bundle was rerun on a clean machine. Output scripts deliberately reject existing run directories. Use a fresh copy when repeating. Keep the frozen protocol unchanged; altered methods on already-examined final participants are exploratory.

1. Retrieve GRABMyo1.1.0 and its official SHA256SUMS.txt into data/public/grabmyo/1.1.0. The included src/grabmyo.py records source URLs and label/trial allocation. Run `python -m scripts.cache_final_coverage grabmyo`. For SeNic, run `python -m scripts.retrieve_senic_final_archives`, which uses the included pinned Git-blob ledger and downloads only the24frozen participants. Then run `python -m scripts.cache_final_coverage senic` in the feature environment. No raw data are redistributed in this bundle.
2. Run `python -m scripts.enroll_final_coverage` with the supplied verified shared-pretraining checkpoints. This uses only each final person's day1 source recordings. Their model inference and calibration follow later.
3. For each seed42,0,1, run decisive_final with `--seed` and `--out research/runs/20260916_decisive_final_seed<seed>`, then verify_decisive_final with `--run` that directory. Inspect failures rather than ignoring them.
4. In the feature environment run tdar_final_cache, then run classical_final_selection and verify_classical_final in the public environment.
5. Run rotation_final, rotation_final_reference and verify_rotation_coverage with `--run research/runs/20260916_rotation_final`; verify the cached features and archive identities as recorded in the audit ledgers.
6. The complete aggregate-analysis program also reads the development-stage runs and full-network control. Those full artifacts are indexed in artifact_index.json rather than shipped. With those available, run analyze_coverage_study, verify_coverage_statistics and write_coverage_manuscript. This distinction prevents presenting a summary-only report rebuild as a fresh scientific replication.

## Interpretation boundaries

No primary superiority claim survived the two-test Holm correction. The activity selector did not reproduce its development advantage. Secondary coverage and omitted-gesture effects are descriptive. Data subsets, scores and optimization settings must not be silently changed. Public-device offline findings do not establish custom-hardware or clinical performance. The discrete log-ratio rotation selector is implemented; the older continuous Schur-information proposal is not, and its general superiority was not tested.
"""
(B / "README.md").write_text(readme)
(O / "REPRODUCIBILITY.md").write_text(readme)
manifest = {
    str(p.relative_to(B)): sha(p)
    for p in sorted(B.rglob("*"))
    if p.is_file() and p.name != "MANIFEST_SHA256.json"
}
(B / "MANIFEST_SHA256.json").write_text(json.dumps(manifest, indent=2) + "\n")
zip_path = O / "EMG_Study_Reproducibility.zip"
with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
    for p in sorted(B.rglob("*")):
        if p.is_file():
            z.write(p, p.relative_to(B))
with zipfile.ZipFile(zip_path) as z:
    assert z.testzip() is None
    for name, h in manifest.items():
        assert hashlib.sha256(z.read(name)).hexdigest() == h
(O / "bundle_verification.json").write_text(
    json.dumps(
        dict(
            passed=True,
            files=len(manifest),
            zip_bytes=zip_path.stat().st_size,
            zip_sha256=sha(zip_path),
            scope="archive integrity and included-file hashes; no clean-environment training rerun",
            created_utc=datetime.now(timezone.utc).isoformat(),
        ),
        indent=2,
    )
    + "\n"
)
print(len(manifest), zip_path.stat().st_size)
