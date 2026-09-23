# Starting sources

These are starting references, not a completed literature review. Verify implementation details against source versions used in the experiment.

- GRABMyo v1.1.0: https://physionet.org/content/grabmyo/1.1.0/
- GRABMyo motion order: https://physionet.org/content/grabmyo/1.1.0/MotionSequence.txt
- EMGBench paper/project: https://emgbench.github.io/
- EMGBench code: https://github.com/jehanyang/emgbench
- Original CORAL explanation/code: https://github.com/VisionLearningGroup/CORAL
- LibEMG: https://github.com/LibEMG/libemg
- Historical project source: GSDSEF Notebook (4).pdf, supplied by Charles. Not bundled here; use the original file if available locally.

For new papers record the exact claim supported, method, data/splits, limitations, and relevance to the proposed contribution. Do not treat an abstract or search snippet as a full-paper review.

## 2026-09-06 — Initial publication-strategy screening

This was a targeted web screening, not a full systematic literature review.

- GRABMyo official metadata: https://physionet.org/content/grabmyo/1.1.0/ — confirms 43 participants, three recording days, repeated trials and 2048 Hz. Loader, channel derivations, rest-label mapping, and archive contents still require verification.
- EMGBench project: https://emgbench.github.io/ — NeurIPS 2024 benchmark spanning nine datasets and adaptation/generalization tasks. Project/abstract inspected; exact protocol and implementation comparison pending. Shows substantial overlap with a generic benchmark contribution.
- Touko et al., Lightweight Test-Time Adaptation for EMG-Based Gesture Recognition (January 2026 preprint): https://arxiv.org/html/2601.04181v1 — inspected abstract/introduction and dataset/method overview; TCN, adaptive normalization, statistical alignment/replay and few-shot meta-learning on NinaPro DB6. Full split/metric/code assessment and peer-review status unverified. Closely overlaps lightweight adaptation framing.
- Le et al., Quantifying Covariate Shift and Improving Electromyography Driven Gesture Recognition with Calibration and Sample Selection (AIM 2024): https://ras.papercept.net/images/temp/AIM/files/0257.pdf — inspected abstract, introduction and selection-method passages. RI selection chooses prior recorded conditions/repetitions; two public datasets. Do not equate it automatically with requesting the next unseen target gesture; review its references and other acquisition literature. Results include small/no gains over unselected adaptation depending on dataset.
- NeurIPS 2026 reviewer guidelines: https://nips.cc/Conferences/2026/ReviewerGuidelines — inspected quality, significance, originality and negative-results criteria; methodological novelty is not the only contribution form, but a failed experiment alone is insufficient.
- ICASSP official scope: https://signalprocessingsociety.org/events/2026-ieee-international-conference-acoustics-speech-and-signal-processing-icassp — signal processing including ML and biomedical applications. Used for venue fit, not a future deadline.
- TNSRE editorial scope: https://www.embs.org/tnsre/for-reviewers/editorial-policy/ — neural systems, rehabilitation engineering and assistive technology.
- IMWUT/UbiComp: https://www.ubicomp.org/ubicomp-iswc-2026/imwut-papers/ — wearable/ubiquitous computing and journal-to-conference publication model; workshop/poster/demo tracks are distinct.
- UIST author guide: https://uist.acm.org/2026/author-guide/ — technical HCI contribution requires evidence appropriate to its claims; user studies are not universally required.
- EMBC full-paper call: https://embc.embs.org/2026/papers/ — biomedical engineering conference with a biomedical signal-processing theme. Used for route comparison only.

## 2026-09-06 — Pivot screening; not a completed novelty review

- GRABMyo official metadata: https://physionet.org/content/grabmyo/1.1.0/ . License, motion order, readme, converter, checksum list and one header downloaded; five files verified. Log: research/runs/20260906_grabmyo_metadata/retrieval.json. Signal decoding remains pending.
- Original window/error/delay study: https://pmc.ncbi.nlm.nih.gov/articles/PMC4241762/ . Reviewed relevant methods passages; window-duration tradeoffs already have prior work, so varying duration alone is not novel.
- EMG-UP author abstract: https://arxiv.org/abs/2509.21589 . Cross-user personalization using contrastive learning and pseudo-label fine-tuning. Detailed comparable splits/code and publication status unverified.
- ReactEMG Stroke author overview: https://roamlab.github.io/reactemg-stroke/ . Head-only/LoRA/full adaptation comparisons already exist for EMG. Overview inspected; different population/task; full independent methods review pending.
- Few-shot prototype adaptation: https://pmc.ncbi.nlm.nih.gov/articles/PMC13086972/ . Method excerpts inspected; prototype/meta-learning adaptation is existing work. The text's sessions/repetitions terminology requires checking against the underlying datasets before reusing its protocol or comparing results.
- Causal benchmark overlap lead: https://www.preprints.org/manuscript/202606.1365 . Not fully reviewed; no claim that the proposed safeguard is novel follows from this screening.
- ICSPS2026 dates/scope and abstract route: https://www.icsps.org/ and https://www.icsps.org/sub.html . Signal Processing Systems, distinct from similarly named conferences.
- SOICT2026 dates/scope: https://soict.org/ and https://soict.org/submission/ . Prior proceedings corroborated by publisher: https://link.springer.com/conference/soict . Detailed current paper instructions, eligibility and acceptance rate unverified.

## 2026-09-06 — Practical motivation and closest-work check

See PROBLEM_AND_SCOPE.md for the evidence table, inspected sections and limits. Added primary sources: Szymaniak et al.2022 https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2022.1061201/full ; Eddy et al.2024 https://www.frontiersin.org/journals/bioengineering-and-biotechnology/articles/10.3389/fbioe.2024.1463377/full . The first publisher page had an open-access error but indexed publisher methods/conclusions were returned; the second full text was accessible. Nature dataset paper page and direct PMC/PubMed opens had errors/challenges; no access bypass attempted.

WFDB4.3.1 physical-signal API checked at https://wfdb.readthedocs.io/en/latest/wfdb.html .

Additional overlap lead: Tirsu et al.2026 https://www.mdpi.com/1424-8220/26/12/3862 — publisher methods excerpts inspected; compact CNN for a different lower-limb movement task. It reinforces that compact CNN use alone is established, not a closest cross-day personalization comparison. No adoption of its protocol or unmeasured embedded estimates.
