# Why this problem is worth studying

Updated September 6, 2026. Focused evidence check, not an exhaustive novelty review.

The practical problem is repeated calibration: an EMG recognizer trained previously may work less well when the user wears the device on another day. The useful outcome is to understand and reduce the labeled recording effort required to recover performance. This matters to researchers building wearable muscle-based interfaces and myoelectric control systems. Our public healthy-participant experiment cannot establish benefit to prosthesis users or live control.

## Evidence and overlap

| Primary paper | Evidence and implications |
| --- | --- |
| Szymaniak, Krasoulis and Nazarpour, [Recalibration of myoelectric control with active learning](https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2022.1061201/full), 2022 | Studies reducing labeling effort using active learning with LDA. Its offline simulated-interaction study supports recalibration burden as a real research concern. Selecting calibration data is already an established idea. Publisher-indexed methods/conclusion passages reviewed; not a reproduction. |
| Eddy, Campbell, Bateman and Scheme, [Big data in myoelectric control](https://www.frontiersin.org/journals/bioengineering-and-biotechnology/articles/10.3389/fbioe.2024.1463377/full), 2024 | Evaluates large cross-user models for discrete gestures, including fine-tuning and data-collection effort. Section 3.8 compares standard and contrastive updates while freezing recurrent layers. Its six dynamic gestures and larger participant pool differ from our held-gesture cross-day design. Introduction and transfer-learning methods/results inspected. A frozen encoder or calibration curve alone is not our novelty. |
| Pradhan, Jiang and He, [GRABMyo v1.1.0 and linked dataset paper](https://physionet.org/content/grabmyo/1.1.0/), 2022 paper / 2024 corrected release | Provides repeated recordings across days for 43 participants and a corrected label mapping. Enables an actual cross-day test. Official metadata, conversion guidance and local records inspected; publisher paper page returned an access error this turn. |
| Touko et al., [Lightweight Test-Time Adaptation for EMG-Based Gesture Recognition](https://arxiv.org/html/2601.04181v1), January2026 preprint | Close overlap: compact TCN, normalization/alignment/replay and few-shot adaptation on NinaProDB6. Architecture, dataset and validation sections inspected. The paper already discusses compute and adaptation-buffer costs. Our shared pretraining and GRABMyo setting require a specific contribution beyond a dataset change. Publication status and code reproduction unverified. |

## What we will build

A compact neural recognizer trained across permitted participants, then personalized to a participant and a later day. Compare no update, head-only updating, full fine-tuning, and classical retraining at identical calibration budgets and fixed scoring trials. Report seconds of recorded calibration and participant-level performance.

The previously suggested update safeguard is not currently justified as the defining contribution. Keep it optional until development evidence and related methods warrant it. Identify the specific finding or improvement before claiming novelty. This is a useful research direction, not proof that a comparison alone will satisfy a venue.

Use only public recordings. Charles explicitly excluded his custom-device data. No artificial corruption or extra model component is needed to motivate this first study. Target submission materials before September 10 and judge readiness from actual runs.
