# SeNic neural comparator fidelity

This stage compares a compact amplitude-preserving temporal CNN with the same model trained using the circular-channel permutation principle from Chamberland et al.2023, *Novel Wearable HD-EMG Sensor with Shift-Robust Gesture Recognition Using Deep Learning*.

Primary source: https://unbscholar.dspace.lib.unb.ca/server/api/core/bitstreams/54407f52-28ec-4abf-95ef-33f55e98ff41/content, Sections IV-A/B, TableI, V-D. Original ABSDA adds circular column permutations to an HD-EMG array; principal range ±2 columns, with original samples retained, on16circumferential columns (nominal ±45degrees). Its rotation cohort has three people and physical shifts through±45degrees. Original uses spatial CNN/AA-CNN on processed HD-EMG maps, not our temporal architecture or sparse sensor.

## Applied adaptation

Eight-channel SeNic raw250ms observations from the same3–5second interval. CNN: temporalConv8->32(k7,s2,p3),ReLU;Conv32->64(k5,s2,p2),ReLU;meanpool;concatenate8log1pMAV values;linear72->7. This explicit amplitude path keeps useful amplitude information. It is a routine compact baseline, not a new architecture claim.

Scalar RMS over unique allowed training samples across channels; no demeaning or per-window normalization. A common scalar commutes with circular permutations and preserves relative channel amplitude. Plain and augmented branches use matching initialization, sampled unaugmented minibatch indices, optimizer, and step counts. Each augmented window independently gets shift−1,0,+1 uniformly, approximating the original ±45degree range on an8channel ring. The original finite augmented dataset is implemented here as stochastic equivalent sampling; identical update counts isolate augmentation rather than added compute. SeNic measured channels are not perfectly uniformly spaced; nominal channel permutations are approximations, not measured physical shifts. Always score real unmodified recordings.

A separately reported all8cyclic-shift source-only sensitivity tests wider synthetic exposure; it is not the paper's default augmentation range. No ruler angles, score labels, other participants or other positions supply fitting data.

## Frozen training and evaluation

Seed42;CPU1thread;Adam lr.001/weightdecay.0001;batch64;300steps for source enrollment or scratch,150steps for fine-tuning. Record last-checkpoint fitting accuracy and all loss values; no scoring-based checkpoint selection. Fine-tuning either target-only or pooledsource+target, retaining source scaler; scratchtarget fits target scaler. No shared pretraining or GRABMyo weights. Source records remain a counted common initial enrollment resource; target budgets count fullCSV durations.

This bounded neural stage covers0/7/14trialbudgets; allseven single-gesture cases were covered classically and are not claimed covered by this neural run. Use all original scoring trials unchanged. Two primaryfamilies each get frozen, targetfine, pooledfine and targetscratch controls. Widerrotation sensitivity is zero-target-data only. Intended readout: calibration frontier relative to strongTDAR LDA, augmentation effect on true±45degree shifts and broader angles, source/fitting convergence, and participant failures. First-stage3point margin remains unchanged; do not tune reserved subjects to make it pass.

Limitations: raw temporal backbone/8channels/7gestures/training durations/sampling differ from original HD-EMG setup. Results measure this documented adaptation, not literal original-paper reproduction or state of the art. Strong LDA controls remain primary comparisons. Li2020 shiftcorrection fullprotocol still unverified. A newer2025SWN normalization paper was identified as relevant future work, but is not added silently to this frozen run; amplitude removal needs a separate ablation and source-accurate causal processing.
