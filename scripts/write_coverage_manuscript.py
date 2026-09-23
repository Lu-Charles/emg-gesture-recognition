"""Generate a results-backed manuscript and scientific figures from audited statistics."""
from pathlib import Path
import json,re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[1];O=R/'research/expanded_study_20260916'
def main():
 s=json.loads((O/'statistics.json').read_text());fig=O/'figures';fig.mkdir(exist_ok=True)
 plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':160,'savefig.bbox':'tight'})
 def one(key,**conditions):return next(z for z in s[key] if all(z[k]==v for k,v in conditions.items()))
 def pct(x):return f'{100*x:.2f}'
 def ci(z):return f"{100*z['mean']:+.2f} [{100*z['ci95'][0]:+.2f}, {100*z['ci95'][1]:+.2f}]"
 def save(name):plt.savefig(fig/(name+'.png'));plt.savefig(fig/(name+'.svg'));plt.close()
 gr=s['final_cnn_replay_k2_vs_frozen']['accuracy'];go=s['final_cnn_replay_k2_vs_frozen']['other_active_accuracy'];se=s['final_selection_primary']
 # Primary contrasts: show participants, then mean and participant interval.
 f,axes=plt.subplots(1,2,figsize=(10,3.5))
 for ax,z,title in zip(axes,[gr,se],['GRABMyo: replay minus frozen','SeNic: active pair minus random pair']):
  d=np.array(z['per_person'])*100;ax.scatter(d,np.linspace(-.18,.18,len(d)),s=24,color='#7c929d',alpha=.85);ax.errorbar(z['mean']*100,.45,xerr=[[100*(z['mean']-z['ci95'][0])],[100*(z['ci95'][1]-z['mean'])]],fmt='o',color='#0c637a',capsize=4,lw=2);ax.axvline(0,color='#ba6043',ls='--');ax.set_yticks([0,.45],['Participants','Mean + 95% CI']);ax.set_ylim(-.3,.75);ax.set_xlabel('Trial accuracy difference (percentage points)');ax.set_title(title+f"\n{z['n']} reserved participants",fontsize=11)
 f.tight_layout();save('primary_contrasts')
 f,axes=plt.subplots(1,2,figsize=(10,3.6))
 for m,label,col in [('cnn_frozen','Frozen','#303c46'),('cnn_naive','Head update','#c65d36'),('cnn_replay','Head + replay','#0c637a'),('cnn_l2','Head + L2','#9b7d35')]:
  axes[0].plot([1,2],[100*one('final_neural',method=m,steps=0 if m=='cnn_frozen' else 100,k=k)['accuracy'] for k in [1,2]],'o-',label=label,color=col)
 for m,label,col in [('frozen','Frozen','#303c46'),('pooled','Pooled LDA','#c65d36'),('cosine_rotation','Rotation only','#9b7d35'),('cosine_gain','Rotation + gain','#0c637a'),('profile_fractional','Profiled rotation + gain','#776096')]:
  axes[1].plot([1,2],[100*one('final_rotation',method=m,k=k)['accuracy'] for k in [1,2]],'o-',label=label,color=col)
 for ax,title in zip(axes,['GRABMyo: two 5-second recordings','SeNic: two complete recordings']):ax.set_xticks([1,2],['One gesture','Two gestures']);ax.set_ylabel('Trial accuracy (%)');ax.set_title(title,fontsize=11);ax.legend(fontsize=8,loc='best');ax.grid(axis='y',alpha=.2)
 f.tight_layout();save('coverage_by_method')
 f,ax=plt.subplots(figsize=(8.4,3.5));xx=np.arange(4);w=.25
 methods=['cnn_frozen','cnn_naive','cnn_replay','cnn_l2']
 for j,(metric,label,color) in enumerate([('calibrated_recall','Calibrated gestures','#b7c8ce'),('other_active_accuracy','Other active gestures','#0c637a'),('accuracy','All 17 classes','#d99558')]):
  ax.bar(xx+(j-1)*w,[100*one('final_neural',method=m,steps=0 if m=='cnn_frozen' else 100,k=2)[metric] for m in methods],w,label=label,color=color)
 ax.set_xticks(xx,['Frozen','Head update','Head + replay','Head + L2']);ax.set_ylabel('Trial accuracy / recall (%)');ax.set_ylim(0,105);ax.legend(ncol=3,fontsize=8,loc='upper center');ax.set_title('GRABMyo held-out cohort: calibrated success versus vocabulary retention',fontsize=11);f.tight_layout();save('vocabulary_retention')
 f,ax=plt.subplots(figsize=(8.4,3.4));labels=['Random pairs','High activity','Cosine diversity','Identifiability','Fixed pair'];xx=np.arange(5)
 for cohort,offset,color,n in [('development',-.16,'#9bafb8',6),('final',.16,'#0c637a',24)]:
  values=[one(cohort+'_rotation',method='cosine_gain',k=2)['accuracy']]+[one(cohort+'_rotation_selection',method='cosine_gain',selector=r)['accuracy'] for r in ['active','diverse','identifiable','fixed']];ax.bar(xx+offset,np.array(values)*100,.3,label=f'{cohort.capitalize()} (n={n})',color=color)
 ax.set_xticks(xx,labels);ax.set_ylabel('Trial accuracy (%)');ax.set_ylim(0,100);ax.legend();ax.set_title('SeNic: selection gains weakened in independent evaluation',fontsize=11);f.tight_layout();save('selection_replication')
 # Tables are created from the same immutable participant summaries as the figures.
 nt='| Method | One gesture | Two gestures | Other gestures, K2 | Calibration recall, K2 | Macro F1, K2 |\n|---|---:|---:|---:|---:|---:|\n'
 for m,label in [('cnn_frozen','Frozen CNN'),('cnn_naive','Head update'),('cnn_replay','Head + replay'),('cnn_l2','Head + L2')]:
  a=one('final_neural',method=m,steps=0 if m=='cnn_frozen' else 100,k=1);b=one('final_neural',method=m,steps=0 if m=='cnn_frozen' else 100,k=2);nt+=f"| {label} | {pct(a['accuracy'])} | {pct(b['accuracy'])} | {pct(b['other_active_accuracy'])} | {pct(b['calibrated_recall'])} | {pct(b['macro_f1'])} |\n"
 ct='| Method | One gesture | Two gestures | High-activity pair, K2 |\n|---|---:|---:|---:|\n'
 for m,label in [('tdar_frozen','Frozen TDAR+RMS LDA'),('amplitude_gain48','Amplitude gain + LDA'),('tdar_gain16','TDAR gain + LDA'),('tdar_pooled','TDAR pooled LDA')]:
  a=one('final_classical',method=m,k=1);b=one('final_classical',method=m,k=2);h=one('final_classical_selection',method=m,selector='active');ct+=f"| {label} | {pct(a['accuracy'])} | {pct(b['accuracy'])} | {pct(h['accuracy'])} |\n"
 rt='| SeNic method | One gesture | Two random gestures | High-activity pair |\n|---|---:|---:|---:|\n'
 for m,label in [('frozen','Frozen'),('pooled','Source + calibration LDA'),('gain_only','Gain only'),('cosine_rotation','Cosine rotation'),('cosine_gain','Cosine rotation + gain'),('profile_integer','Gain-profiled integer rotation'),('profile_fractional','Gain-profiled fractional rotation')]:
  a=one('final_rotation',method=m,k=1);b=one('final_rotation',method=m,k=2);h=one('final_rotation_selection',method=m,selector='active');rt+=f"| {label} | {pct(a['accuracy'])} | {pct(b['accuracy'])} | {pct(h['accuracy'])} |\n"
 st='| Dataset-specific primary contrast | Difference and 95% CI (pp) | Holm-adjusted p |\n|---|---:|---:|\n'
 for label,z,key in [('GRABMyo: K2 replay - frozen',gr,'final_cnn_replay_k2_vs_frozen'),('SeNic: activity - random pair',se,'final_selection_primary')]:st+=f"| {label} | {ci(z)} | {s['primary_tests'][key]['holm_p']:.4f} |\n"
 frozen=one('final_neural',method='cnn_frozen',steps=0,k=2);naive=one('final_neural',method='cnn_naive',steps=100,k=2);replay=one('final_neural',method='cnn_replay',steps=100,k=2)
 cos=one('final_rotation',method='cosine_gain',k=2);act=one('final_rotation_selection',method='cosine_gain',selector='active');ref=one('final_rotation_reference',method='target7');coverage=s['final_rotation_coverage_contrast'];p7=one('final_rotation_reference',method='pooled7')
 full='| Update | Steps | One gesture | Two gestures | Other gestures, K2 |\n|---|---:|---:|---:|---:|\n'
 for m in ['full_naive','full_replay']:
  for steps in [25,100]:
   a=one('development_full_network',method=m,steps=steps,k=1);b=one('development_full_network',method=m,steps=steps,k=2);full+=f"| {m.replace('_',' ')} | {steps} | {pct(a['accuracy'])} | {pct(b['accuracy'])} | {pct(b['other_active_accuracy'])} |\n"
 headsteps='| Update | Steps | One gesture | Two gestures |\n|---|---:|---:|---:|\n'
 for m in ['cnn_naive','cnn_replay','cnn_l2']:
  for steps in [25,100,300]:
   a=one('final_neural',method=m,steps=steps,k=1);b=one('final_neural',method=m,steps=steps,k=2);headsteps+=f"| {m.replace('cnn_','')} | {steps} | {pct(a['accuracy'])} | {pct(b['accuracy'])} |\n"
 text=f'''# Brief EMG Calibration Does Not Guarantee Vocabulary Preservation
## An evaluation across session and electrode-position changes
Charles Lu | Research manuscript draft

### Abstract
Brief recalibration can improve recognition of the gestures used to update an electromyographic classifier while degrading recognition of the remaining vocabulary. We examine this distinction with an existing high-density EMG failure audit, matched-recording-budget development experiments, and independent evaluation on 15 GRABMyo participants and 24 SeNic participants. Whole trials are separated before windowing; scoring sets remain fixed. GRABMyo compares one versus two active calibration gestures using two five-second recordings, shared-pretrained convolutional models across three seeds, replay, weight regularization, and amplitude-preserving classical controls. SeNic compares two-recording calibration under physical electrode-position changes using circular registration, channel gains, pooling, and source-only gesture selection. On the reserved GRABMyo cohort, frozen, ordinary head-update, and replay accuracies were {pct(frozen['accuracy'])}%, {pct(naive['accuracy'])}%, and {pct(replay['accuracy'])}%, respectively, for two-gesture calibration. Replay's paired improvement over the frozen model was {ci(gr)} percentage points; the corresponding change on noncalibrated active gestures was {ci(go)}. On SeNic, rotation-plus-gain accuracy increased by {ci(coverage)} points when the same two-recording budget covered two gestures instead of one. However, high-activity gesture selection did not establish an advantage over an average pair: {ci(se)} points. Its {pct(act['accuracy'])}% accuracy remained below the {pct(ref['accuracy'])}% seven-gesture target-trained reference, which required more recordings. A conditional template analysis and synthetic controls demonstrate why fitting one positive spatial template cannot identify both unrestricted channel gains and a spatial transformation. The study supports evaluating calibration success, omitted-class retention, acquisition cost, and shift assumptions separately. Neither primary improvement claim met the protocol-defined Holm-adjusted 0.05 criterion. The study does not establish a new universally superior adaptation algorithm.

Keywords: electromyography; calibration coverage; domain adaptation; catastrophic forgetting; electrode displacement; reproducibility.

## 1. Introduction
Surface EMG models can become less reliable when electrodes are reattached or recordings are collected on a later day. A short calibration protocol is attractive, but success on a prompted gesture is an incomplete measure of system performance. A classifier that increasingly emits that gesture can obtain excellent recall while making more mistakes elsewhere. This behavior must be distinguished from a fixed permutation of the label encoding and from useful correction of a shared sensor transformation.

Our motivating CSL-HDEMG experiment showed very high recall on a single calibrated gesture alongside substantial errors on other gestures. The original experiment is retained as a diagnostic rather than reused as an untouched test set. The present extension asks: at the same number of new recordings, does collecting different gestures help, which update rules preserve the rest of the vocabulary, and can enrolled source data identify useful calibration gestures before target data are collected?

Prior work already motivates restricted-gesture calibration, spatial registration, and forgetting control. Spatial Adaptation Layers provide structured transformations for biosignal arrays [1]. Chan et al. use benchmark classes to estimate changes affecting a larger gesture set [2]. Replay-based EMG learning and recent lightweight test-time adaptation also address temporal change and preservation [3,4]. Thus, neither source replay nor the concept of choosing calibration gestures constitutes our novelty claim. We contribute a focused empirical comparison with fixed scoring identities, explicit omitted-class outcomes, two distinct shift settings, participant-level uncertainty, and held-out falsification of an apparently promising selection rule. We retain the failed candidate and distinguish each method's assumptions and data requirements.

## 2. Study design and data boundaries
Development and independent evaluation are separate. GRABMyo uses 20 participants for shared representation training, eight for development, and 15 for final evaluation. SeNic uses six development participants and 24 reserved rotation participants. The existing four evaluated CSL-HDEMG people contribute only a post-hoc diagnostic. Repeated sessions, gesture choices, optimizer steps and seeds do not increase the independent participant count. Results from the two new evaluation cohorts are analyzed separately rather than pooled into a single 39-person effect.

The machine-readable local protocol was finalized on June 25, 2026. It documents participants, splits, methods, primary contrasts, optimization settings, feature definitions, and statistical analysis. Its SHA256 is 3545c2f35f02e056bd8a2a9ee65f10ca9f0547f58f7f557728f22b4b0cf74d9c. This is not an externally registered study. Comparisons are described as protocol-defined. Development choices and their negative results remain available.

| Evidence component | Development / representation training | Independent evaluation | Interpretation |
|---|---|---|---|
| CSL-HDEMG audit | Previously evaluated study | None newly reserved | Diagnostic, post-hoc |
| GRABMyo | 20 representation + 8 development people | 15 people, two later days each | Natural cross-day variation |
| SeNic | 6 development people | 24 people, ten target positions each | Electrode-position changes; order confounded |

### 2.1 GRABMyo preprocessing and enrollment
GRABMyo version1.1.0 is used with its documented physical signal calibration [5,6]. The workflow reads forearm channels F1-F16 at2048Hz in millivolts, preserves amplitude, and adds no software filtering. Day1 supplies119 enrollment trials per participant. Each later day has three calibration-eligible and four scoring trials per class under the existing seeded trial allocation. The vocabulary contains16 active gestures and rest. Five-second trials produce35 overlapping512-sample windows with256-sample hops, beginning at sample1024. Whole-trial partitions precede window extraction.

Shared models were pretrained for20epochs on the20 representation participants and enrolled on each evaluated participant's day1 recordings for20epochs. Adam uses learning rate0.001 and batch size128. The fixed channel scaler is fitted only on representation-training participants. The network has three temporal convolution layers with32,64,64 output channels; kernel widths9,7,5; strides4,4,2; ReLU activations; global average pooling; and a17-class linear head. It has no batch normalization or dropout. Seeds42,0,1 define three separately pretrained and enrolled recognizers. This is personalized recognition with substantial initial enrollment, not recognition of completely new users without source data.

### 2.2 Matched calibration and scoring
For each target day, a fixed seeded cyclic ordering of the16 active gestures gives16 paired choices. In K1, gesture g supplies calibration-rank1 and rank2 recordings. In K2, g supplies rank1 and the next gesture h supplies rank2. Both conditions contain two complete recordings: ten seconds of recorded signal and nine unique analyzed seconds after the onset exclusion. The first recording is shared across the paired conditions. Both score exactly the same68 complete trials, four per class, using majority voting over35 windows and the lowest-index label for a tie.

The ten-second quantity is recorded data, not wall-clock setup time. Intertrial rest, prompts, transitions, and user effort are not included. Overlapping windows are not independent observations. Whole-trial voting uses several seconds of evidence and does not demonstrate250ms response latency.

### 2.3 Neural and classical controls
Head-only updates keep the enrolled encoder and scaler fixed. Ordinary adaptation minimizes cross-entropy on64 sampled calibration windows per step. Replay adds a64-window source batch sampled uniformly by class, followed by uniform sampling within the chosen class, with equal weights on source and target cross-entropies. Weight regularization adds the summed squared distance from enrolled head parameters with coefficient1. Adam uses learning rate0.001;100steps is primary, with25 and300 retained as sensitivity endpoints. All methods receive identical target minibatches for a given case. Replay uses additional source storage and computation; no equal-compute claim is made. Its source bank contains all4165 enrolled embeddings, approximately1.07MB in float32, rather than a specially optimized tiny buffer.

A separate eight-person development robustness experiment updates the full network, with and without replay, at learning rate0.0001 and25/100steps. It changes both update scope and learning rate; it is not an isolated estimate of the causal effect of unfreezing the encoder and is not independent final evidence.

Classical controls use equal-prior shrinkage linear discriminant analysis (LDA) with a source-only StandardScaler. The48 amplitude features comprise per-channel mean absolute value, RMS, and mean waveform length. The144-dimensional TDAR+RMS set additionally includes zero crossings, slope changes, and four Burg autoregressive coefficients per channel, using the pinned implementation. Amplitude is retained in primary baselines. Source-plus-calibration pooling refits LDA on all source and permitted calibration windows. Gain correction uses class-matched source/calibration template ratios averaged in log space and clipped to[0.5,2]. TDAR gains are estimated from RMS and applied coherently to MAV, waveform length and RMS, without scaling counts or autoregressive coefficients.

The classical selection experiment exhaustively evaluates all256 ordered gesture pairs per participant/day, including16 same-gesture choices. Random two-gesture performance is the exact average of240 distinct ordered pairs, not the best observed pair. Predeclared source-only selectors choose the greatest cosine template diversity, the two highest summed-RMS gestures, or the fixed pair[0,1]. This exhaustive classical schedule differs from the neural16-pair cycle, so selected-pair classical and random-cycle neural differences do not isolate classifier architecture alone.

### 2.4 SeNic rotation experiment
SeNic is pinned to repository commit a4c12f7daab28a80d557677ae8dbcef0d7871ba2 [7]. Session0 is used for the30 rotation participants. Position0 repetitions0 and1 provide14 source trials. At target positions1-10, repetition2 of all seven gestures is always the scoring set. Calibration uses g repetition0 and h repetition1, covering all49 ordered pairs. K1 has seven same-gesture choices; K2 has42 distinct pairs. Each pair has two whole recordings and four analyzed seconds, but complete recording durations vary and are measured from the files.

Eight-channel signals are used in native Myo counts at200Hz. Samples600:1000 give15 overlapping50-sample windows with25-sample hops. TDAR+RMS supplies72 features. Source scaling and LDA match the classical protocol. Cosine registration aligns class-matched RMS templates over circular shifts from-4 to3.875channels in0.125-channel increments, with deterministic nearest-identity ordering. Feature groups are linearly interpolated in channel coordinates. A subsequent shared gain correction is estimated from aligned templates. Ruler angles are never supplied to either the adapter or gesture selector. Interpolating autoregressive feature groups is an engineering approximation, not an exact inverse of shifted raw physiology.

Controls include no correction, gain only, cosine rotation without gain, source-plus-two-trial pooling, and the gain-profiled candidate described below. Seven-recording target-only and source-plus-target LDA provide higher-budget references on the same scoring trials. These references use the source scaler, all seven target gestures at repetition0, and the same equal-prior shrinkage estimator. Their greater calibration-class coverage and data volume are explicit; they are not same-budget competitors.

### 2.5 Conditional mechanism and candidate falsification
Let u_c and v_c be positive source and target amplitude templates for class c. Consider the idealized relation u_c = diag(g) W(phi) v_c, with a shared spatial transform W and freely varying positive per-channel gains g. For one template and any admissible transform having positive output, choosing g_i = u_i / (W(phi)v)_i produces an exact fit. Therefore a single average template does not identify the spatial transform in this model. Locally, n observations with n gain parameters plus spatial parameters admit compensating gain directions. This is a conditional observation, not a new general theorem about all single-gesture adaptation.

With two templates, component-wise log ratios remove shared log gains: log(u_1)-log(u_2) = log(Wv_1)-log(Wv_2). Additional classes may distinguish transforms, but proportional templates remain ambiguous. Constraints on gains, informative within-gesture temporal structure, external geometry, or informative priors can change the conclusion. Gesture-specific effort gains violate the shared-gain assumption. The relation is not an exact description of every operation in the published SAL implementation.

We tested an engineering candidate that searches integer or fractional circular shifts after profiling out mean log gains, then clips fitted gains. One-gesture profiles are deliberately resolved to identity because their unconstrained residual is uninformative. A source-only candidate selects the gesture pair whose log-RMS ratio has the largest minimum squared separation from its seven nonidentity integer rotations. It is compared with activity, cosine diversity and fixed selection. This criterion and gain profiling are treated as hypothesis-driven controls, not an asserted novel algorithm. The candidate did not outperform the simple approach in development and was retained as a negative control in the final evaluation.

A post-hoc synthetic qualification uses500 independent eight-channel problems per scenario, a separately generated known integer rotation, bounded shared channel gains, and seven positive source templates. It compares diverse templates, proportional calibration templates, and gesture-specific gain perturbations. Because this stress test was added after seeing final SeNic results, it is labeled explanatory rather than confirmatory human evidence.

### 2.6 Outcomes and statistical analysis
Primary accuracy is whole-trial correctness. Scoring sets contain the same number of trials per class, so overall accuracy equals macro-average recall. We separately report calibrated-gesture recall, accuracy on noncalibrated active gestures, false assignment of those gestures into the calibration subset, rest recall where available, and macro F1. A larger calibration subset naturally offers more destinations for false assignment; raw K1 and K2 attraction rates are therefore not interpreted as a pure bias contrast. Within-K method comparisons use identical omitted labels and trials.

Each participant's outcomes are averaged over target sessions or positions and calibration choices. GRABMyo averages recognizer seeds within participant. Paired participant differences produce95% t intervals;10000 participant bootstrap resamples with seed20260916 supply a sensitivity estimate. The two protocol-defined primary tests are K2 head-replay minus frozen accuracy on GRABMyo, and source-activity selection minus uniform distinct-pair accuracy for cosine-plus-gain on SeNic. Two-sided paired t-tests are Holm-adjusted across this two-test family. Secondary intervals are descriptive, with no claim of familywise significance. No equivalence or noninferiority margin was specified. Lack of significance does not establish safety or equivalence.

## 3. Results
### 3.1 The original apparent success reflects label attraction, not a simple global scramble
The previously saved CSL-HDEMG predictions were audited across640 calibration cases from four evaluated people. Spatial-plus-gain adaptation achieved96.99% recall on the calibrated gesture but34.51% accuracy on other gestures, versus60.71% for the frozen model. It assigned36.46% of other-class trials to the calibrated label, compared with5.61% frozen. Complete collapse occurred in6/640 spatial-plus-gain cases and67/640 ordinary fine-tuning cases. Thus, the high recall is partly associated with an expanded tendency to predict the calibrated gesture; it does not mean every case labels everything that way.

All1598 available label/filename identities and the saved prediction-vote construction were checked. The optimal global one-to-one label permutation was identity for every method. This argues against a consistent global label scramble, but does not independently authenticate what a participant physically executed. The audit did not replay the original neural checkpoints and is not an independent replication of SAL's published results.

### 3.2 Frozen independent primary comparisons
{st}

Neither primary contrast meets the familywise 0.05 criterion. The95% intervals are individual, unadjusted intervals and are not simultaneous confidence bounds. The GRABMyo interval excludes zero before multiplicity adjustment, but its Holm-adjusted p-value is0.0646; this is not reported as a confirmed primary superiority result.

![Participant-level primary contrasts](figures/primary_contrasts.png)
Figure1. Each gray point is one participant's paired difference. The colored point and whisker show the mean and95% participant t interval. Sessions, gesture choices and model seeds stay grouped within participant. Dataset effects are not pooled.

### 3.3 Cross-day calibration and vocabulary preservation
{nt}
Table2. GRABMyo reserved participants,15people and three recognizer seeds averaged within person. Values are percentages; K denotes the number of distinct calibration gestures in two recordings. Head results use the protocol-defined100steps. Macro F1 is shown on a0-100 scale.

A post-hoc prediction check further separates recall from reliability: among all outputs naming either calibrated gesture, the exact predicted label was correct in {pct(s['posthoc_calibrated_prediction_correctness']['cnn_naive']['fraction'])}% of ordinary-update outputs, versus {pct(s['posthoc_calibrated_prediction_correctness']['cnn_frozen']['fraction'])}% frozen and {pct(s['posthoc_calibrated_prediction_correctness']['cnn_replay']['fraction'])}% with replay. This pools predictions across the fixed cases and seeds; it is a conditional diagnostic, not a participant-mean precision estimate.

Replay versus the frozen model changed noncalibrated active-gesture accuracy by {ci(go)} points. This quantity is essential: a higher full-vocabulary score can coexist with deterioration on omitted classes if the calibrated classes improve enough. The observed head-replay K2-versus-K1 coverage difference was {ci(s['final_replay_coverage'])} points. Results at all retained step counts appear in AppendixA; the best step for each participant was not selected. Ordinary head updating at25steps reached {pct(one('final_neural',method='cnn_naive',steps=25,k=2)['accuracy'])}% overall, compared with {pct(naive['accuracy'])}% at100steps. The adverse primary result is therefore conditional on the declared optimization budget, not evidence that every brief supervised update must fail.

![Coverage and update method](figures/coverage_by_method.png)
Figure2. Covering more gestures interacts with the chosen update mechanism. Each line uses the same target scoring trials across K. GRABMyo is natural cross-day variation; SeNic concerns recorded electrode positions. Their accuracy levels are not a direct device comparison.

![Vocabulary retention](figures/vocabulary_retention.png)
Figure3. The calibrated and omitted gesture groups must be read together. Calibrated recall alone cannot establish successful vocabulary-wide adaptation. GRABMyo K2, final cohort, primary100steps.

{ct}
Table3. Strong classical GRABMyo controls. K2 columns average all240 ordered distinct pairs; the activity column uses a pair chosen from source data only. All methods preserve amplitude information. These comparisons have different representation-training access from the shared CNN and do not prove intrinsic superiority of either model family.

### 3.4 Measured-position shifts and selection replication
{rt}
Table4. SeNic final cohort,24people; one scoring trial per class at each of ten positions. Percentages are trial accuracy. The random-pair value averages all42 ordered pairs. Every method uses the identical scoring files.

For cosine-plus-gain, two-gesture coverage improved random-pair accuracy over one-gesture coverage by {ci(coverage)} points at the same two-recording count. The gain increment over cosine rotation alone was {ci(s['final_rotation_gain_vs_cosine'])} points. Registration's benefit should not be attributed automatically to extra gain parameters.

The development activity-selector advantage was {ci(s['development_selection_primary'])} points across six people. It weakened to {ci(se)} points across24new people. The final individual effects include{se['positive']} positive and{se['negative']} negative participant differences. A favorable sign count does not remove sensitivity to the magnitudes of harms and gains. The fixed pair and candidate selectors are reported even when their final point estimates happen to exceed the chosen activity rule; final results are not used to switch the deployed rule.

![Selection replication](figures/selection_replication.png)
Figure4. The apparently promising source-activity rule did not establish its development advantage in the reserved cohort. All bars use cosine registration plus gain; the gain-profiled adapter is separately reported in Table4.

Source-activity cosine-plus-gain required an average{act['recorded_seconds']:.2f} recorded seconds and achieved{pct(act['accuracy'])}% accuracy. The seven-recording target-only reference required{ref['recorded_seconds']:.2f} recorded seconds and achieved{pct(ref['accuracy'])}%; source-plus-target pooling with the same seven recordings achieved{pct(p7['accuracy'])}%. This is an acquisition/performance tradeoff, not demonstrated equivalence of two- and seven-gesture calibration. Complete recorded durations include each file's recorded rest but exclude unrecorded interaction overhead.

### 3.5 Mechanistic controls and update-scope sensitivity
In the synthetic shared-gain/diverse-template setting, two templates uniquely recovered the known integer shift in all500examples and reconstructed omitted log-amplitude templates to numerical precision. One template admitted eight equally fitting integer shifts. Proportional pairs also admitted eight minima, despite using two gesture labels. With gesture-specific gain perturbations, the two-template shift was correct in84.8% of examples but mean omitted-template log-RMS error remained0.6764. Identifying a shift therefore does not guarantee that a shared-gain model describes all gestures. These simulations validate a conditional argument; they do not establish the cause of every empirical error.

The gain-profiled candidate underperformed simpler registration in the real-data comparison. Algebraic identifiability under an idealized model is insufficient to establish an effective real adapter or gesture selector. The profile residual is computed before gain clipping, whereas the deployed engineering correction applies clipped gains; exact theoretical template-fit statements refer to unrestricted gains.

{full}
Table5. Development-only full-network sensitivity, eight people, seed42. These results use learning rate0.0001, unlike the head-only0.001experiment. All checkpoints were replayed on CPU from cached raw windows after MPS training. They are controls on update instability, not additional independent final participants. In particular, ordinary full-network updating at25steps reached79.53% with two gestures versus77.48% for the same-seed frozen development recognizer. Update strength and scope matter; this result cautions against a universal adaptation-failure claim.

## 4. Discussion
The original question has a qualified answer. A model can recognize the calibration gesture frequently because its prediction region for that label has expanded. The diagnostic evidence shows this behavior, while the global label-mapping check does not support a simple systematic scramble. Neither observation makes the empirical paper invalid. The scientific weakness would be claiming broad recognition improvement from calibrated recall alone, or portraying an established forgetting mechanism as a new discovery.

The expanded study changes the interpretation of brief calibration in three ways. First, a strong frozen recognizer must be retained: gains over a weak amplitude-only baseline do not establish improvement over shared pretraining. Second, preserving source behavior through replay is a useful comparator, but source preservation does not guarantee accuracy on omitted target gestures. Third, collecting different gestures can supply useful information, yet the value depends on the update model and the nature of the shift. Source-only activity and diversity heuristics must be tested on reserved people rather than judged by their development averages.

Spatial correction has an interpretable role on an eight-channel ring after electrode-position changes. Its success should not be generalized to all natural day-to-day variability, nor should its fitted channel shift be treated as an accurate physical angle without a separate angle-estimation analysis. The present comparison intentionally withholds ruler metadata. Gain profiling illustrates how flexible nuisance parameters can make a small calibration set deceptively easy to fit. Simpler model restrictions can transfer better when real physiology violates a flexible model's assumptions.

The strongest contribution is the controlled empirical separation of calibrated success, omitted-class preservation, gesture coverage, and recording burden, together with a held-out failure of the selected source heuristic. It is not a new replay algorithm, a new general identifiability theorem, or a state-of-the-art leaderboard claim. Recent alignment/replay work [4], restricted benchmark-class calibration [2], and broader EMG distribution-shift benchmarks [8] are complementary precedents. Our configurations have not been matched to every published method, so numerical superiority over those methods is not claimed.

### 4.1 Limitations
Final cohorts contain15 and24 independent people, not the number of repeated model evaluations. GRABMyo has only two later scoring days per person; SeNic has one scoring repetition per class/position. SeNic acquisition order can confound position with elapsed time or fatigue, and a nominal full rotation can return close to the original orientation. Class-balanced, segmented trials and long voting intervals simplify the task relative to continuous use with transitions and rejection of unknown gestures. There is no amputee cohort, physical prosthesis test, custom-hardware validation, or clinical outcome.

The final neural study covers one compact convolutional architecture, a fixed head optimizer, and three training seeds. Full-network sensitivity is development-only and uses a different learning rate. Replay can draw on all source embeddings, and representation pretraining on20other participants is a substantial resource. Pair enumeration measures an expectation under a specified gesture-choice policy rather than a user study of actual choice behavior. The development-based SeNic interval-width estimate derived from only six development people was optimistic; the observed final interval is wider. No threshold for practical noninferiority was prespecified. A statistically uncertain improvement cannot be sold as a guaranteed safe update.

All proposed choices and their negative results remain visible, but the research is not a systematic comparison of every adaptation method. New algorithm development must not use these final cohorts and then relabel them as untouched evidence. The current draft requires expert scientific and related-work review before submission; experiment completion is not a prediction of acceptance.

## 5. Conclusion
High calibrated-gesture recall is not sufficient evidence of useful EMG recalibration. With fixed recorded-data budgets and fixed scoring trials, coverage and preservation depend on the update rule and shift setting. The reported evaluation supports reporting omitted-class outcomes, strong frozen/replay/classical controls, and actual recording costs. It does not establish that an intuitive source-selected gesture pair reliably beats ordinary pair selection, nor that two recordings replace full-vocabulary calibration. These negative boundaries are part of the result.

## Data, code and disclosure
Public data are obtained from GRABMyo version1.1.0 and the pinned SeNic repository [5-7]; raw recordings are not redistributed in the analysis bundle. The local reproducibility package includes code, documented protocol, environment records, participant summaries, verification ledgers and figure sources. It is not a completed clean-machine rerun or a public repository release. Raw caches, trained checkpoints and full prediction archives remain in the canonical project with checksum manifests. Separate verification programs reconstruct saved-model predictions and rederive metrics; this is computational checking by the same research workflow, not an external laboratory replication. No new public release or submission has been made.

OpenAI Codex assisted with literature checking, experimental software, execution, computational verification, analysis and drafting this expanded manuscript. Reported measurements were computed from the recorded labels and saved predictions. Charles Lu should verify the interpretation and disclosure against the intended venue's requirements before submission. No new participant recruitment or custom-device measurements were performed for this extension.

## Appendix A. Prespecified head-update sensitivity
{headsteps}
TableA1. All retained25/100/300step endpoints on the final GRABMyo cohort.100steps remains primary regardless of these outcomes.

## Appendix B. Evaluation identities and audit boundaries
GRABMyo development IDs are3,5,7,20,21,22,24,38; final IDs are1,6,8,9,10,14,15,16,26,29,31,32,33,34,42. SeNic development IDs are0,1,14,18,24,25; final IDs are2,3,4,5,6,7,8,9,10,11,12,13,15,16,17,19,20,21,22,23,26,27,28,29. IDs are dataset-specific and not cross-dataset identities.

Each GRABMyo seed has960final calibration cases and12480metric records. All187425raw embedding windows per seed are reconstructed, and848640saved trial votes per seed are checked. The stronger classical final experiment has7680cases,30720records and2088960trial votes. SeNic has11760two-recording cases,82320records and576240trial votes, plus480seven-recording reference records. Development full-network checking covers2048checkpoints. These counts describe verification scope, not independent sample size.

All5355GRABMyo final trial files have checksum and manual-decoder comparison records;45raw trials are independently reread for cache spot checks. All5544SeNic final trial files have hashes and complete metadata, with72raw feature spot checks and pinned archive verification. TDAR spot checks reuse the pinned Burg implementation, while amplitude formulas are separately rederived. Checkpoint replay is not an independent refit of every covariance estimator.13focused risk tests cover trial leakage, final-cohort access, sampling, parameter preservation, synthetic ambiguity, and numerical selection properties.

## References
[1] J. Pereira, M. Alummoottil, D. Halatsis, D. Farina. Spatial Adaptation Layer: Interpretable Domain Adaptation For Biosignal Sensor Array Applications. arXiv:2409.08058v2,2025 (first version2024). https://arxiv.org/abs/2409.08058

[2] P. Chan, Q. Li, Y. Fang, L. Xu, K. Li, H. Liu, D. S. Yeung. Unsupervised Domain Adaptation for Gesture Identification Against Electrode Shift. IEEE Transactions on Human-Machine Systems52,1271-1280,2022. https://tas-lab.org/publication/2022-unsupervised-domain-adaptation-gesture-identification/

[3] X. Zhang, T. Li, M. Sun, L. Zhang, C. Zhang, Y. Zhang. Replay-Based Incremental Learning Framework for Gesture Recognition Overcoming the Time-Varying Characteristics of sEMG Signals. Sensors24(22),7198,2024. https://pmc.ncbi.nlm.nih.gov/articles/PMC11598278/

[4] N. Touko, M. O. A. Ellis, C. Capone, A. Burrello, E. Donati, L. Manneschi. Lightweight Test-Time Adaptation for EMG-Based Gesture Recognition. arXiv:2601.04181v2,June23,2026. https://arxiv.org/html/2601.04181v2

[5] N. Jiang, A. Pradhan, J. He. Gesture Recognition and Biometrics ElectroMyogram (GRABMyo), version1.1.0. PhysioNet,2024. https://doi.org/10.13026/89dm-f662

[6] A. Pradhan, J. He, N. Jiang. Multi-day dataset of forearm and wrist electromyogram for hand gesture recognition and biometrics. Scientific Data9,733,2022. https://doi.org/10.1038/s41597-022-01836-y

[7] B. Zhu, D. Zhang, Y. Chu, Y. Gu, X. Zhao. SeNic: An Open Source Dataset for sEMG-Based Gesture Recognition in Non-Ideal Conditions. IEEE Transactions on Neural Systems and Rehabilitation Engineering30,1252-1260,2022. https://doi.org/10.1109/TNSRE.2022.3173708

[8] J. Yang, M. Soh, V. Lieu, D. J. Weber, Z. Erickson. EMGBench: Benchmarking Out-of-Distribution Generalization and Adaptation for Electromyography. NeurIPS Datasets and Benchmarks,2024. https://papers.neurips.cc/paper_files/paper/2024/file/59fe60482e2e5faf557c37d121994663-Paper-Datasets_and_Benchmarks_Track.pdf

[9] T. Pollard et al. PhysioNet as a global platform for biomedical research. Nature Health1,792-795,2026. https://doi.org/10.1038/s44360-026-00096-z
'''
 text=re.sub(r'(?<=\d)(?=(?:steps|people|epochs|windows|trials|seconds|examples|channels|participants|records|cases|new|later|active|other|complete|target|ordered|final|source|raw|saved|separate|class|integer|shared|positive|negative|recording|gesture|development|scientific|different|independent|frozen|checks|Hz|MB|ms|layers|evaluated|representation|convolutional|paired|retained|repeat|primary|calibration|metric|per|point|equal|channel|float|experiment|Adam|years|problems|model|reference|seeds|outputs|first|second|third|year|recordings|epochs))', ' ',text)
 text=re.sub(r'(?<=[0-9]),(?=[A-Za-z0-9])',', ',text)
 text=re.sub(r'\b(version|Position|position|repetition|rank|sample|samples|seed|day|session|Session|Figure|Table|Appendix|coefficient|size|rate|widths|strides|layers|rep|from)(?=\d)',r'\1 ',text)
 text=re.sub(r'\b(primary|all|across|from|with|on|in|for|of|the|and|at|over|approximately|retained|contains|supplies|has|uses|provided|only|each|every|to|by)(?=\d)',r'\1 ',text)
 text=re.sub(r',(?=\w)',', ',text)
 (O/'Expanded_EMG_Manuscript.md').write_text(text)
 (O/'tables.json').write_text(json.dumps(dict(neural=nt,classical=ct,rotation=rt,primary=st,full_network=full,step_sensitivity=headsteps),indent=2)+'\n')
 print('Manuscript words',len(text.split()))
if __name__=='__main__':main()
