# Mechanistic model and candidate calibration rule

2026-09-09. Proposed mathematics and implementation specification; no new method has been evaluated. This is a companion to GRADUATE_STUDY_PROTOCOL.md, not a replacement for the actual author-code forward operator.

## A precise ambiguity to test

Let u_c be the source template for gesture c and v_c its target template, with n positive spatial amplitudes. Consider the idealized shared correction

\[
u_c = \operatorname{diag}(g)W_\phi v_c + \epsilon_c,
\]

where g has n position-specific gains and phi has d spatial parameters. Assume interior gains (no clipping), differentiable interpolation and nonzero transformed amplitudes. For one gesture, any admissible nearby phi can be paired with g_i = u_i/(W_phi v)_i to give the same exact template fit. The derivative of the residual with respect to gains is an invertible diagonal matrix. Its n equations therefore have an n+d-parameter Jacobian with at least d null directions. Perfect template fitting does not locally identify the spatial correction under this model.

This is a conditional algebraic observation, not a new general identifiability theorem. It does not say single-gesture adaptation always fails: fixed gains, several informative time-varying templates, constraints, external geometry or a sufficiently strong prior may remove ambiguity. Classification loss is different from exact template reconstruction. The current author implementation also orders gain/normalization/warping differently; the formula is a mechanistic model for a new controlled experiment, not a proof about its implementation.

Stacking C gesture templates produces Cn equations with the same n+d shared parameters. More classes can remove ambiguity if their Jacobians add independent information, but equation counting alone is not sufficient: repeated or proportional templates can remain uninformative. Class-dependent effort gains violate the shared-g model and may defeat the correction even when local rank is full. Those cases belong in the simulation controls and real-data limitations.

## Selection criterion

For each candidate gesture c, compute a Jacobian J_c of predicted template residuals under a declared adapter using **only enrolled source data**, across a small fixed bank of plausible synthetic shifts. Estimate residual scaling from source training repetitions only. In parameter coordinates scaled to physical bounds, write the accumulated information matrix for a chosen set S as

\[
I(S)=\Lambda+\sum_{c\in S}J_c^\top\Sigma_c^{-1}J_c.
\]

If the objective is spatial identification with nuisance gains, use the spatial Schur complement

\[
I_{\phi\mid g}=I_{\phi\phi}-I_{\phi g}I_{gg}^{-1}I_{g\phi}.
\]

Positive prior precision Lambda makes the inverse well-defined; report sensitivity to its strength so the prior cannot silently create apparent data information. Greedily select the next gesture that maximizes the increase in log determinant of this conditional information, averaged across the predeclared synthetic shifts. Use source-side costs or a fixed one-trial cost, and fix tie handling before scores. Never select using unrecorded target gesture templates or observed test accuracy.

Log-determinant experimental design, nuisance-parameter elimination and greedy selection are established techniques. The research question is whether this physically specified information measure selects useful EMG calibration gestures under real session changes, relative to simpler alternatives. Do not advertise the formula itself as a new algorithm.

Compare against seeded random selection, the same fixed gesture sequence for everyone, and farthest-point selection of source RMS templates. The last baseline tests whether any benefit comes merely from template diversity. Exhaustive best-target-scoring selection can be plotted only as a clearly labeled oracle diagnostic and must never enter deployment or main performance claims.

The diagonal-only GRABMyo pilot has no spatial nuisance parameters: do not pretend this spatial criterion applies unchanged. Start the information-based spatial method on SeNic's eight-channel circular geometry and development participants. GRABMyo can test source-template diversity plus regularized classifier adaptation as a separately specified generalization. Keep the common conceptual claim narrower if geometry-specific selection fails to transfer between devices.

## Update and preservation

For a target-only input adapter with a frozen classifier, optimize permitted calibration loss plus a parameter prior toward identity. Source replay cannot constrain a target-only adapter without an explicit shared transformation model. Log gains should be penalized around zero; scale spatial coordinates by their declared bounds before regularization. Record clipping, search boundaries, fitted state and calibration loss independently from transfer accuracy.

For a changing recognizer, a practical comparator is

\[
L(\theta)=L_{\text{target calibration}}(\theta)
+\alpha L_{\text{class-balanced source}}(\theta)
+\beta\|\theta-\theta_0\|^2.
\]

Use the source inputs in source coordinates. Compare plain fine-tuning, replay alone, weight regularization alone and the combination. Hold target presentations and optimizer updates fixed; disclose the extra source forward/backward work. Hyperparameters and any coverage-dependent regularization schedule require grouped development selection. Source preservation is an empirical constraint, not a guarantee for unseen target gestures.

## Falsifiable predictions and necessary controls

1. In noise-free synthetic shared-gain/spatial data, redundant calibration templates permit multiple good fits with different recovered transforms; diverse templates can reduce that ambiguity. Check direct residuals, Jacobian singular values and ground-truth transform error with an independent generator.
2. If poor real transfer is caused mainly by class-label attraction during classifier updates, source replay should reduce false assignment into the calibration subset. It may still harm useful adaptation; report both directions.
3. If source-based information predicts target usefulness, it should outperform random selection at the same complete-recording budget on reserved SeNic people. If only an oracle succeeds, the deployable selection hypothesis is unsupported.
4. If arbitrary class-specific gains dominate session changes, a shared spatial/gain inverse may be inappropriate. Include this misspecification test and retain failure rather than expanding the model until final scores improve.

Before final evaluation, implement these components, prove the stated local rank calculation under its assumptions, compare finite-difference and analytic/autodiff Jacobians, test selection invariance to parameter units after scaling, and run all predetermined development controls. A manuscript should distinguish this conditional analysis, simulations and real public-data results.


## First numerical illustration completed

`research/runs/20260909_template_ambiguity_synthetic/results.json` records a deterministic two-channel example: the identity correction and a mixing correction with compensating gains both exactly match template [1,2], but the latter maps the independent template [2,1] to [1.72727,1.15789]. One-template Jacobian rank is2 for3parameters; two diverse templates give rank3, while proportional templates remain rank2. Centered finite differences agree with the analytic Jacobians. This supports only the stated conditional illustration; larger real-array simulations and a scientific mechanism evaluation remain outstanding.


## September16 outcome update

The frozen empirical extension is complete; see [expanded study](expanded_study_20260916/READ_ME_FIRST.md) and current PLAN/STATUS. GRABMyo15 and SeNic24 reserved cohorts have now been scored. No new superior method was established. The implemented discrete log-ratio rotational-separation selector differs from the original continuous Schur-information proposal above; that proposal was not implemented. Final methods must not be retuned on these outcomes. The expanded manuscript distinguishes prespecified final tests from post-hoc diagnostics/simulations.
