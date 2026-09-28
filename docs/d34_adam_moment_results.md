# Adam moment interventions: a small effect without a persistent two-phase model

Reducing the tracking contribution to Adam's second-moment input often
increases mean normalized slope over the next 20,000 updates. The effect
replicates in direction across most targets, but its magnitude is small and
its sign is not universal. Keeping the checkpoint forces frozen, or repeating
two checkpoint-derived phases, does not predict the ensuing neuronwise
contrast. The evidence supports studying the evolving optimizer state; it
does not yet supply an Adam acquisition-rate theory.

**Terms used below.**

| Symbol or term | Meaning |
|---|---|
| $F,Q$ | Effective and tracking gradients in the full fine-complement, GD-reference decomposition $g=F+Q$ |
| $m,v$ | Adam's first and second moment buffers |
| $m$-only, $v$-only, both | Reduce $Q_a$ by 10% in the indicated moment inputs |
| $\lambda_j=|a_j|/64$ | Normalized slope for construction resolution 128 |
| $\Delta\lambda$ | Intervention's vector of normalized slopes minus ordinary Adam's vector at the same update |

## 1. Example: changing the second-moment input helps often, not always

For the sine target, the $v$-only intervention increases mean $\lambda$ by
$2.57\times10^{-6}$ in development and $2.37\times10^{-6}$ in confirmation
after 20,000 updates. Runge instead changes from $+8.11\times10^{-5}$ to
$-1.42\times10^{-6}$. A shared tendency therefore needs a statement of its
exceptions and size, not just a positive example.

Both cohorts fork at update 600,000 and run to 620,000: development uses seed
0, confirmation seed 1, with the same 13 targets and width 177. Each target
has ordinary Adam and three moment interventions, giving 52 branches per
cohort. These are two seeds of the same target collection; confirmation is
not a new held-out function family or a statistical population estimate.
The settings are $\eta=0.002$, $\beta_1=0.9$, $\beta_2=0.999$ and
$\epsilon=10^{-8}$. Incoming parameters, optimizer age and all moment buffers
are preserved.

**Theory defining the intervention.** Let $P_a$ select slope coordinates and
$P_a^T$ insert them. At every branch's own current state, use

$$
g^{(m)}=g+(\alpha_m-1)P_a^TQ_a,\qquad
g^{(v)}=g+(\alpha_v-1)P_a^TQ_a,
$$

in $m^+=\beta_1m+(1-\beta_1)g^{(m)}$ and
$v^+=\beta_2v+(1-\beta_2)(g^{(v)})^2$.
The arms are $(\alpha_m,\alpha_v)=(0.9,1),(1,0.9),(0.9,0.9)$.
Other coordinate inputs use the ordinary gradient. All parameters and moment
buffers subsequently evolve together.

The second-moment driver includes $(F_a+\alpha_vQ_a)^2$, including its cross
term. Reducing $\alpha_v$ does not necessarily reduce that square. This is
also not a rescaling or reset of the accumulated $v$ buffer.
Here $Q$ comes from the coarse-equilibrium projection of the **full** residual
gradient, as implemented in `adam_forces.split`. Unlike the finite-degree-65
GD experiments, the remaining fine complement is not split into a retained
fine channel and an omitted-mode channel. The projection metric is the
Euclidean GD metric; it is not recomputed using Adam's denominator.

**Measured response.** Each entry below is the signed mean contrast
$10^6\operatorname{mean}(\Delta\lambda)$ at 20,000 additional updates,
shown as development / confirmation. Positive means larger slopes than the
matched ordinary-Adam branch, not necessarily outward motion from the fork.

| Target | $m$-only | $v$-only | Both |
|---|---:|---:|---:|
| sine | 2.797 / 1.024 | 2.569 / 2.372 | 3.189 / 3.793 |
| runge | 4.323 / 1.782 | 81.119 / −1.416 | 15.895 / 6.860 |
| moment3 | 0.250 / 2.670 | 1.754 / 0.688 | 2.250 / 1.339 |
| moment5 | −1.910 / −0.827 | 2.519 / 0.935 | −0.164 / 1.382 |
| moment9 | −5.413 / −1.451 | 16.100 / 3.890 | 10.334 / 2.752 |
| mixed_sine | 8.375 / −1.432 | 8.201 / 0.146 | 17.823 / −0.887 |
| localized_sine | −12.949 / 8.383 | −7.298 / 14.578 | −2.165 / 22.455 |
| chirp | −3.234 / 1.849 | 2.148 / −6.805 | 0.152 / −0.818 |
| moment4 | 1.997 / 1.736 | 1.049 / −2.995 | 0.881 / 1.928 |
| blend_m010 | −1.099 / 7.351 | 2.178 / 2.303 | 0.845 / 1.897 |
| blend_p001 | −1.606 / −1.509 | 3.669 / 1.558 | 2.137 / −0.396 |
| blend_p010 | −0.993 / 0.249 | 3.084 / 0.748 | 2.053 / 0.918 |
| blend_p030 | −0.981 / −2.250 | 0.240 / 3.924 | −1.203 / 1.784 |

**Summary across the 13 targets at that same horizon.** Vector norms retain
neuronwise differences that can cancel in the signed mean.

| Arm | Positive mean effects, dev / confirmation | Same sign across seeds | Median signed mean contrast, dev / confirmation | Median $\|\Delta\lambda\|_2$, dev / confirmation |
|---|---:|---:|---:|---:|
| $m$-only | 5/13 / 8/13 | 8/13 | $-9.93\times10^{-7}$ / $1.02\times10^{-6}$ | $1.16\times10^{-4}$ / $2.48\times10^{-4}$ |
| $v$-only | 12/13 / 10/13 | 9/13 | $2.52\times10^{-6}$ / $9.35\times10^{-7}$ | $1.41\times10^{-4}$ / $1.95\times10^{-4}$ |
| Both | 10/13 / 10/13 | 7/13 | $2.05\times10^{-6}$ / $1.78\times10^{-6}$ | $1.67\times10^{-4}$ / $1.62\times10^{-4}$ |

The strongest replicated directional tendency is therefore in the $v$-only
arm. Neither the numerator-only response nor combining both changes gives a
universal sign. These contrasts do not establish precision fitting, useful
geometry, or newly acquired threshold crossings.

## 2. Example: the correct initial response does not make a persistent driver

For confirmation sine, $v$-only produces a mean contrast of
$2.37\times10^{-6}$ at 20,000 updates. The frozen-driver forecast predicts
$2.96\times10^{-2}$; the two-phase forecast predicts $1.78\times10^{-3}$.
Both get its sign but greatly overpredict its magnitude. This failure is
visible well inside the primary window, without invoking million-step tests.

**Theory being tested.** Both forecasts preserve the complete incoming Adam
state and bias-correction age. One repeats the fork's $F,Q$; the other
alternates those forces with forces evaluated after one virtual ordinary
Adam update. Every arm receives the same prescribed phases. These were issued
before its continuation and were not refitted. They test the persistence of
the imposed drivers, as described in the
[Adam mechanism section](d34_mechanism_rate_theorems.md#5-adam-alternating-force-can-spend-motion-without-acquiring-scale).

For a forecast contrast $\widehat{\Delta\lambda}$, define

$$
\mathrm{skill}=1-\frac{\|\widehat{\Delta\lambda}-\Delta\lambda\|_2^2}
                         {\|\Delta\lambda\|_2^2}.
$$

Positive skill beats predicting no intervention effect. No denominator
cutoff was introduced; the contrasts in these comparisons are nonzero.
Large negative scores can reflect both small actual effects and large
forecast errors, so the absolute example above matters.

**Forecasts beating zero effect, across all three interventions and 13 targets
(39 contrasts per cohort and horizon).**

| Additional updates | Frozen, development / confirmation | Two-phase, development / confirmation |
|---:|---:|---:|
| 1 | 39/39 / 39/39 | 39/39 / 39/39 |
| 10 | 4/39 / 0/39 | 17/39 / 22/39 |
| 1,000 | 4/39 / 1/39 | 7/39 / 1/39 |
| 20,000 | 0/39 / 0/39 | 0/39 / 0/39 |

At 20,000 updates the $v$-only mean sign is predicted correctly for 11/13
development and 10/13 confirmation cases by either model, yet every vector
forecast loses to zero effect. The two-phase model improves some immediate
responses but fails as a persistent quantitative explanation. Scalar temporal
statistics do not resolve this gap: the saved dense diagnostics cover only
the first 2,000 steps and do not retain the raw force-vector phases.

## 3. What this changes in the mechanism picture

These Adam states are not the small-tracking regime of the ordinary-GD
conditional theorem. Across ordinary branches and both cohorts, tracking
accounts for 78.3–99.2% of accumulated raw channel-norm activity and
82.4–99.2% of accumulated processed channel-norm activity. These unsigned
shares do not say which channel produces net scale acquisition.

For example, confirmation sine's mean normalized change is
$2.09\times10^{-5}$. Its recorded effective signed contribution is
$4.92\times10^{-5}$, tracking contributes $-7.83\times10^{-4}$, and the
absolute-value crossing correction is $+7.55\times10^{-4}$. Their cancellation
explains why large activity cannot be read as equally large outward motion.
The crossing correction is a geometric accounting term, not another force.

The supported next theory must evolve geometry, forces and moment buffers
together. The intervention results motivate studying the second-moment input,
while the forecast rejection rules out treating the observed initial phases
as stationary over this window. It does not identify a unique missing feedback
term or prove that denominator growth alone barriers acquisition.

## Verification and source evidence

The [development](../results/checkpoint_D_optimizers/expD34_readout_race/mechanism_refinement/adam/verification_dev.json)
and [confirmation](../results/checkpoint_D_optimizers/expD34_readout_race/mechanism_refinement/adam/verification_confirmation.json)
checks establish bitwise preservation of all fork parameters, moments,
channel histories and counts for all 104 branches. Independently reconstructed
first updates agree with the issued updates to $1.78\times10^{-15}$ in
parameters. A [separate comparison](../results/checkpoint_D_optimizers/expD34_readout_race/mechanism_refinement/adam/analysis/first_step_audit.json)
with the executed GPU first steps gives
maximum discrepancies $1.78\times10^{-15}$ in parameters and
$2.07\times10^{-15}$ across the first-moment histories; counts agree exactly.
These checks distinguish floating-point implementation differences from the
subsequent forecast failures.

Both continuations completed all 20,000 updates with no unresolved coarse
projection. Their maximum moment-reconstruction and step-reconstruction norm
defects are $1.93\times10^{-16}$ and $2.25\times10^{-17}$, respectively.
The [development tables](../results/checkpoint_D_optimizers/expD34_readout_race/mechanism_refinement/adam/analysis/dev/contrasts.csv)
and [confirmation tables](../results/checkpoint_D_optimizers/expD34_readout_race/mechanism_refinement/adam/analysis/confirmation/contrasts.csv)
contain every target, arm and horizon; adjacent `endpoints.csv` and
`summary.json` preserve the activity diagnostics and forecast counts.
The confirmation run and analysis completed as jobs 1255 and 1257.
No new training or forecast fitting was performed for this review.
