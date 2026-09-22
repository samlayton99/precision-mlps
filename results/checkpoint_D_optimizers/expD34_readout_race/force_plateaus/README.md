# Effective-force plateaus: checkpoint audit and queued interventions

The completed audit identifies a useful distinction. In ordinary GD, the hard
degree-9 target has a weak, slowly decaying, almost directionally fixed effective
slope force. A frozen-Jacobian calculation from update 100k predicts its force
vector at 600k with 1.7% median relative error across five seeds. This calculation
fails on several targets that recover. In Adam, a quiet-looking sparse norm
trace can hide rapid force reversals: the adjacent-update effective-force cosine
is negative in 59 of 65 windows starting at 600k. These observations narrow the
mechanisms worth studying; they do not establish a permanent gamma barrier.

This is an **interim evidence report**. The saved-state audit and dense windows
are complete. Long continuations, paired interventions, independent-seed
confirmation, and numerical controls remain queued on Runpod. No results from
those queued stages are claimed here. The user chose to retain the Runpod queue
when all five GPUs were occupied; Hazy access is not being pursued.

## What was measured

All 13 targets from the [GD/Adam study](../adam_force_extension/README.md) are
included: sine, Runge, degrees 3, 4, 5, and 9, mixed sine, localized sine, chirp,
and four cubic/degree-9 mixtures. The five existing seeds give 65 cases per
optimizer. The checkpoint audit covers 20k, 100k, 200k, 400k, and 600k updates:
650 states. Dense windows resume each case for 512 consecutive updates from
100k and 600k, preserving all Adam moments and component histories: 260 windows.

The model, random readout initialization, half-MSE, width 177, FP64 arithmetic,
2048 training midpoints, and equal physical learning rate 0.002 are unchanged.
The complete empirical residual is used. There is no Taylor approximation to
tanh and no truncation of the force to ten polynomial modes. Polynomial modes
are useful diagnostics and define the additional loss interventions; they do
not replace the activation.

The key force is still

$$
F_a=T_a e_H.
$$

Here $e_H$ is the residual beyond its constant and linear components, and
$T_a$ maps that residual into the slope gradient after accounting for the
coarse residual required by instantaneous coarse balance. Thus $F_a$ already
includes the balanced coarse contribution. The tracking correction is the
additional slope gradient caused by departure from that balance. Under GD,
the actual slope update is minus 0.002 times their sum.

The new measurements separate questions that a force norm alone cannot answer:

| Quantity | Meaning |
|---|---|
| $|F_a|$ | How much effective slope gradient is available now? |
| $|e_H|$ and $|F_a|^2/|e_H|^2$ | Is the residual small, or is its coupling to slopes weak? The residual norm uses the empirical mean-square metric. |
| $-operatorname{sign}(a)^T F_a/(sqrt{N}|F_a|)$ | Is the force aligned with increasing mean absolute slope? Negative means an inward contribution. |
| $|F_a|^4/(Nsum_j F_{a,j}^4)$ | How broadly is the force distributed over neurons? |
| Adjacent and two-update vector cosines | Does a stable norm conceal force reversals? |
| $|sum_n v_n|/sum_n|v_n|$ | How much component motion survives cancellation within a window? Here $v_n$ is the force or actual component step being measured. |
| Signed component movement and zero-crossing correction | How does each component contribute to the actual change in mean absolute slope? |

For Adam, raw gradients and actual component steps are separate measurements.
Each component has its own additive first-moment history, but all use the
optimizer's actual shared second-moment denominator. This exactly accounts for
the realized step. It does not describe what an independently modified Adam
optimizer would do.

## Figure 1: which force plateaus actually persist in the existing window?

![Sparse effective-force norms](figures/01_force_norms.png)

Each line is one seed. These five checkpoint samples are a useful overview,
especially for GD, but they cannot establish smooth Adam dynamics between
samples. The vertical scales differ between targets.

The degree-9 GD force changes little from 400k to 600k: its median norm ratio
is 0.947 and its median endpoint cosine is 0.99992. Degree 5 is similarly
persistent, with norm ratio 1.008 and cosine 0.99928. Their fine residual norms
remain about 0.866, so these plateaus are not caused by successful fine fitting.
For degree 9 at 600k, the squared coupling lies between $3.7\times10^{-11}$ and
$1.25\times10^{-10}$ across the five seeds.

The recovering targets are different. Over the same late interval, median
force ratios are 12.79 for sine, 0.454 for degree 3, 0.591 for mixed sine,
0.234 for localized sine, and 1.46 for chirp. A force can revive and later
decline as fitting progresses. There is no common, nearly constant late force
across all targets. See [the per-case windows](diagnostics/windows.csv), rather
than interpreting a visually flat part of a log plot as a universal regime.

## Figure 2: Adam has oscillations that the sparse plots miss

![Adjacent-update force directions and cancellation](figures/02_dense_vectors.png)

Each point describes a 512-update window beginning at 600k. Blue is GD's raw
effective force; orange is Adam's raw effective force; green is the actual
Adam step attributed to that force after momentum and adaptive scaling.

- **Upper left:** adjacent vector cosine. GD stays almost exactly at +1.
  Adam's raw effective force has median cosine −0.981 across cases; 59/65
  cases have negative window-median cosine.
- **Upper right:** two-update cosine. Adam's median is +0.988. Negative
  one-update and positive two-update cosines indicate an alternating component,
  rather than slow, smooth rotation.
- **Lower left:** net vector divided by accumulated vector length. The median
  is approximately 1 for GD, 0.0103 for Adam's raw effective force, and 0.181
  for Adam's effective component steps. Momentum changes how much motion
  survives, but it does not remove all oscillation: 35/65 effective-step windows
  still have negative median adjacent cosine.
- **Lower right:** norm variability within each window. Adam's median raw-force
  coefficient of variation is 1.55, versus 0.00036 for GD. Adam is not merely
  reversing a vector with perfectly constant magnitude; these windows also
  contain substantial amplitude variation.

Tracking cancels even more strongly in these windows. Adam's median coherence
is 0.000798 for raw tracking and 0.00715 for its actual tracking component steps.
The corresponding effective-step coherence is 0.181. This is consistent with
the earlier observation that tracking can dominate path length while contributing
much less net motion. Coherence alone is neither signed gamma growth nor a
causal removal experiment; both the component's amplitude and its direction
relative to the slopes still matter.

These are observations at learning rate 0.002, not a claim about all Adam
settings. [Every window](dense/windows.csv) and the [adjacent scalar samples](dense/samples.csv.gz)
are retained, including the nonoscillating cases.

## Figure 3: a useful checkpoint prediction, with a clear domain of failure

![Frozen-Jacobian forecast errors and signed alignment](figures/03_forecasts_alignment.png)

The left panel asks a concrete question: given only the parameters and residual
at 100k, can we predict the force vector at 600k? The blue calculation freezes
the **full model Jacobian**, evolves the resulting linear model by exact
discrete GD for 500k updates, and projects its predicted gradient using the
same coarse balance. It fits no decay rates or future coefficients. Orange
simply keeps the initial effective force vector constant. The error is
$\|F_{\mathrm{pred}}-F_{\mathrm{actual}}\|/\|F_{\mathrm{actual}}\|$; an error of
1 means the prediction error is as large as the actual force vector.

| Target | Frozen-Jacobian median error | Constant-force median error |
|---|---:|---:|
| Degree 9 | 1.70% | 15.37% |
| Degree 5 | 7.92% | 10.72% |
| Degree 4 | 40.26% | 37.44% |
| Runge | 73.93% | 248.66% |
| Sine | 99.43% | 99.43% |
| Degree 3 | 406.22% | 409.28% |

The frozen calculation captures much of the degree-9 persistence and some of
degree 5. It fails to predict the force evolution during recovery in sine,
mixed/localized sine, and chirp, whose median errors are approximately 97–100%.
Degree 3 fails even more strongly. Thus a fixed feature geometry is a useful
local description of some stalled trajectories, and an inadequate explanation
of recovery. It is not the final theoretical objective.

These are retrospective checkpoint forecasts on the existing five seeds.
Their formulas use only the issuance checkpoint, but this is still the
discovery evidence. The queued new-seed runs will write predictions before
generating the subsequent trajectory.

The right panel explains why a stable force need not grow useful scales.
Degree 9 has negative signed outward alignment in all five seeds, with median
−0.135. Degree 5 is positive in all five, with median +0.162, but its force is
tiny. The median change in mean gamma from 400k to 600k is −0.0000339 for degree
9 and +0.0000635 for degree 5. These are distinct failure mechanisms: slow
contraction and extremely slow expansion. Neither is explained adequately by
the norm alone. Large mean gamma is also not, by itself, proof of a useful
feature geometry; retain the independent approximation-error measurements.

## Figure 4: why is the force changing slowly?

![Contributions to GD force-amplitude change](figures/04_force_drivers.png)

Write the effective force as $F_a=R(\theta)r$, where $r$ is the complete sample
residual and $R(\theta)$ includes the exact slope Jacobian and coarse-balance
projection. Its derivative is the sum of four measured vectors:

$$
\dot F_a =
\underbrace{(\dot R)_{a,b}r}_{\text{shape-map change}}
+\underbrace{(\dot R)_{c,d}r}_{\text{readout-map change}}
+\underbrace{RJv_{\mathrm{effective}}}_{\text{residual change from effective steps}}
+\underbrace{RJv_{\mathrm{tracking}}}_{\text{residual change from tracking steps}}.
$$

$a,b$ are slopes and hidden biases; $c,d$ are readout weights and output bias;
$J$ is the full model Jacobian. The two velocities sum to the actual parameter
velocity. The map derivatives include the changing coarse-balance projection.
For each vector $D_i$, the plotted scalar is $F_a^T D_i/\|F_a\|^2$. Their sum is
$d\log\|F_a\|/dt$, with $t=0.002n$. Positive values replenish the force norm;
negative values deplete it. Points across seeds are not a time series, and the
vertical scale is symmetric logarithmic.

Degree 9 shows **slow depletion**, rather than large replenishment canceling
large depletion. The shape-map, readout-map, and effective-residual terms all
reduce the norm in every seed; the tracking term is tiny. At 600k the median
total logarithmic rate is about $-1.35\times10^{-4}$ per unit $t$. Over 200k
updates, $\Delta t=400$, so keeping that local rate constant would imply about
a 5% decline, consistent with the observed late window. This is a scale check,
not a bound on how long the rate will remain valid. The vector derivative's
norm is also about $1.4\times10^{-4}$ of the force norm per unit $t$, so a
hidden rapid rotation is not responsible for this plateau.

Degree 5 differs: shape-map change replenishes the force while effective
residual evolution depletes it, producing small net growth in four seeds and
decline in one. The terms are still individually slow. Sine and degree 3 have
much faster, seed-dependent contributions and stronger cancellation. In degree
3, residual evolution can **increase** the slope-force norm; residual fitting
does not automatically imply that every projected gradient block decreases.

This decomposition measures mechanisms without proving future bounds on them.
For Adam it remains an exact directional derivative along the next step, but
it should not be read as a continuous norm-decay rate: a finite update can
carry $F_a$ nearly to $-F_a$. The dense-window evidence is essential there.

## Working hypothesis and decisive next comparisons

For the degree-9 GD states, the picture is now more specific than simply
“readout learning suppresses slopes.” A large hard residual survives, its
coupling to slopes is exceptionally weak, and the effective force is dominated
by correcting generated output. At 600k, the generated-output force norm is
$5.3\times10^{-6}$ to $9.7\times10^{-6}$, whereas the target's effective force
is $1.9\times10^{-10}$ to $5.7\times10^{-10}$. Generated output contributes
inward mean motion in all five seeds. The independent
[generated-mode audit](../../../../docs/d34_coarse_balance_stagnation.md#7-what-the-audit-and-continuations-establish) resolves which lower modes
produce that opposition. This is compatible with a slowly evolving feature
geometry and the successful checkpoint forecast.

The queued interventions test the causal parts of this hypothesis:

| Comparison | What an informative outcome would establish |
|---|---|
| Joint GD versus frozen readout | Whether ongoing readout evolution materially maintains or changes the force after the checkpoint. Freezing also changes the active coarse balance, which is measured separately. |
| Clamp only the fine-residual input to effective slope force | Whether residual evolution is needed for depletion/revival, while the current force map and ordinary tracking correction remain active. This is artificial diagnostic dynamics. |
| Remove only slope tracking | Whether a small instantaneous tracking contribution has a consequential accumulated or indirect effect. This is also artificial dynamics. |
| Lower-mode loss weights 0, 0.1, 1, 10 | Whether penalizing generated degrees 2 through $k-1$ is what maintains inward opposition or delays access to the hard target. Weight 1 is ordinary GD. |
| New seeds and longer ordinary training | Whether the observed persistence and checkpoint predictions survive independent trajectories and longer horizons. |

The goal remains to understand what prevents scale acquisition and what enables
recovery. A permanent barrier is not implied by these measurements. Neither is
causal irrelevance of tracking in Adam. A useful next theorem would need to
control the force map and generated-mode opposition over a future interval,
while retaining an explicit route by which those conditions fail during
recovery. The existing movement-budget theorem remains conditional until such
control is established.

## Queue, budget, and verification

All numerical analysis, tests, dense training windows, and plotting ran in
CPU-only Runpod Slurm allocations. The local MacBook was used for editing,
transfers, Git, and image inspection. At the [recorded status check](slurm_status.txt),
the GPU campaign had consumed **0 GPU-hours**; [queue metadata](queue.json)
records the job dependencies and allocation budget.
Slurm's two-GPU user limit applies across its arrays and the other agent's jobs.

| Stage | Runpod job | Cases / allocation | Maximum GPU-hours |
|---|---|---|---:|
| 600k → 6m ordinary continuations | 1025 | 78 cases; two 1-hour allocations | 2 |
| Paired discovery probes | 1036 | 37 cases × 2 forks × 3 seeds; six 30-minute allocations | 3 |
| Independent ordinary trajectories | 1038 | 10 cases × seeds 20–24; five 12-minute allocations | 1 |
| Independent probes from 600k | 1040 | 37 cases × 5 seeds; five 12-minute allocations | 1 |
| Half-step and doubled-grid controls | 1041 | 37 cases × 2 forks × 2 controls; four 20-minute allocations | 1.333 |
| Unallocated reserve | — | Explicitly bounded resumes or follow-up checks | 1.667 |

The total ceiling is 10 GPU-hours, including compilation, failed jobs, and
allocated idle time. The initially queued GPU allocations total 8 1/3 hours.
Jobs save partial checkpoints near their deadlines; there are no automatic
retries. Completion gates 1037 and 1039 stop dependent stages if source bundles
are incomplete, nonfinite, or unresolved. CPU endpoint audits 1042 and 1043
produce data for subsequent interpretation. A queued stage is not a result.

Force, projection, target/generated, and derivative identities close to a
maximum absolute error of $6.84\times10^{-13}$ over the 650 audited states;
there are no unresolved coarse solves. Dense continuations have maximum
accounting error below $10^{-15}$, with no failed or unresolved cases. The
unchanged optimizer kernel reproduces the dense observer's final state.

Runpod job 1034 passed 91 D34 tests. Job 1035 passed five focused intervention
and endpoint tests, including exact disk resume, frozen readout blocks,
weighted-loss gradients against autodiff, and signed-motion reconstruction.
Earlier plotting failed because pandas was absent; the final figure code uses
the existing NumPy environment. An initial full-test invocation lacked explicit
source-commit metadata in the copied checkout; it passed after supplying that
metadata. These were CPU jobs and consumed no GPU budget.

The [implemented protocol](../../../../experiments/expD34_readout_race/README.md)
and `plateau*.py` entrypoints reproduce the analysis. Source/input hashes are in
[the checkpoint audit](diagnostics/audit.json) and [the dense audit](dense/audit.json).
The queued training source is pinned to `0f3385d`. Figure tables retain all
target/seed outcomes; no unsuccessful forecast was excluded.
