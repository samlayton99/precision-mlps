# SSBroyden metric memory and geometry acquisition

This experiment tests whether restoring the prescribed physical learning metric
at measured moments helps a network acquire useful geometry. The primary runs
start with physical-Xavier slopes and reference-scaled Xavier readouts. All
readouts, output bias, and slopes train; centers and corrected halos remain fixed.
No primary run receives the construction bandwidth or solved readout coefficients.

The model, initialization, coordinate maps, FP64 objective, pinned SSBroyden
implementation, and accepted-step correction reuse D35 and D06. Individual and
neighboring coordinates encode the same initial physical model. The primary
targets are sine and mixed sine; selection seeds are 0–1 and confirmation seeds
are 2–4. The new campaign has a hard limit of ten allocated H200 GPU-hours, with
at most two GPUs simultaneously and every remote computation scheduled by Slurm.

## Residual accessibility

For a fixed reference residual $r$ and the native readout matrix $A(\lambda)$,

$$
Q_\tau=\frac{\|e^{-\tau AA^T}r\|^2}{\|r\|^2},\qquad
G_\tau=1-Q_\tau=\frac{b^T f_\tau(C)b}{r^Tr},\quad
C=A^TA,\quad b=A^Tr,\quad f_\tau(x)=\frac{1-e^{-2\tau x}}x.
$$

The continuation is $f_\tau(0)=2\tau$. Computing $G$ directly avoids cancellation
when almost none of the residual is learnable. The derivative of the matrix
function uses divided differences with analytic coincident-eigenvalue limits,
instead of differentiating singular vectors. Independent SVD and finite-difference
checks determine whether a particular diagnostic is resolved.

Horizons are $\tau=K/\|A(\lambda_0)\|_2^2$ for $K=20,000$ and $100,000$. Their
normalization stays fixed along a run. These are readout-flow diagnostic budgets,
not SSBroyden learning rates. The reference residual is held fixed when comparing
candidate geometries. No diagnostic solve changes the live model.

## Metric intervention

At the same state compare $p_N=-Hg$ and $p_S=-sg$, where
$s=\|Hg\|/\|g\|$ matches their native direction norms. The accessibility gain is
$E(p)=\nabla_\lambda G_\tau^T p_\lambda$. A mixture replaces
$H$ by $(1-\beta)H+\beta sI$. When both original directions descend the training
loss, every convex mixture also descends to first order. When $E_N\leq0<E_S$,
a sufficiently large mixture also improves accessibility to first order.

The adaptive rule uses the smallest mixture giving three times the measured
numerical uncertainty at the 100k horizon, without resolved deterioration at
20k. It requires two consecutive qualifying diagnostics and a 100-update
cooldown. Ordinary SSBroyden resumes between interventions. Periodic mixing and
search-history-only replays are separate controls. Emergency non-descent resets
are shared across arms and are not counted as evidence for the allocation claim.

The validity of the curvature approximation and the usefulness of its allocation
are distinct diagnoses. Current Hessian-vector products, local response checks,
and negative-curvature flags distinguish them. A 25% local-model discrepancy is
an operational classification setting, with 10% and 50% sensitivity checks; it
is not a performance criterion or a universal theoretical constant.

The local-model discrepancy compares $d^T\nabla^2L\,d$ with $d^TH^{-1}d$.
For the normalized SSB direction this predicted curvature is computed as
$-d^Tg/\|Hg\|$, avoiding a solve with an ill-conditioned inverse metric. The
separate vector error $\|H\nabla^2L\,d-d\|$ is retained but is not substituted
for the scalar local-model test. The initial baseline records permit the same
scalar prediction to be reconstructed as direction cosine divided by metric scale.

The separate neuron-reset assay changes parameters and therefore tests a different
mechanism. It must include a matched optimizer-state restart, disclose the reset
function jump, and label broader initial bandwidth as injected geometry.

The one-event reset assay branches at update 5000. It selects the lowest 5% of
ordinary neurons by the mean residual-normalized joint readout/bandwidth gradient
at saved updates 4000, 4500, and 5000. This is a three-snapshot selector, not a
continuous utility average. Its five matched branches continue unchanged, restart
optimizer state only, or also replace the selected neurons with zero readouts,
nonzero envelope-scale readouts and small physical-Xavier slopes, or those same
readouts and broader Xavier bandwidths. Every branch gets 20k additional updates.
Sham histories replay the complete, frozen event list from the adaptive run;
they can continue after that source run fails.

An initial control audit found that the inherited non-descent guard classified
a fresh search state's placeholder zero gradient as a bad direction. It overwrote
both requested metric mixtures and history-only restarts with identity before
priming. Those initial intervention trajectories are retained as invalid metric
comparisons. The corrected guard checks only primed states, with a regression
test verifying the direction of the first accepted displacement. Corrected
primary configurations carry `implementation: primed_guard_v2` and start anew.

After the corrected selection runs, the hypothesis is narrowed to finite useful
movement. A positive unit-direction exposure need not produce an appreciable
accepted geometry step. For a native unit direction $d$, let
$a=-g^Td>0$ and $c=d^T\nabla^2L\,d$. When $c>0$, the unconstrained quadratic
model gives length $t_*=a/c$, so predicted accessibility change is
$t_*\nabla_\lambda G_\tau^T d_\lambda$. The follow-up probes compare this
prediction, Gauss–Newton curvature $\|Jd\|^2$, and the actual line-search step.
They also test one tenth and one hundredth of each observed mixture strength.
These are post-selection diagnostic probes, not tuned confirmation settings.
The mixed-target selection trajectories are extended to 100k updates; the
primary analysis remains restricted to the common 20k frontier.

A second, explicitly exploratory timing test uses the earliest resolved single
candidate within the first 100 updates of each N=128 baseline (seeds 0–4).
It does not require persistence across two diagnostics. Each qualifying saved
state has three 20k continuations: unchanged solver state, search-history restart
with the same metric, and the recorded metric mixture. Parameters are identical
at the branch point. This isolates initial acquisition from the much later
switching events of the primary controller. No new target or mixture strength is
selected from the continuation outcomes.

The metric-memory interpretation also has a restricted exact statement. If an
observed secant subspace $U$ is invariant under $H$ and every $s_k,y_k$ lies in
$U$, the SSBroyden rank corrections act only in $U$. Its complementary block
therefore changes as $H_{U^\perp,k+1}=H_{U^\perp,k}/\tau_k$. Its relative prior
anisotropy remains unchanged, even though its overall scalar changes. A focused
test checks this property. The nonlinear network replay measures memory without
assuming that its observed directions form such an invariant subspace.

Every finite scientific run has at least 20k accepted updates. Numerical failures
remain in the ledger. Actual loss, complete-window averages, accessibility,
signed bandwidth movement, physical trajectories, residual frequencies, gradient
decompositions, and cost are evaluated together. More movement alone is not
success; initialization dependence alone is not evidence of bad curvature.
