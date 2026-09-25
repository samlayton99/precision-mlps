## 4.2. Why joint training acquires scale slowly

The frozen-geometry result leaves an important question: why does joint
training not enlarge the slopes and improve access to the remaining error?
The difficulty persists even when slopes grow. For example, over 100k
additional updates on a smooth step target, the effective fine-gradient norm
increases about 2.2-fold, yet relative output error remains approximately
49%. Explaining this behavior requires a bound on useful movement over a
training budget, rather than convergence to a stationary point.

**The force that survives coarse fitting.** Consider the fully trained
network $q_\theta(x)=b+\sum_{j=1}^W w_j\tanh(\gamma_jx+\beta_j)$ with
half mean-squared loss. Split its residual into affine output, spanned by
$1,x$, and the orthogonal remainder. The full gradient decomposes exactly
as $\nabla L=R+F$: $R$ measures deviation from coarse equilibrium, while
the **effective fine gradient** $F$ includes the compensation needed to
preserve coarse output. After the initial transient, measured tracking
effects become small in our GD runs. We restart the analysis there, while
the population's relative slopes $\lambda_j=h|\gamma_j|$ remain small.
Here $h$ is the reference spacing from Section 3, not the spacing between
learned centers.

**Weak force must remain difficult to reinforce.** Large remaining error
can produce little force when it couples weakly to the current features.
For effective flow $\dot\theta=-F$, the exact identity

$$
\frac{d}{dt}\log\|F\|
=-\underbrace{\|J_Hv\|^2}_{\text{residual relaxation}}
+\underbrace{\kappa_H+\kappa_C}_{\text{geometry and compensation feedback}}
\tag{10}
$$

separates fitting away the error that drives the force from changing the
sensitivity to that error. Here $J_H$ is the non-affine output Jacobian and
$v=F/\|F\|$. Residual relaxation always depletes the force norm; feedback
can reinforce it. We bound accumulated reinforcing feedback using the
population's second output response in direction $v$, weighted by the
residual and coarse compensation. It is computed from output derivatives,
rather than from measured net force growth. It allows the features, readouts,
and compensation to evolve; its formula is given in the appendix.

**Theorem 4.2 (Conditional slow scale acquisition; informal).** Restart
noiseless full-batch GD after the tracking transient, and write $t=n\eta$
for elapsed training time. Let $F_0$ be the effective fine gradient at the
restart and $E_0$ the non-affine residual norm. Suppose accumulated
reinforcing feedback is bounded by $B_t$ throughout the interval. Under
the coarse-regularity and disturbance conditions in the appendix,

$$
\begin{aligned}
\|q_{\theta_n}-f\|^2
&\ge E_0^2-\underbrace{2t e^{2B_t}\|F_0\|^2}_{\text{available error reduction}}
-\Delta_{\rm err}(t),\\
p_\delta(t)
&\le\underbrace{\frac{h^2t^2 e^{2B_t}\|F_0\|^2}{W\delta^2}}_{\text{population acquisition allowance}}
+\Delta_{\rm pop}(t).
\end{aligned}
\tag{11}
$$

Here $p_\delta$ is the fraction of neurons whose relative slope has increased
by at least $\delta$ at any time since the restart. The nonnegative
allowances $\Delta_{\rm err},\Delta_{\rm pop}$ account for tracking and
finite GD steps; both vanish for exact effective flow. Norms use the training
measure, except for Euclidean parameter-gradient norms.

**Proof sketch.** Bounded feedback limits effective-force amplification to
$e^{B_t}\|F_0\|$, up to the stated disturbances. Integrating its square
bounds output improvement; integrating its magnitude bounds collective
parameter travel. Counting neurons that require slope travel at least
$\delta/h$ gives the population bound. The appendix proves the GD statement
using its exact interpolated path.

The theorem allows individual escapes and positive slope growth. If most
slopes start below $\lambda_0$, reaching $\lambda_*$ requires an increment
$\delta=\lambda_*-\lambda_0$. More importantly, the output bound directly
tests whether the network reaches a requested accuracy; it does not require
QUILL's geometry to be necessary for every accurate representation.

**The conditions persist over the measured training budget.** Across 23
targets and two seeds at width 705, twice the restart feedback rate supplies
an accumulated allowance covering every saved prefix over 20k additional
updates. Figure 4 examines six targets through 100k additional updates using
a factor-four allowance. Accumulated feedback uses at most 55% of that
allowance, while force may increase or decrease. The resulting effective-flow
RMS slope-movement bounds remain below $8\times10^{-4}$, far below the
reference increment $0.125$; even the simple error floor in (11) remains
above 39% on all six targets. GD closely follows the effective dynamics.
Measured tracking effects and step refinement support the reduction, with
the remaining numerical qualifications reported in the appendix. These are
empirical checks of the conditions, not interval certificates.

Thus, in the observed post-transient regime, weak residual coupling is
reinforced too slowly to supply useful population movement or output
accuracy within the budget. The conclusion permits evolving geometry and
substantial residual error, and does not require the conditions to persist
indefinitely.

<figure>
  <img src="../results/checkpoint_D_optimizers/expD34_readout_race/population_output/evidence/paper_draft_final/population_persistence_paper.png" alt="Six post-transient runs satisfy the sampled feedback allowance while force changes slowly, population slope movement remains small, and output error remains large." style="max-width: 100%;">
  <figcaption><strong>Figure 4: Persistent weak reinforcement limits population acquisition and output progress.</strong> Width 705, six targets, one seed, restarted at 20k updates and continued for 100k updates at learning rate 0.002. (a) Accumulated directional feedback along effective flow divided by the allowance $4d_{\rm dir}(0)t$; the dashed line marks the condition's boundary. (b) GD (solid) and effective flow (dotted) nearly overlap despite differing signs of force growth across targets. (c) Endpoint RMS changes in relative slopes and their conditional effective-flow upper bounds. (d) Raw relative output errors and conditional effective-flow lower bounds, using the actual attached readouts. Effective-flow time is divided by 0.002 for comparison with GD updates. All bounds use the same factor-four allowance; their empirical evaluation is not a continuous-time certificate.</figcaption>
</figure>

## Appendix: Conditional persistence under evolving geometry

### A. Definitions and exact force decomposition

Let $\mu$ be the equally weighted empirical measure on the training inputs
in $[-1,1]$, with nonzero variance. All output inner products use $L^2(\mu)$;
parameter space has its ordinary Euclidean metric. The target is
$f\in L^2(\mu)$, and

$$
q_\theta(x)=b+\sum_{j=1}^W w_j\tanh(\gamma_jx+\beta_j),\qquad
L(\theta)=\tfrac12\|q_\theta-f\|^2.
$$

All slopes, hidden biases, readouts, and the output bias are trained. Let
$Q_C$ map two orthonormal affine coordinates into output space, and put
$P_H=I-Q_CQ_C^*$. Define

$$
e_C=Q_C^*(q_\theta-f),\qquad e_H=P_H(q_\theta-f),\qquad
J_C=D_\theta e_C,\qquad J_H=D_\theta e_H.
$$

The term "fine" denotes the full orthogonal complement of affine output.
It does not select target-dependent polynomial modes. Assume $J_C$ has full
row rank on the interval under consideration. Then

$$
\begin{aligned}
\Pi&=I-J_C^*(J_CJ_C^*)^{-1}J_C,\\
\ell&=(J_CJ_C^*)^{-1}J_CJ_H^*e_H,\\
F&=\Pi J_H^*e_H=J_H^*e_H-J_C^*\ell,\\
R&=J_C^*(e_C+\ell),\qquad \nabla L=F+R.
\end{aligned}
\tag{A1}
$$

The compensation $-J_C^*\ell$ is part of $F$ and generally persists after
tracking $R$ becomes small. Under $\dot\theta=-F$, coarse output is
preserved because $J_CF=0$. This effective ODE removes tracking while
retaining the evolving coupling among readouts and hidden geometry.

### B. A structural measure of reinforcement

At $F\ne0$, let $v=F/\|F\|$ and define

$$
\begin{aligned}
\kappa_H&=-\langle e_H,D^2e_H[v,v]\rangle,\\
\kappa_C&=\ell\cdot D^2e_C[v,v],\\
d_{\rm dir}&=\|e_H\|\,\|D^2e_H[v,v]\|
+\|\ell\|\,\|D^2e_C[v,v]\|.
\end{aligned}
\tag{A2}
$$

Consequently $\kappa_H+\kappa_C\le d_{\rm dir}$, without a favorable-sign
assumption. The second output response is evaluated in a unit direction;
$d_{\rm dir}$ is not a bound obtained by dividing observed net force growth
by the force. An empirically useful nondecreasing allowance $B_t$ satisfies

$$
\int_0^t d_{\rm dir}(s)ds\le B_t,\qquad B_0=0.
\tag{A3}
$$

**Derivation of (10).** Write $g_H=J_H^*e_H$. Differentiating $F=\Pi g_H$
along effective flow gives

$$
\frac12\frac{d}{dt}\|F\|^2
=-F^*D\Pi[F]g_H-\langle e_H,D^2e_H[F,F]\rangle-\|J_HF\|^2.
$$

Since $\Pi$ is an orthogonal projector, $\Pi D\Pi[F]\Pi=0$.
Differentiating $\Pi J_C^*=0$ and substituting $g_H=F+J_C^*\ell$ gives
$F^*D\Pi[F]g_H=-\ell\cdot D^2e_C[F,F]$. Dividing by $\|F\|^2$
proves (10). Independently,

$$
\frac{d}{dt}\|e_H\|^2=-2\langle J_H^*e_H,F\rangle=-2\|F\|^2.
\tag{A4}
$$

These are identities for the moving nonlinear network. They require no
frozen Jacobian, Taylor truncation of tanh, or equilibrium of the slopes.

### C. Precise theorem, including GD disturbances

**Theorem A.1.** Let $\theta:[0,T]\to\mathbb R^{3W+1}$ be an absolutely
continuous path satisfying $\dot\theta=-F-S$, with the coarse projector
defined along the path. Suppose (A3) holds for every prefix. Set
$s_0=\|F(\theta(0))\|>0$, $E_0=\|e_H(\theta(0))\|$, and define

$$
\begin{aligned}
U(t)&=\int_0^t\|DF[S]\|\,ds,\\
V(t)&=\int_0^t\|S\|\,ds,\\
Z(t)&=\int_0^t[\langle e_H,J_HS\rangle]_+\,ds,\\
D_A(t)&=t e^{B_t}U(t)+V(t),\\
A(t)&=t e^{B_t}s_0+D_A(t).
\end{aligned}
\tag{A5}
$$

All integrals are assumed finite; nondecreasing upper budgets may replace
their exact values. For $\lambda_j(t)=h|\gamma_j(t)|$ and $\delta>0$, let

$$
p_\delta(t)=\frac1W\#\left\{j:
\sup_{0\le s\le t}\bigl(\lambda_j(s)-\lambda_j(0)\bigr)\ge\delta\right\}.
$$

Then, for every $t\le T$,

$$
\begin{aligned}
\|F(\theta(t))\|&\le e^{B_t}[s_0+U(t)],\\
\int_0^t\|\dot\theta(s)\|ds&\le A(t),\\
\|q_{\theta(t)}-f\|^2&\ge
\left[E_0^2-2t e^{2B_t}[s_0+U(t)]^2-2Z(t)\right]_+,\\
p_\delta(t)&\le\min\left\{1,\frac{h^2A(t)^2}{W\delta^2}\right\}.
\end{aligned}
\tag{A6}
$$

In particular (11) holds with the explicit nonnegative allowances

$$
\begin{aligned}
\Delta_{\rm err}(t)
&=2t e^{2B_t}\bigl(2s_0U(t)+U(t)^2\bigr)+2Z(t),\\
\Delta_{\rm pop}(t)
&=\frac{h^2}{W\delta^2}
\bigl[2t e^{B_t}s_0D_A(t)+D_A(t)^2\bigr].
\end{aligned}
\tag{A7}
$$

For a nonzero target, a sufficient condition for relative error to remain
above $\varepsilon$ throughout $[0,T]$ is

$$
2T e^{2B_T}[s_0+U(T)]^2+2Z(T)
<E_0^2-\varepsilon^2\|f\|^2.
\tag{A8}
$$

If fraction $p_0$ starts above $\lambda_0<\lambda_*$, the fraction ever
reaching $\lambda_*$ is at most
$\min\{1,p_0+p_{\lambda_*-\lambda_0}(t)\}$. No maximum-neuron or
force-concentration assumption is needed for these population conclusions.
Centers may move freely; the proof works in the regular coordinates
$(\gamma,\beta,w)$ without division by a slope.

**Proof.** The directional identity implies, along the disturbed path,

$$
D^+\|F\|\le d_{\rm dir}\|F\|+\|DF[S]\|.
$$

Write $\mathcal B(t)=\int_0^t d_{\rm dir}(s)ds$. The integrating-factor inequality
gives

$$
\|F(t)\|\le e^{\mathcal B(t)}
\left[s_0+\int_0^t e^{-\mathcal B(s)}\|DF[S]\|ds\right]
\le e^{B_t}[s_0+U(t)].
$$

At a zero of $F$, the same Dini-derivative inequality holds with the
effective-flow term zero; the nonnegative directional rate can be set to
zero there. Monotonicity of $B,U$ and triangle inequality now yield
$\int_0^t\|\dot\theta\|\le t e^{B_t}[s_0+U(t)]+V(t)=A(t)$.

For output error, (A1) gives the exact disturbed energy identity

$$
\frac{d}{dt}\|e_H\|^2=-2\|F\|^2-2\langle e_H,J_HS\rangle.
$$

Integrating and using $\|q_\theta-f\|\ge\|e_H\|$ proves the third line
of (A6). For acquisition, put $a_j(t)=\int_0^t|\dot\gamma_j(s)|ds$.
Minkowski's inequality gives

$$
\left(\sum_j a_j(t)^2\right)^{1/2}
\le\int_0^t\|\dot\gamma(s)\|_2ds\le A(t).
$$

Each label counted by $p_\delta$ requires $a_j\ge\delta/h$. Counting
those labels proves the final line of (A6). Expanding the two squares yields
(A7), and monotonicity of the budgets proves (A8). $\square$

**Discrete GD is included exactly.** For
$\theta_{n+1}=\theta_n-\eta g(\theta_n)$, $g=\nabla L$, use the linear
interpolant $\theta(t)=\theta_n-(t-n\eta)g(\theta_n)$ on each step. It
satisfies $\dot\theta=-F(\theta)-S$ with

$$
S(t)=R(\theta(t))+g(\theta_n)-g(\theta(t)).
\tag{A9}
$$

The two terms are tracking and the finite-step defect. Consequently (A6)
applies at every GD iterate if their aggregate budgets and (A3) hold along
the interpolant. This is a conditional GD theorem, not an identification of
GD with effective flow. The main-text evaluations set $S=0$ for their
effective-flow bounds and compare the actual GD trajectories separately.

**Sharper bounds are optional.** Integrating the time-dependent force
envelope instead of replacing it by its terminal value improves (A6).
Accumulated fourth-power concentration can also sharpen population counting.
The simpler second-power counting bound suffices for the main-text figure;
it avoids adding that structural premise. To see directly what panel (c)
bounds, note that

$$
\left(\frac1W\sum_j|\lambda_j(t)-\lambda_j(0)|^2\right)^{1/2}
\le\frac{hA(t)}{\sqrt W}.
\tag{A10}
$$

The numerical lower bounds are deliberately conservative. Their purpose is
to rule out useful output accuracy over a budget, not to forecast the exact
small amount of error reduction.

### D. Empirical scope and methods

The claim begins at a post-transient checkpoint. It does not prove that
initialization enters the regime or that a checkpoint determines its entire
duration. The assumption audit measures the evolving feedback quantity (A2)
and tests (A3) throughout the sampled interval. Low endpoint error or small
endpoint slopes are not substitutes for this check.

**Matched six-target continuations.** We use width $W=705$, reference
resolution $N=512$, $h=2/512$, and seed 30. The training measure consists of
2,048 equally weighted midpoint inputs in $[-1,1]$. Initial slopes, hidden
biases, and readouts are independent uniform draws on
$[-\sqrt{6/(W+1)},\sqrt{6/(W+1)}]$, with output bias zero. Initializations
are paired across targets. All parameters train simultaneously with
deterministic full-batch GD at learning rate 0.002. The restarts are the
unmodified states after 20k updates, and the continuation lasts a further
100k updates, ending at age 120k. No readout refitting or checkpoint selection
by final accuracy is used.

The pair $(W,N)=(705,512)$ is the archived comparison convention. These
randomly initialized networks do not use the fixed grid or the particular
halo count in Theorem 3.1. The value of $h$ supplies a common relative-scale
reference; neither the output bound nor the training dynamics depends on
interpreting it as an actual learned center spacing.

The functions before empirical RMS normalization are:

- Degree five: $0.3q_0+0.4q_1+\sqrt{0.75}\,q_5$, with $q_k$ empirical
  orthonormal polynomials of positive leading coefficient.
- Mixed sine: $\sin(2\pi x)+0.5\sin(6\pi x)+0.25\sin(14\pi x)$.
- Gaussian: $\exp(-((x+0.35)/0.22)^2)$.
- Compact bump: $\exp(1-1/(1-u^2))$ for $|u|<1$ and zero otherwise,
  with $u=(x-0.35)/0.22$.
- Smooth step: $\tanh(14(x-0.31))$.
- Kink: $|x+0.23|$.

The polynomial is already normalized. The other targets are divided by
their RMS on the original training grid, with this normalization held fixed.
The degree-five label is not the monomial $x^5$. The compact bump and kink
are additional stress tests for the training mechanism, outside the analytic
target class of the construction theorem. The persistence theorem itself
requires only a square-integrable target.

Effective flow is integrated with RK4 steps 0.02 and 0.01. GD is compared
at learning rates 0.002 and 0.001 over the same flow-time duration $T=200$;
the smaller-step control consequently uses twice as many updates. Scalar
diagnostics are retained at 201 equally spaced times. These comparisons
test the effective reduction and discretization sensitivity, rather than
equating optimizer update counts across different learning rates.

**A common allowance, with visible slack.** Figure 4 uses
$B_t=4d_{\rm dir}(0)t$ on every target. The factor four comes from the
previously tested sensitivity family $1,2,4,8$; its coverage is an empirical
finding, not an initial-data prediction. The largest sampled ratio of actual
accumulated feedback to the allowance is 0.5436. The conditional
effective-flow relative-error floors from (11) range from 0.3928 to 0.9124.
Their RMS slope-movement allowances range from $2.51\times10^{-7}$ to
$7.74\times10^{-4}$. Observed GD RMS changes range from
$2.61\times10^{-8}$ to $5.25\times10^{-5}$. For increment $0.125$, the
largest conditional ever-acquired-fraction bound is $3.84\times10^{-5}$.
These are rounded numerical evaluations, not interval-arithmetic certificates.

The initial relative slopes are all below $0.001$, so the reference
increment $0.125$ already falls short of reaching the construction benchmark
$\lambda=0.25$. The benchmark is an illustrative construction regime,
not a necessary or sufficient criterion for all accurate networks. The
independent raw-error conclusion is therefore essential. We use the lenient
1% relative-error criterion to establish failure well before machine precision.

**Breadth and robustness.** The separate width-705 baseline contains 23
targets and two seeds, including polynomial, oscillatory, localized, rational,
exponential, and transition functions. Twice the restart directional rate
covers all retained prefixes of all 46 branches over 20k further updates;
the largest required multiplier is 1.367. Twelve width-1409 reference
branches also pass that check. These families include related variants, and
the six dense trajectories are a subset of the broader investigation, not
six additional independent target families.

The largest endpoint raw relative-error difference between refined GD
and effective flow across the six targets is $1.72\times10^{-6}$.
On the sampled GD paths, accumulated tracking-induced derivative loading
is at most 0.0993% of the restart force. Incorporating sampled tracking
effects into the sharper time-integrated energy calculation changes its
relative-error floor by at most $4.16\times10^{-6}$. This last number concerns
that sharper calculation, not a certified value of $\Delta_{\rm err}$ in
(A7). The finite-step defect in (A9) has not been enclosed by interval
arithmetic. Halving the GD step changes endpoint parameters by less than
$1.98\times10^{-6}$ in Euclidean norm; halving the effective-flow RK4 step
changes them by less than $2.4\times10^{-14}$.

The broad archive is sparsely sampled, whereas the matched continuations
provide dense temporal checks. Neither supplies a rigorous enclosure between
every saved state. This distinction concerns empirical verification of the
premises; the conditional implication in Theorem A.1 is exact. All reported
force and error quantities use the training measure and actual attached
readouts. No continuous-input generalization or quantitative Adam guarantee
is asserted.

### E. Mechanism checks and the domain of the explanation

The conditional theorem should be read alongside interventions that
distinguish explanations of slow motion. Function-preserving fourfold
cloning divides each readout among four identical neurons. It leaves the
represented function unchanged while reducing each copy's geometry gradient
by four and increasing aggregate readout mobility by four. Correcting both
learning rates recovers the original trajectory to numerical precision.
Across the tested 23-target cohorts, correcting geometry mobility largely
restores slope motion, whereas correcting readout mobility alone does not.
Faster readout fitting alone is therefore insufficient to explain this
controlled slowdown.

Likewise, the smallest tested outward pulses retain 99.0–100.3% of their
directional offset after 20k updates. This supports slow motion without
strong restoration in those tested directions; it does not establish a
global attractor or behavior under arbitrary perturbations. Much larger
geometry injections can leave the weak-force regime. Under primary coarse
repair, the median initial effective-force norm rises from about
$6.04\times10^{-4}$ at baseline to 2.64 at 100-fold injection; the corresponding
weak-force error bound is then uninformative. A failed bound does not show
successful training.

These tests constrain the mechanism without requiring that every target
share one signed force contribution. Correcting generated low-order error
can oppose expansion, but neither net slope contraction nor that particular
error pattern is an assumption of Theorem 4.2. Its claim is limited useful
reinforcement over a stated interval, with the observed domain and failures
of that premise reported separately.
