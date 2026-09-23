# Conditional acquisition bounds from the effective fine force

The question is whether the force left after coarse balance can move enough
neurons to a large slope scale within a given number of GD updates. Our
experiments identify that force: the **effective fine force**, whose driving
errors and sensitivities evolve together. A useful theorem should retain this
structure and charge the small coarse-tracking correction only for the motion
it can actually add.

This note proves conditional results with that purpose. The leading movement
budget comes from the coupled effective model. The ordinary-GD theorem adds
separate allowances for sensitivity evolution, coarse tracking, omitted modes,
and finite steps. It preserves signed cancellation in the sensitivity response.
A neighborhood lemma explains how to establish the assumptions without using
the future true trajectory. A final lemma gives conditions under which small
coarse tracking itself persists.

**What is new, and what is not yet established.** Small tracking enters both
the slope equation and the equation for the errors driving later motion. It
therefore reduces two explicit terms in the bound. The algebra and conditional
proofs below are complete; useful uniform constants across long intervals have
not yet been certified. We do not assert that these results already exclude
scale 64 for millions of updates on every target.

Read Section 1 for the empirical premises, Sections 2–3 for the mechanism,
Section 4 for the ordinary-GD theorem, and Sections 5–6 for how its assumptions
could be proved. The [central exposition](d34_coarse_balance_stagnation.md)
supplies the broader experimental narrative.

**Notation. Gradients use empirical half-MSE; GD velocity has the opposite sign.**

| Symbol | Meaning |
|---|---|
| $\theta=(a,b,c,d)$ | Signed slopes, hidden biases, readouts, and output bias. |
| $\gamma_j=\lvert a_j\rvert$, $W$ | Neuron scale and network width. |
| $e_C$, $e=e_H$ | Coarse residual coefficients (degrees 0–1) and retained fine coefficients (degrees 2–65). |
| $J_C$, $J_H$ | Jacobians of those coefficients with respect to **all** parameters; modes are rows. |
| $B$, $z_C=e_C+Be$ | Coarse-balance map and departure from instantaneous coarse balance. |
| $T$, $T_a$, $S=T^TT$ | Effective fine-gradient map, its slope rows, and its residual coupling. |
| $r_C=J_C^Tz_C$, $g_\perp$ | Coarse-tracking gradient and gradient from the omitted residual. |
| $n$, $N$, $\eta$ | Additional-update index from a specified checkpoint, horizon, and GD step size. |
| A hat | The forecast using the full effective map frozen at that checkpoint. |
| $\Gamma$ | The slope threshold being tested; 64 is an application, not a universal requirement for approximation. |

## 1. Which empirical findings should the theorem use?

**Example: accurate coarse balance leaves substantial geometry.** In the
cross-function audit, the balanced coarse contribution to the slope force has
median magnitude 0.249–0.430 relative to the effective force at +200k updates,
depending on the cohort. Yet the departure from that balance contributes a
much smaller force. We must retain the former while treating the latter as a
controlled correction. Dropping everything called “coarse” would change the
leading mechanism.

The audit covers 23 function instances and 333 starting states: 195 original
starts, 78 fresh-seed starts on the same functions, and 60 starts on ten new
functions. The new panel includes exponentials, Gaussians, compact bumps,
tanh steps, and kinks. Starts and seeds are repeated observations, not
independent functions. All use width 177, 2048 fixed training points, and
noiseless full-batch GD with $\eta=0.002$. The attribution audit is retrospective;
the underlying forecasts were issued before their continuations.

**Evidence, proposed assumptions, and their precise role.** Measurements below
describe saved states, not every intervening update. Provenance is in Section 7.

| Finding | What the theorem retains or assumes | What this buys |
|---|---|---|
| At all 1,665 saved states, the slope-tracking norm ratio is below 1.24%. | Bound the direct force $(r_C)_a$ over the interval. | A small direct tracking-displacement allowance. |
| The ratio $\lVert J_Hr_C\rVert/\lVert Se\rVert$ is below 0.007 at all 1,665 saved states. | Independently bound $J_Hr_C$. | A small correction to the residual evolution that determines future slope force. |
| The balanced coarse contribution remains appreciable. | Keep $T=J_H^T-J_C^TB$. | Preserve the actual effective sensitivity. |
| Median map-force defects at +200k are 0.504–0.704 relative to the effective slope force across cohorts. | Bound the signed response to changing $T$ and $S$, rather than assume that both barely change. | Allow substantial sensitivity evolution without abandoning the framework. |
| On new functions, the median norm of the summed force-forecast defects is 0.351 times the sum of their norms. | Combine signed reference responses before bounding uncertainty. | Avoid spending the budget on contributions that cancel. |
| One compact-bump state has omitted residual-forcing ratio 0.0169 but omitted slope-force ratio 0.0000633. | Give omitted modes separate allowances in the two equations. | Avoid assigning residual evolution an unjustifiably small slope-derived tolerance. |

These observations motivate conditional hypotheses; they do not certify them.
Relative ratios also become unreliable when their denominators are near zero.
The theorem therefore uses absolute force allowances. Relative bounds can supply
them once their denominators are bounded on the proposed domain.
The measured cancellation concerns pointwise force-forecast defects. It
motivates preserving signs, but does not establish that the time-propagated
paired response in (11), or its uncertainty, is small.

**Prediction.** Holding the other estimates fixed, tightening either
tracking-force bound reduces its contribution to the acquisition allowance.
Whether that contribution stays small over a long interval is a calculation,
not a consequence of the sampled ratios. The theorem does not assume that
every target stalls.

## 2. The exact coupled system and what small tracking removes

**Example: a small direct correction can still change tomorrow's force.** A
correction may act mostly on readouts or biases and therefore barely move
slopes in the current step. It can nevertheless change the fine errors that
drive subsequent slope updates. This is why the second tracking measurement
in Section 1 is essential.

Let

$$
f_\theta(x)=d+\sum_{j=1}^W c_j\tanh(a_jx+b_j),\qquad
L(\theta)=\tfrac12\|f_\theta-y\|_m^2.
$$

The inner product averages over the fixed training points. The modal errors
are projections onto a fixed empirical orthonormal polynomial basis; they are
not a Taylor expansion of the network. The omitted residual remains in
$g_\perp$. The target can be any fixed label vector on those points; no
polynomial target is assumed. Assume $K_C=J_CJ_C^T$ is invertible and define

$$
B=K_C^{-1}J_CJ_H^T,\qquad z_C=e_C+Be,\qquad
T=J_H^T-J_C^TB.
\tag{1}
$$

Substituting $e_C=z_C-Be$ in the gradient gives the exact decomposition

$$
g=Te+r_C+g_\perp,\qquad r_C=J_C^Tz_C.
\tag{2}
$$

Writing $\Pi=I-J_C^TK_C^{-1}J_C$ shows that $T=\Pi J_H^T$,
$J_CT=0$, and

$$
J_HT=J_H\Pi J_H^T=T^TT=S.
\tag{3}
$$

Thus $z_C=0$ removes $r_C$; it does **not** remove $-J_C^TB$ from $T$.
This balance is not an optimal least-squares refit of the readouts. Lowercase
$c$ continues to mean readout weights; a subscript $c$ on a gradient refers
to that parameter block, not to “coarse.”

For actual discrete GD, set

$$
\tau_n=e_H(\theta_n-\eta g_n)-e_H(\theta_n)+\eta J_{H,n}g_n.
$$

This is the exact finite-step Taylor remainder. Equations (2)–(3) yield

$$
\begin{aligned}
\theta_{n+1}&=\theta_n-\eta(T_ne_n+r_{C,n}+g_{\perp,n}),\\
e_{n+1}&=(I-\eta S_n)e_n
-\eta J_{H,n}r_{C,n}-\eta J_{H,n}g_{\perp,n}+\tau_n.
\end{aligned}
\tag{4}
$$

The first equation includes the slopes. The second explains how today's
update changes their future drive. Sensitivities depend on all parameter
blocks, so small slope corrections alone do not close the coupled system.

**Why small tracking is insufficient by itself.** Consider the simple
quadratic model with parameters $(a,q)$ and errors
$e_C=q$, $e=\sigma(a-A)$, where $A>64$, $\sigma>0$, and $a_0=q_0=0$.
Here $B=0$, tracking remains exactly zero, and there is no omitted residual
or Taylor remainder. Nevertheless,

$$
a_n=A[1-(1-\eta\sigma^2)^n]
$$

crosses 64 for sufficiently large $n$ when $0<\eta\sigma^2\le1$.
This is a counterexample to an inference from tracking alone, not a model
of the measured tanh trajectories. A barrier needs an additional restriction
on the motion the effective force can supply.

**Prediction.** The relevant restriction should depend on error loadings,
sensitivities, their evolution, and elapsed time. It cannot follow just from
the statement that coarse balance is accurate.

## 3. Theorem 1: a finite movement budget for the coupled effective model

**Example: sensitivity can be weak, or its driving error can run out.** In
one singular direction of a fixed effective map, let $\sigma>0$ be the
sensitivity, $\alpha$ the initial error loading, and $u_a$ its slope direction.
Assume $0<\eta\sigma^2\le1$. The coordinate movement allowance after $N$
updates is

$$
|u_{a,j}|\frac{|\alpha|}{\sigma}
[1-(1-\eta\sigma^2)^N]
\le |u_{a,j}|\min\{\eta N\sigma|\alpha|,\ |\alpha|/\sigma\}.
\tag{5}
$$

Weak sensitivity restricts motion during a finite window. Appreciable
sensitivity can instead remove its driving error quickly, leaving a small
total displacement when $|\alpha|/\sigma$ is small. Increasing sensitivity
does not necessarily increase the total motion supplied by a fixed error.
This illustrates why weak response to a hard mode is only one possible
mechanism; it does not identify a one-mode explanation of sine.

**Statement.** Freeze the full effective map at the starting state and evolve

$$
\widehat e_{n+1}=(I-\eta S_0)\widehat e_n,\qquad
\widehat\theta_{n+1}=\widehat\theta_n-\eta T_0\widehat e_n,
\qquad(\widehat\theta_0,\widehat e_0)=(\theta_0,e_0).
\tag{6}
$$

This moves all parameter blocks and evolves all retained fine errors. It is
the reduced forecast, not the slope-only freeze-map intervention. Write the
positive singular directions as $T_0v_i=\sigma_i u_i$, set
$\alpha_i=v_i^Te_0$, and assume $0<\eta\sigma_i^2\le1$. Define

$$
\Phi_n(\lambda)=\eta\sum_{k=0}^{n-1}(1-\eta\lambda)^k,
\qquad \Phi_0(\lambda)=0.
$$

For $\lambda>0$, this is $[1-(1-\eta\lambda)^n]/\lambda$;
at zero it equals $\eta n$. Then

$$
\widehat\theta_n-\theta_0
=-\sum_{\sigma_i>0}u_i\sigma_i\alpha_i\Phi_n(\sigma_i^2).
\tag{7}
$$

In particular, for every $n\le N$,

$$
|\widehat a_{j,n}|\le |a_{j,0}|+U_j(N),\qquad
U_j(N)=\sum_{\sigma_i>0}|u_{a,j,i}\sigma_i\alpha_i|
\Phi_N(\sigma_i^2).
\tag{8}
$$

**Proof.** Project the residual recurrence onto $v_i$ to obtain
$v_i^T\widehat e_n=(1-\eta\sigma_i^2)^n\alpha_i$.
Substitution into the parameter update and summation gives (7). Under the
stated step condition these scalar factors are nonnegative, so the sum of
absolute coordinate increments is at most (8). This also bounds every
intermediate coordinate displacement and positive scale travel, including
slope sign crossings. An exact null direction has $T_0v_i=0$ and produces
no movement. Small positive singular values retain their finite-time factor;
they must not be discarded as numerical nulls. The one-mode estimate (5)
uses $1-(1-x)^N\le\min\{Nx,1\}$ for $0\le x\le1$.

**Prediction and limit.** The initial loadings and geometry now give a
per-neuron movement allowance, rather than charging the entire loss to
possible slope motion. All 333 audited starts meet the nonoscillating step
condition in FP64. Their uncorrected allowances are compatible with actual
scale-1 acquisitions at +10k, but fail for 43 starts at +200k. This does
not falsify Theorem 1, which concerns (6); it establishes that transferring
it to GD requires the corrections below. No particular horizon is privileged.

## 4. Theorem 2: ordinary-GD acquisition with explicit mechanism corrections

**Example: a sizeable relative force error can still leave acquisition
impossible.** If a forecast and its entire uncertainty interval stay near
scale 1, that interval can exclude scale 64 even when it does not resolve a
small displacement accurately. Conversely, a tiny correction repeated for
long enough can matter. We need its propagated displacement, not just a
small ratio at the starting state.

### First keep the direct and delayed responses together

For matrix arguments, write
$\Phi_m(S_0)=\eta\sum_{\ell=0}^{m-1}(I-\eta S_0)^\ell$.
Define the exact defects

$$
\begin{aligned}
\delta g_k&=(T_k-T_0)e_k+r_{C,k}+g_{\perp,k},\\
\xi_k&=(S_k-S_0)e_k+J_{H,k}r_{C,k}
+J_{H,k}g_{\perp,k}-\tau_k/\eta.
\end{aligned}
\tag{9}
$$

Then the full parameter error is exactly

$$
\theta_n-\widehat\theta_n
=-\eta\sum_{k<n}\delta g_k
+\eta\sum_{k<n}T_0\Phi_{n-1-k}(S_0)\xi_k.
\tag{10}
$$

The first sum is direct movement. The second is the later response to
changed residual evolution. The last residual update has no effect on the
current parameter position because $\Phi_0=0$.

To preserve the cancellation between changing sensitivity and its residual
feedback, define their **paired response**, a full parameter vector:

$$
\mathcal K_{n,k}(\theta)
=-[T(\theta)-T_0]e_H(\theta)
+T_0\Phi_{n-1-k}(S_0)[S(\theta)-S_0]e_H(\theta).
\tag{11}
$$

Its actual contribution is $\eta\sum_{k<n}\mathcal K_{n,k}(\theta_k)$.
Both terms can be large and oppose each other. We will approximate and
bound their combination, not assume small map drift.

Choose reference states $\bar\theta_k$ from initial information alone. The
default is $\bar\theta_k=\widehat\theta_k$, using (6), and define

$$
\overline D(n)=\eta\sum_{k<n}\mathcal K_{n,k}(\bar\theta_k).
\tag{12}
$$

Here $e_H(\bar\theta_k)$ means the residual of the actual network at that
reference parameter vector. It need not equal the auxiliary forecast
$\widehat e_k$. Evaluating the network on a forecast is allowed; supplying
future true states would instead make (12) retrospective. The correction
need not be small, and evaluating it costs more than running (6) alone.

### Statement and the assumptions that determine its tightness

Assume (1) is defined along the GD trajectory through the required interval.
Suppose the following nonnegative allowances hold for each update $k<N$ and
each $k<n\le N$. The index $\ell$ below labels any parameter coordinate.

$$
\begin{aligned}
|\mathcal K_{n,k}(\theta_k)_\ell
-\mathcal K_{n,k}(\bar\theta_k)_\ell|&\le\omega_{\ell,n,k},\\
|(r_{C,k})_\ell|\le u^C_{\ell,k},\quad
\|J_{H,k}r_{C,k}\|_2&\le v^C_k,\\
|(g_{\perp,k})_\ell|\le u^\perp_{\ell,k},\quad
\|J_{H,k}g_{\perp,k}\|_2&\le v^\perp_k,\\
\|\tau_k\|_2&\le t_k.
\end{aligned}
\tag{13}
$$

These are conditional hypotheses on a full interval. Section 5 supplies
sufficient conditions to establish them without assuming the future path.
They do not follow merely by setting the allowances to sampled maxima.

Let

$$
L_{\ell,n,k}
=\|\operatorname{row}_\ell(T_0\Phi_{n-1-k}(S_0))\|_2
$$

and define four separately reported uncertainty budgets:

$$
\begin{aligned}
E^{\rm gain}_\ell(n)&=\eta\sum_{k<n}\omega_{\ell,n,k},\\
E^C_\ell(n)&=\eta\sum_{k<n}
\left(u^C_{\ell,k}+L_{\ell,n,k}v^C_k\right),\\
E^\perp_\ell(n)&=\eta\sum_{k<n}
\left(u^\perp_{\ell,k}+L_{\ell,n,k}v^\perp_k\right),\\
E^{\rm step}_\ell(n)&=\sum_{k<n}L_{\ell,n,k}t_k,\qquad
E_\ell(n)=E^{\rm gain}_\ell(n)+E^C_\ell(n)
+E^\perp_\ell(n)+E^{\rm step}_\ell(n).
\end{aligned}
\tag{14}
$$

Then

$$
|\theta_{\ell,n}-\widehat\theta_{\ell,n}-\overline D_\ell(n)|
\le E_\ell(n).
\tag{15}
$$

For a neuron $j$, use the slope coordinate $\ell=(a,j)$ and put

$$
M_j(N)=\max_{0\le n\le N}
\left(|\widehat a_{j,n}+\overline D_{a,j}(n)|+E_{a,j}(n)\right).
\tag{16}
$$

With all sums zero at $n=0$, the population of distinct neurons ever
acquiring scale $\Gamma$ satisfies

$$
\#\{j:\max_{0\le n\le N}|a_{j,n}|\ge\Gamma\}
\le\#\{j:M_j(N)\ge\Gamma\}.
\tag{17}
$$

Consequently at least $\#\{j:M_j(N)<64\}/W$ of the neurons remain below
scale 64 at **every** update through $N$. Restrict both sets in (17) to
$|a_{j,0}|<\Gamma$ when counting only new acquisitions. An initially large
neuron is otherwise included automatically.

### Proof

Write $A_0=I-\eta S_0$. Subtracting the forecast residual recurrence from
(4) gives

$$
e_n-\widehat e_n=-\eta\sum_{k<n}A_0^{n-1-k}\xi_k.
$$

Next sum the parameter update differences. Substituting this residual error
and interchanging the two finite sums gives (10). Split its right-hand side
using (9). The changing-map contribution is (11). The tracking contribution is

$$
\eta\sum_{k<n}
\left[-r_{C,k}+T_0\Phi_{n-1-k}(S_0)J_{H,k}r_{C,k}\right],
\tag{18}
$$

with the same expression for the omitted force. The Taylor contribution is
$-\sum_{k<n}T_0\Phi_{n-1-k}(S_0)\tau_k$, with no additional factor of
$\eta$ outside $\Phi$. Subtract (12), apply (13), and bound each row-vector
product by its Euclidean row norm. This proves (14)–(15). Finally,
$|a_{j,n}|\le M_j(N)$ at every update, so any true crossing requires
$M_j(N)\ge\Gamma$. This proves (17), including sign changes.

### Exactly how the empirical assumptions improve the result

Small tracking shrinks **both** $u^C$ and $v^C$ in (14). The latter is
multiplied by the explicit frozen-effective propagation kernel. For example,
domain-wide relative estimates
$\|(r_C)_a\|\le\varepsilon_a\|T_ae\|$ and
$\|J_Hr_C\|\le\varepsilon_H\|Se\|$, together with upper bounds
$V^a_k\ge\|T_ae\|$ and $V^H_k\ge\|Se\|$, give
$u^C_{a,j,k}=\varepsilon_a V^a_k$ and $v^C_k=\varepsilon_H V^H_k$.
Coordinate bounds can improve the first choice. Small $z_C$ alone does not
supply either force bound without controlling its multiplying matrices.

The leading path retains signed modal cancellation in (7). The reference
correction retains cancellation between the two terms in (11), and across
updates in (12). Only the uncertainty about that response is charged to
$E^{\rm gain}$. This does not preserve every possible cancellation: (14)
still bounds the other channels separately and sums their uncertainty
magnitudes. Joint bounds on (18), or correlated uncertainty sets, can be
sharper when the explicit version is insufficient.

A simpler consequence of (8) and (15) replaces (16) by

$$
|a_{j,0}|+U_j(N)
+\max_{n\le N}\left(|\overline D_{a,j}(n)|+E_{a,j}(n)\right).
\tag{19}
$$

Equation (16) is preferable when cancellation matters. It bounds an
ever-crossing event directly; a trajectory enclosure alone does not bound
total positive travel, which can include repeated oscillations. Likewise,
an instantaneous Euclidean population-distance bound must not be mistaken
for a count of distinct neurons ever crossing at different times.

**Prediction.** Report the excluded fraction as a function of $N$, with
the four budgets and the signed reference correction shown separately.
An informative result need not recover every tiny displacement. It must
keep the resulting envelope below the chosen acquisition threshold for
the claimed neurons. No uniform inward force, universal cubic mechanism,
or monotone readout effect is assumed.

## 5. Lemma 3: close the allowances without assuming a nearby trajectory

**Example: a measured forecast error is not a prospective allowance.** Five
saved states can reveal that a forecast worked, or that it missed an
acquisition. They cannot bound the largest error between those states.
Similarly, drawing a narrow neighborhood around an observed future path
does not establish that the path had to stay there. We need estimates that
imply their own neighborhood containment.

**Statement.** Compute the reference path from the initial state and choose
nonnegative coordinate radii $r_{\ell,k}$ defining full-parameter boxes

$$
\mathcal D_k=\{\theta:
|\theta_\ell-\widehat\theta_{\ell,k}|\le r_{\ell,k}
\text{ for every }\ell\},\qquad 0\le k\le N.
\tag{20}
$$

Use $\bar\theta_k=\widehat\theta_k$ in (12). Require the coarse Gram matrix
to have a strictly positive eigenvalue lower bound on these boxes. Establish
the bounds in (13) uniformly on $\mathcal D_k$, replacing $\theta_k$ by
any $\theta\in\mathcal D_k$ and defining the Taylor remainder directly as

$$
\tau(\theta)=e_H(\theta-\eta g(\theta))-e_H(\theta)
+\eta J_H(\theta)g(\theta).
\tag{21}
$$

If, for every parameter coordinate and every $1\le n\le N$,

$$
|\overline D_\ell(n)|+E_\ell(n)\le r_{\ell,n},
\tag{22}
$$

then the actual GD trajectory lies in $\mathcal D_n$ throughout the interval.
Consequently Theorem 2 applies with these independently established allowances.

**Proof.** The initial state equals $\widehat\theta_0$, so it lies in
$\mathcal D_0$. Suppose all preceding states through $n-1$ lie in their
boxes. Each term of (10) then obeys the domain bounds used in (13).
The same algebra that proves (15) gives

$$
|\theta_{\ell,n}-\widehat\theta_{\ell,n}|
\le |\overline D_\ell(n)|+E_\ell(n)\le r_{\ell,n}.
$$

Thus the next state lies in its box, completing the induction. No future
trajectory containment was assumed. Coordinate boxes are one sufficient
construction; another domain shape must come with its own inclusion proof.

### What must actually be bounded on the boxes?

For the sensitivity response, a valid choice is

$$
\omega_{\ell,n,k}
\ge\sup_{\theta\in\mathcal D_k}
|\mathcal K_{n,k}(\theta)_\ell
-\mathcal K_{n,k}(\widehat\theta_k)_\ell|.
\tag{23}
$$

One sufficient derivative bound is
$\sup_{\theta\in\mathcal D_k}\sum_q
r_{q,k}|\partial_q\mathcal K_{n,k}(\theta)_\ell|$.
Differentiate the combined expression (11) before taking this bound.
Separately bounding its direct and delayed terms can lose the cancellation
we are trying to preserve. The coarse-Gram lower bound controls the inverse
in these derivatives; invertibility only at the initial point is insufficient.

Tracking and omitted-force allowances are suprema of the respective
quantities in (13) on the same domains. Coarse tracking may require a much
thinner domain in some directions than in others. A large isotropic ball
need not exploit the empirically narrow tracking regime. The two measured
tracking ratios concern slope motion and residual forcing; closing a
full-parameter box additionally requires bounds for nonslope coordinates.
Those bounds are not inferred from the slope measurements.

For the discrete correction, suppose $\|g(\theta)\|_2\le G_k$ on
$\mathcal D_k$, and the second derivative of $e_H$ has bilinear operator
norm at most $M_{2,k}$ on every segment
$\theta-t\eta g(\theta)$, $0\le t\le1$, from that domain. Taylor's
formula gives

$$
t_k=\tfrac12\eta^2M_{2,k}G_k^2.
\tag{24}
$$

The whole step segment must be covered. Bounds at its endpoints alone do
not suffice. An output-Hessian bound in empirical RMS supplies $M_{2,k}$
because projection onto the orthonormal retained basis is contractive.
The companion's [tanh derivative bounds](d34_coarse_balance_stagnation_details.md#appendix-a-computable-neighborhood-bounds)
give a starting point, but its earlier sample-Jacobian enclosure is a
different forecast and must not be relabeled as a certificate for this one.

**Prediction and limitation.** This lemma turns domain-wide assumptions into
a prospective ordinary-GD statement. It does not guarantee that simple boxes
or derivative bounds will be sufficiently sharp. If (22) fails, the proposed
certificate stops there; it does not imply that acquisition occurs. The
remaining problem is to obtain useful allowances, especially for evolving
sensitivities, without replacing their measured structure by an expensive
global bound. Blockwise evaluation is possible only if its estimates cover
every intermediate update.

## 6. Lemma 4: when can small coarse tracking persist?

**Example: fast coarse relaxation can absorb a slowly moving equilibrium.**
The small observed $z_C$ is plausible if the coarse variables restore balance
faster than the fine evolution moves that balance. This is an additional
dynamical statement, not a consequence of seeing small tracking once. The
following recurrence specifies both the restoring part and its driving terms.

Define the coarse Taylor remainder by

$$
\tau_{C,n}=e_C(\theta_n-\eta g_n)-e_{C,n}+\eta J_{C,n}g_n.
$$

With $K_{C,n}=J_{C,n}J_{C,n}^T$, the exact recurrence is

$$
z_{C,n+1}=A_{C,n}z_{C,n}+f_n,\qquad
A_{C,n}=I-\eta(I+B_nB_n^T)K_{C,n},
\tag{25}
$$

where

$$
\begin{aligned}
f_n={}&(B_{n+1}-B_n)e_{n+1}-\eta B_nS_ne_n\\
&-\eta(J_{C,n}+B_nJ_{H,n})g_{\perp,n}
+\tau_{C,n}+B_n\tau_n.
\end{aligned}
\tag{26}
$$

**Statement.** Suppose a single specified norm $\|\cdot\|_*$ on the coarse
coefficient space satisfies, throughout the interval,
$\|A_{C,n}\|_{*\to *}\le r<1$ and $\|f_n\|_*\le h$. Then

$$
\|z_{C,n}\|_*
\le Z_n:=r^n\|z_{C,0}\|_*+
\frac{h(1-r^n)}{1-r}.
\tag{27}
$$

If domain bounds also give

$$
\|\operatorname{row}_\ell(J_C^T)\|_{*\to|\cdot|}
\le A_{\ell,n},\qquad
\|J_HJ_C^T\|_{*\to2}\le H_n,
$$

then $u^C_{\ell,n}=A_{\ell,n}Z_n$ and $v^C_n=H_nZ_n$ supply the two
tracking allowances in Theorem 2. This states exactly what additional
estimates could replace an assumed interval-wide small-tracking condition.

**Proof.** Since $J_CT=0$, the coarse residual update is

$$
e_{C,n+1}=e_{C,n}-\eta K_{C,n}z_{C,n}
-\eta J_{C,n}g_{\perp,n}+\tau_{C,n}.
$$

Use (4) and the identity $J_HJ_C^T=B^TK_C$. In
$z_{C,n+1}=e_{C,n+1}+B_{n+1}e_{n+1}$, write
$B_{n+1}e_{n+1}=B_ne_{n+1}+(B_{n+1}-B_n)e_{n+1}$.
Collecting terms gives (25)–(26). Taking the common norm yields
$\|z_{C,n+1}\|_*\le r\|z_{C,n}\|_*+h$; summing this scalar geometric
recurrence proves (27). Applying the stated operator bounds proves the
two force estimates.

**Prediction and empirical status.** If the changing balance and fine drive
keep $h/(1-r)$ small, tracking stays small after its initial transient.
The experiments support the resulting small forces at audited states; they
have not established these contraction and forcing bounds on a closed domain.
The $B$-drift term in (26) must itself be controlled, not replaced by measured
future values in a prospective proof.

Although $(I+BB^T)K_C$ has positive eigenvalues, it is generally nonsymmetric.
Those eigenvalues alone do not prove contraction in a common norm as the
matrices change. A varying norm would require explicit comparison factors
between consecutive norms. This lemma is therefore a candidate route to
proving persistence of the premise, not a claim that the premise has already
been discharged.

To combine Lemmas 3 and 4, a bound on the reachable trajectory must not be
silently treated as a bound at every point in a box. Either establish the
uniform tracking suprema required by Lemma 3, or extend its induction to
both $\theta_n\in\mathcal D_n$ and $\|z_{C,n}\|_*\le Z_n$. In the latter
case, the known tracking bound encloses the next parameter state first;
bounds on (26) over all admissible consecutive-state pairs then enclose
the next tracking state. These pairs must include the true GD update and
have defined coarse inverses at both endpoints. Merely checking $f_n$ at
the reference states does not close this argument.

## 7. What these theorems establish, and the next calculation

The mechanism-specific conclusion is conditional but concrete: **a neuron
cannot acquire a scale that lies outside its entire corrected effective-model
envelope**. Equation (17) turns that statement into an excluded population.
Equation (14) shows exactly what small tracking contributes. Equations
(11)–(12) retain the coupled response that would be lost by assuming fixed
sensitivities or summing the magnitudes of every force component.

There is no claim of permanent trapping. For a desired excluded fraction,
the admissible horizon is the largest prefix for which the domain conditions
and population inequality both hold. It may be short or long, and need not
match an experimental endpoint. A loose but valid envelope can still exclude
scale 64 for a substantial population. The current work establishes the
conditional theorem, not numerical values for that population or horizon.

The next calculation is to evaluate its assumptions, not launch another broad
training campaign:

1. Compute the initial effective spectra and loadings, then the corrected
   reference paths and their acquisition margins across the archived targets.
   Retain the existing diagnostic thresholds as checks and evaluate 64 as the
   requested larger-scale application.
2. Bound each of the four uncertainty channels on proposed domains. Report
   which condition first fails and whether loss of precision comes from real
   sensitivity evolution or conservative uncertainty bounds.
3. Compare against archived states as falsification checks, then rigorously
   enclose promising instances. Sampling cannot replace the all-update bounds;
   a formal numerical certificate requires controlled numerical errors as well.

The alternative [local inaccessible-loss theorem](d34_coarse_balance_stagnation_details.md#a-stronger-bound-from-the-loss-the-network-cannot-yet-remove)
remains useful in the degree-9 regime. It subtracts a local unavoidable loss
floor before applying descent, and can give a strong separate confinement
result. A generic bound based on total loss discards that advantage and also
discards the effective spectra and cancellations used here. Neither should
replace the mechanism-specific theorem as the general explanation.

### Evidence provenance and proof checks

The existing [campaign synthesis](../results/checkpoint_D_optimizers/expD34_readout_race/effective_feedback/README.md)
and [new-function report](../results/checkpoint_D_optimizers/expD34_readout_race/effective_feedback/analysis/heldout/README.md)
record the training and forecast comparisons. The cross-function attribution
is archived in the same repository's Git history: audit implementation and
summary code at commit `2143173`, with curated evidence and exposition
`docs/d34_cross_function_audit.md` at commit
`131323bfa2c2363d34422a0ac30ecb6fb3cdc295`. Its base campaign is `f69e1bb`.
These files need not be present in the current checkout to be recovered from
those commits.

The audit's evidence root is
`results/checkpoint_D_optimizers/expD34_readout_race/cross_function_audit/`.
The cohort table comes from `summary_final/report_metrics.json`; the
all-sampled tracking maxima were independently checked against all 1,665 rows
of `summary_final/forces_combined.csv.gz`. Each start contributes the fork and
an additional 1k, 10k, 50k, and 200k updates. The maximal slope-tracking ratio
is 0.0123246 and the maximal residual-tracking ratio is 0.00690079. The paired
compact-bump example is seed 23, fork 600k, additional 200k. These are
ordinary-GD force attributions, not intervention effects or a statistical
significance test. They were not used to certify any constants in (13).

Independent in-memory checks verified all 900 intermediate comparison
identities in 100 random varying-map systems with nonzero tracking, omitted
forcing, and residual step defects. The largest absolute reconstruction
discrepancy was $1.23\times10^{-15}$. One-mode geometric motion, exact-null
behavior, and a slope sign crossing were also checked. These synthetic
systems verify algebra and indexing; they do not establish that arbitrary
prescribed maps arise from a tanh training trajectory.

The 16 existing effective-force and forecast tests also passed with
`.venv/bin/python -m pytest -q tests/test_expD34_effective_feedback_kernel.py tests/test_expD34_effective_feedback_predict.py`.
They check the implemented decomposition, derivative comparisons, discrete
forecasts, and existing enclosure examples. They do not numerically certify
the new theorem's interval-wide assumptions.

All proofs above concern exact real-arithmetic GD. Existing FP64 trajectories
and identity checks are numerical evidence, not directed-rounding interval
certificates. Certifying a particular implementation also requires enclosing
its arithmetic errors. No new training or long-horizon certificate was
produced for this note.
