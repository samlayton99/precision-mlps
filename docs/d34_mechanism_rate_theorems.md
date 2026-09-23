# Conditional mechanisms and rates of scale acquisition

The effective fine force identifies what moves the slopes. The next question
is why that force remains small, changes sign, or restores a perturbed geometry.
These are different mechanisms and make different predictions. This note gives
conditional results for distinguishing them, then translates their predictions
into rates of acquiring the normalized scale $\lambda=h|a|$.

The results concern noiseless, simultaneous updates with step size $\eta>0$. They do not assert an
attracting slow manifold, a universal stalled regime, or a width law without
additional hypotheses. The proposed response models use information available
at a checkpoint; fitting their coefficients to the continuation would change
the scientific question. Numerical measurements on saved states can test these
models, but do not establish uniform bounds between those states.

The [existing acquisition theorem](d34_effective_force_acquisition_theorems.md)
already supplies the exact signed defect convolution and a neighborhood-based
transfer to ordinary GD. This companion develops mechanisms that can supply
its leading forecast and improve its error allowances. No new experimental
outcomes are asserted here.

**Table 1. The quantities whose coupled evolution determines acquisition.**

| Symbol | Meaning |
|---|---|
| $a,b,c,d$ | Signed slopes, hidden biases, readouts, and output bias; $g_c$ is the readout gradient. |
| $e$, $T$ | Retained fine residual coefficients and their effective gradient map. |
| $r_C$, $g_\perp$ | Coarse disequilibrium and omitted-residual gradient corrections. |
| $n$, $\eta$ | Additional updates after the checkpoint and the physical-coordinate step size. |
| $W$, $N_{\mathrm{ref}}$ | Actual neuron count and construction resolution, respectively. |
| $\lambda=h|a|$, $h=2/N_{\mathrm{ref}}$ | Normalized slope scale; $\lambda=0.25$ is the reference acquisition level. |

## 1. The coupled object that must be predicted

**Example.** A tracking correction can have almost no slope component yet
change the readouts and therefore tomorrow's driving error. Small tracking
must be checked in both equations, not inferred from the slope equation alone.

Use $\theta=(a,b,c,d)$ and empirical half-MSE for
$f_\theta(x)=d+\sum_j c_j\tanh(a_jx+b_j)$. Let $e_C$ contain the coarse
residual coefficients (the two modes of degrees zero and one) and $e=e_H$ the retained fine coefficients in a fixed
empirical orthonormal basis. Their Jacobians $J_C,J_H$ have modes as rows and
all parameters as columns. Assume $J_CJ_C^T$ is invertible and set

$$
B=(J_CJ_C^T)^{-1}J_CJ_H^T,\quad z_C=e_C+Be,\quad
T=J_H^T-J_C^TB,\quad S=T^TT=J_HT.
$$

Then, exactly,

$$
g=\nabla L=Te+r_C+g_\perp,\qquad r_C=J_C^Tz_C.
\tag{1}
$$

Here $g_\perp$ is the gradient from modes omitted by the retained basis;
$F_a=T_ae$ is the effective fine slope gradient. Gradients have the opposite
sign to GD velocities. One discrete step satisfies

$$
\begin{aligned}
\theta_{n+1}&=\theta_n-\eta(T_ne_n+r_{C,n}+g_{\perp,n}),\\
e_{n+1}&=(I-\eta S_n)e_n-\eta J_{H,n}r_{C,n}
                         -\eta J_{H,n}g_{\perp,n}+\tau_n,
\end{aligned}\tag{2}
$$

where $\tau_n=e_H(\theta_n-\eta g_n)-e_H(\theta_n)+\eta J_{H,n}g_n$.
This is an identity for any fixed target, not an assumption about polynomial
degree. If the operator norm of the second derivative of $e_H$ is at most
$M_e$ on the update segment, then
$\|\tau_n\|\le \eta^2M_e\|g_n\|^2/2$.

**Prediction and scope.** Conditional small absolute bounds on $(r_C)_a$ and
$J_Hr_C$ remove two specific sources of motion. They do not remove the
evolution of $T$, and they do not bound $D r_C$. This distinction matters for
the perturbation experiment below. Omitted modes and finite-step errors need
their own bounds in (2).

## 2. Depletion: sensitivity can be appreciable while its error load disappears

**Example.** A feature may respond strongly to an error at the checkpoint but
correct that error before moving far. Weak sensitivity is therefore not the
only reason for limited acquisition.

Freeze the full effective map at $T_0$ and remove the corrections in (2).
The resulting model is

$$
\widehat e_n=(I-\eta S_0)^ne_0,\qquad
\widehat\theta_n=\theta_0-\eta\sum_{k<n}T_0\widehat e_k.
\tag{3}
$$

Write a singular decomposition $T_0=\sum_i\sigma_i u_iv_i^T$ over positive
singular values and $\alpha_i=v_i^Te_0$. Assume
$0<\eta\sigma_i^2\le1$. Summing the geometric series proves

$$
\widehat\theta_n-\theta_0
=-\sum_i u_i\frac{\alpha_i}{\sigma_i}
       [1-(1-\eta\sigma_i^2)^n].
\tag{4}
$$

Null directions of $S_0$ do not move parameters because $T_0$ kills them.
For a slope coordinate, a valid displacement envelope is

$$
U_j(N)=\sum_i |u_{a,j,i}|\frac{|\alpha_i|}{\sigma_i}
                   [1-(1-\eta\sigma_i^2)^N].
\tag{5}
$$

Each summand is at most
$|u_{a,j,i}|\min\{\eta N\sigma_i|\alpha_i|,|\alpha_i|/\sigma_i\}$.
Thus the relevant quantities are sensitivity, loading, and depletion time
together. Equation (5) bounds displacement and, under this spectral step-size
condition, also bounds the sum of absolute coordinate increments by the
triangle inequality. The exact endpoint (4) can be much smaller because of
signed cancellation.

**Prediction.** A depletion explanation must predict both the residual change
and the resulting slope motion with the checkpoint loadings. Matching just
the current slope force is insufficient. If the measured map evolution
produces motion outside a prospectively computed correction allowance, this
particular frozen-map explanation fails. It does not invalidate identity (2).
Multimode (3) can itself produce delayed changes of force; it must not be
identified with the scalar no-restoration hypothesis in the next section.

## 3. A matched kick distinguishes delayed restoration from simple force decay

**Example.** Move the geometry outward while keeping its immediate slope
gradient unchanged to first order. If the subsequent response returns toward
the original trajectory, the experiment probes delayed feedback rather than
an immediate change of slope force. Whether this response is well described
by two coordinates is a testable hypothesis, not a consequence of balance.

At the checkpoint choose a direction $v$ satisfying

$$
J_Cv=0,\qquad Dz_C[v]=0,\qquad Dg_a[v]=0.
\tag{6}
$$

The last derivative is the full loss-Hessian block, including residual
curvature. A Gauss--Newton replacement would define a different experiment.
The stacked constraints have at most $W+4$ rows in $3W+1$ dimensions, so their
nullspace has dimension at least $2W-3$. This does not ensure an outward
direction: the projected outward covector may vanish. Record such a case as
unavailable rather than relaxing the matching conditions after seeing results.

For $\theta_0\pm\rho v$, each constraint changes by $O(\rho^2)$, while its
central difference divided by $2\rho$ has error $O(\rho^2)$ under the required
smoothness. Amplitude halving must test this behavior above the numerical
resolution floor. The finite perturbations are not exactly matched states.
In particular $Dg_a[v]=0$ implies
$D(T_ae)[v]=-D(r_C+g_\perp)_a[v]$; a small remainder value does not justify
dropping its derivative.

Let $G(\theta)=\theta-\eta g(\theta)$ and follow the unperturbed trajectory.
The infinitesimal pulse obeys the exact variational equation

$$
v_{n+1}=(I-\eta H_n)v_n,\qquad H_n=Dg(\theta_n).
\tag{7}
$$

Choose a fixed slope covector $w$, including normalization by $h$ if desired,
and define
$q_n=w^TP_av_n$, $p_n=w^TDg_a(\theta_n)v_n$.
Here $P_a$ selects slope coordinates. Exactly $q_{n+1}=q_n-\eta p_n$.
The proposed additional closure is

$$
p_{n+1}=(1-\eta r)p_n+\eta\kappa q_n+d_n.
\tag{8}
$$

**Conditional response proposition.** If (8) holds, set
$M=\left(\begin{smallmatrix}1&-\eta\\\eta\kappa&1-\eta r\end{smallmatrix}\right)$.
Then

$$
\binom{q_n}{p_n}=M^n\binom{q_0}{p_0}
  +\sum_{k<n}M^{n-1-k}\binom{0}{d_k}.
\tag{9}
$$

For $p_0=0$ and $d_k=0$, the hypothesis $\kappa=0$ predicts $q_n=q_0$.
For $\kappa>0$, $q_1=q_0$ and $q_2=(1-\eta^2\kappa)q_0$.
The two eigenvalues are strictly inside the unit disk precisely when

$$
\kappa>0,\qquad r>\eta\kappa,\qquad
4-2\eta r+\eta^2\kappa>0.
\tag{10}
$$

These are the quadratic Jury conditions. They establish decay for this
constant-coefficient closed response, not for the full nonlinear network.
Given independently justified $|d_k|\le\delta_k$, its position error is bounded
by $\sum_{k<n}|(M^{n-1-k})_{12}|\delta_k$. Finite-amplitude or approximate
matching errors can be included as a two-component forcing in (9).

**Coefficient identification and rejection.** Fix $w$ and the identification
directions before continuation. Where possible, choose two directions with
$(q,p)=(1,0)$ and $(0,1)$ and evaluate the derivative of one GD step at the
checkpoint. Its second observable identifies $\eta\kappa$ and $1-\eta r$.
This uses checkpoint derivatives and one virtual update, not a regression
against future motion. Report rank and conditioning; if these observable
directions cannot be resolved, this closure is not identified.

Identification along two directions does not establish closure elsewhere.
The defect to check or bound on the reachable response directions is

$$
\begin{aligned}
\mathcal D_\theta(v)={}&w^TDg_a(G(\theta))(I-\eta Dg(\theta))v\\
 &-(1-\eta r)w^TDg_a(\theta)v-\eta\kappa w^TP_av.
\end{aligned}\tag{11}
$$

Then $d_n=\mathcal D_{\theta_n}(v_n)$. This definition is target-general;
smallness of the defect is an additional hypothesis. A wrong predicted sign,
rate, or response curve beyond the declared numerical and closure allowances
rejects the proposed closure. It cannot be rescued by fitting new coefficients
to that same curve. Uniform control over a neighborhood would upgrade the
local experiment into a conditional finite-time response theorem. It would
still not prove existence of a slow manifold.

Full-field and effective-field response forecasts must use the same initial
matching convention. If an effective predictor omits $DR_a$, expose its
initial-force mismatch or anchor it at each pulse's measured full gradient
and charge the omitted derivative thereafter. Otherwise apparent restoration
could merely reflect inconsistent initialization of the predictors.

## 4. Exact cloning separates readout allocation from a width effect

**Example.** Replace a neuron by $k$ identical copies with readout $c/k$.
The function is unchanged, but ordinary GD gives each copy a slope gradient
$1/k$ as large. A wider representation can therefore change the optimizer
without changing its initial approximation problem.

**Exact conjugacy proposition.** Split every neuron this way and preserve the
output bias. Give slope and hidden-bias blocks learning rate $k\eta$, readouts
rate $\eta/k$, and output bias rate $\eta$. Starting from identical copies,
every subsequent copy has the original slope and bias and $1/k$ of the
original readout. The aggregate function and parameter trajectory coincide
with the unsplit ordinary-GD run.

The proof is induction: copy slope/bias gradients divide by $k$, copy readout
gradients equal the original, and output-bias gradients agree. The stated
rates cancel these factors. Deterministic identical copies do not diversify
spontaneously, so this experiment can be run exactly in the smaller quotient
representation. Ordinary-rate clones instead correspond in that quotient to
slope/bias rate $\eta/k$ and readout rate $k\eta$.

The effective map must be recomputed for these rates. For a fixed positive
diagonal metric $M$, with update $-\eta Mg$, define

$$
\begin{aligned}
B_M&=(J_CM J_C^T)^{-1}J_CM J_H^T,& z_M&=e_C+B_Me,\\
T_M&=J_H^T-J_C^TB_M,& S_M&=J_HM T_M=T_M^TM T_M.
\end{aligned}\tag{12}
$$

Then $J_CM T_M=0$, the effective velocity is $-M T_Me$, and the tracking
channels are $(MJ_C^Tz_M)_a$ and $J_HMJ_C^Tz_M$. Using the unchanged
Euclidean balance map for a modified-rate branch would misattribute its force.

**Prediction.** Compensated cloning must reproduce the original physical
trajectory, up to measured numerical error. Failure is an implementation
problem. Ordinary-rate cloning tests the readout/geometry timescale change;
it is not an independent test of approximation capacity. If simultaneously
$N_{\rm ref}$ increases by $k$, then $h$ divides by $k$: its initial signed
normalized-slope update divides by $k^2$ under ordinary rates and by $k$
under compensated rates. For absolute slopes this statement holds away from
sign crossings. Even compensated clones have $\lambda$ smaller by $k$ at
every corresponding time. None of these exact representation identities
establishes the scaling of independently initialized wider networks.

## 5. Adam: alternating force can spend motion without acquiring scale

**Example.** A deterministic gradient can alternate while its average remains
small. Adam retains that alternation in its second moment. Large absolute
travel can then coexist with little net slope acquisition, without SGD noise.

Take $\eta>0$, $0\le\beta_1,\beta_2<1$, and $\epsilon>0$. For each
coordinate, preserve the optimizer age and incoming moments:

$$
m_{t+1}=\beta_1m_t+(1-\beta_1)g_t,\quad
v_{t+1}=\beta_2v_t+(1-\beta_2)g_t^2,\quad
\theta_{t+1}=\theta_t-\eta\frac{\widehat m_{t+1}}
 {\sqrt{\widehat v_{t+1}}+\epsilon}.
\tag{13}
$$

Hats apply the usual age-dependent bias corrections. The first moment can be
decomposed linearly into histories of $Te$, $r_C$, and $g_\perp$, plus the
decayed incoming moment. The second moment contains their cross terms and
cannot be obtained by adding separate channel variances. Adam therefore
requires an extended-state response model; it is not ordinary GD with a
diagonal metric applied to the current gradient.

For an imposed two-phase gradient sequence $g_+=\mu+A$, $g_-=\mu-A$, its
limiting periodic moments satisfy

$$
m_\pm=\mu\pm d_1A,\qquad
v_\pm=\mu^2+A^2\pm2d_2\mu A,\qquad
d_i=\frac{1-\beta_i}{1+\beta_i}.
\tag{14}
$$

Solving the two linear recurrences proves (14). On that periodic orbit,
without bias correction or in its infinite-age limit, the exact two-step
displacement is

$$
-\eta\left[
\frac{\mu+d_1A}{\sqrt{\mu^2+A^2+2d_2\mu A}+\epsilon}
+\frac{\mu-d_1A}{\sqrt{\mu^2+A^2-2d_2\mu A}+\epsilon}
\right].\tag{15}
$$

At $\mu=0$, this displacement is zero although each step has magnitude
$\eta d_1|A|/(|A|+\epsilon)$. Near zero mean, with $A\ne0$, the derivative
of the average step with respect to $\mu$ is

$$
-\frac{\eta}{|A|+\epsilon}
 \left(1-\frac{d_1d_2|A|}{|A|+\epsilon}\right).
\tag{16}
$$

**Prediction.** Estimate $g_+$ from the saved checkpoint and $g_-$ after one
virtual ordinary Adam step with its preserved moments. Freeze those two input
phases and propagate (13), including the actual bias-correction indices.
This gives a finite-history forecast; (15) is its limiting interpretation,
not a substitute for that propagation. Its predictive assumption is that
the local phases persist. Check phase means, alternating amplitudes, and net
normalized displacement separately. Do not label $A$ a coarse force unless
the measured channel histories support that attribution.

A denominator intervention with the same numerator has the exact immediate
contrast $-\eta\widehat m(1/D'-1/D)$, where $D,D'>0$. This alone says nothing
about later acquisition: altered states change both future moments. A valid
mechanism test forecasts that coupled continuation or explicitly limits its
claim to the immediate mobility response.

## 6. Turn a mechanism forecast into normalized acquisition rates

**Example.** Positive mean outward motion does not mean a population reaches
a useful scale. Initial large tails and the distance each remaining neuron
must travel both matter.

Define $h=2/N_{\rm ref}$ and $\lambda_j=h|a_j|$. The construction resolution
$N_{\rm ref}$ is distinct from physical width, which may include halo neurons.
The threshold $\lambda_*=0.25$ means $|a|=N_{\rm ref}/8$; it is not always
$|a|=64$. Normalizing the reported coordinate does not change the GD update.

Suppose a prospective mechanism forecast supplies $\widehat a_{j,n}$, a
signed correction $D_{j,n}$, and a proved conditional error allowance
$|a_{j,n}-\widehat a_{j,n}-D_{j,n}|\le E_{j,n}$. Define

$$
M_j^\lambda(N)=h\max_{0\le n\le N}
       (|\widehat a_{j,n}+D_{j,n}|+E_{j,n}).
\tag{17}
$$

Then the fraction that has ever reached $\lambda_*$ by update $N$ is at most

$$
\frac1W\#\{j:M_j^\lambda(N)\ge\lambda_*\}.
\tag{18}
$$

If this is below a desired fraction $p$, its first attainment time exceeds
$N$. For newly acquired neurons restrict the count to
$\lambda_{j,0}<\lambda_*$. These statements follow coordinate by coordinate;
a bound on the simultaneous norm at each time does not automatically bound
the union of neurons hitting at different times.

The instantaneous normalized rate, away from a zero crossing, is
$-\eta h\,\operatorname{sign}(a_{j,n})g_{a,j,n}$ per update. At crossings use
the exact difference $h(|a_{j,n}-\eta g_{a,j,n}|-|a_{j,n}|)$. Report physical
training time $\eta n$, additional updates, and total updates distinctly.
There is no general $1/W$ acquisition law without control of the effective
map, residual loadings, readouts, initialization, and step sizes.

### What makes the allowance prospective?

For the frozen reference (3), put $A_0=I-\eta S_0$ and
$\Phi_m=\eta\sum_{\ell<m}A_0^\ell$. Define

$$
\delta g_k=(T_k-T_0)e_k+r_{C,k}+g_{\perp,k},\qquad
\xi_k=(S_k-S_0)e_k+J_{H,k}r_{C,k}+J_{H,k}g_{\perp,k}-\tau_k/\eta.
$$

Direct substitution into (2), followed by exchanging the two finite sums,
gives the exact full-parameter identity

$$
\theta_n-\widehat\theta_n
=-\eta\sum_{k<n}\delta g_k
 +\eta\sum_{k<n}T_0\Phi_{n-1-k}\xi_k.
\tag{19}
$$

Preserve signs when combining map change with its induced residual response.
For example, a reference contribution at time $k$ is
$-(T-T_0)e+T_0\Phi_{n-1-k}(S-S_0)e$, evaluated on a predeclared forecast path.
Do not assume both map changes are tiny: the existing audit contradicts that
as a general long-window hypothesis.

Tracking contributes a coordinate allowance no larger than

$$
\eta\sum_{k<n}\left[
u_{j,k}+\|\operatorname{row}_j(T_0\Phi_{n-1-k})\|\,v_k
\right],\tag{20}
$$

provided $|(r_C)_j|\le u_{j,k}$ and $\|J_Hr_C\|\le v_k$.
The omitted residual has an analogous two-channel allowance; the finite-step
allowance is $\sum_{k<n}\|\operatorname{row}_j(T_0\Phi_{n-1-k})\|\,\|\tau_k\|$.
Map-response uncertainty is charged after combining its signed reference
terms. Small tracking alone does not make the total budget small.

For a genuine conditional enclosure, these bounds must hold on predeclared
full-parameter neighborhoods and candidate update segments. Establish
self-consistent inclusion by induction or first exit using (19); do not assume
the unknown path already stays near the forecast. The campaign can condition
on persistent small tracking in both channels without claiming to prove entry
into that regime. It still must control all other terms and all parameter
blocks needed for inclusion. Checkpoint derivatives or sampled defects alone
are empirical diagnostics, not uniform bounds.

**Deliverable and failure criterion.** The useful output is an excluded-fraction
curve versus horizon for $\lambda_*=0.25$, with the controlling term identified.
An empirical forecast, an FP64-evaluated conditional enclosure, and a validated
enclosure with rigorous numerical error control are different deliverables.
If the map-response or closure allowance consumes the acquisition margin,
the proposed theorem is uninformative for that case. That is a concrete reason
to refine its coupled mechanism, not permission to weaken its hypotheses or
infer a barrier from the small tracking measurements alone.
