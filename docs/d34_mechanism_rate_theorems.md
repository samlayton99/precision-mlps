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

## 7. A loaded-response condition that closes the ordinary-GD envelope

**Example.** The frozen reference can leave a large distance to $\lambda=0.25$
while its map changes appreciably. Requiring a small unweighted $\|T-T_0\|$
would discard this case. A more relevant question is whether the *loaded,
time-propagated response* varies enough inside the proposed neighborhood to
consume that distance. This section gives an explicit sufficient test.

All references below are constructed from the checkpoint and prescribed
model calculations, not selected to follow future true states. Every bound
is conditional on its stated neighborhood estimates; samples of a Jacobian
are not a supremum over that neighborhood.

### A single-state functional, including the finite-step error

For $0\le k<n$, abbreviate the fixed propagation matrix by
$Q_{n,k}=T_0\Phi_{n-1-k}$. Define functions of one parameter state $\theta$:

$$
\begin{aligned}
\mathcal K_{n,k}(\theta)
 &=-[T(\theta)-T_0]e_H(\theta)
     +Q_{n,k}[S(\theta)-S_0]e_H(\theta),\\
\mathcal C_{n,k}(\theta)
 &=-r_C(\theta)+Q_{n,k}J_H(\theta)r_C(\theta),\\
\mathcal O_{n,k}(\theta)
 &=-g_\perp(\theta)+Q_{n,k}J_H(\theta)g_\perp(\theta),\\
\tau(\theta)
 &=e_H(G(\theta))-e_H(\theta)+\eta J_H(\theta)g(\theta),\qquad
 G(\theta)=\theta-\eta g(\theta).
\end{aligned}\tag{21}
$$

In particular, $e_H(\theta)$ is always the nonlinear network residual
projected into the fixed fine basis. It is not an independent auxiliary
variable in (21). Evaluating $G(\theta)$ is one analytic virtual GD update;
$\tau(\theta)$ does not require an unknown next trajectory state. The exact
identity (19) is equivalently

$$
\theta_n=\widehat\theta_n
 +\eta\sum_{k<n}\bigl[\mathcal K_{n,k}(\theta_k)
       +\mathcal C_{n,k}(\theta_k)+\mathcal O_{n,k}(\theta_k)\bigr]
 -\sum_{k<n}Q_{n,k}\tau(\theta_k).
\tag{22}
$$

All sums start at the checkpoint, $k=0$. At $k=n-1$, $Q_{n,n-1}=0$,
so the most recent residual update has no propagated parameter contribution.
The final sum has no extra factor of $\eta$.

### Proposition: a full-parameter neighborhood inclusion test

Choose reference states $\bar\theta_0=\theta_0,\ldots,\bar\theta_N$ and
coordinate radii $r_k\ge0$. Let

$$
\mathcal D_k=\{\theta:|\theta-\bar\theta_k|\le r_k\},\qquad
b_n=\widehat\theta_n+\eta\sum_{k<n}\mathcal K_{n,k}(\bar\theta_k)
       -\bar\theta_n.
\tag{23}
$$

Absolute values and inequalities between vectors are coordinatewise. Assume
the coarse solve remains invertible on each box. Suppose nonnegative matrices
$L_{n,k}$ and nonnegative vectors $c_n$ satisfy

$$
\begin{aligned}
|D\mathcal K_{n,k}(\theta)|&\le L_{n,k}
       &&\text{for every }\theta\in\mathcal D_k,\\
\left|\eta\sum_{k<n}(\mathcal C_{n,k}(\theta_k)
                  +\mathcal O_{n,k}(\theta_k))
       -\sum_{k<n}Q_{n,k}\tau(\theta_k)\right|&\le c_n
       &&\text{for every choice }\theta_k\in\mathcal D_k.
\end{aligned}\tag{24}
$$

The separate two-channel allowances of (20), together with omitted-mode and
finite-step allowances, are one sufficient way to obtain $c_n$. In particular,
uniform small tracking shrinks it through both $(r_C)_a$ and $J_Hr_C$.
For full-parameter inclusion the other parameter coordinates also need
bounds; small slope tracking does not supply them automatically.

If

$$
|b_n|+c_n+\eta\sum_{k<n}L_{n,k}r_k\le r_n
\quad\text{for every }1\le n\le N,
\tag{25}
$$

then ordinary GD stays in these boxes through $N$. More precisely it satisfies

$$
\left|\theta_n-\widehat\theta_n
 -\eta\sum_{k<n}\mathcal K_{n,k}(\bar\theta_k)\right|
\le c_n+\eta\sum_{k<n}L_{n,k}r_k.
\tag{26}
$$

**Proof.** The initial state is in its box. Suppose all previous states are
in theirs. Each box is convex, so the mean-value integral and (24) bound
$|\mathcal K_{n,k}(\theta_k)-\mathcal K_{n,k}(\bar\theta_k)|$
by $L_{n,k}r_k$. Substituting into (22) proves (26). Adding $|b_n|$ then gives
(25), placing the next state in its box. Induction proves both conclusions.
If the derivative estimates require smoothness beyond the box because of
$\tau$, bound the image $G(\mathcal D_k)$ and its connecting update segments
as part of those estimates; this is not supplied by trajectory inclusion alone.

The center in (26) retains the signed reference correction. The boxes in
(23) are centered at the independently chosen $\bar\theta_k$. These need not
coincide. In particular, choosing $\bar\theta=\widehat\theta$ requires the
box radius to pay for the entire reference correction $b_n$ even when the
final acquisition envelope benefits from its sign.

### Scalar and block conditions an implementation can check

For common radii $r_k=r$ choose a nonnegative matrix $A$ and vector $d$ with

$$
A\ge\eta\sum_{k<n}L_{n,k},\qquad d\ge |b_n|+c_n
\quad\text{for all }n\le N.
\tag{27}
$$

Then $d+Ar\le r$ suffices. If the spectral radius of $A$ is less than one,
$(I-A)^{-1}$ is nonnegative and $r=(I-A)^{-1}d$ is a candidate. The estimates
must be established on the boxes with that candidate radius; evaluating $A$
on a smaller box and enlarging the radius afterward is invalid.

For a cheaper scalar criterion choose positive coordinate scales $w$ and
$W_d=\operatorname{diag}(w)$. Use boxes $r=Rw$ and establish

$$
a\ge\max_{n\le N}\eta\sum_{k<n}
 \sup_{\theta\in\mathcal D_k}
 \|W_d^{-1}D\mathcal K_{n,k}(\theta)W_d\|_\infty,
\qquad d_w\ge\max_{n\le N}\|W_d^{-1}(|b_n|+c_n)\|_\infty.
\tag{28}
$$

The conditions $a<1$ and $R\ge d_w/(1-a)$ imply inclusion. Block norms for
slopes, biases, readouts, and output bias give an intermediate small
nonnegative matrix condition. The sequential criterion (25) can be sharper
than either common-radius reduction. Failure of these sufficient conditions
does not establish that the network escapes; it identifies a failed enclosure.

One can improve the reference centers without inspecting the true future.
For example, a prescribed map-only Volterra reference is defined causally by

$$
\bar\theta_0=\theta_0,\qquad
\bar\theta_n=\widehat\theta_n
 +\eta\sum_{k<n}\mathcal K_{n,k}(\bar\theta_k).
\tag{29}
$$

Here $b_n=0$. Equation (29) still needs its own computation and may be costly;
it is a nonlinear surrogate, not a free consequence of freezing $T$. Its
omission of tracking, omitted modes, and finite-step terms is charged by
$c_n$. If instead one recursively includes every exact channel of (22), the
reference reproduces ordinary GD itself. That is not an independent reduced
model or a mechanistic simplification.

### Exact loaded derivatives and the missing numerical ingredient

For any direction $v$, with all unmarked quantities evaluated at $\theta$,

$$
\begin{aligned}
D\mathcal K_{n,k}[v]
={}&-(DT[v])e-(T-T_0)J_Hv\\
 &+Q_{n,k}\{[(DT[v])^TT+T^TDT[v]]e+(S-S_0)J_Hv\}.
\end{aligned}\tag{30}
$$

Thus the quantity to control contains the current residual load and the
paired response. Bounding $DT$ and $e$ separately can lose decisive modal
cancellation. At the checkpoint the terms with $T-T_0$ and $S-S_0$ vanish,
but $DT[v]e$ generally does not.

These derivatives are computable without a future trajectory. Write
$K_C=J_CJ_C^T$. Then

$$
\begin{aligned}
DK_C[v]&=(DJ_C[v])J_C^T+J_C(DJ_C[v])^T,\\
DB[v]&=K_C^{-1}\{(DJ_C[v])J_H^T+J_C(DJ_H[v])^T-DK_C[v]B\},\\
DT[v]&=(DJ_H[v])^T-(DJ_C[v])^TB-J_C^TDB[v].
\end{aligned}\tag{31}
$$

For a version that also keeps signed reference tracking or finite-step
corrections, differentiate those functions in (21). In particular,

$$
D\tau(\theta)[v]
=J_H(G(\theta))(I-\eta H(\theta))v-J_H(\theta)v
 +\eta\{DJ_H(\theta)[v]g(\theta)+J_H(\theta)H(\theta)v\}.
\tag{32}
$$

This makes explicit the full loss Hessian and the nonlinear projection
consistency; neither can be silently replaced by an auxiliary residual state.
Signed reference corrections can be moved into $b_n$, while their derivative
uncertainty is added to $L_{n,k}$. For the finite-step channel this means adding
a bound for $D[-Q_{n,k}\tau]/\eta$, because (25) places an outer $\eta$
before $L_{n,k}$ and the last sum in (22) has none. Uniform upper bounds still need proof on
the proposed boxes, for example from analytic derivative bounds or validated
interval evaluation. FP64 point derivatives and refinement of the reference
sum are useful diagnostics of that task, not its completion.

**Prediction and useful failure report.** Apply (17)–(18) with the signed center
in (26) and its explicit uncertainty. Report whether loss of exclusion comes
from the reference correction itself, the two tracking channels, omitted or
finite-step terms, or the loaded-response neighborhood gain. This separates
a mechanism that actually predicts acquisition from an error bound that is
too loose to decide. It also states precisely what must improve before the
strong frozen-model exclusion can become an ordinary-GD theorem.

## 8. A width-dependent rate without freezing the effective map

**Example.** In the independent-width experiment, the slopes, hidden biases,
and readouts all remain comparable to $W^{-1/2}$ at the 20k forks. Increasing
width then makes the normalized slope motion much slower, even though a
constant-force forecast works about as well as evolving the fine residual.
This suggests a different question from depletion: can the remaining force
already be small because the features are almost affine?

The following result answers that question conditionally. It uses the exact
tanh model, permits any target values on the empirical training grid, and
does not freeze $T$. Its width orders are bounds within a stated parameter
regime, not a claim that all training trajectories enter or stay there.

### Proposition: small parameters suppress the balanced fine force

Assume $|x_i|\le1$. In this section only, let $H$ be the **entire** empirical
orthogonal complement of $\operatorname{span}\{1,x\}$, so $g_\perp=0$.
The inner product is still the empirical mean. Use any orthonormal coordinates
for this complement; the bounds do not depend on that choice. Suppose

$$
|a_j|\le\frac{A_a}{\sqrt W},\qquad
|b_j|\le\frac{A_b}{\sqrt W},\qquad
|c_j|\le\frac{A_c}{\sqrt W},\qquad
\lambda_{\min}(J_CJ_C^T)\ge\kappa_C>0.
\tag{33}
$$

There is no smallness requirement on the output bias $d$. Define

$$
\begin{aligned}
U&=A_a+A_b,\qquad
L_H=\sqrt{2A_c^2U^4+U^6/9},\\
Y_H&=\|P_Hy\|_m,\qquad
E_W=Y_H+\frac{A_cU^3}{3W},\\
K_a=K_b&=E_W A_c\left(U^2+\frac{L_H}{\sqrt{\kappa_C}}\right),\\
K_c&=E_W\left(\frac{U^3}{3}
                  +\frac{U L_H}{\sqrt{\kappa_C}}\right).
\end{aligned}\tag{34}
$$

Then the exact balanced force obeys, for every neuron,

$$
|(Te_H)_{a,j}|\le\frac{K_a}{W^{3/2}},\quad
|(Te_H)_{b,j}|\le\frac{K_b}{W^{3/2}},\quad
|(Te_H)_{c,j}|\le\frac{K_c}{W^{3/2}}.
\tag{35}
$$

Moreover,

$$
\|J_H\|_F\le\frac{L_H}{W},\qquad
\|e_H\|\le E_W,\qquad
\|S\|\le\frac{L_H^2}{W^2}.
\tag{36}
$$

**Proof.** Write $u_j=a_jx+b_j$, so $|u_j|\le U/\sqrt W$.
The exact inequalities

$$
|\tanh u-u|\le |u|^3/3,\qquad
|\operatorname{sech}^2u-1|=\tanh^2u\le u^2
\tag{37}
$$

follow by integrating $\tanh'(u)-1=-\tanh^2u$ and using
$|\tanh u|\le|u|$. Thus this argument is a remainder bound, not an
uncontrolled replacement of tanh by its cubic Taylor polynomial.

The projection $P_H$ annihilates $1,x,u_j$. The fine parts of the slope,
bias, and readout Jacobian columns consequently have norms at most
$A_cU^2/W^{3/2}$, $A_cU^2/W^{3/2}$, and $U^3/(3W^{3/2})$, respectively.
The output-bias column has no fine component. Summing their squared norms
proves the first part of (36). Applying (37) to
$P_Hf=\sum_j c_jP_H(\tanh u_j-u_j)$ proves its residual bound.

For the coarse correction, the slope and bias columns of $J_C$ have norm
at most $A_c/\sqrt W$, and its readout columns have norm at most
$U/\sqrt W$. Since

$$
\|(J_CJ_C^T)^{-1}J_C\|\le\kappa_C^{-1/2},\qquad
\|Be_H\|\le\frac{L_HE_W}{\sqrt{\kappa_C}W},
$$

bounding each column of $J_H^Te_H-J_C^TBe_H$ proves (35). This explicitly
retains the balanced coarse contribution. Finally,
$T=(I-J_C^T(J_CJ_C^T)^{-1}J_C)J_H^T$ is an orthogonal projection of
$J_H^T$, so $\|S\|=\|T^TT\|\le\|J_H\|^2\le L_H^2/W^2$.

### Corollary: conditional slow acquisition and a regime that can be closed

Suppose coarse tracking satisfies the absolute component bounds

$$
|(r_C)_{\ell,j}|\le\frac{\delta_\ell}{W^{3/2}},
\qquad \ell\in\{a,b,c\}.
\tag{38}
$$

For every update in this regime, including slope sign crossings,

$$
|\lambda_{j,n+1}-\lambda_{j,n}|
\le\frac{\eta h(K_a+\delta_a)}{W^{3/2}}.
\tag{39}
$$

This follows directly from $||a-\eta g_a|-|a||\le\eta|g_a|$.
When $N_{\rm ref}$ is proportional to $W$, $h$ is proportional to $W^{-1}$,
so the conditional normalized rate is $O(\eta W^{-5/2})$ per update.
This order requires $A_a,A_b,A_c,\kappa_C^{-1},\delta_a,\delta_b,\delta_c$
and $Y_H$ bounded uniformly across widths. Without those uniform hypotheses
and the stated width convention, the precise statement is (39).

There is also a first-exit version. Suppose the initial bounds in (33) hold
with smaller constants $A_{a,0},A_{b,0},A_{c,0}$. Assume the coarse
conditioning and tracking hypotheses continue to hold through the proposed
interval whenever the three parameter blocks satisfy the outer bounds (33).
If

$$
\frac{\eta N}{W}(K_\ell+\delta_\ell)
\le A_\ell-A_{\ell,0}
\quad\text{for }\ell=a,b,c,
\tag{40}
$$

the outer parameter bounds hold through $N$ updates. Indeed, assuming them
at preceding states, summing (35) and (38) bounds each coordinate's movement
by $\eta N(K_\ell+\delta_\ell)/W^{3/2}$, which fits its available margin.
Induction closes the three blocks. Every slope then has
$\lambda_{j,n}\le hA_a/\sqrt W$ for $n\le N$. If that value is below
$\lambda_*$, none can acquire the specified normalized scale during the
interval. The constants in this sufficient bound can be conservative; it is
not a numerical claim about a particular width until evaluated there.

These assumptions also address the second tracking channel. For
$\delta=\max_\ell\delta_\ell$, the non-output-bias tracking vector has
norm at most $\sqrt3\delta/W$. Because the fine Jacobian annihilates the
output-bias direction,
$\|J_Hr_C\|\le\sqrt3L_H\delta/W^2$.
Small slope tracking alone would not give this conclusion.

### Why geometry can evolve before the fine error appreciably relaxes

The same assumptions distinguish two timescales. Put
$G^2=\sum_{\ell=a,b,c}(K_\ell+\delta_\ell)^2$,
$\delta_2=(\delta_a^2+\delta_b^2+\delta_c^2)^{1/2}$, and
$L_2=4A_cU+\sqrt2U^2$. Through a closed interval from (40),

$$
\|e_{H,n}-e_{H,0}\|
\le\frac{\eta n}{W^2}
 \left(L_H^2E_W+L_H\delta_2+\frac{\eta L_2G^2}{2W}\right).
\tag{41}
$$

To prove the finite-step term, subtract
$\sum_j c_j(a_jx+b_j)$ before differentiating the fine output. Its fine
projection is zero. The slope/bias Hessian block of the remainder has norm
at most $4A_cU/W$, and its mixed readout–geometry block has norm at most
$\sqrt2U^2/W$. The block-diagonal neuron structure therefore gives
$\|D^2 e_H\|\le L_2/W$ on the convex parameter box. The output bias has
no fine derivative, so its movement does not enter this estimate. The other
three blocks move by at most $\eta G/W$ in Euclidean norm per update.
Consequently $\|\tau_n\|\le\eta^2L_2G^2/(2W^3)$. Sum the exact residual
equation (2), using (36) and
$\|J_Hr_C\|\le L_H\delta_2/W^2$, to obtain (41).

On a window with $\eta n$ proportional to $W$, the permitted parameter
motion is an order-one fraction of the initial $W^{-1/2}$ parameter scale,
whereas the fine residual changes by at most $O(W^{-1})$. Thus changing
geometry and readout correlations can matter before the driving error has
substantially depleted. This is a conditional separation of timescales; it
does not assert that parameters actually move by their upper bounds or that
the same box remains valid for times proportional to $W^2$.

**Prediction and scope.** At comparable states with bounded rescaled
parameters and coarse conditioning, $W^{3/2}$ times the per-neuron effective
force and $W\|J_H\|_F$ should remain comparable across widths. The fixed-map
residual clock is at most $\eta N L_H^2/W^2$, so short windows at large width
need not exhibit appreciable depletion. Those predictions are distinct from
the later-checkpoint evidence that residual evolution improves forecasts.
Neither requires readouts to dominate dissipation, and neither implies that
the network can never leave the small-parameter regime. The hypothesis about
coarse tracking and its persistence remains explicit in (38)–(40).
