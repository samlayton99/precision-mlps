# Why a small effective force can persist

A small slope gradient explains slow motion at one checkpoint. It does not
explain why the motion stays slow. The missing question is whether the
network's subsequent motion strengthens or depletes its own effective fine
force. This note identifies the two signed contributions to that feedback,
gives a conditional discrete-GD persistence theorem, and derives interventions
whose initial effects have a prescribed sign and magnitude.

The theorem applies to the full fine residual, including targets with
accessible low-order components such as sine. It does not require ninth-degree
orthogonality, a frozen feature map, or universal contraction. Its substantive
assumption is a regional bound on **loaded force amplification**. That bound
must be established separately; a value measured at one checkpoint is not a
proof of its persistence. No new numerical results are asserted here.

For mechanisms that derive slow rates from a specified coupled ODE, use the
[companion mechanism note](d34_coupled_ode_mechanisms.md). It proves passage
delays from readout compensation and signed target/error competition, and
derives algebraic slowing when correction weakens its own sensitivity. The
system need not approach an equilibrium. Its assumption ledger distinguishes
the proved reduced-system statements from the unverified conditions needed
for a heterogeneous tanh-GD population. The role of the present note is to
turn a justified persistence condition into discrete motion and acquisition
bounds.

**Notation.** All norms below are Euclidean norms in orthonormal empirical
coordinates; the loss uses the empirical mean normalization.

| Symbol | Meaning |
|---|---|
| $\theta=(a,b,c,d)$ | All slopes, hidden biases, readouts and output bias |
| $e_C,e_H$ | Constant/linear residual coefficients and their full empirical orthogonal complement |
| $J_C,J_H$ | Parameter Jacobians of these residual coordinates |
| $F$ | Effective fine **gradient**; its learning velocity is $-F$ |
| $R$ | Coarse-disequilibrium correction to the full gradient |
| $V=\|F\|^2/2$ | Effective-force energy, distinct from training loss |
| $\mathcal D,\mathcal C$ | Nonnegative residual-relaxation term and signed loaded-curvature term |
| $W$, $h=2/N_{\rm ref}$ | Physical neuron count and construction spacing; these are distinct |
| $\lambda_j=h|a_j|$ | Normalized slope scale of neuron $j$ |

## 1. An accessible target component need not produce outward motion

Consider the leading small-parameter transport model from
[the rescaled transport note](d34_rescaled_transport_model.md). On a symmetric
input grid, take an odd target, zero rescaled hidden biases, and write
$X=(\alpha,0,\zeta)=\sqrt W(a,0,c)$. Its nonzero cubic target loading is
$m_3=-\langle y_H,x^3\rangle_m$. Define

$$
U=\mathbb E(\zeta^2+\alpha^2),\qquad
M=\mathbb E(\zeta^2\alpha^2+\alpha^4/3),\qquad
\mu=M/U.
$$

In the time variable $\tau=\eta n/W$, the balanced characteristic equations are

$$
\alpha'=m_3\zeta(\alpha^2-\mu),\qquad
\zeta'=m_3\alpha(\alpha^2/3-\mu).
\tag{1}
$$

The coarse coefficient $p=\mathbb E(\alpha\zeta)$ is conserved, since
$p'=m_3(M-\mu U)=0$. But the squared slope of an individual particle changes
with the sign of

$$
\alpha\alpha'=m_3\alpha\zeta(\alpha^2-\mu).
\tag{2}
$$

Even when $m_3$ is appreciable, the distribution's fourth moments and the
alignment of readouts with slopes decide this sign. The compensation in (1)
is part of the effective fine force. It is present at exact coarse balance;
it is not a neglected disequilibrium force.

Here is a sharp illustration. In distribution A, put equal mass at
$(\alpha,\zeta)=(1,1)$ and $(-1,-1)$. In distribution B, put mass $1/4$ at
each of those points, mass $1/12$ at each of
$(\sqrt3,\sqrt3)$ and $(-\sqrt3,-\sqrt3)$, and mass $1/3$ at $(0,0)$.
Both distributions have zero means and
$\mathbb E\alpha^2=\mathbb E\zeta^2=\mathbb E\alpha\zeta=1$.
Thus they have identical second moments, coarse slope and leading coarse
conditioning. Nevertheless, $\mu_A=2/3$ and $\mu_B=4/3$. The shared particle
$(1,1)$ has velocities $m_3/3$ and $-m_3/3$, respectively. Second moments and
coarse fit alone do not close even the instantaneous slope equation.

For one atom, the same model can be solved far enough to exhibit a hitting
time. Let $\alpha_0>0$ and $p=\alpha\zeta\ne0$. Then

$$
\alpha'=\chi\frac{\alpha^5}{\alpha^4+p^2},\qquad
\chi=\frac{2m_3p}{3},\qquad \zeta=p/\alpha.
\tag{3}
$$

If $\chi<0$, slopes contract while readout magnitude grows. If $\chi>0$, the model
reaches any fixed $A>\alpha_0$ at

$$
\tau_{\rm hit}(A)=\frac{1}{\chi}\left[
\log\frac{A}{\alpha_0}
+\frac{p^2}{4}\left(\alpha_0^{-4}-A^{-4}\right)\right].
\tag{4}
$$

Equation (4) follows by integrating
$d\tau=(\alpha^{-1}+p^2\alpha^{-5})d\alpha/\chi$. It exhibits a long delay
when $\alpha_0^4\ll p^2$, despite a nonzero target loading. It also rules out
a universal contraction theorem based only on exact coarse balance.
These are exact statements about the leading polynomial model. Its
approximation to tanh GD requires the bounded-domain and tracking assumptions
in the linked transport note. A large physical acquisition threshold may lie
outside that approximation regime; (4) is not by itself a tanh hitting-time
theorem.

**Prediction suggested by this example.** A persistence theory should track
how the loaded effective map changes, including readout/geometry correlations.
It should not infer the sign of slope motion solely from target accessibility,
readout RMS, or coarse equilibrium. The next section identifies an exact
quantity for making this distinction without a polynomial truncation.

## 2. Exact mechanism: relaxation competes with loaded geometry

Work with the full empirical fine space and write

$$
L=\tfrac12\|e_C\|^2+\tfrac12\|e_H\|^2,\qquad
K=J_CJ_C^T,\qquad
B=K^{-1}J_CJ_H^T,\qquad \ell=Be_H.
$$

Here $\ell$ is the two-component coarse compensation multiplier. Assume $K$
is invertible and set

$$
\Pi=I-J_C^TK^{-1}J_C,\qquad
F=\Pi J_H^Te_H=J_H^Te_H-J_C^T\ell,
\qquad z_C=e_C+\ell,\qquad R=J_C^Tz_C.
\tag{5}
$$

Thus $\nabla L=F+R$ and $J_CF=0$. The effective flow preserves the coarse
outputs continuously; ordinary GD differs by $R$. The fine-residual equation
along ordinary gradient flow is

$$
\dot e_H=-J_HF-J_HR.
\tag{6}
$$

This displays the two familiar tracking channels: $R_a$ affects slope motion
directly and $J_HR$ affects residual evolution. A finite polynomial basis
does not equal the full empirical complement. If it is used for computation,
its omitted gradient must be retained as an additional correction; it cannot
silently disappear from (5).

Define the two scalar feedback terms

$$
\mathcal D=\|J_HF\|^2,\qquad
\mathcal C=
\langle e_H,D^2e_H[F,F]\rangle
-\langle \ell,D^2e_C[F,F]\rangle.
\tag{7}
$$

**Proposition 1 (force-energy identity).** Along the effective flow
$\dot\theta=-F$,

$$
\dot V=-\mathcal D-\mathcal C.
\tag{8}
$$

Along ordinary gradient flow $\dot\theta=-F-R$,

$$
\dot V=-\mathcal D-\mathcal C-F^TDF[R].
\tag{9}
$$

The first term is always damping: reducing the currently loaded residual
reduces the force it supplies. The second can have either sign: changing
features and their coarse compensation can strengthen or weaken that force.
Both terms belong to the effective fine dynamics. Small coarse tracking does
not remove $\mathcal C$.

**Proof.** Differentiate $F=\Pi J_H^Te_H$. For an orthogonal projector,
$F^TD\Pi[v]F=0$ whenever $\Pi F=F$. Differentiating
$\Pi J_C^T=0$ gives
$D\Pi[v]J_C^T=-\Pi D(J_C^T)[v]$. Since
$J_H^Te_H=F+J_C^T\ell$, contraction of the derivative with $F$ gives

$$
F^TDF[v]=
\langle J_HF,J_Hv\rangle
+\langle e_H,D^2e_H[v,F]\rangle
-\langle \ell,D^2e_C[v,F]\rangle.
\tag{10}
$$

Set $v=F$ and then apply $\dot V=F^TDF[\dot\theta]$.

When $F\ne0$, its intrinsic logarithmic speed rate is

$$
k(\theta)=-\frac{\mathcal D+\mathcal C}{\|F\|^2}.
\tag{11}
$$

Negative $k$ means depletion dominates amplification at that state. Positive
$k$ means the force strengthens. Neither sign determines whether individual
slopes move outward. At $F=0$, use (8) directly rather than assigning a rate
by dividing by a numerical floor.

There is one additional requirement when transferring this mechanism to
ordinary GD. Writing $T=\Pi J_H^T$ gives

$$
DF[R]=(DT[R])e_H+T(J_HR).
\tag{12}
$$

Small $R_a$ and $J_HR$ alone do not bound the first term in (12). A theorem
must also control the loaded response of the map to the correction, or bound
the full correction and a regional derivative. This is a transfer condition,
not evidence that tracking replaces the effective force as the driver.

## 3. A discrete persistence theorem with a first-exit closure

The identity becomes predictive if amplification is bounded on a specified
region around the checkpoint. The region and its constants are declared
before following the future trajectory. This permits a negative curvature
margin and a finite amplification clock; contraction is not built into the
hypotheses.

Let $\mathcal U$ be the open ball of radius $d>0$ about $\theta_0$ and let
$\mathcal U^+$ be the ball of radius $d+\delta$, with $\delta>0$.
Assume $F$ is twice continuously differentiable and $K$ is uniformly positive
definite on a neighborhood of the closure of $\mathcal U^+$. Suppose

$$
\mathcal D(\theta)+\mathcal C(\theta)
\ge m\|F(\theta)\|^2\quad(\theta\in\mathcal U),
\qquad
\|D^2V(\theta)\|_{\rm op}\le L_V
\quad(\theta\in\mathcal U^+),
\tag{13}
$$

where $m$ may be negative and $L_V\ge0$. For step size $\eta>0$, require

$$
r^2=1-2\eta m+\eta^2L_V\ge0,\qquad r=\sqrt{r^2}.
\tag{14}
$$

**Proposition 2 (effective discrete flow).** Put $s_0=\|F(\theta_0)\|$,
$\bar s_n=r^ns_0$, and $S_N=\max_{0\le n\le N}\bar s_n$. If

$$
\eta S_N<\delta,\qquad
\eta\sum_{k=0}^{N-1}\bar s_k<d,
\tag{15}
$$

then the Euler iterates $\theta_{n+1}=\theta_n-\eta F(\theta_n)$ remain in
$\mathcal U$ through step $N$ and satisfy

$$
\|F(\theta_n)\|\le\bar s_n,\qquad
\|\theta_n-\theta_0\|\le\eta\sum_{k<n}\bar s_k.
\tag{16}
$$

**Proof.** Whenever the current state is in $\mathcal U$ and its outgoing
segment is in $\mathcal U^+$, Taylor's theorem and (8), (13) give

$$
V(\theta-\eta F)
\le V(\theta)-\eta m\|F\|^2
+\tfrac12\eta^2L_V\|F\|^2
=r^2V(\theta).
$$

The padding condition in (15) contains that segment. The travel condition
places its endpoint in $\mathcal U$. Induction proves both claims, without
assuming that the unknown iterates stay in the region.

If $r<1$, the travel sum is bounded by $\eta s_0/(1-r)$. If $r=1$, it is
$\eta Ns_0$; if $r>1$, it is $\eta s_0(r^N-1)/(r-1)$. Thus the same theorem
distinguishes damping, nearly constant force and amplification. A positive
continuous-time margin $m$ alone is insufficient for discrete contraction:
the $\eta^2L_V$ term must also be controlled.

The regional condition in (13) is mechanistic: it compares residual depletion
with signed, residual-loaded curvature. It is not an assumption that future
forces are small. One sufficient, stronger condition is a lower spectral
bound on

$$
J_H^TJ_H+\sum_\ell e_{H,\ell}D^2e_{H,\ell}
-\sum_i \ell_iD^2e_{C,i}
$$

on the region. Bounding only its quadratic form in the current $F$ direction
can be substantially sharper. Neither condition is established by plotting
(11) along an already observed trajectory.

### Ordinary GD: an explicit correction allowance

For ordinary GD, additionally suppose $\|R(\theta)\|\le\rho$ on
$\mathcal U$. Assume an absolute loaded-response bound $u\ge0$ such that

$$
\left\|DF\big[\theta-\eta F(\theta)-t\eta R(\theta)\big]
\,R(\theta)\right\|\le u,
\qquad 0\le t\le1,
\tag{17}
$$

whenever $\theta\in\mathcal U$ and this segment is contained in
$\mathcal U^+$. Here $DF[x]v$ denotes the derivative at $x$ applied to $v$;
$R(\theta)$ is held fixed along the segment. A conservative sufficient
choice is $u=L_F\rho$, with $\|DF\|_{\rm op}\le L_F$ on $\mathcal U^+$.
A bound on the loaded derivative in (17) can retain much more structure.

**Proposition 3 (ordinary GD).** Define

$$
\bar s_0=s_0,\qquad \bar s_{n+1}=r\bar s_n+\eta u,
\qquad S_N=\max_{0\le n\le N}\bar s_n.
\tag{18}
$$

If

$$
\eta(S_N+\rho)<\delta,\qquad
\eta\sum_{k<N}(\bar s_k+\rho)<d,
\tag{19}
$$

then ordinary GD stays in $\mathcal U$ through $N$, with
$\|F(\theta_n)\|\le\bar s_n$ and
$\|\theta_n-\theta_0\|\le\eta\sum_{k<n}(\bar s_k+\rho)$.

**Proof.** First compare with the effective Euler endpoint
$\theta^E=\theta-\eta F(\theta)$, for which Proposition 2's one-step
estimate gives $\|F(\theta^E)\|\le r\|F(\theta)\|$. The actual endpoint
is $\theta^E-\eta R(\theta)$. Integrating (17) between these endpoints gives
$\|F(\theta_{n+1})\|\le r\|F(\theta_n)\|+\eta u$.
Since $r,u\ge0$, this scalar comparison is monotone in its input. Conditions
(19) close the same first-exit induction. The comparison sequence itself
need not increase with $n$; this is why the padding uses $S_N$.

For $r\ne1$ the explicit allowance is
$\bar s_n=r^ns_0+\eta u(1-r^n)/(1-r)$, and for $r=1$ it is
$s_0+n\eta u$. Persistent tracking can leave a speed floor even when $r<1$.
The theorem does not prove entry into low disequilibrium or its indefinite
persistence. It specifies the absolute regional allowances sufficient to
transfer the effective mechanism to ordinary GD.

## 4. From force persistence to normalized-scale acquisition rates

Force energy measures total mobility. Acquisition additionally requires
motion in the appropriate slope coordinates. To give a conservative
population statement, suppose the hypotheses of Proposition 3 hold and
$\|R_a\|\le\rho_a$ on $\mathcal U$. For $0\le\lambda_0<\lambda_*$,
let $p_{\rm init}$ be the fraction with $\lambda_j(0)>\lambda_0$, and let
$p_{\rm ever}(N)$ count distinct neurons that reach $\lambda_*$ at any step
up to $N$, including neurons already there initially. Then

$$
p_{\rm ever}(N)\le\min\left\{1,\;
p_{\rm init}
+\frac{h^2\eta^2}{W(\lambda_*-\lambda_0)^2}
\left[\sum_{k<N}(\bar s_k+\rho_a)\right]^2\right\}.
\tag{20}
$$

Indeed, every new hit from at most $\lambda_0$ requires slope path length at
least $(\lambda_*-\lambda_0)/h$. The vector of per-neuron absolute path
lengths has Euclidean norm at most
$\eta\sum_{k<N}\|F_{a,k}+R_{a,k}\|$, by the triangle inequality. Bound
the number of coordinates exceeding the required length by its squared norm
divided by that length squared. This counts distinct ever-hit labels, not
only the occupancy at the final step. The weaker action form replaces the
squared sum in (20) by $N\sum_{k<N}(\bar s_k+\rho_a)^2$; retaining the
sum before squaring is useful when force decays geometrically.

A per-neuron correction allowance $|R_{a,j}|\le\rho_{a,j}$ also gives

$$
|\lambda_j(n+1)-\lambda_j(n)|
\le h\eta(\bar s_n+\rho_{a,j}),\qquad
\lambda_j(n)\le\lambda_j(0)
+h\eta\sum_{k<n}(\bar s_k+\rho_{a,j}).
\tag{21}
$$

Choose the largest $N$ satisfying the regional conditions and the desired
population allowance in (20). This is a conditional acquisition-time lower
bound, derived from the state and its feedback constants. It does not select
a long horizon first and then infer stability from failure to acquire.
Neither an increase in $\|F\|$ nor a failure of these upper bounds proves
outward acquisition. Signed slope motion must be measured separately.
The normalization uses $h=2/N_{\rm ref}$; replacing it by $1/W$ without
accounting for the construction's physical neuron count would change the
claim.

## 5. Matched feedback interventions with signed predictions

The mechanism admits a sharper test than deleting arbitrary mode groups.
We can change geometry feedback and residual relaxation independently while
preserving the entire initial effective force. At a fork $\theta_0$, set
$F_0=F(\theta_0)$ and define the following fields at a variable state:

$$
Z(\theta)=\Pi_\theta F_0,\qquad
A(\theta)=\Pi_\theta J_H(\theta)^Te_H(\theta_0),\qquad
E(\theta)=F(\theta)-A(\theta),
$$

$$
F_{\kappa,\nu}(\theta)
=Z(\theta)+\kappa\{A(\theta)-Z(\theta)\}+\nu E(\theta).
\tag{22}
$$

These all equal $F_0$ at the fork and remain tangent to the current coarse
level sets. The six arms are:

| Arm | $\kappa$ | $\nu$ |
|---|---:|---:|
| Natural | 1 | 1 |
| Projected constant | 0 | 0 |
| No geometry feedback | 0 | 1 |
| Double geometry feedback | 2 | 1 |
| No residual relaxation | 1 | 0 |
| Double residual relaxation | 1 | 2 |

The applied gradient is $F_{\kappa,\nu}+R=\nabla L+F_{\kappa,\nu}-F$.
The natural arm is exactly ordinary GD. The others are controlled vector
fields, generally not gradients of the original loss. “Projected constant”
means a fixed vector projected onto the changing coarse tangent space; it
does not mean a globally constant parameter velocity. Euler steps preserve
coarse outputs only to first order, not exactly.

For a pure effective-flow intervention, the initial energy derivative is

$$
\left.\frac12\frac{d}{dt}\|F_{\kappa,\nu}\|^2\right|_0
=-\kappa\mathcal C_0-\nu\mathcal D_0.
\tag{23}
$$

For the applied field including $R$, define

$$
\mathcal C_{R,0}=
\langle e_{H,0},D^2e_H[R_0,F_0]\rangle
-\langle \ell_0,D^2e_C[R_0,F_0]\rangle,
\qquad
\mathcal D_{R,0}=\langle J_HF_0,J_HR_0\rangle.
$$

Then (23) becomes
$-\kappa(\mathcal C_0+\mathcal C_{R,0})
-\nu(\mathcal D_0+\mathcal D_{R,0})$.
Although $\mathcal D_0\ge0$, the correction $\mathcal D_{R,0}$ can be
negative and must be evaluated rather than discarded.

For completeness, let
$G_0=J_{H,0}^TJ_{H,0}$ and
$H_{\rm load,0}=\sum e_{H,\ell}D^2e_{H,\ell}
-\sum \ell_iD^2e_{C,i}$ at the fork. Differentiating (22) gives

$$
DF_{\kappa,\nu}[v]
=D\Pi[v]F_0+\Pi_0(\kappa H_{\rm load,0}+\nu G_0)v.
\tag{24}
$$

Contracting with $F_0$ proves (23). All arms have the identical first GD
update $\theta_1=\theta_0-\eta\nabla L(\theta_0)$. Their exact second-step
contrast with the natural arm is

$$
\theta_{2,\kappa,\nu}-\theta_{2,1,1}
=-\eta\{(\kappa-1)(A(\theta_1)-Z(\theta_1))
+(\nu-1)E(\theta_1)\}.
\tag{25}
$$

Its leading expansion is
$\eta^2\Pi_0\{(\kappa-1)H_{\rm load,0}
+(\nu-1)G_0\}\nabla L(\theta_0)+O(\eta^3)$.
Equation (25) is a fork-only, finite-step prediction; no future derivative
fit is required.

**What would distinguish the explanations?** Before continuation, compute
the signed contributions and forecast their contrasts. If
$\mathcal C_0+\mathcal C_{R,0}<0$, geometry feedback initially amplifies
force energy: removing it must reduce the initial growth rate and doubling
it must increase that rate by the specified amount. If it is positive, the
predeclared ordering reverses. Relaxation has its own independently computed
ordering. The sign is classified before seeing each arm's outcome; opposite
observations cannot both support the same forecast.

Initial identities and first-step equality verify implementation. They do
not establish persistence. The scientific test is whether the checkpoint's
predicted rate, vector direction and intervention contrasts remain accurate
over a specified finite window, and whether a regional bound such as (13)
can be closed there. A scalar forecast
$F_n\approx\exp(\eta k_{\rm ord,0}n)F_0$, with
$k_{\rm ord,0}=-(\mathcal D_0+\mathcal C_0+F_0^TDF[R_0])/\|F_0\|^2$,
assumes both a nearly fixed rate and direction. Adding the direct frozen
$R_0$ displacement does not remove either assumption. A resolved wrong
contrast sign, or a vector error outside its justified remainder allowance,
rejects that forecast. It does not reject the exact identity (9).

## 6. Physical kicks: change feedback without changing the initial slope update

The matched fields deliberately modify the equations. A complementary test
perturbs an actual network and then releases it under ordinary GD. Seek a
feasible direction $v$ satisfying, at the fork,

$$
v_a=0,\qquad J_Cv=0,\qquad Dz_C[v]=0,\qquad
D(\nabla_aL)[v]=0,\qquad D\|F\|^2[v]=0.
\tag{26}
$$

Within this nullspace, choose the direction of the projected gradient of
the intrinsic rate (11), using the declared parameter metric. Opposite
kicks $\theta_0\pm\epsilon v$ change the intrinsic rate by
$\pm\epsilon Dk[v]+O(\epsilon^2)$ while leaving slopes exactly unchanged
and matching coarse outputs, disequilibrium, the actual initial slope
gradient, and effective speed to first order. They do not match the whole
effective-force vector. If the projected rate gradient vanishes, this
particular contrast is unavailable; that is not support for persistence.

Halving the amplitude checks that the rate contrast scales linearly and the
matched quantities differ quadratically, down to a resolved numerical
floor. The ordinary-GD rate includes the loaded $R$ correction in (9), so
it must be recomputed at both perturbed states before predicting an ordering.
A change in the intrinsic rate alone is insufficient if that correction
reverses it. The perturbation also changes residuals and correlations; its
specific claim is the predicted loaded-response contrast, not isolation of
one parameter block as a universal cause.

The first nonzero slope response makes the missing coupling explicit. Let
$g=\nabla L$, $H=Dg$, and let $\xi(t)$ be the derivative of the full-loss
gradient-flow trajectory with respect to its initial perturbation
$\theta_0+\epsilon v$. Then $\dot\xi=-H(\theta(t))\xi$. Although
$v_a=0$ and $(H_0v)_a=0$ imply $\xi_a(0)=\dot\xi_a(0)=0$, differentiation
once more gives

$$
\ddot\xi_a(0)=\bigl[H_0^2v+DH_0[g_0]v\bigr]_a.
$$

The same distinction holds for discrete GD. If $\xi_n$ denotes its
initial-condition derivative, the exact two-step response is

$$
\xi_2=(I-\eta H_1)(I-\eta H_0)v,\qquad
\theta_1=\theta_0-\eta g_0,\qquad H_1=H(\theta_1),
$$

and, under the two matching conditions,

$$
(\xi_2)_a=\eta^2\bigl[H_0^2v+DH_0[g_0]v\bigr]_a+O(\eta^3).
$$

The virtual first update and its Hessian are computable from the fork;
this identity needs no fitted future trajectory. Freezing the full Hessian
retains the $H_0^2v$ coupling but omits the changing-curvature term at the
same order in $\eta$. Thus even a hidden-bias/readout/output-bias kick
with matched initial slope update can produce a slope response that its
scalar force-norm rate does not determine. These are derivatives of the raw
slopes with respect to infinitesimal pulse amplitude. Derivatives of
$|a_j|$ additionally require nonzero slopes with fixed signs; sign crossings
must instead be handled by the actual absolute-scale increments.

This yields a falsifiable progression: verify the local identities, test
the precomputed finite-window contrasts under release, then seek uniform
regional bounds for the observed mechanism. Failure at the second stage
means the proposed local closure misses important state evolution. Success
there motivates the third stage but cannot substitute for it. A persistence
theorem for ordinary GD requires the regional feedback and correction
allowances explicitly stated above.

## 7. Width can slow reinforcement without providing strong damping

There is a useful width law even when the signed feedback is not damping.
If slopes, hidden biases and readouts remain bounded after multiplication
by $\sqrt W$, then the effective force is $O(W^{-1})$ and its possible
relative amplification rate is $O(W^{-1})$. Residual relaxation contributes
only $O(W^{-2})$ to the relative rate. Thus a wide network can move slowly
because it cannot rapidly reinforce its current force, even when that force
is growing. The following bounds make this statement conditional and
target-general, without dividing by a possibly zero force.

Use the exact tanh network, the full empirical fine complement, $|x_i|\le1$,
and the parameter bounds

$$
|a_j|\le A_a/\sqrt W,\qquad |b_j|\le A_b/\sqrt W,\qquad
|c_j|\le A_c/\sqrt W,\qquad K\succeq\kappa I.
\tag{27}
$$

No bound on the output bias is needed. Following
[Section 8 of the rate note](d34_mechanism_rate_theorems.md#8-a-width-dependent-rate-without-freezing-the-effective-map),
put

$$
\begin{aligned}
U&=A_a+A_b,&
J_*&=\sqrt{2A_c^2U^4+U^6/9},\\
E_*&=\|P_Hy\|_m+A_cU^3/(3W),&
H_*&=4A_cU+\sqrt2U^2,\\
M_*&=\sqrt2+4A_cU/W,&
C_*&=E_*\left(H_*+\frac{J_*M_*}{\sqrt\kappa}\right),\\
L_*&=E_*\left(H_*+\frac{2J_*M_*}{\sqrt\kappa}\right)+\frac{J_*^2}{W}.
\end{aligned}
\tag{28}
$$

**Proposition 4 (uniform width bounds).** On this regime, with
$s(\theta)=\|F(\theta)\|$,

$$
s\le\frac{J_*E_*}{W},\qquad
0\le\mathcal D\le\frac{J_*^2}{W^2}s^2,\qquad
|\mathcal C|\le\frac{C_*}{W}s^2,\qquad
\|DF\|_{\rm op}\le\frac{L_*}{W}.
\tag{29}
$$

These inequalities hold also at $F=0$. Their constants remain bounded across
widths if the rescaled parameter bounds, fine target norm and
$\kappa^{-1}$ do. They are analytic bounds, not numerical evaluations or
certificates for the archived checkpoints.

**Proof.** Section 8 of the rate note supplies
$\|J_H\|\le J_*/W$, $\|e_H\|\le E_*$ and
$\|D^2e_H[v,w]\|\le(H_*/W)\|v\|\|w\|$.
The last bound uses the fine projection to remove the affine part of tanh
before differentiating. For the coarse Hessian, no such removal is available,
but a width-independent bound suffices. At each input, one neuron's
geometry–geometry Hessian block has norm at most $4A_cU/W$, using
$|\tanh''u|\le2|u|$. Its mixed readout–geometry block has norm at most
$\sqrt2$, using $|\tanh' u|\le1$. The full Hessian is block diagonal in
neurons and has a zero output-bias row and column. Taking the empirical
mean norm and projecting onto the coarse modes therefore gives

$$
\|D^2e_C[v,w]\|\le M_*\|v\|\|w\|,
\qquad \|\ell\|\le\frac{J_*E_*}{\sqrt\kappa W}.
$$

Consequently the loaded Hessian in (24) has operator norm at most $C_*/W$.
This proves the curvature bound in (29); the force and relaxation bounds
follow directly from the fine Jacobian estimate. Finally the projector
derivative obeys

$$
\|D\Pi[v]\|\le\frac{2\|DJ_C[v]\|}{\sqrt\kappa}
\le\frac{2M_*}{\sqrt\kappa}\|v\|.
$$

Differentiating $F=\Pi J_H^Te_H$ now gives the stated $L_*/W$ bound.
This calculation includes the output-bias coordinate of $F$ and the full
coarse compensation; neither was dropped to obtain the width orders.

For effective continuous flow, (8) and (29) imply

$$
V(t)\le V(0)e^{2C_*t/W},\qquad
s(t)\le s(0)e^{C_*t/W}
\tag{30}
$$

while the region remains valid. This proves an amplification clock of order
$W$ in GD time. It does not prove amplification actually occurs, nor does
it claim the larger residual-relaxation clock alone determines acquisition.

### Discrete GD and an explicit closure of the parameter region

For ordinary GD, condition on the following absolute component tracking
allowances at its actual iterates $0\le n<N$:
$|R_{a,j}|\le\delta_a/W^{3/2}$,
$|R_{b,j}|\le\delta_b/W^{3/2}$, and
$|R_{c,j}|\le\delta_c/W^{3/2}$; set
$\delta_2=(\delta_a^2+\delta_b^2+\delta_c^2)^{1/2}$.
Persistent low tracking is a separate premise here, not a conclusion of
the geometric first-exit argument below. These allowances are not asserted
uniformly over arbitrary output biases: $R$ depends on $d$, although $F$
does not.
The field $F$ is independent of $d$, so its change along an update depends
only on these three blocks. On a connecting segment satisfying (27), (29)
gives the discrete comparison

$$
s_{n+1}\le\left(1+\frac{\eta L_*}{W}\right)s_n
+\frac{\eta L_*\delta_2}{W^2}.
\tag{31}
$$

Equivalently, $Q_n=Ws_n$ satisfies

$$
Q_n\le (Q_0+\delta_2)
\left(1+\frac{\eta L_*}{W}\right)^n-\delta_2
\le (Q_0+\delta_2)e^{L_*\eta n/W}-\delta_2.
\tag{32}
$$

This conservative discrete bound uses the full derivative estimate; it does
not retain the signed improvement available from (13). It nevertheless
proves the same width-dependent amplification clock for any $\eta>0$ for
which the region can be closed. At zero initial force it remains meaningful.
The second tracking channel satisfies
$\|J_HR\|\le J_*\delta_2/W^2$, and the loaded map response satisfies
$\|DF[R]\|\le L_*\delta_2/W^2$. An unconstrained $R_d$ affects neither
bound, because both $e_H$ and $F$ are independent of $d$.

Here is a sufficient first-exit test that also checks coarse conditioning.
Let the initial rescaled bounds be $A_{\ell,0}<A_\ell$ for
$\ell\in\{a,b,c\}$ and let
$\lambda_{\min}(K(\theta_0))\ge\kappa_0>\kappa$. Define

$$
\begin{aligned}
K_a=K_b&=E_*A_c\left(U^2+J_*/\sqrt\kappa\right),\\
K_c&=E_*\left(U^3/3+UJ_*/\sqrt\kappa\right),\\
G&=\left[\sum_{\ell=a,b,c}(K_\ell+\delta_\ell)^2\right]^{1/2},
\qquad J_{C,*}=\sqrt{1+2A_c^2+U^2}.
\end{aligned}
$$

Define the initial-state neighborhood by
$\|\theta_{a,b,c}-\theta_{0,a,b,c}\|\le\eta NG/W$, with unrestricted
output bias $d$. It is a three-block ball times the output-bias line, hence
a cylinder in the full parameter space. Use the analytic fine-force and
conditioning estimates on its intersection with the outer parameter box;
the tracking premise concerns only the actual iterates as stated above.
The fine-force coordinate bounds
from Section 8 of the rate note and the elementary bound
$\|DK[v]\|\le2J_{C,*}M_*\|v_{a,b,c}\|$ show that it suffices to require

$$
\frac{\eta N}{W}(K_\ell+\delta_\ell)<A_\ell-A_{\ell,0}
\quad(\ell=a,b,c),\qquad
2J_{C,*}M_*\frac{\eta NG}{W}<\kappa_0-\kappa.
\tag{33}
$$

Indeed, each parameter coordinate moves by at most
$\eta N(K_\ell+\delta_\ell)/W^{3/2}$ and the three-block vector moves
by at most $\eta NG/W$. Every connecting segment remains in the convex
outer parameter box and in this initial-state cylinder. The coarse derivative
bound then preserves $K\succeq\kappa I$ there, without assuming the entire
centered small-parameter box is conditioned. Induction closes the states
and the segments required by (31), conditional on the stated tracking
allowances. Thus (33) closes the geometric and conditioning requirements;
it proves neither entry into low tracking nor persistence of low tracking.

On the resulting interval the normalized per-neuron acquisition rate obeys
$|\lambda_{j,n+1}-\lambda_{j,n}|\le
\eta h(K_a+\delta_a)/W^{3/2}$. The path-length proof of (20) also applies
with $\bar s_n$ given by the right side of (32) divided by $W$ and
$\rho_a=\delta_a/W$; no output-bias tracking bound is needed for this
slope-only conclusion once (33) has closed the three-block region.
If $h$ is proportional to $W^{-1}$, the component rate is
$O(\eta W^{-5/2})$; the constants and the construction-to-width relation
are part of that statement. This is a conditional mechanism for slow
acquisition, not an inference from width alone.

**Prediction.** Comparable bounded rescaled states can show geometry
feedback on the clock $\eta n/W$ while their fine residual changes little.
This is consistent with the nonlinear transport model, whose coefficients
depend on the evolving particle distribution even under nearly fixed target
loading. When the relevant quadratic and cubic target loadings vanish,
the leading transport field can vanish and a longer clock can emerge, as
described in the linked transport note. The $W$ clock proved here is a
target-general upper bound on reinforcement; it is not a universal lower
bound on the rate of change, and does not replace those higher-order cases.

## 8. A forecast can miss the trajectory and still give a useful upper bound

Suppose a forecast predicts twice the observed force speed. It is inaccurate
as a trajectory model, but its speed prediction can still be a useful upper
bound. Even an underprediction by a bounded factor may leave ample distance
to the acquisition threshold. The relevant question for exclusion is
therefore one-sided: can we bound the actual speed by a fixed multiple of
the forecast throughout the interval?

This is a different objective from the vector-forecast tests in Section 5.
An incorrect force direction or intervention contrast can reject that
trajectory closure without invalidating a separately proved speed envelope.
No vector-direction accuracy is required for the path-length bound below.
Conversely, a small collection of observed speed ratios does not prove a
uniform envelope: intermediate peaks and future changes still matter.

### Proposition 5: bounded excess amplification relative to a reference

Let $\widehat q_0,\ldots,\widehat q_N>0$ be any preissued reference speeds,
independent of the actual future trajectory, and put $q_n=\|F(\theta_n)\|$.
At step $n$, suppose the regional hypotheses of Propositions 2–3 hold with
constants $m_n,L_{V,n},u_n,\rho_n$, giving

$$
q_{n+1}\le r_nq_n+\eta u_n,\qquad
r_n=\sqrt{1-2\eta m_n+\eta^2L_{V,n}}\ge0,
\qquad \|R(\theta_n)\|\le\rho_n.
\tag{34}
$$

The square root is required to be real, $u_n,\rho_n\ge0$, and the
constants must be valid on declared state regions and the necessary
outgoing neighborhoods. For example, they can all be bounds on the same
balls $\mathcal U,\mathcal U^+$ from Section 3. Bounds on distinct
time-indexed regions require a separate inclusion argument for those
regions. They are not fitted curvature values from the future iterates.
More generally, the conclusion below holds for any justified nonnegative
recurrence multiplier $r_n$ in (34), including the derivative-based
multiplier $1+\eta L_*/W$ from (31).

Define the excess amplification factors and the propagated tracking allowance

$$
\alpha_n=r_n\frac{\widehat q_n}{\widehat q_{n+1}},\qquad
b_0=0,\qquad b_{n+1}=r_nb_n+\eta u_n.
\tag{35}
$$

Suppose a finite $M\ge0$ satisfies

$$
\frac{q_0}{\widehat q_0}\prod_{i=0}^{n-1}\alpha_i\le M
\qquad\text{for every }0\le n\le N,
\tag{36}
$$

where an empty product is one. Then, on the closed interval of validity,

$$
\boxed{\quad q_n\le M\widehat q_n+b_n\quad(0\le n\le N).\quad}
\tag{37}
$$

For the usual reference initialized with $\widehat q_0=q_0>0$, (36)
starts at one. It bounds accumulated excess amplification; individual
$\alpha_n$ may exceed one. When all factors are positive, it is equivalent
to an upper bound on their partial log sums. The product form also handles
$r_n=0$ without taking a logarithm of zero.

**Proof.** Write $P_{n,k}=\prod_{i=k}^{n-1}r_i$, with $P_{n,n}=1$.
Unrolling (34) gives

$$
q_n\le P_{n,0}q_0
+\eta\sum_{k=0}^{n-1}P_{n,k+1}u_k,
\qquad
b_n=\eta\sum_{k=0}^{n-1}P_{n,k+1}u_k.
\tag{38}
$$

The reference ratios telescope:
$P_{n,0}q_0/\widehat q_n
=(q_0/\widehat q_0)\prod_{i<n}\alpha_i$.
Equation (36) therefore proves (37). If any reference speed is zero,
retain the absolute formula (38) instead of dividing by that speed or
introducing a numerical floor.

**Closing the region.** To make this a first-exit theorem rather than a
bound conditional on unproved containment, use the common balls of Section
3 and define $Q_n=M\widehat q_n+b_n$. It suffices that

$$
\eta\max_{0\le k<N}(Q_k+\rho_k)<\delta,
\qquad
\eta\sum_{k<N}(Q_k+\rho_k)<d.
\tag{39}
$$

At each induction step the first inequality contains the outgoing segments,
the regional Taylor and loaded-response estimates imply (34), and the
second inequality places the actual endpoint back inside $\mathcal U$.
Thus all iterates through $N$ satisfy (37). Coarse conditioning is part of
the regional hypotheses and must also be established there. In the
small-parameter setting, Section 7 provides an alternative three-block
closure, conditional on its explicitly stated persistent-tracking premise.
Small slope tracking alone cannot close a full-parameter region.

### Consequence: the factor costs linearly in travel, quadratically in population

Suppose also $\|R_a(\theta_n)\|\le\rho_{a,n}$. Define

$$
\mathcal A_N=\eta\left[
M\sum_{k<N}\widehat q_k+\sum_{k<N}b_k+\sum_{k<N}\rho_{a,k}
\right].
\tag{40}
$$

The same path-length proof as (20) gives the distinct-ever acquisition bound

$$
p_{\rm ever}(N)\le\min\left\{1,\;
p_{\rm init}(\lambda>\lambda_0)
+\frac{h^2\mathcal A_N^2}{W(\lambda_*-\lambda_0)^2}
\right\},\qquad 0\le\lambda_0<\lambda_*.
\tag{41}
$$

Per-neuron bounds follow by replacing $\rho_{a,k}$ by an absolute allowance
for $|R_{a,j}(\theta_k)|$ in (40) and adding $h\mathcal A_{j,N}$ to
$\lambda_j(0)$. The $b_k$ terms account for tracking's propagated effect
on effective speed; the $\rho_{a,k}$ terms account for its direct slope
motion. Neither can be omitted merely because the other is small.

In the absence of these additive corrections, multiplying a reference by
$M$ multiplies its travel allowance by $M$ and its additional population
allowance by $M^2$. Exact trajectory prediction is therefore unnecessary
when the acquisition margin can absorb this cost. A useful width statement
requires $M$ and the relevant correction constants to remain bounded
uniformly across the widths and time windows claimed. Observing modest
ratios at finitely many endpoints suggests such a bound; it does not prove
(36), (39), or uniformity in width.

The [ODE rate results](d34_coupled_ode_mechanisms.md#4-a-heterogeneous-population-can-have-a-limited-correction-travel-budget)
offer a different starting point from a close forecast: a structural law
relating generated error to its changing sensitivity can bound accumulated
motion directly. For example, with zero competing drive their cubic degeneracy
condition yields a $T^{1/6}$ path allowance in the reduced clock, conditional on the stated
regional assumptions. Establishing such a law for the trained population is
a mechanistic persistence problem, not merely an improvement in forecast
accuracy. Its transfer to this discrete-GD argument must retain the relevant
tracking, approximation and time-normalization terms.

**An explicit structural width example.** Suppose Section 7's hypotheses
close through $\eta N/W\le T$, with width-uniform constants. For the
constant reference $\widehat q_n=q_0>0$, choose
$r_n=1+\eta L_*/W$. Then a prospective choice is

$$
M=e^{L_*T},\qquad
b_n\le\frac{\delta_2}{W}
\left[e^{L_*\eta n/W}-1\right].
\tag{42}
$$

For an exponential reference $\widehat q_n=q_0e^{k_0\eta n}$, the lower
bound $k_0\ge-K_0/W$, with $K_0\ge0$, instead permits
$M=e^{(L_*+K_0)T}$. These factors
come from structural regional bounds, not ratios measured after training.
They are $O(1)$ in width but need not be small: the constants in the
exponential can be enormous, and the first-exit conditioning margin can
allow only a very short interval. Thus this example proves a width order,
not a useful finite-width exclusion by itself. Actual constant evaluation
and closure remain necessary before claiming a practical acquisition bound.
The [campaign report](../results/checkpoint_D_optimizers/expD34_readout_race/mechanism_persistence/README.md)
keeps that finite-width audit separate from the structural width statement.

Finally, two different appearances of $W$ must be kept separate. At fixed
normalized threshold $\lambda_*$, the required physical slope is
$\gamma_*=\lambda_*/h$, which is proportional to $W$ when
$N_{\rm ref}$ is proportional to $W$. The reinforcement clock from Section
7 is a statement about time, $\eta n$ of order $W$. A slope threshold and
a time scale have different units. Neither implies that acquisition occurs
after $O(W)$ updates, and small-parameter rate bounds cannot be extrapolated
beyond their established first-exit interval to claim such a hitting time.
