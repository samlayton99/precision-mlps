# Accumulated outward motion: what follows from the actual GD structure

November 13, 2026. Theory and read-only saved-state checks; no new training. The requested useful, essentially all-time bound for the default-initialization runs has **not** been established. This note records an explicit obstruction, a structural saturation bound with unusable constants on the training grid, and the limits of a local-convexity argument. These are research derivations, not an independently reviewed replacement theorem.

## 1. The exact quantity needing control

For slope energy A_n=||gamma_n||² and full GD gradient g_n, native GD gives

\[
A_N-A_0=-2\sum_{n<N}\eta_n\langle\gamma_n,g_{\gamma,n}\rangle
+\sum_{n<N}\eta_n^2\|g_{\gamma,n}\|^2.
\]

Sufficient loss decrease controls sum eta_n ||g_n||². With bounded step sizes it therefore controls the last term. It does not control the first term or its positive part. Square-summable increments need not be summable. Requiring a fitted ratio between these terms to stay bounded simply introduces a new persistence premise, as D65 did.

A finite bound on total positive outward movement is stronger than a bound on the largest slope: a bounded trajectory can still oscillate. The counterexample below is stronger than that distinction, because its slopes themselves diverge.

## 2. An explicit native-GD tanh counterexample

Use exactly the joint one-hidden-layer tanh architecture, with all parameters trainable, width W=196, and constant learning rate eta=0.001. Train on the four equally weighted points

\[
x\in\{-1,-\tfrac12,\tfrac12,1\},\qquad
f(x)=\frac{7x-4x^3}{3}.
\]

The target is a smooth cubic and its values on the training set are (-1,-1,1,1), with training RMS one.

Initialize every slope at a_0=1/2, every hidden bias at zero, every readout at c_0/W, and the output bias at zero. Set

\[
t_1=\tanh(a_0/2),\quad t_2=\tanh(a_0),\qquad
c_0=\frac{t_1+t_2}{t_1^2+t_2^2}=2.584786\ldots.
\]

Each readout is 0.013188..., inside the PyTorch-default interval +/-1/sqrt(196); slopes, hidden biases, and output bias are also strictly inside their default ranges. This is a correlated symmetric initialization, **not** an independent random draw.

Identical neurons remain identical. Odd target/output symmetry makes every hidden-bias gradient and the output-bias gradient zero. These parameters remain trainable; their zero updates follow from symmetry. Thus q_n(x)=c_n tanh(a_n x), and the exact full-network GD update reduces to

\[
a_{n+1}=a_n-\frac\eta W\,\partial_a L(a_n,c_n),\qquad
c_{n+1}=c_n-\eta W\,\partial_c L(a_n,c_n),
\]

where

\[
L(a,c)=\frac14\left[(c\tanh(a/2)-1)^2+(c\tanh a-1)^2\right].
\]

The unequal reduced rates are essential: this is native GD on all W neurons, not an altered shared-parameter optimizer.

### The profiled loss has no finite positive stationary point

Minimizing over c gives c_*(a)=(t_1+t_2)/(t_1²+t_2²). With s=tanh(a/2),

\[
F(a)=\min_c L(a,c)
=\frac{(1-s^2)^2}{4[(1+s^2)^2+4]}.
\]

It satisfies F(0+)=1/20, F(infinity)=0, and

\[
\frac{dF}{ds}=
-\frac{2s(1-s^2)(s^2+3)}{(s^4+2s^2+5)^2}<0
\quad(0<s<1).
\]

At a finite positive critical point, L_c=0 would force c=c_*(a), while L_a=F'(a)<0. Hence no such critical point exists.

### The specified finite GD step is stable for all iterations

Initially L_0=F(1/2)=0.04311584...<1/20, so E_0=sqrt(2L_0)<1/3.

Inductively suppose a>=1/2, c>0, and L<=L_0. Since tanh(1/4)>0.24 and ||q||<=1+E_0<4/3,

\[
\frac23<c<6.
\]

Let T=tanh²(a/2)+tanh²(a) and S=tanh(a/2)+tanh(a). The readout update is

\[
c_{n+1}=(1-\eta WT/2)c_n+(\eta WT/2)c_*(a_n).
\]

Since eta W=0.196<1 and 0<c_*(a)<=1/tanh(a/2)<6, this is a convex combination of two positive numbers below six. It keeps c between zero and six.

The slope update obeys |a_next-a| <= eta c E_0/W < 2 eta/W < 1.1e-5. Its whole segment therefore stays at a>=1/4. On the full parameter segment, the readout norm is at most 6/sqrt(W), the largest individual readout at most 6/W, and the residual RMS at most seven. The standard global tanh derivative bounds give

\[
\|\nabla^2 L\|\le W+1+\frac{72}{W}
+7\left(\frac{24}{W}+\sqrt2\right)
=208.124\ldots<210.
\]

Thus eta ||H||<0.21<1 and

\[
L_{n+1}\le L_n-\frac\eta2\|g_n\|^2.
\]

In particular L_next<=L_0. Since L_next>=F(a_next) and F strictly decreases on positive a, this also restores a_next>=a_0. The induction is closed without assuming bounded future slopes.

### Consequence: slopes diverge under native GD

Loss decrease implies sum ||g_n||²<infinity, so g_n tends to zero. The aggregate readout remains in [2/3,6]. If a_n had any bounded subsequence, compactness would produce a finite positive limiting pair with L_a=L_c=0, contradicting the profiled-loss calculation. Therefore

\[
a_n\longrightarrow\infty,\qquad c_n\longrightarrow1.
\]

Every slope equals a_n, so gamma_RMS also tends to infinity. Both net slope-energy growth and accumulated positive outward movement are unbounded. All neuron concentration factors remain exactly one, since the neurons stay identical.

This establishes that smooth targets, bounded initial slopes, distributed parameter energy, trainable readouts, and stable GD do not by themselves imply a finite total-outward-motion budget.

It is **not** a claim about typical independent PyTorch draws, our 1024-point grid, or the 24 saved trajectories. For any fixed finite slope threshold, finite-iterate continuity gives an open neighborhood of this initialization that also crosses it eventually, hence positive probability under the default initialization distribution for this four-point problem. No useful lower bound on that probability or quantitative crossing time is asserted. Divergence alone does not rule out extremely long escape times.

## 3. A structural saturation bound without a fitted radial constant

For distinct inputs x_i in [-a,a], 0<a<=1, including +/-a, set

\[
n_i=\frac{(1,-x_i)}{\sqrt{1+x_i^2}},\qquad
\Phi(v)=\max_i|n_i\cdot v|,\quad v=(\gamma,\beta).
\]

Let Delta be the smallest projective angular separation of these normals and delta=sin(Delta/2)>0. The endpoint normals imply |gamma|<=sqrt(1+a²) Phi(v).

At a maximizing facet k, n_k is a closest normal to v in projective angle. Every other normal is at angle at least Delta/2 from v. Thus

\[
|\gamma x_i+\beta|\ge\delta\Phi(v)\quad(i\ne k).
\]

The gradient contribution of sample k is perpendicular to n_k and does not increase the active facet. All other sample contributions are bounded using sech²(u)<=4 exp(-2|u|).

For gradient flow this gives

\[
D^+\Phi(v_j)\le4\sqrt2 E|w_j|e^{-2\delta\Phi(v_j)}.
\]

For a native GD step of size eta, let d_j=||v_j,next-v_j||. Apply the same facet argument along the straight update segment; preactivations at its start differ from those along it by at most sqrt(2)d_j. Integration yields

\[
e^{2\delta\Phi(v_{j,n+1})}
\le e^{2\delta\Phi(v_{j,n})}
+8\sqrt2\delta\eta_n E_n|w_{j,n}|e^{2\sqrt2 d_{j,n}}.
\tag{S1}
\]

Assume the ordinary sufficient GD loss decrease, eta_n<=eta_max, and write E_start=sqrt(2L_start). It gives E_n<=E_start and d_j,n<=D=E_start sqrt(eta_max). For readout energy V=||w||²+b_out², the exact update and bounded tanh features imply

\[
V_{n+1}-V_n
\le\eta_n\|f\|^2/2+\eta_n^2(W+1)E_{\rm start}^2.
\]

Consequently, with S_n=sum_{k<n}eta_k and c=||f||²/2+eta_max(W+1)E_start²,

\[
V_n\le V_0+cS_n,
\]

and summing (S1) gives

\[
|\gamma_{j,n}|\le\frac{\sqrt{1+a^2}}{2\delta}
\log\left[
 e^{2\delta\Phi(v_{j,0})}
+8\sqrt2\delta e^{2\sqrt2D}E_{\rm start}S_n\sqrt{V_0+cS_n}
\right].
\tag{S2}
\]

This is logarithmic in cumulative step size with explicit constants and no fitted radial-gain assumption. It bounds current and running-maximum slopes, not their total positive variation. For GD it still requires sufficient descent over the claimed interval; it does not establish that a fixed step size remains descending indefinitely. Gradient flow has monotone loss automatically, with the corresponding simpler integral version and no discrete segment factor.

### The constants do not solve our problem

On the saved 1024-midpoint grid, delta=0.000489236 and the front factor sqrt(1+a²)/(2delta)=1444.62. On width-196 sine at LR 0.001, restarting at 30k, (S2) gives an RMS ceiling of about **3565** at 200k, compared with QUILL reference **18.375**.

The dense grid contains a sample near almost any neuron center, so saturation does not provide a useful global barrier at the desired scales. The logarithmic asymptotic order does not rescue these constants.

## 4. The current saved states are not in a locally convex region

As another check, the exact loss Hessian was assembled at width 196, LR 0.001, and checked against autograd Hessian-vector products. Its smallest eigenvalues are:

| Target | 30k | 200k |
|---|---:|---:|
| x² | -0.001740 | -0.000900 |
| sin(4 pi x) | -0.016742 | -0.003055 |
| 1/(1+100x²) | -0.016413 | -0.015999 |

These are numerical checks, not interval certificates. They rule out simply asserting a convex neighborhood containing these exact checkpoints. Negative curvature does not itself prove escape or rule out nonconvex trapping; a more specific basin argument would still have to be established.

Analyticity alone also does not supply the missing confinement. For example, [Dereich and Kassing's analytic-network convergence result](https://arxiv.org/abs/2102.09385) explicitly conditions convergence on the trajectory remaining bounded. [Sontag's earlier report](https://www.sontaglab.org/FTPDIR/sycon88-02.pdf) already discusses divergent sigmoid weights and slow saturation; it is context, not the source of the joint-GD counterexample or (S2) above.

## 5. Current conclusion

No informative, essentially all-time accumulated-outward-motion bound for the saved default-init runs has been obtained. A finite universal total-motion budget is false in the architecture class. The explicit structural saturation bound is too loose at our reference scale. A valid very-long-time statement must use genuine information specific to the target, sample geometry, initialization distribution, or a proved trapping region; it cannot be obtained by choosing a larger empirical K or by invoking analytic convergence while assuming boundedness.

The counterexample does not disprove a useful huge finite escape-time bound for our specific experiments. That narrower statement remains unresolved.

Checks: `check_accumulated_motion.py`, with machine-readable results in `results/checkpoint_D_optimizers/expD65_radial_loss_budget/structural_audit/checks.json`.
