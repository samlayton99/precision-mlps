# A certified finite window for one stalled instance

The effective-force reference can now be transferred to ordinary GD with
rounding-controlled arithmetic for one empirical problem. Starting from the
`moment9` development checkpoint, all 177 neurons satisfy
$\lambda_j<0.002973994$ throughout the next 20,000 updates. This excludes the
threshold $\lambda_*=0.25$ over that window. It does not establish a longer
window, a statement across initializations, or a population-loss theorem.

**Notation for this certificate.**

| Symbol | Meaning |
|---|---|
| $\theta=(a,b,c,d)$ | All 532 network parameters |
| $\lambda_j=|a_j|/64$ | Normalized slope; construction resolution $N_{\rm ref}=128$ |
| $q_n$ | Prescribed reference state at additional update $n$ |
| $r_n$ | Certified Euclidean bound on $\|\theta_n-q_n\|$ |
| $\beta_b(R)$ | Uniform Lipschitz bound for one GD step in a ball about $b$ |

## 1. Example: a reference with substantial distance to acquisition

The case is `moment9`, seed 0, width 177, at the archived 600,000-update
checkpoint. The loss is empirical half-MSE,

$$
L(\theta)=\frac{1}{2m}\sum_{i=1}^m
\left[d+\sum_{j=1}^{177}c_j\tanh(a_jx_i+b_j)-y_i\right]^2,
\qquad \theta_{n+1}=\theta_n-\eta\nabla L(\theta_n).
$$

Here $m=2048$, $|x_i|\le1$, and $\eta$ is the exact binary64 number represented
by `0.002`. Every archived $x_i,y_i$ and initial parameter is treated as its
exact binary64 value. The checkpoint defines $n=0$; the certificate ends at
the nominal total-update index 620,000 and does not certify the preceding
training history. The archive hashes in the
[provenance record](../results/checkpoint_D_optimizers/expD34_readout_race/mechanism_refinement/certificates/moment9_dev_20k/provenance.json)
specify the empirical problem and reference unambiguously.

The reference endpoints come from the checkpoint-issued frozen effective
model in [the mechanism note](d34_mechanism_rate_theorems.md#2-depletion-sensitivity-can-be-appreciable-while-its-error-load-disappears).
Between selected endpoints we use exact rational linear interpolation.
The endpoints' stored binary64 values define the reference; their generating
eigensolve and floating-point updates need not themselves be exact. The
actual reference increment enters the defect bound below, so its numerical
generation error is not silently discarded.

This implements the neighborhood-transfer requirement of
[Section 7](d34_mechanism_rate_theorems.md#7-a-loaded-response-condition-that-closes-the-ordinary-gd-envelope)
using a conservative full-gradient ball bound. It certifies that this reference
excludes acquisition for actual GD. It does not separately certify small
tracking or identify which force channel causes the exclusion.

## 2. Bound: control the full GD map throughout each reference segment

Write $f_\theta$ for the sample-output vector, with empirical norm
$\|u\|_m^2=m^{-1}\sum_i u_i^2$. At a reference endpoint $b$, let $J_b$
be the output Jacobian divided by $\sqrt m$, and decompose the loss Hessian as

$$
H_b=J_b^TJ_b+C_b,\qquad
C_b=\frac1m\sum_i(f_b(x_i)-y_i)D^2f_b(x_i).
$$

The residual-curvature matrix $C_b$ consists of independent symmetric
$3\times3$ neuron blocks, plus a zero output-bias block. Compute bounds
$j_b\ge\|J_b\|_F$, $s_b\ge\|f_b-y\|_m$,
$\nu_b\ge\max(0,-\lambda_{\min}(C_b))$ and $\rho_b\ge\|C_b\|$.
These retain the sign of residual curvature instead of treating the positive
Gauss--Newton term as a source of instability.

For the Euclidean ball of radius $R$ about $b$, set
$c_* = \|c(b)\|_\infty+R$. Uniform scalar-output derivative bounds are

$$
M_2(R)=\frac{8c_*}{3\sqrt3}+\sqrt2,\qquad
M_3(R)=4\sqrt2\,c_*+\frac8{\sqrt3}.
$$

They follow from $|\tanh'|\le1$, $|\tanh''|\le4/(3\sqrt3)$,
$|\tanh'''|\le2$ and $\|(x,1)\|\le\sqrt2$.
The independent neuron blocks introduce no additional width factor in these
operator bounds. Taylor's formula and the product rule then give

$$
\begin{aligned}
t(R)&=j_bR+\tfrac12M_2(R)R^2,\\
v(R)&=M_2(R)t(R)+M_3(R)s_bR,\\
\nu(R)&=\nu_b+v(R),\\
U(R)&=[j_b+M_2(R)R]^2+\rho_b+v(R).
\end{aligned}
$$

Indeed, $t$ bounds residual change and $v$ bounds $\|C_\theta-C_b\|$.
Consequently $-\nu(R)I\preceq H_\theta\preceq U(R)I$ throughout the ball,
and $U(R)\ge\nu(R)$ also bounds $\|H_\theta\|$.
For $G(\theta)=\theta-\eta\nabla L(\theta)$, averaging the symmetric Hessian
along a line inside this convex ball proves

$$
\|G(u)-G(v)\|\le\beta_b(R)\|u-v\|,\qquad
\beta_b(R)=\max\{1+\eta\nu(R),\ |1-\eta U(R)|\}.
$$

Both spectral endpoints are required for discrete GD; no continuous-time
substitution or unverified step-size restriction is used.

For a segment of $M$ updates with endpoints $b,b'$, define
$\Delta=b'-b$, $D\ge\|\Delta\|$ and $q_{s+t}=b+t\Delta/M$.
For every integer $0\le t<M$, the gradient Lipschitz bound on $B(b,D)$ gives

$$
\|G(q_{s+t})-q_{s+t+1}\|
\le d_b:=\left\|\eta\nabla L(b)+\frac{\Delta}{M}\right\|
                  +\eta U(D)D.
$$

This controls the entire segment, rather than a sampled maximum. Starting
with $r_0=0$, propagate outward-rounded upper bounds by

$$
r_{n+1}\ge\beta_b(r_n+D)r_n+d_b.
$$

**Induction.** If $\|\theta_n-q_n\|\le r_n$, both states and their connecting
line lie in $B(b,r_n+D)$. Apply the GD-map bound and then the defect bound.
This proves $\|\theta_{n+1}-q_{n+1}\|\le r_{n+1}$ without assuming future
trajectory containment. At segment boundaries the next reference endpoint
is identical to the preceding one, and the radius is carried forward.

Since $\beta_b\ge1$ and $d_b\ge0$, these radii are nondecreasing. A coordinate's
absolute value on a linear segment is no larger than its endpoint maximum.
Thus, for every neuron and every time in a completed segment,

$$
\lambda_{j,n}\le\frac1{64}
  \left[\max\{|b_{a,j}|,|b'_{a,j}|\}+r_{s+M}\right].
$$

Taking the maximum over all segments includes distinct neurons hitting at
different times. No conversion from simultaneous occupancy to ever-hits is
needed: this is a bound for every neuron at every update.

## 3. Verification result and its numerical meaning

The immutable
[certificate helper](../experiments/expD34_readout_race/mechanism_arb_certificate.py)
uses python-flint 0.9.0, with 100-bit Arb ball arithmetic, for gradients,
curvature blocks, norms, analytic constants and radius propagation.
Floating-point eigenvalues only propose block spectral endpoints: interval
Sylvester tests prove positivity of $C_j-\ell I$ and $uI-C_j$ before accepting
$[\ell,u]$. The computation therefore does not trust those eigenvalues or
ordinary `libm` evaluations as rigorous bounds.

**Two bounds for the same empirical GD problem and 20,000-update window.
Displayed bounds are rounded upward. Chunk length changes the auxiliary
reference interpolation, not the GD step size or the asserted dynamics.**

| Reference segment length | Final full-parameter radius | Maximum normalized slope over all neurons and updates | CPU seconds |
|---:|---:|---:|---:|
| 1,000 updates | 0.00705758 | 0.00307850 | 27 |
| 100 updates | 0.000372332 | 0.002973994 | 271 |

The raw [1,000-step](../results/checkpoint_D_optimizers/expD34_readout_race/mechanism_refinement/certificates/moment9_dev_20k/chunk1000.json)
and [100-step](../results/checkpoint_D_optimizers/expD34_readout_race/mechanism_refinement/certificates/moment9_dev_20k/chunk100.json)
records retain enclosing interval strings for each segment.
Both runs exclude all 177 neurons from $\lambda\ge0.25$; here that threshold
is $|a|\ge16$, because $N_{\rm ref}=128$.
Independent mathematical review checked the derivative constants, block
spectral verification, enlarged-ball induction and prefix bound. Source and
artifact hashes were checked against the executed files.

This is a rounding-controlled certificate for real-arithmetic GD on the
specified finite dataset, subject to the arithmetic library and executed
code being correct. It does not enclose roundoff in a floating-point training
implementation, ideal target-generation errors, or a population distribution.
No longer certified window is reported here. The gain over the earlier FP64
diagnostic is that numerical rounding and all intermediate states now enter
the proof; the remaining scientific task is to extend useful certified
windows and determine how broadly the effective-force mechanism applies.
