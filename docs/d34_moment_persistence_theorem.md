# Slow reinforcement from persistent moments

The [population audit](../results/checkpoint_D_optimizers/expD34_readout_race/population_coverage/README.md)
found that the aligned, zero-bias sector does not contain the archived
training states. The next theorem should therefore allow the mixed signs
and hidden biases that those states actually have. This note proves a
finite-time result for the **exact tanh effective fine flow**, retaining
both. It does not replace tanh by a polynomial network.

The mechanism is a coupled constraint on movement. At small physical
parameters, affine removal makes the fine gradient proportional to slope
size. Orthogonal coarse compensation cannot increase its total norm. These
facts control growth of the total parameter moment and decay of the moment
that keeps the coarse fit adjustable. The latter gives a lower bound on
coarse conditioning. We then close a support bound and derive a force-growth
bound from those structural estimates. Small future force is a conclusion,
not an assumption.

This is a conditional theorem with explicit tests on initial data and a
chosen horizon. Its usefulness requires evaluating those tests. In
particular, a horizon proportional to width is not automatically a practical
many-update guarantee. The result concerns effective fine flow; transfer
to ordinary GD requires tracking and discretization control separately.

| Symbol | Meaning |
|---|---|
| $W$ | Physical number of neurons. |
| $X_j=(\alpha_j,\beta_j,\zeta_j)=\sqrt W(a_j,b_j,c_j)$ | Rescaled slope, hidden bias, and readout. |
| $\mathsf X=(X_1,\ldots,X_W,d)$ | Complete state, with unscaled output bias $d$. |
| $R=\max_j|X_j|$ | Euclidean support radius of the rescaled particles. |
| $M=\mathbb E_W|X_j|^2$, $E_s=\mathbb E_W(\alpha_j^2+\zeta_j^2)$ | Total particle moment and slope–readout moment. |
| $e_H$, $Y_0$ | Full fine residual and its initial empirical norm. |
| $F$ | Exact effective fine gradient in physical Euclidean coordinates. |
| $t$, $\tau=t/W$ | Physical time and reduced time. |
| $q=W\|F\|$ | Speed of the effective flow in the rescaled metric. |
| $h$, $\lambda_j=h|a_j|$ | Construction spacing and normalized slope scale. |

## 1. Example: why mixed signs need not destroy conditioning

Imagine positive and negative slope–readout products nearly canceling, so
that $\mathbb E_W\alpha\zeta$ is close to zero. A proof based on this signed
quantity staying away from zero would fail even if slopes and readouts
have appreciable squared size. Instead, the coarse Gram matrix can be
bounded using $E_s$, which adds their squared sizes. The dynamics cannot
rapidly destroy $E_s$ without moving through the same weak fine gradient
that drives learning.

This is different from assuming that the coarse Gram matrix stays well
conditioned. We will prove a lower bound from $E_s(0)>0$ and the coupled
evolution. It also differs from the generic Jacobian-drift argument whose
constants closed only a very short interval in the
[earlier audit](../results/checkpoint_D_optimizers/expD34_readout_race/mechanism_persistence/README.md).
Whether the replacement constants give a useful interval is a numerical
question, not a consequence asserted here.

### Exact flow and metric

Use equally weighted samples $x_i\in[-1,1]$ on a centered symmetric grid,
with $v=\langle x^2\rangle_m\in(0,1]$. The target values are arbitrary.
Let $P_H$ be the empirical orthogonal projection away from $1,x$. The
network, fine residual, and fine loss are

$$
f(\mathsf X;x)=d+\frac1{\sqrt W}\sum_{j=1}^W
\zeta_j\tanh\!\left(\frac{\alpha_jx+\beta_j}{\sqrt W}\right),
\qquad e_H=P_H(f-y),\qquad L_H=\frac12\|e_H\|_m^2.
\tag{1}
$$

Equip the state with the label metric

$$
\|\dot{\mathsf X}\|_{\mathcal H}^2
=\mathbb E_W|\dot X_j|^2+|\dot d|^2.
\tag{2}
$$

The linear change from physical parameters to $\mathsf X$ is an isometry
from physical Euclidean norm to (2). In particular, the output bias is not
multiplied by $\sqrt W$. All adjoints and operator norms below use (2) and
the empirical sample norm.

Define the exact coarse coordinates and their Jacobian by

$$
C(\mathsf X)=\left(\langle f\rangle_m,
\frac{\langle xf\rangle_m}{\sqrt v}\right),\qquad
J_C=DC,\qquad K=J_CJ_C^*,\qquad
\Pi=I-J_C^*K^{-1}J_C.
\tag{3}
$$

Whenever $K$ is positive definite, $\Pi$ is the orthogonal projection onto
the tangent space preserving the exact coarse output. Put

$$
G=\nabla_{\mathcal H}(W L_H),\qquad
\frac{d\mathsf X}{d\tau}=V(\mathsf X):=-\Pi G.
\tag{4}
$$

This is precisely physical effective fine flow $d\theta/dt=-F(\theta)$
under the isometry and $t=W\tau$. It preserves $C$, not the affine
approximation $d+\mathbb E_W\zeta\beta,\mathbb E_W\zeta\alpha$.
That distinction matters when comparing with the earlier cubic ODE.

## 2. Theorem: an initial-data test for slow reinforcement

**What this statement shows.** Choose an outer support radius and a reduced
time interval. Explicit initial moments determine whether the dynamics
remain inside that radius. If the test passes, the exact fine force cannot
grow faster than the stated exponential on that interval.

Let $R(0)=r_0$, $M(0)=M_0$, $E_s(0)=E_{s,0}>0$, and
$Y_0=\|e_H(0)\|_m$. Choose $r>r_0$ and $T>0$, and define

$$
M_*=M_0e^{8Y_0r^2T},\qquad
E_{s,*}=E_{s,0}e^{-8Y_0r^2T},
\tag{5}
$$

$$
\delta_*=
\sqrt{\frac{vE_{s,*}}{1+2M_*}}-\frac{3r^3}{W}.
\tag{6}
$$

Require **$\delta_*>0$ before squaring**, and set $\kappa_* =\delta_*^2$.
Define the support-speed bound

$$
B_*=4Y_0r^3\left[1+\sqrt{\frac{2M_*}{\kappa_*}}\right].
\tag{7}
$$

Suppose the strict first-exit condition holds:

$$
T B_*<r-r_0.
\tag{8}
$$

Then (4) exists throughout $0\le\tau\le T$, and

$$
R(\tau)\le r_0+B_*\tau<r,\quad
M(\tau)\le M_*,\quad E_s(\tau)\ge E_{s,*},\quad
K(\tau)\succeq\kappa_* I.
\tag{9}
$$

Let

$$
L_*=
\frac{9r^6}{W}+6\sqrt2Y_0r^2
+8Y_0r^2\sqrt{\frac{M_*}{\kappa_*}}
\left(\sqrt2+\frac{6\sqrt2r^2}{W}\right).
\tag{10}
$$

The force-growth and path bounds are

$$
q(\tau)\le q(0)e^{L_*\tau},\qquad
\int_0^\tau\|V(s)\|_{\mathcal H}\,ds
\le\mathcal A(\tau):=
\begin{cases}
q(0)(e^{L_*\tau}-1)/L_*,&L_*>0,\\
q(0)\tau,&L_*=0.
\end{cases}
\tag{11}
$$

Equivalently, for physical time $0\le t\le WT$,

$$
\|F(\theta(t))\|\le\|F(\theta(0))\|e^{L_*t/W}.
\tag{12}
$$

There is no target parity, target-degree, neuron-sign, readout-alignment,
or zero-hidden-bias assumption. The initial effective speed need not be
postulated small: projection and the estimates below give
$q(0)\le4Y_0r_0^2\sqrt{M_0}$. Small physical speed follows when these
rescaled quantities are bounded, since $\|F(0)\|=q(0)/W$.

### Proof, part 1: affine removal makes the raw fine gradient small

First, along (4),

$$
\frac{dL_H}{d\tau}=-\frac1W\|\Pi G\|_{\mathcal H}^2\le0.
\tag{13}
$$

Thus $Y=\|e_H\|_m\le Y_0$ without an assumption on future residuals.
The $d$ component of $G$ vanishes because $e_H$ is orthogonal to constants.

We claim the particlewise estimate

$$
|G_j|\le\sqrt{11}\,Yr^2|\alpha_j|
\le4Yr^2|\alpha_j|\qquad\text{whenever }R\le r.
\tag{14}
$$

To see it, work momentarily in physical coordinates
$a=\alpha/\sqrt W$, $b=\beta/\sqrt W$, $c=\zeta/\sqrt W$ and put
$s(u)=\operatorname{sech}^2u$. Fine-residual pairing annihilates every
affine function of $x$. In the slope derivative $cxs(b+ax)$, subtract
$cxs(b)$. Since $|s'(u)|\le2|u|$, its remainder has absolute value at most
$2|ca|(|a|+|b|)$. In the hidden-bias derivative $cs(b+ax)$, subtract
$cs(b)+cas'(b)x$. Since $|s''(u)|\le2$, the remainder is at most
$|c|a^2$. In the readout derivative $\tanh(b+ax)$, subtract
$\tanh b+ax s(b)$; the remainder is at most $a^2(|a|+|b|)$.

The rescaled gradient $G_j$ multiplies each such physical gradient
component by $W\sqrt W$. Cauchy–Schwarz in the empirical sample metric
therefore bounds its three components respectively by

$$
2\sqrt2Yr^2|\alpha_j|,\qquad
Yr^2|\alpha_j|,\qquad
\sqrt2Yr^2|\alpha_j|.
$$

Their squared sum proves (14). Consequently, with
$A=\mathbb E_W\alpha^2$,

$$
\|G\|_{\mathcal H}\le4Yr^2\sqrt A,
\qquad \|V\|_{\mathcal H}\le\|G\|_{\mathcal H}.
\tag{15}
$$

The second inequality uses orthogonality of the coarse projection. It is
valid for mixed signs and nonzero biases.

### Proof, part 2: coupled moments keep the coarse fit adjustable

Differentiating the two particle moments and using (15) gives

$$
M'\le2\sqrt M\,\|V\|_{\mathcal H}
\le8Y_0r^2M,
\qquad
E_s'\ge-2\sqrt{E_s}\,\|V\|_{\mathcal H}
\ge-8Y_0r^2E_s.
\tag{16}
$$

The last inequality uses $A\le E_s$, not just $A\le M$. Up to support
exit and through time $T$, integration yields (5).

For the affine network $f_0=d+\mathbb E_W\zeta(\alpha x+\beta)$,
write $B=\mathbb E_W\beta^2$, $Z=\mathbb E_W\zeta^2$ and
$D=\mathbb E_W\alpha\beta$. Its coarse Gram matrix is

$$
K_0=
\begin{pmatrix}
1+B+Z&\sqrt v D\\
\sqrt v D&v(A+Z)
\end{pmatrix}.
\tag{17}
$$

Since $D^2\le AB$,

$$
\det K_0\ge v[(1+Z)(A+Z)+BZ]\ge vE_s,
\qquad \operatorname{tr}K_0\le1+2M.
$$

Thus

$$
\lambda_{\min}(K_0)\ge\frac{vE_s}{1+2M}.
\tag{18}
$$

This bound does not divide by the signed moment
$\mathbb E_W\alpha\zeta$. It remains useful when that moment vanishes.

The derivative of the exact network differs from that of $f_0$ by at most
$3r^3/W$ in operator norm. Indeed,
$|\tanh u-u|\le|u|^3/3$ and $|s(u)-1|\le u^2$, while
$|u|\le\sqrt2r/\sqrt W$. In physical coordinates the squared Frobenius
bound, summed over neurons, is
$[(2\sqrt2)^2+(2\sqrt2/3)^2]r^6/W^2<9r^6/W^2$.
Projecting to coarse coordinates cannot increase it. The metric isometry
gives the same estimate in (2).

The singular-value perturbation inequality and (18) now imply

$$
\sigma_{\min}(J_C)
\ge\sqrt{\frac{vE_s}{1+2M}}-\frac{3r^3}{W}
\ge\delta_*>0.
\tag{19}
$$

Only after verifying positivity may we square this inequality to obtain
$K\succeq\kappa_*I$.

### Proof, part 3: close support and continuation

Write the projected velocity as
$V=-G+J_C^*\ell_G$, where $\ell_G=K^{-1}J_CG$. Then

$$
|\ell_G|\le\frac{\|G\|_{\mathcal H}}{\sqrt{\kappa_*}}
\le4Y_0r^2\sqrt{\frac{M_*}{\kappa_*}}.
\tag{20}
$$

For each particle the exact coarse-adjoint block has norm at most
$\sqrt2|X_j|$. To verify this, its output derivatives, represented in
the label metric, are
$(\zeta_jx s(u_j),\zeta_js(u_j),\sqrt W\tanh u_j)$.
Their squared pointwise sum is at most
$2\zeta_j^2+2(\alpha_j^2+\beta_j^2)$. Projection onto the two
orthonormal coarse functions is a contraction.

Combining this block bound with (14) and (20) yields

$$
|V_j|\le4Y_0r^3+\sqrt2r|\ell_G|\le B_*.
\tag{21}
$$

Therefore the upper right derivative of $R$ is at most $B_*$.
The strict inequality (8) prevents a first support exit before $T$.
Equations (16) and (19) then prevent a first loss of conditioning.
For completeness, $C$ is conserved because $J_CV=0$. Its constant
component bounds $d$: bounded particles bound their contribution to the
network, leaving $d=C_0-\langle f-d\rangle_m$ bounded as well.
At finite $W$, the smooth vector field can therefore be continued through
the whole interval. This proves (9).

### Proof, part 4: derive reinforcement rather than assume it

Let $J_H=De_H$. The same affine subtraction gives

$$
\|J_H\|\le\frac{3r^3}{W},\qquad
\|D^2e_H\|\le\frac{6\sqrt2r^2}{W},\qquad
\|DJ_C\|\le\sqrt2+\frac{6\sqrt2r^2}{W}.
\tag{22}
$$

Here the second-derivative norm is the norm of a bilinear map into the
empirical sample space. To check the constants, subtract the affine
network's Hessian. For a single physical particle, put $z=(x,1)$.
The remaining hidden-parameter block is $c\tanh''(u)zz^T$, with norm
at most $4\sqrt2r^2/W$. The readout cross-block is
$(s(u)-1)z$, whose symmetric off-diagonal matrix has norm at most
$2\sqrt2r^2/W$. The affine Hessian itself has norm at most $\sqrt2$.
Block diagonality across particles, followed by the sample norm and
orthogonal projection, proves (22).

Since $G=WJ_H^*e_H$, differentiation gives

$$
\|DG\|
\le W\bigl(\|J_H\|^2+\|D^2e_H\|\,Y_0\bigr)
\le\frac{9r^6}{W}+6\sqrt2Y_0r^2.
\tag{23}
$$

For a full-row-rank Jacobian, differentiating its orthogonal kernel
projector yields

$$
\|D\Pi[u]\|
\le\frac{2\|DJ_C[u]\|}{\sigma_{\min}(J_C)}.
\tag{24}
$$

One can verify (24) by writing the derivative as the sum of the two
off-diagonal terms between the kernel and row space, each bounded by
$\|DJ_C[u]\|/\sigma_{\min}(J_C)$. Thus, from $V=-\Pi G$,
(15), and (19)–(24),

$$
\|DV\|\le\|DG\|+\|D\Pi\|\,\|G\|_{\mathcal H}\le L_*.
\tag{25}
$$

Along the autonomous flow, $V'=DV[V]$. Hence
$D^+\|V\|_{\mathcal H}\le L_*\|V\|_{\mathcal H}$, proving
(11)–(12) by scalar comparison and integration. This completes the proof.

If $q(0)=0$, local uniqueness makes the effective-flow solution stationary;
no division by $q$ or logarithm of zero is needed. If $Y_0=0$, this case
holds automatically. If $E_{s,0}=0$, the two-coordinate coarse Gram matrix
is singular initially, and the theorem does not apply; no pseudoinverse
extension is asserted here.

## 3. Prediction: bounded force reinforcement limits scale acquisition

**What to measure next.** The theorem predicts a bounded amplification of
the full effective force and a limited path length when its initial-data
test passes. It does not predict universal slope contraction. A particle
may move outward throughout the interval.

Let $\alpha_{\max,0}=\max_j|\alpha_j(0)|$. For physical time $t=W\tau$,
(21) gives the direct support bound

$$
\max_j\lambda_j(t)
\le\frac{h}{\sqrt W}\bigl(\alpha_{\max,0}+B_*\tau\bigr).
\tag{26}
$$

For a chosen threshold $\lambda_*$, suppose
$g_*:=\sqrt W\lambda_*/h-\alpha_{\max,0}>0$. The fraction of distinct neurons
that ever reach this threshold by physical time $W\tau$ satisfies

$$
p_{\rm ever}(W\tau;\lambda_*)
\le\min\left\{1,\frac{\mathcal A(\tau)^2}{g_*^2}\right\}.
\tag{27}
$$

Indeed, each such neuron has
$\sup_{s\le\tau}|\alpha_j(s)-\alpha_j(0)|\ge g_*$. Minkowski's
inequality applied to the integral of the labelwise velocity bounds the
RMS of these suprema by $\int_0^\tau\|V(s)\|_{\mathcal H}ds$.
This proves (27) for asynchronous first hits, not merely the population
above threshold at one sampled time. If $B_*\tau<g_*$, (26) excludes
every hit.

The factor $1/W$ in the physical reinforcement exponent follows from a
bounded rescaled support and the affine cancellation. The theorem does
not establish a width-dependent extension of $T$, a $W^2$ clock for all
targets, or an enormous optimizer-update horizon. These would require
stronger structural estimates or favorable evaluated constants. It also
does not claim that training reaches the tested initial state from random
initialization.

## 4. Ordinary GD: derive tracking and contain every update segment

The [ordinary-GD companion](d34_moment_persistence_gd.md) proves a discrete
version directly for the exact tanh network. It starts from the same moments,
the initial total loss, and the measured initial disequilibrium $z_0$.
Its scalar recurrences bound support, both moments, tracking, and fine-force
growth together. There is no required width power for $z_0$ and no assumed
future tracking history.

Two differences from the continuous effective-flow proof matter. Ordinary
GD need not decrease fine loss, so the companion proves total-loss descent
and bounds the residual on each update segment. Also, the moment and support
recurrences contain the entire segment before derivative bounds are used
on it. The lower moment region is not convex; endpoint membership alone
would not justify that step.

There is no polynomial-trajectory approximation error in either theorem.
The tanh inequalities bound the exact Jacobian and its derivatives. The
discrete result still has tracking, finite-step, and containment conditions;
substituting $t=\eta n$ into (12) alone does not prove it.

The [central explanation](d34_coarse_balance_stagnation.md) gives the
empirical motivation. The [evaluation protocol](../results/checkpoint_D_optimizers/expD34_readout_race/moment_persistence/PROTOCOL.md)
separates applicability of the continuous theorem from that of its discrete
companion. Only a passing discrete calculation bounds archived GD motion.
