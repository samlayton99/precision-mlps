# A generic energy baseline for finite-time scale exclusion

A 20,000-update exclusion of a distant slope threshold can follow from
ordinary GD descent alone. It need not identify the effective-force mechanism.
The argument below makes this baseline explicit before interpreting the
[much tighter moving-reference certificate](d34_certified_instance.md).
It proves a statement for initial states satisfying stated norm and loss
bounds. Outward-rounded checks now verify those assumptions for all 18
seed-30 independent-width checkpoints listed in Section 4, so their entire
next 20,000-update windows admit this generic exclusion.

**Notation.**

| Symbol | Meaning |
|---|---|
| $\theta=(a,b,c,d)$ | Parameters of $f_\theta(x)=d+\sum_{j=1}^Wc_j\tanh(a_jx+b_j)$. |
| $L=\frac12\|f_\theta-y\|_m^2$ | Empirical half mean squared error; $|x_i|\le1$. |
| $M_0=\|(a_0,b_0,c_0)\|_2$ | Initial combined hidden-parameter/readout norm, excluding $d_0$. |
| $R$ | Proposed bound on cumulative full-parameter travel from the initial state. |
| $\eta,N$ | Ordinary GD step size and number of additional updates. |

## 1. Example: a coarse threshold can already be inaccessible

Suppose $M_0\le3$, $L(\theta_0)\le1$, and
$0<\eta\le1/490$. These assumptions permit the binary64 value used for
the decimal step size $0.002$, when interpreted as an exact real number.
Then, for every width and every empirical target satisfying these initial
conditions, all full-GD iterates through 20,000 additional updates obey

$$
\sum_{n<k}\|\theta_{n+1}-\theta_n\|_2<8,
\qquad
\max_j|a_{j,k}|<11,
\qquad 0\le k\le20{,}000.
\tag{1}
$$

Consequently no neuron reaches $|a|=16$. With the campaign normalization
$\lambda=2|a|/N_{\rm ref}$ and $N_{\rm ref}\ge128$, this excludes
$\lambda=0.25$ throughout the entire window. This is an **ever-hit**
exclusion, not merely a statement about the last iterate. Physical width
$W$ and the normalization parameter $N_{\rm ref}$ remain distinct.

The initial output bias $d_0$ need not be small: the initial loss controls
the residual, while the derivative bounds below do not depend on $d$.

## 2. The bound: close descent using a one-step neighborhood

Here is a general sufficient condition. Choose an initial loss upper bound
$L_\star\ge L(\theta_0)$ and a proposed travel radius $R>0$. Set

$$
J_R=\sqrt{1+2(M_0+R)^2},\qquad
\delta=\eta J_R\sqrt{2L_\star},\qquad R_+=R+\delta,
$$

$$
J_+=\sqrt{1+2(M_0+R_+)^2},\qquad
M_2=\sqrt2+\frac{8(M_0+R_+)}{3\sqrt3},
$$

$$
r_+=\sqrt{2L_\star}+J_+\delta,\qquad
H=J_+^2+M_2r_+,\qquad \mu=1-\eta H/2.
\tag{2}
$$

**Proposition.** If $\mu>0$ and

$$
Q_N:=\sqrt{\frac{\eta N L_\star}{\mu}}<R,
\tag{3}
$$

then all iterates through update $N$ have loss at most $L_\star$ and
cumulative parameter travel at most $Q_k$ at update $k$. In particular,
$\max_j|a_{j,0}|+Q_N<\Gamma$ excludes every crossing of slope threshold
$\Gamma$ through that horizon. Upper bounds replacing the constants in
(2) give the same conclusion with their resulting lower bound on $\mu$.

**Derivative estimates.** For a single input, write $u_j=a_jx+b_j$.
Using $|\tanh u|\le|u|$ and $|\tanh' u|\le1$ gives

$$
\|\nabla_\theta f(x)\|_2^2
=1+\sum_j\left[\tanh^2u_j+c_j^2(1+x^2)\tanh'^2u_j\right]
\le1+2\|(a,b,c)\|_2^2.
\tag{4}
$$

The output Hessian consists of independent three-parameter neuron blocks
and a zero output-bias block. Within a block its geometry/readout mixed
part has norm at most $\sqrt{1+x^2}\le\sqrt2$. Its geometry/geometry
part has norm at most
$|c_j|(1+x^2)\sup|\tanh''|\le8|c_j|/(3\sqrt3)$.
Thus its operator norm is bounded by $M_2$ on the enlarged ball.

The empirically normalized sample Jacobian has Frobenius norm at most the
same $J$ as in (4). The loss Hessian identity

$$
\nabla^2L=J^TJ+\langle(f-y)\nabla^2f\rangle_m
\tag{5}
$$

therefore bounds its norm by $J_+^2+M_2\|f-y\|_m$ wherever the
corresponding residual bound is available.

**Bootstrap proof.** Assume all previous steps have total travel below
$R$ and the old state has loss at most $L_\star$. Its gradient norm is
at most $J_R\sqrt{2L_\star}$, so the entire next GD step lies in the
radius-$R_+$ ball. Along this particular segment, integrating the output
Jacobian gives $\|f-y\|_m\le r_+$. Consequently (5) bounds the loss
Hessian by $H$ on the entire step, and the descent lemma gives

$$
L_{n+1}\le L_n-\eta\mu\|\nabla L_n\|_2^2.
$$

Summing descent and using $L\ge0$, followed by Cauchy–Schwarz, yields

$$
\sum_{n<k}\|\theta_{n+1}-\theta_n\|_2
\le\sqrt{k\sum_{n<k}\eta^2\|\nabla L_n\|_2^2}
\le\sqrt{\frac{\eta kL_\star}{\mu}}\le Q_N<R.
\tag{6}
$$

The new state still has loss at most $L_\star$ and travel below $R$,
closing induction from $k=0$. No low-loss assumption on the entire ball
was used: low loss holds inductively at the old state, and the residual
bound is required only along its next step.

## 3. Verify the example and interpret the gain

For $R=8$, $M_0\le3$ and $L_\star=1$, (4) gives $J_R^2\le243$
and $J_R<16$. The next step has length at most
$16\sqrt2/490<0.05$, so we may use the larger radius $R_+=8.05$.
There,

$$
J_+^2\le1+2(11.05)^2<246,\quad J_+<16,\quad M_2<19,
$$

$$
\|f-y\|_m\le\sqrt2(1+256/490)<2.18,
\quad H<246+19(2.18)<288.
$$

Hence $\mu\ge1-144/490=346/490>0.7$ and

$$
Q_{20{,}000}\le\sqrt{\frac{20{,}000}{490(0.7)}}<8.
$$

This proves (1) with explicit, width-independent constants. It does not
explain the direction of slope motion, distinguish target families, or
show persistence for arbitrarily long times. The horizon in (3) grows only
like the squared allowed travel divided by available loss and step size.

The existing [local inaccessible-loss argument](d34_coarse_balance_stagnation_details.md#a-stronger-bound-from-the-loss-the-network-cannot-yet-remove)
and its [implementation](../experiments/expD34_readout_race/persistence_energy.py)
use this same descent/Cauchy principle, strengthened by a positive local
loss floor. The present variant uses the universal zero floor and derives
curvature on each low-loss outgoing step instead of bounding residuals
throughout a large ball. It is not a new force-attribution theorem.

The [Arb certificate](d34_certified_instance.md) instead confines an
identified instance around a moving effective-model reference to radius
less than $0.000373$ through 20,000 updates. That is a much sharper
trajectory statement than a travel budget of eight, and checks a specified
empirical instance with rounding control. Its distant-threshold exclusion
alone should not be presented as evidence that only the proposed mechanism
could establish that horizon: for the 18 states below, the elementary
baseline already supplies that exclusion.

## 4. Verified empirical instances

The six targets `moment5`, `mixed_sine`, `gauss_left`, `bump_right`,
`step_right`, and `kink_abs` were checked at seed 30 and the archived
20,000-update checkpoint, at each of three independent widths. Every case
satisfies $M_0\le3$, $L(\theta_0)\le1$, and
$\eta\le1/490$. The table displays maxima across all six targets at each
width, rounded **upward** from the retained Arb bounds.

| $N_{\rm ref}$ | Physical width $W$ | Verified states | Upper bound on $M_0$ | Upper bound on $L(\theta_0)$ |
|---:|---:|---:|---:|---:|
| 128 | 177 | 6 | 2.681334 | 0.416601 |
| 512 | 705 | 6 | 2.607185 | 0.416404 |
| 1024 | 1409 | 6 | 2.631417 | 0.416352 |

The unchanged [Arb helper](../experiments/expD34_readout_race/mechanism_arb_certificate.py)
evaluates an outward-rounded upper bound $s_0\ge\|f_{\theta_0}-y\|_m$.
The loss bound is the outward-rounded upper endpoint of $s_0^2/2$;
$s_0$ is nonnegative. The norm check uses exactly the first $3W$ archived
parameters, $(a,b,c)$, excluding the output bias. Arb also verifies that the
exact binary64 number represented by `0.002` is at most $1/490$.
No fine-force or coarse-tracking assumptions enter these checks.

The [18 per-state records](../results/checkpoint_D_optimizers/expD34_readout_race/mechanism_refinement/energy_baseline_arb/conditions.json),
[summary](../results/checkpoint_D_optimizers/expD34_readout_race/mechanism_refinement/energy_baseline_arb/summary.json),
and [executed command](../results/checkpoint_D_optimizers/expD34_readout_race/mechanism_refinement/energy_baseline_arb/verification.sbatch)
preserve enclosing interval strings, input hashes, and the original source
hash. Job 1295 used 100-bit Arb arithmetic and 103 seconds on one CPU.

These verified losses also give a stronger horizon than the generic
20,000-update example. Every loss upper bound is below $0.417$. Keep the
radius $R=8$ and the segment bounds $H<288$, $\mu>0.7$ proved using
$L_\star=1$, but use the sharper initial loss in the summed descent budget:

$$
Q_{50{,}000}^2
\le\frac{50{,}000(0.417)}{490(0.7)}
=\frac{20{,}850}{343}<64.
$$

The same induction therefore proves, for **every one of these 18 empirical
instances**, that total parameter travel remains below eight and no neuron
reaches $|a|=16$ or $\lambda=0.25$ at any prefix of the next **50,000
updates**. This extends from the archived update-20,000 state through nominal
update 70,000. It is an analytic extension from verified initial conditions;
no new training or trajectory comparison is needed or claimed.

The statement concerns exact real-arithmetic GD on the archived binary64
empirical data and initial state; it does not certify preceding training or
floating-point training roundoff. It establishes a coarse finite-time
exclusion. The much sharper predictions of signed slope motion and the
evolving effective-force geometry still require the mechanism analysis.
