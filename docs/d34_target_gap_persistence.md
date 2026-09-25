# A target gap can preserve weak population sensitivity

Broad tanh features can initially respond to an unwanted cubic component much
more strongly than to the fine structure in a target. The question is why
coupled training cannot rapidly change that situation. This note proves one
answer: **if the target has no low-degree fine components, then the loss that
training can release inside a broad-feature population is small. Gradient
flow cannot move that population far without spending this limited loss.**
Consequently, weak sensitivity persists for a provable duration. This is an
exact, finite-width result for evolving features, readouts, and biases.

The result starts at a specified checkpoint, typically after the coarse
tracking transient. It does not prove entry into that regime. Its target
assumption is stronger than the assumptions of the broader conditional
feedback theorem. The purpose is to derive persistence for an identifiable
family, then evaluate whether the resulting duration is useful at our saved
checkpoints. A proof of the implication and a useful numerical guarantee are
separate questions.

**Notation.** All parameter coordinates use the ordinary GD metric.

| Symbol | Meaning |
|---|---|
| $f_\theta=d+\sum_{j=1}^W c_j\tanh(a_jx+b_j)$ | Network; $a_j$ is the slope also denoted $\gamma_j$. |
| $y=y_C+g$ | Target split into affine and non-affine parts. |
| $P_H$ | Orthogonal projection removing affine functions. |
| $z=P_Hf_\theta$, $e_C=P_C(f_\theta-y)$ | Generated non-affine output and coarse residual. |
| $M_4=W\sum_j(a_j^2+b_j^2+c_j^2)^2$ | Rescaled population fourth moment. |
| $q=M_4^{1/4}$ | Fourth-moment radius; distinct from the older output-capacity symbol $\mathcal Q$. |
| $F,R$ | Effective fine gradient, including compensation, and coarse tracking gradient. |
| $D_*$ | Upper bound on the loss available to finance movement before exit. |

## 1. Example: correcting a generated cubic cannot unlock unlimited movement

Take an affine target plus a normalized degree-nine orthogonal polynomial.
Orthogonality is with respect to the actual input measure, not monomials
treated as if they were orthogonal. At a broad-feature checkpoint,

$$
c_j\tanh(a_jx+b_j)
=c_j(a_jx+b_j)-\frac{c_j}{3}(a_jx+b_j)^3+\cdots.
$$

The affine part can fit the coarse target. The cubic part introduces error
without correlating with the degree-nine target. Reducing that generated
error releases some loss and can move the parameters. But that reservoir is
small when the entire population has a moderate fourth moment. The fine
target can supply additional loss reduction only through higher terms.

The theorem below makes this argument without replacing tanh by a polynomial
ODE. Taylor's theorem supplies **global inequalities for exact tanh**, valid
even if some neurons leave the small-argument region. A population moment
bounds their total contribution; no maximum-neuron restriction is imposed.

The prediction is finite-time slow acquisition, not restoration toward an
attractor. Slopes may grow, and isolated neurons may escape. The conclusion
limits aggregate movement and output improvement over a stated duration.

## 2. The exact dynamics and the structural target assumption

Let $\mu$ be a probability measure on $[-1,1]$ for which constants and linear
functions are independent in $L^2(\mu)$. All norms and inner products below
use this space. Uniform integration and a finite training grid are both
allowed, but they define different orthogonality conditions. Set

$$
L(\theta)=\frac12\|f_\theta-y\|^2,
\qquad \dot\theta=-\nabla L(\theta).
\tag{1}
$$

Writing $J_C=D_\theta(P_Cf_\theta)$ and $J_H=D_\theta(P_Hf_\theta)$, the
usual decomposition, when $J_CJ_C^*$ is invertible, is

$$
\begin{aligned}
\Pi&=I-J_C^*(J_CJ_C^*)^{-1}J_C,\\
F&=\Pi J_H^*(z-g),\\
R&=J_C^*\left[e_C+(J_CJ_C^*)^{-1}J_CJ_H^*(z-g)\right],\\
\nabla L&=F+R.
\end{aligned}
\tag{2}
$$

Thus compensation remains part of $F$. The proof for the full flow (1)
does not require inversion of the coarse Gram matrix or a future bound on
$R$. Instead it charges the coarse residual present at the checkpoint to
the movement budget. A small $R$ alone does **not** imply a small $e_C$;
the latter must be measured separately.

Let $\mathcal P_5$ denote polynomials of degree at most five. Our main family
is

$$
y=y_C+g,\qquad y_C\in\mathcal P_1,\qquad
g\perp\mathcal P_5.
\tag{3}
$$

This contains arbitrary square-integrable mixtures of orthogonal polynomial
degrees six and above, with arbitrary affine parts. It is not confined to
degree nine. An ordinary sine target generally does not satisfy (3).
Section 6 quantifies leakage into the omitted low degrees rather than
silently treating those targets as members of the exact-gap family.

## 3. Exact population bounds: the available loss is small

Write $p_j=(a_j,b_j,c_j)$, and define

$$
A_3=\frac{\sqrt6}{8},\qquad
A_7=\frac{17}{315}\frac{2^{7/2}7^{7/2}}{8^4}.
\tag{4}
$$

**Lemma 1 (output capacity and target coupling).** For every parameter state,

$$
\|z\|\le A_3\frac{q^4}{W}.
\tag{5}
$$

If (3) holds, then also

$$
|\langle g,f_\theta\rangle|
\le A_7\|g\|\frac{q^8}{W^2}=:B(q).
\tag{6}
$$

**Proof.** The global remainder inequalities are

$$
|\tanh u-u|\le |u|^3/3,
\qquad
|\tanh u-u+u^3/3-2u^5/15|\le 17|u|^7/315.
\tag{7}
$$

The first follows by integrating $|\tanh'(u)-1|=\tanh^2u\le u^2$.
For the second, $\sup|\tanh^{(7)}|\le272$; a direct verification appears
in the appendix. Taylor's theorem through degree six gives $272/7!=17/315$.

For odd $k$, maximizing $|c|(|a|+|b|)^k$ on a unit Euclidean sphere gives

$$
|c|(|a|+|b|)^k\le
\frac{2^{k/2}k^{k/2}}{(k+1)^{(k+1)/2}}\,|p|^{k+1}.
\tag{8}
$$

Indeed, $|a|+|b|\le\sqrt2(a^2+b^2)^{1/2}$, and the remaining maximum
occurs at $c^2=1/(k+1)$. Projection removes the linear term in (7), and
is a contraction, giving (5) after summation. In (6), orthogonality removes
every polynomial term of degree at most five, including all bias-generated
terms. Cauchy–Schwarz and (8) give $A_7\|g\|\sum_j|p_j|^8$.
Finally, $\sum_j|p_j|^8\le(\sum_j|p_j|^4)^2=q^8/W^2$. $\square$

The loss has the exact decomposition

$$
L=\frac12\|g\|^2+\frac12\|e_C\|^2
  +\frac12\|z\|^2-\langle g,z\rangle.
\tag{9}
$$

The generated error contributes a nonnegative term. The fine target lowers
the loss only through the last term, which (6) limits throughout a moment
region. Thus, in $q\le q_*$,

$$
L\ge\frac12\|g\|^2-B(q_*).
\tag{10}
$$

This is the mechanism behind the proof: a large unlearned target error is
not automatically a large energy supply for parameter motion.

## 4. Persistence follows from energy dissipation

**Theorem 1 (target-gap persistence for full gradient flow).** At a specified
checkpoint $\theta_0$, let $q_0=M_4(\theta_0)^{1/4}$ and choose $q_*>q_0$.
Assume (3), and define

$$
D_*=L(\theta_0)-\frac12\|g\|^2+B(q_*),
\qquad
T_* =\frac{(q_*-q_0)^2}{\sqrt W D_*}.
\tag{11}
$$

Then $D_*\ge0$. If $D_*>0$, the exact flow satisfies $q(t)<q_*$ for
$0\le t<T_*$. Throughout this interval,

$$
\begin{aligned}
\int_0^t\|\dot\theta\|^2\,ds&\le D_*,&
\|\theta(t)-\theta_0\|&\le\sqrt{tD_*},\\
\frac{\|a(t)\|}{\sqrt W}&\le\frac{q_*}{\sqrt W},&
\|f_{\theta(t)}-y\|&\ge
\left[\|g\|-A_3q_*^4/W\right]_+.
\end{aligned}
\tag{12}
$$

If $D_*=0$, the flow is stationary. No future force, neuronwise bound,
frozen Jacobian, or equilibrium assumption is a premise.

**Proof.** Up to the first exit from $q<q_*$, (1) and (10) give

$$
\int_0^t\|\dot\theta\|^2ds=L(\theta_0)-L(\theta(t))\le D_*.
\tag{13}
$$

The block fourth norm satisfies the triangle inequality. Its change is at
most the Euclidean movement times $W^{1/4}$. Therefore

$$
q(t)\le q_0+W^{1/4}\|\theta(t)-\theta_0\|
\le q_0+W^{1/4}\sqrt{tD_*}.
\tag{14}
$$

An exit before $T_*$ would contradict (14), including its continuous limit
at the exit time. For fixed finite $W$, bounded $q$ bounds all hidden
parameters, while bounded loss bounds $d$ on this region. The smooth ODE
therefore continues up to every time strictly below $T_*$. The same argument
with $D_*=0$ forces zero velocity and a stationary solution.

Moreover $\sum_j a_j^2\le\sum_j|p_j|^2\le q_*^2$, giving the RMS bound.
The reverse triangle inequality applied to $z-g$, with (5), gives the output
bound. $\square$

**What has been proved about sensitivity?** The elementary columnwise
Jacobian estimate in the appendix gives

$$
\|J_H(t)\|_{\mathrm{HS}}^2
\le9\sum_j|p_j(t)|^6
\le9q_*^6/W^{3/2},\qquad 0\le t<T_*.
\tag{15}
$$

This is a consequence of the persistence proof, not an assumption on the
future trajectory. Since $\|\Pi\|\le1$, it also bounds the effective fine
sensitivity wherever (2) is defined. It need not give an accurate force
forecast: residual alignment and coarse compensation can reduce the force
substantially further.

**Rates and acquisition.** If $q_0,q_*,\|g\|$ are width-independent,
$q_*-q_0$ is bounded below, and $\|e_C(0)\|^2=O(W^{-2})$, equations
(5), (6), and (9) imply $D_*=O(W^{-2})$, hence
$T_*=\Omega(W^{3/2})$ in physical gradient-flow time. The explicit formula
(11), rather than this asymptotic notation, determines numerical usefulness.
The normalization is the unscaled sum network in the glossary; changing
the parameter metric or network normalization changes the rate.

For a spacing $h$, put $\lambda_j=h|a_j|$. Then the RMS scale is at most
$hq_*/\sqrt W$, and at every time before $T_*$,

$$
\frac1W\#\{j:\lambda_j\ge\lambda_*\}
\le \min\{1,h^2q_*^2/(W\lambda_*^2)\}.
\tag{16}
$$

There is also an **ever-acquired** bound. For any $\delta>0$, the fraction
of labels for which $\sup_{s\le t}h|a_j(s)-a_j(0)|\ge\delta$ is at most
$h^2tD_* /(W\delta^2)$. This follows by applying Cauchy–Schwarz to each
coordinate's path and summing its action. It is an aggregate conclusion,
not a premise that every neuron moves slowly.

For the effective flow $\dot\theta=-F$, the same proof uses
$L_H=\|z-g\|^2/2$ instead of $L$. It removes the initial coarse residual
from $D_*$ because $dL_H/dt=-\|F\|^2$. This variant holds on intervals
where the projected flow is defined; the full-flow theorem needs no
coarse-rank condition.

## 5. Ordinary GD: a discrete proof with an explicit step guard

A flow theorem is not a theorem about a particular number of optimizer
updates. The following result handles the finite GD step without assuming
that a step remains in the desired moment region.

**Theorem 2 (GD transfer).** Use the same target and checkpoint as in
Theorem 1. Choose $q_*>q_0$, and put

$$
\begin{aligned}
r&=(q_*-q_0)/W^{1/4},& q_o&=2q_*-q_0,\\
j_o&=\sqrt{1+2q_o^2},&
H_o&=j_o^2+(\sqrt{2L_0}+2rj_o)(\sqrt2+4q_o),\\
D_o&=L_0-\tfrac12\|g\|^2+B(q_o),&
G_0&=\sqrt{1+2q_*^2}\sqrt{2L_0}.
\end{aligned}
\tag{17}
$$

Suppose $\eta H_o\le1$ and $\eta G_0\le r$. For ordinary GD
$\theta_{k+1}=\theta_k-\eta\nabla L(\theta_k)$, every integer $n$ satisfying

$$
n\eta<\frac{r^2}{2D_o}
\tag{18}
$$

has $\|\theta_n-\theta_0\|<r$, and hence $q_n<q_*$. The RMS and output
bounds (12) hold at those iterates. The action and displacement bounds become
$\sum_{k<n}\|\theta_{k+1}-\theta_k\|^2/\eta\le2D_o$ and
$\|\theta_n-\theta_0\|\le\sqrt{2n\eta D_o}$.

**Proof.** The Euclidean ball of radius $2r$ about $\theta_0$ lies in
$q\le q_o$. Throughout this ball the exact output Jacobian satisfies
$\|Df\|\le j_o$ and $\|D^2f\|\le\sqrt2+4q_o$; the appendix derives
these constants. The residual norm is at most $\sqrt{2L_0}+2rj_o$ by
integration along a straight segment from $\theta_0$. Hence
$\|\nabla^2L\|\le H_o$ throughout that ball.

Until the first iterate leaving the radius-$r$ ball, loss descent gives
$\|\nabla L(\theta_k)\|\le G_0$. The guard $\eta G_0\le r$ places the
whole next step in the radius-$2r$ ball, even if its endpoint is the first
exit. Taylor's integral formula and $\eta H_o\le1$ then give

$$
L_{k+1}\le L_k-\frac{1}{2\eta}\|\theta_{k+1}-\theta_k\|^2.
\tag{19}
$$

The endpoint still has $q\le q_o$, so (10), now with $q_o$, bounds the
total loss decrease by $D_o$. Summing (19) and applying Cauchy–Schwarz
contradicts a first exit at any $n$ satisfying (18). This also establishes
loss descent inductively, so its use above is not circular. $\square$

The factor two and outer radius are explicit costs of this simple transfer.
The theorem concerns plain GD. Adam has a different metric and momentum;
none of these Euclidean action bounds are asserted for Adam.

## 6. How much target mismatch can the proof tolerate?

For a general target, decompose its fine part orthogonally as

$$
g=\ell+g_{>5},\qquad
\ell\in\mathcal P_5\cap\mathcal P_1^\perp,\qquad
g_{>5}\perp\mathcal P_5,\qquad \delta=\|\ell\|.
\tag{20}
$$

The same proofs hold with

$$
B(q)=\delta A_3q^4/W+A_7\|g_{>5}\|q^8/W^2.
\tag{21}
$$

This follows from $|\langle\ell,z\rangle|\le\delta\|z\|$ and Lemma 1.
The energy baseline remains $\|g\|^2/2$, not $\|g_{>5}\|^2/2$. Leakage
of order $W^{-1}$ preserves the $W^{3/2}$ lower time scale under the other
conditions above. Fixed leakage can reduce the guaranteed scale to
$\Omega(\sqrt W)$ with these estimates. Sine targets therefore require
their actual low-degree content to be included; this theorem alone is not
a sharp explanation of their observed long plateaus.

A useful intermediate family is $g\perp\mathcal P_3$, which includes the
degree-five target. The global fifth-order remainder gives

$$
B(q)=A_5\|g\|q^6/W^{3/2},\qquad
A_5=\frac{2}{15}\frac{2^{5/2}5^{5/2}}{6^3}.
\tag{22}
$$

Here $\sup|\tanh^{(5)}|\le16$, and $\sum_j|p_j|^6\le(q^4/W)^{3/2}$.
With the same initial coarse-error scaling, the resulting guaranteed flow
time is $\Omega(W)$. Leakage into degrees two and three adds the first term
of (21), with $\delta$ defined for that smaller polynomial space.

## 7. Relation to the broader mechanism and to existing theory

The broader feedback theorem explains trajectories whose accumulated
reinforcing curvature and force concentration remain controlled. This note
provides a different closure for a narrower target family. It does not
bound each curvature term or prove every fitted response-to-travel premise.
Instead it bounds what the entire evolving system can accomplish with its
available loss. In particular, it proves persistence of aggregate sensitivity
even when the local force grows or changes direction.

This connects the spectral-bias intuition to an ODE statement. Missing
low-degree target components weaken the immediate learning signal. The
additional step here is that the exact dynamics cannot quickly create the
population moment that would remove that weakness. The moment also implies
an output-error floor, which is the operational failure criterion.

The energy-versus-movement argument is related to the general analysis of
slow gradient flows by [Otto and Reznikoff, *Slow Motion of Gradient
Flows*](https://webdoc.sub.gwdg.de/ebook/serien/e/sfb611/275.pdf).
Their slow-manifold framework uses additional transverse relaxation
conditions. We do not assume those conditions or infer an attracting
manifold from our perturbation experiments; the elementary energy identity
and a first-exit argument suffice here.

[Berthier, Montanari, and Zhou, *Learning time-scales in two-layers neural
networks*](https://arxiv.org/abs/2303.00055) study learning stages and missing
target coefficients in a high-dimensional single-index setting, combining
rigorous results with asymptotic derivations. This supports examining target
coefficient gaps, but their parameterization and time scales do not directly
supply the theorem above. [Yang, *Curse of Dimensionality in Neural Network
Optimization*](https://arxiv.org/abs/2502.05360) connects population moment
growth to approximation obstructions in a different, high-dimensional
setting. The useful common pattern is to bound population growth and then
convert it into an approximation limitation.

The new mathematical conclusion is therefore deliberately scoped: an exact
tanh, finite-width, post-checkpoint persistence theorem for a specified
target family, with a leakage allowance and a separate GD transfer. The
empirical audit must still determine how much of the observed duration its
explicit constants explain.

## Appendix: elementary derivative bounds used in the proofs

Put $s=\tanh^2u\in[0,1]$. The seventh derivative equals

$$
-272+3968s-12096s^2+13440s^3-5040s^4.
$$

Its degree-four Bernstein coefficients on $[0,1/2]$ are
$(-272,224,216,124,53)$, and on $[1/2,1]$ are
$(53,-18,-68,8,0)$. A polynomial lies between the smallest and largest
Bernstein coefficient on its interval. Thus its absolute value is at most
$272$. Similarly, the fifth derivative is $16-136s+240s^2-120s^3$.
Its degree-three Bernstein coefficients on the two half intervals are
$(16,-20/3,-28/3,-7)$ and $(-7,-14/3,8/3,0)$, giving the bound $16$.
These establish the global remainder bounds, including at large arguments.

For (15), set $u=ax+b$ and use $|u|\le\sqrt2|p|$ and
$|\tanh'(u)-1|\le u^2$. Removing the affine parts of the three Jacobian
columns gives pointwise upper bounds $2|p|^3$, $2|p|^3$, and
$2\sqrt2|p|^3/3$, respectively. Their squared sum is at most $9|p|^6$.
The output-bias column projects to zero. Orthogonal projection cannot
increase the column norms.

For the GD guard, before projection the squared norm of the three columns
is at most $2c^2+2(a^2+b^2)=2|p|^2$. The output-bias column adds one, so
$\|Df\|^2\le1+2\sum_j|p_j|^2\le1+2q^2$.
The Hessian of one feature $c\tanh(ax+b)$ has a geometry block bounded
by $4|c|$ and a readout–geometry cross block bounded by $\sqrt2$.
The full parameter Hessian of the scalar output is block diagonal across
neurons, hence $\|D^2f\|\le\sqrt2+4(\sum_j|p_j|^2)^{1/2}
\le\sqrt2+4q$. These inequalities concern the whole parameter norm;
they impose no separate restriction on any neuron.
