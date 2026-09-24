# Population persistence and output error in an evolving tanh network

An inaccurate network need not return toward its initial slopes to remain
inaccurate. Its whole population can continue moving, while the nonlinear
output sensitivity needed to correct the residual develops too slowly. This
note makes that distinction precise for the exact tanh gradient-flow ODE.

The first result is a **closed, initial-data persistence theorem**. A finite
population moment bounds how quickly that same moment can increase; a
collective travel bound preserves coarse conditioning; and together they
bound output-error reduction. No future concentration bound, frozen Jacobian,
Taylor surrogate, or maximum-neuron support assumption is needed. The
sixth-moment version gives a width-dependent window proportional to
$W^{2/3}$ under bounded initial rescaled moments. A higher finite-moment
version improves this to $W^{1-2/p}$ for fixed $p\ge6$.

The preferred mechanistic refinement is **Theorem 12**, proved in the final
appendix. One future structural premise, bounded concentration of the
effective force across the population, closes the fourth-moment evolution
and gives a conditional time proportional to $W$. It assumes neither small
future force nor bounded future moments. This sharper result and the
initial-data result answer different questions: the former identifies the
population property whose preservation remains to be proved, while the
latter supplies a shorter sufficient window without that future premise.

These are rigorous sufficient bounds, not yet claims of quantitatively useful
certification on every experimental checkpoint. The cleanest theorem concerns
the effective fine flow. A final initial-data comparison system also includes
tracking and finite steps for ordinary GD. In particular, small
tracking-induced slope motion does not automatically imply small
tracking-induced output progress.

For a reader seeking the main logical chain: Section 2 translates population
structure into an output-error floor; Sections 4–5 prove that structure
persists under the effective ODE; Section 6 includes tracking and discrete
GD. The width exponents describe sufficient asymptotic windows. Whether
their constants give a useful training-budget guarantee at a particular
checkpoint is a separate numerical question.

| Symbol | Meaning |
|---|---|
| $W$ | Number of neurons. |
| $p_j=(a_j,b_j,c_j)$ | Physical slope, hidden bias, and readout of neuron $j$. |
| $X_j=\sqrt W p_j$ | Rescaled particle; this is a change of coordinates, not a change of optimizer. |
| $e_C,e_H$ | Affine and non-affine parts of output error. |
| $F,R$ | Effective fine gradient and coarse tracking correction; the full gradient is $F+R$. |
| $Y=\|e_H\|$ | Non-affine output-error norm. |
| $M=\sum_j|p_j|^2$ | Total particle second moment, also $\mathbb E_W|X|^2$. |
| $E_s=\sum_j(a_j^2+c_j^2)$ | Slope–readout second moment controlling coarse conditioning. |
| $M_p=\mathbb E_W|X|^p$ | Rescaled population moment of order $p$. |
| $B=(\sum_j|p_j|^6)^{1/6}$ | Physical population sixth norm; $B=M_6^{1/6}/W^{1/3}$. |

## 1. The exact evolving system

Let $\mu$ be a symmetric probability measure supported on $[-1,1]$, with
$v=\int x^2\,d\mu>0$. A symmetric training grid with empirical weights is
one example. All function inner products and norms below use $L^2(\mu)$.
Assume $y\in L^2(\mu)$. The network and loss are

$$
f_\theta(x)=d+\sum_{j=1}^Wc_j\tanh(a_jx+b_j),
\qquad L(\theta)=\tfrac12\|f_\theta-y\|^2.
$$

The parameter metric is ordinary Euclidean distance in $(a,b,c,d)$.
Let $P_H$ remove the orthonormal affine coordinates $1,x/\sqrt v$,
and set

$$
e_C=\left(\langle f-y,1\rangle,
\langle f-y,x/\sqrt v\rangle\right),\qquad
e_H=P_H(f-y),\qquad J_C=D_\theta e_C,\quad J_H=D_\theta e_H.
$$

Adjoints use the stated parameter and output metrics. Wherever
$K=J_CJ_C^*$ is positive definite, define

$$
\begin{aligned}
\Pi&=I-J_C^*K^{-1}J_C,& g_H&=J_H^*e_H,\\
\ell&=K^{-1}J_Cg_H,& F&=\Pi g_H=g_H-J_C^*\ell,\\
z&=e_C+\ell,&R&=J_C^*z.
\end{aligned}
\tag{1}
$$

Thus $\nabla L=F+R$. The compensating coarse gradient $-J_C^*\ell$
belongs to $F$; it does not disappear when tracking is small.

The **effective flow** is $\dot\theta=-F$. It exactly preserves the current
coarse output because $J_CF=0$, and it satisfies

$$
\dot e_H=-S(\theta)e_H,\qquad
S=J_H\Pi J_H^*\succeq0,\qquad
\frac{d}{dt}\frac{Y^2}{2}=-\|F\|^2.
\tag{2}
$$

Everything in $S$ evolves. Equation (2) is not a frozen-feature model.
It says that effective training dissipates fine error through the sensitivity
currently supplied by the population.

Equivalently, the empirical population
$\rho_t=W^{-1}\sum_j\delta_{X_j(t)}$ obeys the transport equation
$\partial_t\rho+\nabla_X\cdot(\rho V[\rho,d])=0$ in its weak sense,
coupled to the output-bias ODE. Its velocity includes the current residual
and coarse compensation. For a smooth test function $\psi$,

$$
\frac{d}{dt}\int\psi(X)\,d\rho_t(X)
=\int\nabla\psi(X)\cdot V[\rho_t,d_t](X)\,d\rho_t(X).
$$

The moment comparisons below are bounds on this transport dynamics. They
require neither an infinite-width limit nor a separate forecast for each
particle. Coupling is retained through the evolving vector field.

## 2. Why population structure directly limits output accuracy

Consider a target with a substantial non-affine component. If the collective
nonlinear output of the features is too small, inaccurate output follows
regardless of the optimizer that produced those parameters. This is a state
constraint, distinct from a claim about how long the state persists.

**Proposition 1: instantaneous output bound.** Define

$$
\mathcal Q=\sum_j|c_j|a_j^2\left(|b_j|+\frac{|a_j|}{3}\right).
$$

For every parameter state,

$$
\boxed{\quad
\|f-y\|\ge\|e_H\|
\ge\left[\|P_Hy\|-\mathcal Q\right]_+.
\quad}
\tag{3}
$$

**Proof.** For $\phi=\tanh$, $|\phi''(u)|\le2|u|$ for all real $u$.
Taylor's integral identity about the current bias gives

$$
\phi(b+ax)-\phi(b)-ax\phi'(b)
=(ax)^2\int_0^1(1-s)\phi''(b+sax)\,ds.
$$

Its absolute value is at most $a^2(|b|+|a|/3)$ on $[-1,1]$.
The first two terms are affine in $x$, so projection removes them.
Contractivity of $P_H$, the triangle inequality, and the reverse triangle
inequality give $\|P_Hf\|\le\mathcal Q$ and (3). This uses an exact
remainder bound, not an approximate training model. $\square$

Readouts matter in (3). Small slopes alone do not imply inaccurate output:
large readouts or a concentrated exceptional group can supply nonlinear
output. Conversely, the bound can be loose because it discards cancellations.
Its usefulness must be measured, not inferred from the theorem alone.

The same argument separates an already-correct lower-order output from a
still-missing target tail. Let $P_{\ge k}$ project away from polynomials of
degree less than $k$, and choose a global derivative bound
$C_k\ge\sup_{u\in\mathbb R}|\tanh^{(k)}u|$. Then

$$
\|f-y\|\ge
\left[\|P_{\ge k}y\|
-\frac{C_k}{k!}\sum_j|c_j||a_j|^k\right]_+.
$$

To prove this, subtract the degree-$(k-1)$ Taylor polynomial about each
current bias; the exact remainder is at most $C_k|a_j|^k/k!$ on the
input interval. The target need not be a polynomial. For sine and other
non-polynomial targets, $\|P_{\ge k}y\|$ measures how much output is
missing beyond the chosen lower-order space. Evaluating several $k$ values
reveals whether the difficulty lies in generating any nonlinear output or
in generating its remaining higher-order portion.

## 3. Weak nonlinear sensitivity gives slow error reduction

The next estimate explains the width dependence. After affine output is
removed, derivatives of a broad tanh feature start at cubic order in its
slope, bias, and readout. Summing their squares produces a sixth population
moment.

**Lemma 2: global sensitivity estimate.** For all parameter states,

$$
\|J_H\|^2\le9B^6=\frac{9M_6}{W^2},\qquad
\|F\|\le3YB^3.
\tag{4}
$$

**Proof.** Write $u_j=a_jx+b_j$, $r_j=|p_j|$. The output-Jacobian
columns for one neuron are
$c_jx\operatorname{sech}^2u_j$, $c_j\operatorname{sech}^2u_j$, and
$\tanh u_j$. Subtract the affine columns $c_jx,c_j,u_j$.
The global inequalities

$$
|1-\operatorname{sech}^2u|\le u^2,
\qquad |\tanh u-u|\le|u|^3/3,
\qquad |u_j|\le\sqrt2r_j
$$

bound the sum of squared column remainders by
$c_j^2(1+x^2)u_j^4+u_j^6/9\le(80/9)r_j^6<9r_j^6$.
The output-bias column is removed exactly. Summing and using projection
contractivity bounds the Hilbert–Schmidt norm, hence the operator norm.
Finally $\Pi$ is an orthogonal projector. $\square$

Along the effective flow, (2) and (4) imply

$$
\boxed{\quad
Y(t)\ge Y_0\exp\left[-9\int_0^tB(s)^6\,ds\right].
\quad}
\tag{5}
$$

Indeed $\dot Y=-\|F\|^2/Y\ge-9B^6Y$ whenever $Y>0$;
comparison gives the claim. This is already an exact evolving-feature
error bound. However, inserting a measured future integral makes it a
retrospective bound. The next theorem supplies that integral from initial
data alone.

## 4. A closed collective persistence theorem

The key observation is elementary but consequential. The population sixth
norm cannot grow faster than the total parameter speed. That speed is itself
cubic in the same sixth norm. This closes a differential inequality without
assuming that concentration stays bounded.

**Theorem 3: sixth-moment persistence from initial data.** Suppose
$B_0>0$, $Y_0>0$. Choose $q>1$ and define

$$
k=6Y_0B_0^2,\qquad T_q=\frac{1-q^{-2}}{k},\qquad
A_q=(q-1)B_0.
$$

Assume the following explicit initial-data test is positive:

$$
\delta_q=
\sqrt{\frac{v(\sqrt{E_{s,0}}-A_q)_+^2}
{1+2(\sqrt{M_0}+A_q)^2}}
-3q^3B_0^3>0.
\tag{6}
$$

Alternatively, one may replace $\delta_q$ by the larger of the expression
in (6) and

$$
\sigma_{\min}(J_C(\theta_0))
-\left[\sqrt2+4(\sqrt{M_0}+A_q)\right]A_q.
$$

The latter uses the initial coarse singular value directly. Both are
collective rank bounds; an audit should say which one it evaluates.

Then the exact effective flow is defined through $T_q$, its coarse Gram
matrix satisfies $K(t)\succeq\delta_q^2I$, and for $0\le t\le T_q$,

$$
\begin{aligned}
B(t)&\le b(t):=\frac{B_0}{\sqrt{1-kt}},\\
\int_0^t\|\dot\theta(s)\|\,ds
&\le A(t):=B_0\left[(1-kt)^{-1/2}-1\right],\\
Y(t)&\ge Y_0\exp\left\{
-\frac{9B_0^6}{2k}\left[(1-kt)^{-2}-1\right]\right\}.
\end{aligned}
\tag{7}
$$

In particular,

$$
Y(T_q)\ge Y_0
\exp\left[-\frac{3B_0^4}{4Y_0}(q^4-1)\right].
\tag{8}
$$

No premise in this theorem concerns a future moment, future force, or future
tracking history. It is a theorem about the effective-flow ODE itself.

**Proof: close the moment inequality.** On any interval where $K$ is
invertible, $Y\le Y_0$ by (2). The derivative of a norm is bounded by
the norm of its velocity, so

$$
D^+B\le\left(\sum_j|\dot p_j|^6\right)^{1/6}
\le\left(\sum_j|\dot p_j|^2\right)^{1/2}
\le\|F\|\le3Y_0B^3.
\tag{9}
$$

Here $D^+$ denotes the upper right derivative; this also handles zero
particle coordinates. Scalar comparison yields $B\le b$.
Integrating $\|F\|\le3Y_0b^3=b'$ gives the path-length bound in (7).
Substitution in (5) gives the last line of (7).

**Proof: preserve conditioning and continue the flow.** For the affine
network $d+\sum_jc_j(a_jx+b_j)$, let its coarse Jacobian be $J_{C,0}$.
Writing $A=\sum a_j^2$, $Z=\sum c_j^2$, $B_b=\sum b_j^2$, and
$D=\sum a_jb_j$, its Gram matrix is

$$
J_{C,0}J_{C,0}^*=
\begin{pmatrix}1+B_b+Z&\sqrt vD\\\sqrt vD&v(A+Z)\end{pmatrix}.
$$

Since $D^2\le AB_b$, its determinant is at least $vE_s$ and its
trace is at most $1+2M$. Thus
$\sigma_{\min}(J_{C,0})\ge\sqrt{vE_s/(1+2M)}$.
The same column-remainder calculation as in Lemma 2 gives
$\|J_C-J_{C,0}\|\le3B^3$. Therefore

$$
\sigma_{\min}(J_C)\ge\sqrt{\frac{vE_s}{1+2M}}-3B^3.
\tag{10}
$$

Travel at most $A(t)$ implies
$\sqrt M\le\sqrt{M_0}+A(t)$ and
$\sqrt{E_s}\ge(\sqrt{E_{s,0}}-A(t))_+$ by the triangle inequality.
Consequently (6) bounds the right side of (10) below by $\delta_q$.
A second valid estimate follows from
$\|DJ_C\|\le\sqrt2+4\sqrt M$ and the same path-length bound, proving
the alternative stated after (6).
A first-exit argument prevents loss of rank before $T_q$. All parameters,
including $d$, remain bounded by the total path length; the smooth vector
field on this rank-preserving region therefore continues through $T_q$.
This completes the proof. $\square$

If $Y_0=0$, the effective flow is stationary wherever it is defined.
If $B_0=0$, all particle parameters vanish and the two-dimensional coarse
solve is singular; this degenerate state is outside the theorem.

**What width scaling this actually proves.** If initial $M_6$ and $M$
are uniformly bounded, $E_s$ is bounded below, and $Y_0$ stays between
positive constants, then fixed $q>1$ satisfies (6) at all sufficiently large
widths. In that case

$$
T_q=\Theta(W^{2/3}),\qquad A_q=O(W^{-1/3}),\qquad
Y(T_q)/Y_0\ge\exp[-O(W^{-4/3})].
\tag{11}
$$

Time is physical gradient-flow time; comparison with step size $\eta$
uses $t=n\eta$ only after accounting for discretization. The result allows
concentration to change and individual particles to behave differently.
It establishes slow population evolution, not contraction or attraction.

There is also a direct capacity interpretation of the same envelope:

$$
\|P_Hf(t)\|
\le\frac{2\sqrt2}{3}\sum_j|p_j(t)|^4
\le\frac{2\sqrt2}{3}W^{1/3}B(t)^4
\le\frac{2\sqrt2q^4}{3W}M_6(0)^{2/3}.
$$

The first inequality subtracts $d+\sum c_j(a_jx+b_j)$ and bounds the
exact tanh remainder; the second is a finite-sum norm inequality. Therefore
through the certified interval,

$$
\|f(t)-y\|\ge
\left[\|P_Hy\|-\frac{2\sqrt2q^4}{3W}M_6(0)^{2/3}\right]_+.
$$

This conclusion does not require accuracy to imply any particular slope
threshold. It directly says that the persisting population cannot yet
produce enough non-affine output. For relative $L^2$ error, divide each
output-error floor in this note by $\|y\|$, assumed nonzero.

This capacity conclusion also transfers to a different evaluation measure
supported on $[-1,1]$: use that measure's affine projection and target norm
on the right side. The population envelope is established by training
dynamics, while the tanh remainder bound holds pointwise across the whole
interval. In contrast, the residual-dissipation formula (2) and its progress
bound concern the measure used to train; they do not automatically describe
an independent evaluation loss.

## 5. Higher finite moments give a longer collective window

The sixth-norm argument uses the total Euclidean speed to bound population
growth. A finite higher moment can retain more of the distribution's shape
without controlling every neuron. The next theorem makes this tradeoff
explicit; taking arbitrarily high moments is not the proposed empirical
strategy.

**Theorem 4: persistence from an initial $p$th moment.** Assume $Y_0>0$,
fix a finite $p\ge6$, and put

$$
L_p=\left(\sum_j|p_j|^p\right)^{1/p},\qquad
D_p=W^{1/2-3/p},\qquad l_0=L_p(0)>0.
$$

Choose $q>1$ and a coarse singular-value margin $\sigma_*>0$. Define

$$
\begin{aligned}
c_*&=1+\frac{\sqrt2D_pql_0}{\sigma_*},&
k_p&=6Y_0c_*l_0^2,\\
T_p&=\frac{1-q^{-2}}{k_p},&
A_p&=\frac{D_p}{c_*}(q-1)l_0.
\end{aligned}
$$

Assume

$$
\sqrt{\frac{v(\sqrt{E_{s,0}}-A_p)_+^2}
{1+2(\sqrt{M_0}+A_p)^2}}
-3D_pq^3l_0^3>\sigma_*.
\tag{12}
$$

As in Theorem 3, the left side may be replaced by its maximum with
$\sigma_{\min}(J_C(\theta_0))-[\sqrt2+4(\sqrt{M_0}+A_p)]A_p$.

Then through $T_p$ the effective flow has
$\sigma_{\min}(J_C)>\sigma_*$, and

$$
\begin{aligned}
L_p(t)&\le\frac{l_0}{\sqrt{1-k_pt}},\\
\int_0^t\|F\|\,ds
&\le\frac{D_pl_0}{c_*}\left[(1-k_pt)^{-1/2}-1\right],\\
Y(t)&\ge Y_0\exp\left\{
-\frac{9D_p^2l_0^6}{2k_p}
\left[(1-k_pt)^{-2}-1\right]\right\}.
\end{aligned}
\tag{13}
$$

**Proof.** Norm comparison gives $B^3\le D_pL_p^3$. The
uncompensated fine-gradient block obeys
$|(g_H)_j|\le3Yr_j^3$ by the proof of Lemma 2.
The coarse-compensation block obeys
$|(J_C^*\ell)_j|\le\sqrt2r_j|\ell|$: its three Jacobian columns have
joint squared norm at most $2r_j^2$. Moreover,
$|\ell|\le\|g_H\|/\sigma_*\le3YD_pL_p^3/\sigma_*$ while the rank
margin holds. Differentiating the summed $p$th moment and using
$\sum r_j^{p+2}\le L_p^{p+2}$ gives

$$
D^+L_p\le3YL_p^3+\sqrt2|\ell|L_p
\le3Y_0L_p^3\left(1+\frac{\sqrt2D_pL_p}{\sigma_*}\right).
$$

Until $L_p$ reaches $ql_0$, scalar comparison bounds this by
$3Y_0c_*L_p^3$. Integrating
$\|F\|\le3Y_0D_pL_p^3$ gives the travel bound in (13), and (5)
gives the error bound. Equation (10) with $B^3\le D_pL_p^3$ and
(12) preserves the rank margin. The resulting simultaneous first-exit
argument closes both premises. $\square$

For bounded initial $M_p$, $l_0=M_p^{1/p}W^{1/p-1/2}$.
If the other initial quantities admit the same uniform bounds as before,
one can choose fixed $q,\sigma_*$ and obtain

$$
T_p=\Theta(W^{1-2/p}),\qquad
A_p=O(W^{-2/p}),\qquad
Y(T_p)/Y_0\ge\exp[-O(W^{-1-2/p})].
\tag{14}
$$

For example $p=12$ gives a $W^{5/6}$ window. The theorem uses only
an initial twelfth moment and aggregate conditioning. No individual support
constraint is part of its statement or continuation argument. Larger
moments can nevertheless be dominated by rare particles, so a better
asymptotic exponent does not guarantee a better numerical certificate.

For any $k+1\le p$, the same envelope also gives

$$
\sum_j|c_j(t)||a_j(t)|^k
\le W^{1-(k+1)/p}L_p(t)^{k+1}
\le q^{k+1}M_p(0)^{(k+1)/p}W^{-(k-1)/2}.
$$

Substitution in the target-tail bound from Section 2 yields an explicit
output-error lower bound throughout the interval. This connects a collective
dynamical theorem to target-dependent accuracy requirements without
restricting the theory to one target family.

## 6. What full gradient flow and ordinary GD add

The effective-flow theorem is a mechanism result, not permission to discard
tracking whenever convenient. Under the full gradient flow,

$$
\begin{aligned}
\dot\theta&=-F-R,\\
\dot e_H&=-Se_H-J_HR,\\
\dot z&=-Kz+\dot\ell,\\
\frac{d}{dt}\frac{Y^2}{2}
&=-\|F\|^2-\langle e_H,J_HR\rangle.
\end{aligned}
\tag{15}
$$

The forcing $\dot\ell$ in the third identity is the total derivative
along the evolving full flow. Freezing it would change the problem.
Let $\omega=\|J_HR\|$ and $H(t)=9\int_0^tB^6ds$. Then

$$
Y(t)\ge
e^{-H(t)}\left[Y_0-\int_0^te^{H(s)}\omega(s)\,ds\right]_+,
\qquad
D^+B\le3YB^3+\|R\|.
\tag{16}
$$

These distinguish two disturbances: $\|R\|$ affects the population
envelope, whereas $\|J_HR\|$ affects output progress directly. Integrating
either along an archived trajectory is an audit, not an initial-data proof
that its future budget stays small. A full-flow theorem must close these
budgets through the tracking ODE or an independently justified bound.

For ordinary GD with $g_n=F_n+R_n$,
$\theta_{n+1}=\theta_n-\eta g_n$, the exact output relation is

$$
e_{H,n+1}=(I-\eta S_n)e_{H,n}-\eta J_{H,n}R_n+r_n,
\tag{17}
$$

where $r_n$ is the nonlinear one-step output remainder. If
$9\eta B_n^6\le1$, it follows that

$$
Y_{n+1}\ge(1-9\eta B_n^6)Y_n
-\eta\|J_{H,n}R_n\|-\|r_n\|,
\qquad
B_{n+1}\le B_n+\eta(3Y_nB_n^3+\|R_n\|).
\tag{18}
$$

An explicit global bound that retains weak nonlinear sensitivity is

$$
\|r_n\|\le3\sqrt2\eta^2
\left[B_n+\eta\|(g_n)_{abc}\|\right]^2\|g_n\|^2.
\tag{19}
$$

To check (19), subtract the affine network before differentiating twice.
The remaining Hessian block for neuron $j$ has geometry–geometry norm at
most $4\sqrt2r_j^2$ and geometry–readout cross-block norm at most
$2\sqrt2r_j^2$. Consequently $\|D^2e_H\|\le6\sqrt2B^2$.
Taylor's integral formula and $B$'s norm inequality along the GD segment
give (19). No individual-particle hypothesis is used in this estimate.

Neither an Euler discretization nor the full gradient is silently identified
with the effective ODE here. The following certificate controls both from
initial data, although its numerical usefulness remains a separate question.

### A closed comparison system including tracking

**Proposition 5: full-flow and ordinary-GD certificate.** Choose $q>1$,
set $B_*=qB_0$, $A_*=(q-1)B_0$, and suppose $\sigma=\delta_q>0$
from (6), with its stated alternative rank bound allowed. Let

$$
\begin{aligned}
\overline Y&=\sqrt{2L(\theta_0)},&
J_*&=\sqrt{1+2(\sqrt{M_0}+A_*)^2},\\
H_C&=\sqrt2+4(\sqrt{M_0}+A_*),&
u_*&=3\overline YB_*^3.
\end{aligned}
$$

For a residual bound $Y_*$ define

$$
D_\ell(Y_*)=
\frac{6H_CY_*B_*^3}{\sigma^2}
+\frac{9B_*^6+6\sqrt2Y_*B_*^2}{\sigma}.
\tag{20}
$$

For full gradient flow, solve the scalar linear system

$$
\dot Z=-(\sigma^2-D_\ell(\overline Y)J_*)Z
+D_\ell(\overline Y)u_*,\qquad
\dot A=u_*+J_*Z,
\quad Z(0)=\|z_0\|,\ A(0)=0.
\tag{21}
$$

Every interval with $A(t)<A_*$ is certified: the full flow exists there,
$B\le B_*$, $K\succeq\sigma^2I$, $\|z\|\le Z$, and total parameter
travel is at most $A$. Equation (16) gives an output-error floor with
$H(t)=9B_*^6t$ and $\omega(t)=3B_*^3J_*Z(t)$.

For ordinary GD, set $Y_{\rm seg}=\overline Y+J_*A_*$ and require

$$
\eta\left[J_*^2+Y_{\rm seg}H_C\right]\le1.
\tag{22}
$$

Starting at $Z_0=\|z_0\|$, $A_0=0$, iterate

$$
\begin{aligned}
v_n&=u_*+J_*Z_n,\\
A_{n+1}&=A_n+\eta v_n,\\
Z_{n+1}&=(1-\eta\sigma^2)Z_n
+\eta D_\ell(Y_{\rm seg})v_n
+\tfrac12H_C\eta^2v_n^2.
\end{aligned}
\tag{23}
$$

Accept each update only when $A_{n+1}<A_*$. For all accepted updates,
the actual total loss is nonincreasing, total travel is at most $A_n$,
$B_n\le B_*$, $K_n\succeq\sigma^2I$, and $\|z_n\|\le Z_n$.
If also $9\eta B_*^6\le1$, an output lower bound follows from
$y_0=Y_0$ and

$$
y_{n+1}=\left[
(1-9\eta B_*^6)y_n
-3\eta B_*^3J_*Z_n
-3\sqrt2\eta^2B_*^2v_n^2
\right]_+,
\qquad Y_n\ge y_n.
\tag{24}
$$

**Proof.** Within total travel $A_*$, the triangle inequality gives
$B\le B_*$, $\sqrt M\le\sqrt{M_0}+A_*$, and the conditioning estimate
in Theorem 3 gives $\sigma_{\min}(J_C)\ge\sigma$. Direct differentiation gives
$\|D_\theta f\|\le J_*$ and $\|D^2_\theta f\|\le H_C$ there.
The latter follows from the full neuron Hessian bounds
$4|c_j|+\sqrt2$ and block diagonality in the parameters.

For a parameter direction $w$, differentiating
$\ell=(J_CJ_C^*)^{-1}J_Cg_H$ gives the useful identity

$$
D\ell[w]=K^{-1}(DJ_C[w])F
+K^{-1}J_CDg_H[w]
-K^{-1}J_C(DJ_C[w])^*\ell.
$$

Use $\|F\|\le3Y_*B_*^3$,
$\|\ell\|\le3Y_*B_*^3/\sigma$, and
$\|Dg_H\|\le9B_*^6+6\sqrt2Y_*B_*^2$ to obtain
$\|D\ell\|\le D_\ell(Y_*)$. The total flow dissipates $L$, so
$\|e_H\|\le\overline Y$. Equations (15), the coercivity of $K$, and
$\|R\|\le J_*\|z\|$ imply (21) by scalar comparison. The same
first-exit argument used earlier closes the travel premise.

For GD, consider a proposed step with $A_{n+1}<A_*$. Its entire line
segment stays in the initial parameter ball of radius $A_*$, where the
total residual norm is at most $Y_{\rm seg}$. The loss Hessian there is
bounded by $J_*^2+Y_{\rm seg}H_C$, so (22) implies descent and preserves
the nodal bound $\|e_H\|\le\overline Y$. Taylor expansion of the coarse
output gives

$$
z_{n+1}=(I-\eta K_n)z_n
+(\ell_{n+1}-\ell_n)+r_{C,n},\qquad
\|r_{C,n}\|\le\tfrac12H_C\eta^2\|g_n\|^2.
$$

Condition (22) implies $\eta\|K_n\|\le1$, hence
$\|I-\eta K_n\|\le1-\eta\sigma^2$.
The bound on $D\ell$ along the segment gives (23).
Finally (17), $\|J_HR\|\le3B_*^3J_*Z_n$, and
$\|D^2e_H\|\le6\sqrt2B_*^2$ along the accepted segment give (24).
This proves the induction. $\square$

The coefficient $\sigma^2-D_\ell(\overline Y)J_*$ in (21) describes
an actual stability balance: coarse dissipation competes with the rate at
which the evolving population changes its compensating response. Positivity
implies decay toward a forced tracking allowance; negativity does not
invalidate the comparison, but may make its certified interval short.
This is one explicit way to derive small tracking rather than assume it.

**Corollary 6: an ordinary-GD population window from initial data.** Consider a
sequence of widths for which $v$ and initial $E_s$ are bounded below by
positive constants, initial $M_6$ and total loss are bounded above, and

$$
\|z_0\|=o(W^{-1/3}).
$$

Then there exist width-independent constants $c>0$ and $\eta_*>0$ such
that, for all sufficiently large $W$ and constant GD steps
$0<\eta\le\eta_*$, the certificate in Proposition 5 accepts every update
with $n\eta\le cW^{2/3}$. Throughout that interval, $M_6(n)$ remains
bounded uniformly in width, total particle travel is $O(W^{-1/3})$, and

$$
\|f_n-y\|\ge[\|P_Hy\|-O(W^{-1})]_+.
$$

If also $Y_0$ is bounded below, the progress estimate sharpens to
$Y_n\ge Y_0-O(W^{-4/3})$. Constants may depend on the uniform initial
bounds. This is an asymptotic theorem, not a claim that its unspecified
constants certify the experimental widths; those require (23).

In plain language: start after coarse tracking has become sufficiently small,
with a broad population whose sixth rescaled moment is bounded and whose
coarse response is nondegenerate. Ordinary GD then preserves a broad
population for a width-growing duration. A target with substantial
non-affine output consequently retains substantial error during that
duration. The statement permits individual neurons to move differently; it
does not assert that slopes shrink or that no neuron escapes.

**Proof.** Fix $q>1$. The initial moment bounds give
$B_0=\Theta(W^{-1/3})$, $A_*=\Theta(W^{-1/3})$,
$\sigma\ge\sigma_0>0$, and bounded $J_*,H_C,Y_{\rm seg}$ for large
$W$. Consequently $u_*=O(W^{-1})$ and
$D_\ell(Y_{\rm seg})=O(W^{-2/3})$. Choose $\eta_*$ small enough for
(22) and $\eta_*\sigma^2\le1$ uniformly.

In a region $Z\le\varepsilon W^{-1/3}$, the terms of (23) linear or
quadratic in $Z$ can, for large $W$, be absorbed into half the contraction
$\eta\sigma^2Z$. Thus for constants independent of width,

$$
Z_{n+1}\le(1-\eta\sigma_0^2/2)Z_n
+C\eta W^{-5/3}+C\eta^2W^{-2}.
$$

This region is preserved from $Z_0=o(W^{-1/3})$. Summing the geometric
bound through physical time $T=n\eta$ gives

$$
\eta\sum_{k<n}Z_k
\le C Z_0+CTW^{-5/3}+C\eta TW^{-2}.
$$

The travel recurrence is therefore at most
$C TW^{-1}+o(W^{-1/3})$ for $T=O(W^{2/3})$.
Choosing $c$ sufficiently small keeps it strictly below $A_*$, closing
the induction and proving the moment and capacity statements. In (24),
the accumulated effective contraction is $O(TW^{-2})=O(W^{-4/3})$;
the tracking allowance is $O(W^{-1})\eta\sum Z_k=o(W^{-4/3})$.
Finally $\eta\sum Z_k^2=o(W^{-2/3})+O(TW^{-10/3})$, so the
accumulated remainder is at most
$C\eta W^{-2/3}[TW^{-2}+\eta\sum Z_k^2]=o(W^{-4/3})$.
This proves the progress statement. $\square$

The initial tracking condition is a post-transient premise. It neither
assumes that future tracking stays small nor controls the early training
episode that produces the checkpoint. An ordinary-GD theorem on that
early episode is a different question.

Adam requires its own update identity with the actual momentum and adaptive
denominator. Proposition 1 remains valid for Adam states; the gradient-flow
and ordinary-GD persistence certificates do not cover Adam's evolution.

## 7. The remaining mechanistic refinement

The new initial-data theorems establish a baseline: broad populations have
limited capacity to increase their own nonlinear sensitivity rapidly. They
permit outward motion and do not rely on a restoring force. Their weakest
step is an absolute norm inequality, which ignores the direction of learning.

One refinement is already provable without a signed-drift hypothesis. It
retains the actual initial nonlinear sensitivity instead of replacing it by
the worst-case moment bound.

**Proposition 7: propagating the actual initial sensitivity.** Assume
$Y_0>0$, choose $q>1$, and let $j_0=\|J_H(\theta_0)\|_{\rm HS}>0$,
where the square of the Hilbert–Schmidt norm is the sum of squared output
norms of all parameter-Jacobian columns. For effective flow, define the
scalar comparison system

$$
\dot b=Y_0s,\qquad
\dot s=6\sqrt2Y_0b^2s,\qquad
b(0)=B_0,\quad s(0)=j_0.
$$

Before loss of the certified coarse rank margin,

$$
B(t)\le b(t),\quad \|J_H(t)\|_{\rm HS}\le s(t),\quad
\int_0^t\|F\|\,ds\le b(t)-B_0.
$$

The scalar system has the relation

$$
s=j_0+2\sqrt2(b^3-B_0^3).
$$

Consequently, if the rank test (6), or its stated alternative, is positive,
the effective flow remains in this envelope through

$$
T_q^{\rm sens}=
\int_{B_0}^{qB_0}
\frac{du}{Y_0[j_0+2\sqrt2(u^3-B_0^3)]}.
$$

On reaching $b=qB_0$, the fine-error floor is

$$
Y(T_q^{\rm sens})\ge Y_0\exp\left\{-\frac1{Y_0}
\left[j_0(q-1)B_0+
2\sqrt2B_0^4\left(\frac{q^4-1}{4}-(q-1)\right)\right]\right\}.
$$

**Proof.** For the affine-subtracted neuron Hessian used in (19),
$\|H_j(x)\|\le6\sqrt2r_j^2$ pointwise. Because a Jacobian column
block depends only on its own particle,

$$
\|DJ_H[w]\|_{\rm HS}^2
\le72\sum_jr_j^4|w_j|^2
\le72B^4\|w\|^2.
$$

Projection of the output cannot increase this norm. Thus, with
$j=\|J_H\|_{\rm HS}$,
$D^+j\le6\sqrt2B^2\|F\|\le6\sqrt2Y_0B^2j$ and
$D^+B\le\|F\|\le Y_0j$. The right sides of the two comparison
equations are nondecreasing in both nonnegative variables, proving the
coupled comparison. Integrating speed gives travel at most $b-B_0$,
so the same initial-data rank argument closes. The error inequality
$\dot Y\ge-j^2Y$ and the substitution $dt=db/(Y_0s)$ give the
stated floor. $\square$

If $j_0=0$, $F(\theta_0)=0$ and the effective flow is stationary;
the comparison has $s=0$, $b=B_0$. This limiting case is not interpreted
by evaluating the singular integral above. The case $Y_0=0$ is also
stationary and is excluded from the integral formula for the same reason.

This proposition bounds how a weak sensitivity can **reinforce itself**.
The Jacobian is free to rotate and change its entries; it is not held close
to its initial matrix. If $j_0/B_0^3$ is especially small, the scalar
growth has a longer initial slow phase than the uniform sixth-moment
envelope. Target-specific alignment can still make the effective force much
smaller than the Hilbert–Schmidt sensitivity, so this refinement does not
exhaust the possible mechanisms.

The more specific hypothesis is that the actual vector field increases
concentration and readout-weighted nonlinear moments much less rapidly than
this worst-case bound allows. Let $C_6=M_6/M^3$ and
$X_j=\sqrt Wp_j$. Its exact signed drift is

$$
\frac{d}{dt}\log C_6
=6\left[
\frac{\mathbb E_W(|X|^4X\cdot\dot X)}{M_6}
-\frac{\mathbb E_W(X\cdot\dot X)}{M}
\right].
\tag{25}
$$

The first term asks whether particles carrying the sixth moment grow faster
than particles carrying the second moment. A small difference can preserve
the population's shape even while both groups expand. This is the relevant
stability question; it is not whether the network remains near its initial
Jacobian.

Generated-error correction, target loading, and coarse compensation should
be retained with their signs in the two population averages. A successful
refinement would bound their difference, or a better readout-weighted
quantity, by an initial-data comparison system. Merely observing that
$C_6$ stays small, or assuming that it will, is insufficient to prove that
refinement. Likewise a positive instantaneous drift does not falsify
persistence: its integrated size and dependence on the current state matter.

The empirical questions are consequently concrete. Do the unconditional
moment envelopes yield useful durations? Which signed terms make observed
growth slower than their envelopes? When geometry is increased by orders of
magnitude, does nonlinear output sensitivity begin to reinforce itself and
reduce output error, or does the population enter another slowly evolving
region? These tests discriminate mechanisms at the population level without
requiring a theory of every neuron's path.

## Appendix. Frozen-feature GD as an output-error lower bound

The frozen-feature question is simpler and should be stated directly in
terms of accuracy: **does the initial output error contain energy in
directions that readout GD cannot correct within the training budget?**
A large condition number alone does not answer this question.

Fix $m$ inputs and all hidden features
$\phi_j(x_i)=\tanh(a_jx_i+b_j)$. Train only the readouts and output bias,
collected in $w=(c_1,\ldots,c_W,d)$. Define the RMS-normalized matrix and
target

$$
A_{ij}=\frac{\phi_j(x_i)}{\sqrt m}\quad(1\le j\le W),\qquad
A_{i,W+1}=\frac1{\sqrt m},\qquad
\widetilde y_i=\frac{y(x_i)}{\sqrt m}.
$$

Then $r=Aw-\widetilde y$ has Euclidean norm equal to the raw output RMS
error, and $L(w)=\|r\|_2^2/2$ is the same mean-square loss convention
used above. The relative $L^2$ error is $\|r\|_2/\|\widetilde y\|_2$,
assuming $\widetilde y\ne0$.

**Proposition 8: residual-weighted frozen-feature barrier.** Put
$K_{\rm fr}=AA^*$; this output-space matrix is distinct from the
two-dimensional coarse Gram matrix $K$ in the main text. Let its
orthonormal eigenvectors be $u_i$ and eigenvalues be $\kappa_i\ge0$.
For constant-step readout GD with
$0<\eta\le1/\kappa_{\max}$,

$$
r_n=(I-\eta K_{\rm fr})^nr_0,\qquad
\boxed{\quad
\|r_n\|_2^2=\sum_{i=1}^m(1-\eta\kappa_i)^{2n}
|\langle u_i,r_0\rangle|^2.
\quad}
$$

For a chosen slow-band cutoff $0<s\le\kappa_{\max}$, define

$$
E_0=\|P_{\ker A^*}r_0\|_2^2
=\|P_{\ker A^*}\widetilde y\|_2^2,\qquad
E_{\rm slow}(s)=\sum_{0<\kappa_i\le s}|\langle u_i,r_0\rangle|^2.
$$

Every update $0\le n\le N$ then satisfies

$$
\frac{\|r_n\|_2}{\|\widetilde y\|_2}
\ge
\frac{\sqrt{E_0+(1-\eta s)^{2N}E_{\rm slow}(s)}}
{\|\widetilde y\|_2}.
$$

If this lower bound exceeds an accuracy requirement $\varepsilon$, no
checkpoint through update $N$ attains relative error at most
$\varepsilon$. Residual energy outside the feature range, $E_0$, never
decays; energy inside the range but in the slow band can require a long
budget even when exact fitting is possible.

**Proof.** The GD step is $w_{n+1}=w_n-\eta A^*r_n$, so
$r_{n+1}=(I-\eta AA^*)r_n$. Diagonalizing this fixed symmetric matrix
gives the exact identity. Its multipliers lie in $[0,1]$. On the slow
band each is at least $1-\eta s$, and its $n$th power is at least its
$N$th power when $n\le N$. Retain these terms and the nullspace terms
in the squared norm. Finally $Aw_0$ is orthogonal to $\ker A^*$,
which proves the formula for $E_0$. $\square$

The weights are initial **residual** energies. For zero initial readouts
and bias, they are exactly the target's modal energies; for a trained
restart, they measure what remains to be corrected. A poorly conditioned
matrix with negligible residual energy in its slow directions gives no
substantial barrier by this argument. The spectrum and residual energy
must therefore be measured together.

This proposition uses a fixed feature matrix and ordinary GD. Its modal
identity does not transfer to evolving features or to Adam's adaptive,
momentum-dependent updates. Sections 4–7 address the distinct question of
whether evolving population structure itself preserves slow learning.

**Proposition 8b: an arbitrary-witness frozen-GD certificate.** Keep the
RMS-normalized fixed matrix $A$ above, including the output-bias column,
and set $L_*=W+1$. For tanh features, $\|A\|^2\le\|A\|_F^2\le L_*$.
Suppose $0<\eta L_*<2$. For any nonzero output-space vector $v$, every
ordinary readout-GD checkpoint $0\le n\le N$ satisfies

$$
\|r_n\|_2\ge
\frac{\left[|\langle v,r_0\rangle|-
\|A^*v\|_2\|r_0\|_2
\sqrt{\frac{\eta N}{2-\eta L_*}}\right]_+}{\|v\|_2}.
$$

Divide by $\|\widetilde y\|_2>0$ to obtain a relative output-error floor.
If it exceeds the chosen tolerance, every checkpoint through $N$ fails
that tolerance. This bound permits any witness: an approximate SVD may
suggest $v$, but the certificate only needs verified direct norms and
inner products involving $v,r_0,A^*v$.

**Proof.** Since $r_{j+1}=r_j-\eta AA^*r_j$,

$$
\|r_j\|_2^2-\|r_{j+1}\|_2^2
\ge\eta(2-\eta L_*)\|A^*r_j\|_2^2.
$$

Sum over $j<n$ to bound
$\sum_{j<n}\|A^*r_j\|_2^2\le\|r_0\|_2^2/[\eta(2-\eta L_*)]$.
Telescoping the witness projection and applying Cauchy–Schwarz gives

$$
|\langle v,r_n-r_0\rangle|
\le\eta\|A^*v\|_2\sum_{j<n}\|A^*r_j\|_2
\le\|A^*v\|_2\|r_0\|_2
\sqrt{\frac{\eta n}{2-\eta L_*}}.
$$

The triangle inequality, $n\le N$, and
$|\langle v,r_n\rangle|\le\|v\|_2\|r_n\|_2$ prove the claim.
$\square$

This is a statement about ideal real-arithmetic GD on the empirical problem
defined by the encoded sample locations, targets, and frozen parameters.
Outward-rounded evaluation can certify its arithmetic inputs and bound;
it does not automatically certify the roundoff of a particular training
implementation, population error between samples, evolving geometry, or Adam.

## Appendix. How rapidly can the actual effective force reinforce itself?

The nonlinear sensitivity of a population need not be well aligned with
its remaining target error. Therefore even the measured Jacobian norm can
overestimate the force actually driving training. The following refinement
starts from the measured effective-force norm and bounds its subsequent
reinforcement through the evolving constrained-gradient ODE. It is an
effective-flow result; it does not by itself certify ordinary GD or Adam.

**Proposition 9: initial-force persistence.** Consider the exact effective
flow $\dot\theta=-F$, with $Y_0>0$ and
$f_0=\|F(\theta_0)\|>0$. Choose $q>1$ and let

$$
A_q=(q-1)B_0,\qquad
H_C=\sqrt2+4(\sqrt{M_0}+A_q).
$$

Choose a positive coarse singular-value margin $\sigma$ no larger than
the maximum of the two valid initial-data lower bounds from Theorem 3.
Define, for $b\ge B_0$,

$$
U(b)=f_0+2\sqrt2Y_0(b^3-B_0^3)
+\frac{3Y_0H_C}{4\sigma}(b^4-B_0^4),\qquad
T_q^F=\int_{B_0}^{qB_0}\frac{db}{U(b)}.
$$

Let $b(t)$ solve $\dot b=U(b)$, $b(0)=B_0$. Through $T_q^F$,
the effective flow remains defined and satisfies

$$
B(t)\le b(t),\qquad
\|F(t)\|\le U(b(t)),\qquad
\int_0^t\|F(s)\|\,ds\le b(t)-B_0,
\qquad \sigma_{\min}(J_C(t))\ge\sigma.
$$

Its fine-error floor is

$$
Y(t)\ge\left[Y_0^2-2I(b(t))\right]_+^{1/2},
$$

where the integrated force allowance is explicit:

$$
\begin{aligned}
I(b)={}&f_0(b-B_0)\\
&+2\sqrt2Y_0\left[\frac{b^4-B_0^4}{4}-B_0^3(b-B_0)\right]\\
&+\frac{3Y_0H_C}{4\sigma}
\left[\frac{b^5-B_0^5}{5}-B_0^4(b-B_0)\right].
\end{aligned}
$$

The output-capacity and population-crossing bounds already proved in the
note apply with the same travel allowance $b(t)-B_0$. If the energy floor
above becomes zero, it is simply uninformative; the capacity bound may
still be useful.

**Proof: retain the exact geometry and compensation feedback.** Let
$A=J_C$ temporarily, so $\Pi A^*=0$ and $\Pi^2=\Pi$.
Differentiating these identities in direction $F$ gives

$$
(D\Pi[F])A^*=-\Pi(DA[F])^*,\qquad
\Pi(D\Pi[F])\Pi=0.
$$

Since $g_H=F+A^*\ell$, differentiating $F=\Pi g_H$ along
$\dot\theta=-F$ yields the exact identity

$$
\boxed{\quad
\frac12\frac{d}{dt}\|F\|^2
=-\|J_HF\|^2
-\langle e_H,D^2e_H[F,F]\rangle
+\ell\cdot D^2e_C[F,F].
\quad}
$$

The first term is residual relaxation. The second and third are the
changing nonlinear geometry and coarse-compensation contributions; their
signs are not fixed. No projector-derivative term has been omitted.

Inside the proposed travel region, $Y\le Y_0$,
$\|D^2e_H\|\le6\sqrt2B^2$,
$\|D^2e_C\|\le H_C$, and
$|\ell|\le3Y_0B^3/\sigma$. Dropping the nonpositive first term and
bounding the other two gives, for $f=\|F\|>0$,

$$
\dot f\le Y_0\left(6\sqrt2B^2+
\frac{3H_C}{\sigma}B^3\right)f,
\qquad D^+B\le f.
$$

The right sides are nondecreasing in the nonnegative comparison variables.
Consequently the scalar system

$$
\dot b=u,\qquad
\dot u=Y_0\left(6\sqrt2b^2+
\frac{3H_C}{\sigma}b^3\right)u,
\quad (b(0),u(0))=(B_0,f_0)
$$

bounds $(B,f)$ from above while the coarse margin holds. Dividing the
second equation by the first and integrating gives $u=U(b)$.
Its integrated speed is $b-B_0$, so the same first-exit argument as in
Theorem 3 preserves the coarse margin through $b=qB_0$. This closes the
comparison without a future-force assumption. Finally (2) gives

$$
Y(t)^2=Y_0^2-2\int_0^t f(s)^2\,ds
\ge Y_0^2-2\int_{B_0}^{b(t)}U(u)\,du,
$$

which is the claimed formula. $\square$

If $f_0=0$ and the initial coarse projector is defined, the initial state
is an equilibrium of the effective-flow ODE and remains stationary. This
includes $Y_0=0$. The positive-force integral is not used in that case.
The proposition does not claim that the observed stalled states have
exactly zero force or form an attracting equilibrium.

When $f_0$ is much smaller than $Y_0B_0^3$, the comparison begins with
that smaller speed. It then allows the current force to grow at
a state-dependent rate. A longer interval, if obtained numerically, would
therefore describe suppressed reinforcement in the evolving ODE, not
continued agreement with a frozen-force forecast. Whether these absolute
curvature bounds retain the empirically observed slowness for a useful
duration remains a numerical question.

## Appendix. A conditional population theorem weighted by the actual force

The previous proofs bound curvature in every possible parameter direction.
They can therefore lose information about which part of the population is
actually moving. A more targeted structural quantity weights particle size
by its share of the current effective-force energy:

$$
f=\|F\|,\qquad
\Omega_F=
\frac{W\sum_j|p_j|^2|F_j|^2}{f^2}
=\frac{\sum_j|X_j|^2|F_j|^2}{f^2},
\qquad M_4=\mathbb E_W|X|^4=W\sum_j|p_j|^4.
$$

Here $F_j=(F_{a,j},F_{b,j},F_{c,j})$ contains the three particle
coordinates; the denominator includes the output-bias force as well.
$\Omega_F$ is unchanged if all force coordinates are multiplied by the
same nonzero scalar. Thus bounding it constrains **where the force acts**,
not how small its amplitude must be. Large particles are allowed if they
carry sufficiently little of the force energy.

The following statement is conditional in a new, explicit way. Unlike
Theorems 3–4 and Proposition 9, it does not establish all its structural
conditions from initial data. It assumes that $\Omega_F$ stays bounded,
and derives the force rate, a fourth-moment envelope, and output persistence.
Proving preservation of that force-weighted condition remains open.

**Proposition 10: slow reinforcement under a force-weighted population
condition.** Consider noiseless exact effective flow with
$Y_0>0$, $f_0=\|F(\theta_0)\|>0$, and
$\sigma_0=\sigma_{\min}(J_C(\theta_0))>0$.
Choose a travel allowance $A_*>0$ such that

$$
\sigma_*=
\sigma_0-(\sqrt2+4\sqrt{M_0})A_*-2A_*^2>0.
$$

Assume $\Omega_F(t)\le\Omega_*$, with $\Omega_*>0$, along this
effective trajectory until its first exit from total parameter travel
$A_*$. Set

$$
M_{4,*}=\left(\sqrt{M_4(0)}+2\sqrt{\Omega_*}A_*\right)^2,
$$

$$
\beta=Y_0\left[
6\sqrt2\Omega_*+
\frac{3\sqrt2M_{4,*}}{\sigma_*^2}
\left(\sqrt2+4\sqrt{\frac{\Omega_*}{W}}\right)
\right],\qquad
T_* = \frac{W}{\beta}
\log\left(1+\frac{\beta A_*}{Wf_0}\right).
$$

Through $T_*$, the coarse projector remains defined and

$$
\begin{aligned}
\|F(t)\|&\le f_0e^{\beta t/W},\\
\int_0^t\|F(s)\|\,ds&\le
A(t):=\frac{Wf_0}{\beta}(e^{\beta t/W}-1),\\
M_4(t)&\le
\left(\sqrt{M_4(0)}+2\sqrt{\Omega_*}A(t)\right)^2,\\
\|f_{\theta(t)}-y\|&\ge
\left[\|P_Hy\|-\frac{2\sqrt2}{3W}M_{4,*}\right]_+.
\end{aligned}
$$

The first $f$ in the proposition denotes force norm; the subscripted
$f_\theta$ in the last line denotes network output. The fine-error progress
bound is also explicit:

$$
Y(t)^2\ge
\left[Y_0^2-
\frac{Wf_0^2}{\beta}(e^{2\beta t/W}-1)\right]_+.
$$

In particular, suppose initial $M_0,M_4(0),Y_0$ and $Wf_0$ are bounded
above uniformly in width, $\sigma_0$ is bounded below, and the structural
allowance $\Omega_*$ is width-independent. A fixed sufficiently small
$A_*$ then gives $T_*\ge cW$ for a width-independent $c>0$.
Bounded initial $M_6$ is one sufficient way to obtain $f_0=O(W^{-1})$
from Lemma 2, but no bound on **future** $M_6$ is required here.
For a target with substantial non-affine output, the retained error is
therefore substantial on this conditional $O(W)$ interval.

**Proof: retain force-weighted curvature.** The neuronwise Hessian estimate
used earlier gives a sharper bound when applied to the current force:

$$
\|D^2e_H[F,F]\|
\le6\sqrt2\sum_j|p_j|^2|F_j|^2
=\frac{6\sqrt2}{W}\Omega_F f^2.
$$

The coarse Hessian satisfies a corresponding directional estimate. Its
particle block is bounded by $\sqrt2+4|c_j|$, so Cauchy–Schwarz gives

$$
\|D^2e_C[F,F]\|
\le\sqrt2\sum_j|F_j|^2+4\sum_j|c_j||F_j|^2
\le\left(\sqrt2+4\sqrt{\frac{\Omega_F}{W}}\right)f^2.
$$

To control compensation, one need not introduce a future sixth moment.
The coarse Jacobian block for particle $j$ has norm at most
$\sqrt2|p_j|$, while its uncompensated fine-gradient block is at most
$3Y|p_j|^3$. The output-bias component of $g_H$ vanishes exactly. Hence

$$
|\ell|\le\frac{\|J_Cg_H\|}{\sigma_*^2}
\le\frac{3\sqrt2Y}{\sigma_*^2}\sum_j|p_j|^4
=\frac{3\sqrt2Y M_4}{\sigma_*^2W}.
$$

Substitute these estimates into the exact norm identity in Proposition 9
and discard the nonpositive residual-relaxation term. While the proposed
region holds,

$$
\frac{d}{dt}\log f
\le\frac{Y_0}{W}\left[
6\sqrt2\Omega_F+
\frac{3\sqrt2M_4}{\sigma_*^2}
\left(\sqrt2+4\sqrt{\frac{\Omega_F}{W}}\right)
\right].
$$

**Proof: close the population and rank bounds.** Since $\dot p_j=-F_j$,

$$
\left|\dot M_4\right|
\le4W\sum_j|p_j|^3|F_j|
\le4f\sqrt{M_4\Omega_F},\qquad
D^+\sqrt{M_4}\le2f\sqrt{\Omega_*}.
$$

Thus travel at most $A$ implies
$\sqrt{M_4}\le\sqrt{M_4(0)}+2\sqrt{\Omega_*}A$.
It also implies $\sqrt M\le\sqrt{M_0}+A$. Integrating
$\|DJ_C\|\le\sqrt2+4\sqrt M$ along the path gives

$$
\sigma_{\min}(J_C(t))
\ge\sigma_0-(\sqrt2+4\sqrt{M_0})A-2A^2.
$$

These are the claimed fourth-moment and coarse-rank envelopes. They bound
the logarithmic force rate by $\beta/W$, giving the exponential force
bound and its integrated travel. A first-exit argument closes the region
through $A(T_*)=A_*$. The output floor follows from
$\|P_Hf_\theta\|\le(2\sqrt2/3)M_4/W$; the fine-error progress bound
follows by integrating $dY^2/dt=-2f^2$. This proves the proposition.
$\square$

As before, $f_0=0$ gives a stationary effective flow when the initial
projector is defined. It is not evaluated by the positive-force time formula.
No ordinary-GD or Adam persistence claim follows automatically from this
effective-flow proposition.

**The associated population scale statement.** For any spacing $h>0$, set
$\lambda_j=h|a_j|$. Let $p_0$ be the fraction initially above
$\lambda_0<\lambda_*$ and let $p_{\rm ever}(t)$ count labels that have
reached $\lambda_*$ at any time up to $t$. Any total parameter-travel
bound $A(t)$ proved in this note gives

$$
p_{\rm ever}(t)\le
\min\left\{1,\ p_0+
\frac{h^2A(t)^2}{W(\lambda_*-\lambda_0)^2}\right\}.
$$

Indeed, Minkowski's inequality gives
$\sum_j(\int_0^t|\dot a_j|\,ds)^2\le A(t)^2$.
Each new crossing requires slope travel at least
$(\lambda_*-\lambda_0)/h$, so counting by squared travel proves the
claim. This allows individual escapes while controlling their population
fraction, and needs no individual-neuron forecast.

### What remains to propagate: force-energy redistribution

The hypothesis on $\Omega_F$ is not supplied by Proposition 10. Its exact
derivative identifies the remaining collective dynamical question. Let
$G=\dot F=-DF[F]$. Then

$$
\begin{aligned}
\dot\Omega_F={}&
-\frac{2W}{f^2}\sum_j(p_j\cdot F_j)|F_j|^2\\
&+\frac{2W}{f^2}\sum_j|p_j|^2(F_j\cdot G_j)
-\frac{2\Omega_F}{f^2}(F\cdot G).
\end{aligned}
$$

The first line is movement of the force-carrying population. The second
line is redistribution of force energy among particles. To express this
without separate particle predictions, introduce weights
$\omega_j=|F_j|^2/f^2$ and the output-bias weight
$\omega_d=F_d^2/f^2$, which together sum to one. Assign sizes
$q_j=|X_j|^2$, $q_d=0$, and block growth rates
$\nu_j=F_j\cdot G_j/|F_j|^2$ and $\nu_d=G_d/F_d$ where their weights
are nonzero. Their values at zero-weight blocks do not affect the formula.
Then the same identity is

$$
\dot\Omega_F
=-2W\sum_j\omega_j(p_j\cdot F_j)
+2\operatorname{Cov}_{\omega}(q,\nu).
$$

The covariance is positive when force energy moves preferentially toward
larger particles, even if their positions barely change. This is a
population feedback condition, distinct from controlling every neuron's
trajectory.

For example, define hidden-force concentration
$I_F=W\sum_j\omega_j^2$. The movement term obeys

$$
\left|2W\sum_j\omega_j(p_j\cdot F_j)\right|
\le2f\sqrt{\Omega_F I_F}.
$$

This is Cauchy–Schwarz applied to
$\sum_j|p_j|\omega_j^{3/2}$, using
$\sum_j|p_j|^2\omega_j=\Omega_F/W$ and
$\sum_j\omega_j^2=I_F/W$.

If $I_F$ is bounded and $f=O(W^{-1})$, this part changes at rate
$O(W^{-1})$. But $\Omega_F$ alone does not bound $I_F$ or the covariance.
A generic Cauchy–Schwarz estimate of the covariance introduces a
force-weighted fourth moment and the variation of the block growth rates;
it does not close an initial-data theorem at the desired rate.
The remaining task is to derive a sufficiently small signed redistribution
bound from the coupled vector field, or identify another closed collective
quantity. Measuring these two aggregate terms and perturbing their
population alignment provides a direct test of that hypothesis.

## Appendix. An optional diagnostic: redistribution by residual relaxation

For $f=\|F\|>0$, retain the weights and sizes from the preceding appendix,
including the output-bias block with $q_d=0$. Define

$$
\Omega=\sum_{j,d}\omega_jq_j,\qquad
\Omega_2=\sum_{j,d}\omega_jq_j^2,\qquad
V_F=\Omega_2-\Omega^2,\qquad
I_F=W\sum_{j=1}^W\omega_j^2.
$$

Here $\Omega=\Omega_F$, and $\Omega_2$ is the force-weighted fourth moment
of the rescaled particle size. The bias weight is included in the first
three sums, but excluded from $I_F$.
The bound $I_F\le I_*$ prevents hidden force energy from concentrating
excessively: equal weights on all hidden neurons give $I_F=1$, whereas
equal weights on only $m$ hidden neurons give $I_F=W/m$ when the bias force
vanishes. Force concentration is unchanged by multiplying the entire force
vector by a nonzero scalar.

### Residual relaxation and redistribution have different signs

The negative term $-\|J_HF\|^2$ in the force-norm identity reduces total
force energy. It need not reduce the size of the population carrying that
energy. The distinction can be quantified without controlling individual
neurons.

**Lemma 11 (relaxation-induced redistribution).** At a state with defined
coarse projector, let

$$
\mathcal A=\Pi J_H^*J_H\Pi
$$

act on the coarse tangent space $\ker J_C$. Let $\Delta_H$ be its spectral
diameter on that space. The contribution $G_{\rm relax}=-\mathcal A F$
to $G=\dot F$ produces size redistribution satisfying

$$
|\dot\Omega_{\rm relax}|
\le\Delta_H\sqrt{V_F}
\le\frac{9M_6}{W^2}\sqrt{V_F}.
$$

**Proof.** Let $Q$ be the block diagonal parameter operator with entries
$q_jI_3$ and bias entry zero. Set $U=\Pi(Q-\Omega I)F$. Then
$\langle U,F\rangle=0$ and $\|U\|\le f\sqrt{V_F}$. The redistribution
term is $-2\langle U,\mathcal A F\rangle/f^2$. Subtracting the midpoint
of the spectrum times the identity from $\mathcal A$ does not change this
inner product, and its remaining operator norm is $\Delta_H/2$.
Cauchy–Schwarz proves the first bound. The second follows from
$\Delta_H\le\|J_H\|^2\le9M_6/W^2$. $\square$

For example, consider the two-dimensional linear flow
$\dot F=-\operatorname{diag}(1,0)F$ with $F(0)=(1,1)$ and fixed size
operator $Q=\operatorname{diag}(1,4)$. The force norm decreases, but

$$
\Omega(t)=\frac{e^{-2t}+4}{e^{-2t}+1}
$$

increases from $5/2$ toward $4$. This is a counterexample to an inference
from positive-semidefinite residual relaxation alone; it is not claimed
as a trajectory of the tanh network. Under bounded $M_6$ and $V_F$, the
lemma nevertheless bounds this redistribution by $O(W^{-2})$. The
curvature terms below can act at the larger $O(W^{-1})$ rate.

## Appendix. The preferred conditional closure: distributed force alone

Proposition 10 can be closed with a single structural premise: **bounded force concentration
$I_F$ alone propagates the fourth population moment.** Neither a future
force-weighted size bound nor a future kurtosis bound is needed. This is
the preferred conditional population theorem in this note. The initial-data
theorems earlier remain distinct: the concentration assumption here still
concerns a future interval.

The key observation is that the effective gradient is an orthogonal
projection of the fine gradient. Its strength cannot be arbitrary relative
to the current population. Write

$$
q=M_4^{1/4}=W^{1/4}\left(\sum_j|p_j|^4\right)^{1/4}.
$$

This $q$ is a scalar population norm, distinct from the individual squared
sizes $q_j$ used in the preceding appendices. For the current effective
force, the following three inequalities are exact:

$$
D^+q\le I_F^{1/4}f,\qquad
f\le\frac{3Y I_F^{1/4}q^3}{W},\qquad
\Omega_F\le\sqrt{I_F M_4}.
\tag{P1}
$$

They explain the closure. A population whose fourth moment is bounded has
weak nonlinear force if that force is sufficiently distributed. Changing
that population moment requires the same weak force. The force can reinforce
itself as the population changes, but the two inequalities bound that coupled
reinforcement directly.

**Concentration lemma (effective force and population motion).**
The bounds (P1) hold wherever the effective projector is defined and $f>0$.

**Proof.** The derivative of a norm is bounded by the norm of the velocity.
Using $\dot p_j=-F_j$ and
$\sum_j|F_j|^4=I_F f^4/W$ gives

$$
D^+q\le W^{1/4}\left(\sum_j|F_j|^4\right)^{1/4}
=I_F^{1/4}f.
$$

The identity $F=\Pi g_H$ with $\Pi$ an orthogonal projector implies
$f^2=\langle F,g_H\rangle$. The output-bias component of $g_H$ is zero,
and each hidden block satisfies $|(g_H)_j|\le3Y|p_j|^3$ by the global
fine-sensitivity bound. Hölder's inequality therefore gives

$$
\begin{aligned}
f^2
&\le3Y\sum_j|F_j||p_j|^3\\
&\le3Y\left(\sum_j|F_j|^4\right)^{1/4}
          \left(\sum_j|p_j|^4\right)^{3/4}
=\frac{3Y I_F^{1/4}q^3}{W}f.
\end{aligned}
$$

Finally, $\sum_j\omega_j^2=I_F/W$ and
$\sum_j(W|p_j|^2)^2=WM_4$. Cauchy–Schwarz applied to
$\Omega_F=\sum_j\omega_j W|p_j|^2$ proves the third bound.
The output-bias weight contributes zero to this sum. $\square$

**Theorem 12 (conditional persistence from distributed effective force).**
Consider the exact effective flow of Section 1 with $f_0>0$ and
$\sigma_0=\sigma_{\min}(J_C(0))>0$. Choose a constant $I_*>0$ and
a factor $\chi>1$. Define

$$
s=I_*^{1/4},\qquad q_0=M_4(0)^{1/4},\qquad
A_\chi=\frac{(\chi-1)q_0}{s}.
$$

Assume the initial rank margin is positive:

$$
\sigma_\chi=
\sigma_0-(\sqrt2+4\sqrt{M_0})A_\chi-2A_\chi^2>0.
\tag{P2}
$$

The sole future population-shape assumption is
$I_F(t)\le I_*$ on the interval in question. In particular, the theorem
does **not** assume future bounds on $M_4,M_6,\Omega_F$, force-weighted
size kurtosis, or $f$.
Let

$$
k=\frac{6Y_0\sqrt{I_*}\,q_0^2}{W},\qquad
T_\chi=\frac{1-\chi^{-2}}{k},\qquad
\bar q(t)=\frac{q_0}{\sqrt{1-kt}}.
$$

Then, for $0\le t\le T_\chi$, provided the concentration assumption
holds through that time,

$$
\begin{aligned}
q(t)&\le\bar q(t)\le\chi q_0,\\
f(t)&\le\frac{3Y(t)s\bar q(t)^3}{W},\\
\int_0^t\|\dot\theta(v)\|\,dv
&\le\frac{\bar q(t)-q_0}{s}\le A_\chi,\\
\sigma_{\min}(J_C(t))&\ge\sigma_\chi.
\end{aligned}
\tag{P3}
$$

The resulting output-error bounds are

$$
\begin{aligned}
\|f_{\theta(t)}-y\|
&\ge\left[\|P_Hy\|-
\frac{2\sqrt2}{3W}\bar q(t)^4\right]_+,\\
Y(t)
&\ge Y_0\exp\left\{
-\frac{9\sqrt{I_*}\,q_0^6}{2kW^2}
\left[(1-kt)^{-2}-1\right]\right\}.
\end{aligned}
\tag{P4}
$$

With a common fixed allowance $I_*>0$, width-independent upper bounds on
$M_0,M_4(0),Y_0$, and a positive uniform lower bound on $\sigma_0$,
one can choose a fixed
$\chi>1$ sufficiently close to one so that (P2) holds uniformly. The
conditional persistence interval is then at least $cW$ for some $c>0$.
On that interval the non-affine output is $O(W^{-1})$, and squared fine
error decreases by at most $O(W^{-1})$. A target with a fixed nonzero
non-affine component consequently retains a positive output-error floor
at sufficiently large width. This is a width-dependent effective-flow
statement, not a claim about a fixed number of GD or Adam updates.
For a relative tolerance $\varepsilon$ and $\|y\|>0$, failure is guaranteed
whenever the first lower bound in (P4) exceeds $\varepsilon\|y\|$.
A nonzero non-affine target component alone does not imply failure at every
chosen tolerance.

**Proof.** Since $Y\le Y_0$, the first two bounds (P1) imply

$$
D^+q\le\frac{3Y_0\sqrt{I_*}}Wq^3.
$$

Its scalar comparison solution is $\bar q$. The force bound follows
from (P1). Integrating
$f\le3Y_0s\bar q^3/W=\dot{\bar q}/s$ proves the travel bound.
As before, travel at most $A_\chi$ preserves the coarse rank by (P2).
This closes the argument by first exit, and bounds all particle parameters
and the output bias over this finite interval. No breakdown of the smooth
effective ODE can precede it while the concentration assumption holds.

The capacity inequality
$\|P_Hf_\theta\|\le(2\sqrt2/3)M_4/W$ proves the first line of (P4).
For the second, $d(Y^2/2)/dt=-f^2$ and the force bound give

$$
\frac d{dt}\log Y\ge
-\frac{9\sqrt{I_*}}{W^2}\bar q(t)^6.
$$

Integration gives (P4). For the uniform-width conclusion, choose $\chi$
using the upper bound on $q_0$ and the initial rank margin. The formula
for $T_\chi$ then bounds its denominator by a width-independent constant
times $W^{-1}$. The force bound is $O(W^{-1})$ on a fixed $cW$ interval,
so integrating $2f^2$ bounds squared-error reduction by $O(W^{-1})$.
If $f_0=0$, the effective flow is stationary and the positive-force
formulas are unnecessary. $\square$

### Retaining the measured initial force without extra shape assumptions

The explicit theorem above bounds $f_0$ by Hölder's inequality. A much
smaller measured $f_0$ can also be retained without additional shape bounds.
The following scalar comparison uses the force-curvature identity already
proved in Proposition 10.

**Corollary 13 (actual-force comparison under the same concentration
assumption).** Choose any $A_*>0$ for which

$$
\sigma_*=
\sigma_0-(\sqrt2+4\sqrt{M_0})A_*-2A_*^2>0.
$$

Set $s=I_*^{1/4}$, $q(A)=q_0+sA$ and $g_0=Wf_0>0$. Define

$$
\begin{aligned}
\beta(A)&=Y_0\left[
6\sqrt2s^2q(A)^2+
\frac{6q(A)^4}{\sigma_*^2}+
\frac{12\sqrt2s q(A)^5}{\sigma_*^2\sqrt W}\right],\\
G(A)&=g_0+\int_0^A\beta(v)\,dv\\
&=g_0+2\sqrt2Y_0s\left[q(A)^3-q_0^3\right]
+\frac{6Y_0}{5\sigma_*^2s}\left[q(A)^5-q_0^5\right]
+\frac{2\sqrt2Y_0}{\sigma_*^2\sqrt W}
\left[q(A)^6-q_0^6\right].
\end{aligned}
\tag{P5}
$$

Let $\bar A$ solve $d\bar A/d\tau=G(\bar A)$, $\bar A(0)=0$,
where $\tau=t/W$. Its travel endpoint occurs at physical time

$$
T_*=W\int_0^{A_*}\frac{dA}{G(A)}.
\tag{P6}
$$

Subject only to $I_F\le I_*$ through that interval, actual travel is
at most $\bar A(t/W)$, $q(t)\le q(\bar A(t/W))$,
and $f(t)\le G(\bar A(t/W))/W$. In particular,

$$
\begin{aligned}
\|f_{\theta(t)}-y\|
&\ge\left[\|P_Hy\|-
\frac{2\sqrt2}{3W}q(\bar A(t/W))^4\right]_+,\\
Y(t)^2
&\ge\left[Y_0^2-
\frac2W\int_0^{\bar A(t/W)}G(A)\,dA\right]_+.
\end{aligned}
\tag{P7}
$$

**Proof.** Write actual total travel as $A$. The first bound (P1) gives
$q\le q_0+sA$, and the third gives $\Omega_F\le s^2q^2$.
Substitution in Proposition 10's logarithmic force bound yields
$\dot f\le\beta(A)f/W$. In rescaled time, with $g=Wf$,
$A'=g$ and $g'\le\beta(A)g$. Therefore
$g\le g_0+\int_0^A\beta(v)\,dv=G(A)$, and scalar comparison bounds
travel by $\bar A$. The chosen travel allowance preserves rank.
The capacity bound gives the first line of (P7). For the second,
integrate $dY^2/d\tau=-2g^2/W$ using
$g(\tau)\le G(\bar A(\tau))$ and $d\bar A/d\tau=G(\bar A)$.
This proves the corollary. $\square$

The simpler mechanism therefore leaves one specific population question:
does effective fine force remain distributed across enough neurons?
Bounded $I_F$ is a substantive future assumption and can fail when force
concentrates on a few particles. It does not assume a weak force amplitude,
small future slopes, or a frozen feature model. Theorem 12 and Corollary 13 derive
the resulting weak force and slow population motion. Neither theorem alone
transfers this conclusion to full GD or Adam.
