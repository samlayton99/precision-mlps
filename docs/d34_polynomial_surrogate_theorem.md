# Conditional accuracy of an evolving polynomial effective force

A polynomial surrogate can retain the changing geometry and readouts while
approximating tanh analytically. The useful theorem is an absolute error bound
for this evolving field, followed by a finite-time comparison. It does not
assert that a cubic model resolves a force whose leading term is fifth order.
This note gives checkpoint-based error bounds for that model comparison;
the observed forecast errors are not assumptions in the proof.

**Notation.**

| Symbol | Meaning |
|---|---|
| $P_p$ | Odd Taylor polynomial of tanh, degree $p\in\{3,5\}$ |
| $F,F_p$ | Exact-tanh and polynomial balanced fine forces, each evaluated at its own current parameters |
| $\Pi,\Pi_p$ | Orthogonal projections onto the nullspaces of their coarse Jacobians |
| $\alpha=(p+1)/2$ | Width exponent in the full-force approximation error |
| $t=\eta N$ | Additional GD time, distinct from update count $N$ |

## 1. Example: preserve coupling without retaining every tanh term

The [small-parameter regime](d34_mechanism_rate_theorems.md#8-a-width-dependent-rate-without-freezing-the-effective-map)
has $|a_j|\le A_a/\sqrt W$, $|b_j|\le A_b/\sqrt W$ and
$|c_j|\le A_c/\sqrt W$. Put $U=A_a+A_b$. Let the fine space be the **whole**
empirical orthogonal complement of $\{1,x\}$, with $|x_i|\le1$.
Projecting polynomial features onto modes through degree $p$ is equivalent
for their force: all their higher-mode Jacobian rows vanish. The target need
not be a polynomial.

The cubic network makes the retained coupling concrete. After its affine
part is removed, its only monomial coefficients are

$$
x^2:\quad-\sum_j c_ja_j^2b_j,\qquad
x^3:\quad-\frac13\sum_j c_ja_j^3.
$$

Their projections onto the fixed empirical modes determine the generated
quadratic and cubic outputs. Subtracting the target's fixed modal coefficients
gives the two driving errors. As every particle's slope, bias and readout
changes, these sums and their sensitivities change too. The full polynomial
coarse output, including its Taylor corrections, determines the compensating
projection. This is a closed model in the particle parameters; two error
coefficients alone do not close their own dynamics. The quintic model adds
the corresponding terms through degree five.

For example, let $r_H$ be the cubic model's own fine residual function, whose
modal coefficients are $e_H$, and put $m_2=\langle r_H,x^2\rangle_m$ and
$m_3=\langle r_H,x^3\rangle_m$. The cubic model's raw fine gradient is

$$
\begin{aligned}
(g_H)_{a,j}&=-2c_ja_jb_jm_2-c_ja_j^2m_3,\\
(g_H)_{b,j}&=-c_ja_j^2m_2,\\
(g_H)_{c,j}&=-a_j^2b_jm_2-\tfrac13a_j^3m_3,
\qquad (g_H)_d=0.
\end{aligned}
$$

The applied effective force is $\Pi_3g_H$, not these raw terms alone.
This displays both ingredients of the mechanism: the target and generated
errors set $m_2,m_3$, while the evolving readout–geometry products set how
each neuron responds. Coarse compensation remains in the projection.

At the same state $\theta$, define

$$
F=\Pi J_H^Te_H,\quad F_p=\Pi_pJ_{H,p}^Te_{H,p},\qquad
\Pi=I-J_C^T(J_CJ_C^T)^{-1}J_C,
$$

and use the corresponding polynomial definition for $\Pi_p$.
Assume both coarse Grams have minimum eigenvalue at least $\kappa>0$ on
the domain used for the comparison. No bound on the output bias is needed:
the fine residual and both fields are independent of that bias.

### Same-state force error

Let $M_{p+2}$ bound $|\tanh^{(p+2)}|$ on $[-U,U]$. Define
$c_r=M_{p+2}/(p+2-r)!$, for $r=0,1,2$.
Taylor's theorem, using the vanishing even coefficients, gives

$$
|\tanh^{(r)}(u)-P_p^{(r)}(u)|\le c_r|u|^{p+2-r}.
$$

Set

$$
C_e=A_cc_0U^{p+2},\quad
C_J=\sqrt{2A_c^2c_1^2U^{2p+2}+c_0^2U^{2p+4}},\quad
C_2=2A_cc_2U^p+\sqrt2c_1U^{p+1}.
$$

Uniformly in the parameter regime,

$$
\|e_H-e_{H,p}\|\le C_eW^{-\alpha},\quad
\|J_H-J_{H,p}\|,\|J_C-J_{C,p}\|\le C_JW^{-\alpha},\quad
\|D(J_H-J_{H,p})\|,\|D(J_C-J_{C,p})\|\le C_2W^{-\alpha}.
\tag{1}
$$

For example, a slope column's Jacobian error is at most
$A_cc_1U^{p+1}W^{-(p+2)/2}$. Summing squared column bounds gives $C_J$.
The second-derivative difference has independent neuron blocks: its geometry
block is bounded by $2A_cc_2U^pW^{-\alpha}$ and its mixed block by
$\sqrt2c_1U^{p+1}W^{-\alpha}$. This proves the last inequality without an
extra width factor.

Write $E=\|e_H\|$, $A=\|J_H\|$, $A_p=\|J_{H,p}\|$ and
$\varepsilon_J=C_JW^{-\alpha}$, $\varepsilon_e=C_eW^{-\alpha}$.
The elementary projector estimate

$$
\|\Pi-\Pi_p\|\le\frac{2\varepsilon_J}{\sqrt\kappa}
$$

follows by splitting the difference into the two cross-projections and using
the norms of the coarse pseudoinverses. Consequently,

$$
\boxed{\ \|F-F_p\|\le
\varepsilon_JE+A_p\varepsilon_e
+\frac{2\varepsilon_J}{\sqrt\kappa}A_p(E+\varepsilon_e).\ }
\tag{2}
$$

Section 8 gives $E=O(1)$ and $A=O(W^{-1})$, so (2) is
$O(W^{-\alpha})$ with width-independent constants under uniform parameter,
target and conditioning bounds. Every $a,b,c$ coordinate has the sharper
error $O(W^{-\alpha-1/2})$; the output-bias coordinate need not share that
sharper order. To see the coordinate improvement, write
$F_i=(J_H)_i^Te_H-(J_C)_i^Tz$, where
$z=(J_CJ_C^T)^{-1}J_CJ_H^Te_H$. Each $a,b,c$ coarse column is $O(W^{-1/2})$,
$z=O(W^{-1})$, its polynomial difference is $O(W^{-\alpha})$, and the
direct fine-column remainder has order $W^{-\alpha-1/2}$.

## 2. Theory: compare two evolving states, not two forces at one checkpoint

On a convex comparison domain, assume the same uniform bounds and both
coarse conditioning bounds. The exact fine Hessian is $O(W^{-1})$ by Section
8; (1) gives the same order for the polynomial fine Hessian. Furthermore
$\|D\Pi_p[v]\|\le2\|DJ_{C,p}[v]\|/\sqrt\kappa=O(\|v\|)$.
Differentiating $F_p=\Pi_pJ_{H,p}^Te_{H,p}$ therefore gives

$$
\|DF_p\|\le
\frac{2\|DJ_{C,p}\|}{\sqrt\kappa}\|J_{H,p}\|\|e_{H,p}\|
+\|DJ_{H,p}\|\|e_{H,p}\|+\|J_{H,p}\|^2\le\frac{L}{W}.
$$

Let exact ordinary GD use $F(\theta_n)+r_{C,n}$, and let the surrogate use
$F_p(\vartheta_n)$, starting from identical parameters. If (2) is bounded by
$C W^{-\alpha}$ and $\|r_{C,n}\|\le R_n$, then

$$
\|\theta_N-\vartheta_N\|\le
\eta\sum_{k<N}(1+\eta L/W)^{N-1-k}
       (CW^{-\alpha}+R_k).
\tag{3}
$$

This follows by subtracting the two discrete updates and induction. For
effective-only trajectories, $R_k=0$. On a window $\eta N\le cW$, (3) is
$O(W^{1-\alpha})=O(W^{-(p-1)/2})$, plus the displayed tracking allowance.
For a comparison restricted to the three neuron blocks, that allowance can
use $\|(r_C)_{a,b,c}\|$: the effective fields do not depend on $d$.

This comparison yields an acquisition envelope. Let $E_n$ be the right side
of (3), or a valid sharper bound. Then every neuron satisfies

$$
\max_{n\le N}\lambda_{j,n}
\le \max_{n\le N}h\bigl(|\vartheta_{a,j,n}|+E_n\bigr).
$$

The fraction ever reaching $\lambda_*$ is therefore at most the fraction
whose right side is at least $\lambda_*$. The predicted trajectory comes
from the checkpoint and the chosen polynomial field; the allowance must
come from uniform bounds on its comparison domain. A small measured endpoint
error cannot substitute for that allowance.

**Closure matters.** The centered small-parameter box contains $a=c=0$,
where the coarse Gram is singular. Conditioning must not be assumed uniformly
on that entire box. A conditioned convex subdomain, or a separate argument
for every comparison segment, is required. Trajectory containment must also
be established by initial margins and displacement bounds, rather than read
from the subsequent trajectories.

On a symmetric grid, a concrete conditioning bootstrap is available. Put
$v=\langle x^2\rangle_m>0$. In the coarse basis $1,x/\sqrt v$, the affine
feature model has

$$
J_{C,\mathrm{aff}}J_{C,\mathrm{aff}}^T
\succeq\operatorname{diag}(1+\|c\|^2,v\|c\|^2).
$$

The remaining readout-column contribution is positive semidefinite. With
$L_H$ from Section 8, Weyl's inequality gives a common sufficient lower bound

$$
\sigma_{\min}(J_C),\sigma_{\min}(J_{C,p})
\ge\sqrt v\,(\|c_0\|-B_c)-L_H/W-C_JW^{-\alpha},
\tag{4}
$$

whenever $\|c-c_0\|\le B_c$. Requiring the right side to exceed
$\sqrt\kappa$ on a convex neighborhood of the initial readouts establishes
conditioning there. Section 8's readout displacement bound
$B_c\le\eta N(K_c+\delta_c)/W$, or its polynomial counterpart, supplies a
first-exit test. Use the larger bound for both trajectories and their
connecting segments; the readout ball and outer coordinate box are convex.
The tracking hypothesis remains separate.

## 3. Prediction: what initial force anchoring can and cannot improve

Define $\delta F=F-F_p$ and the checkpoint-anchored field
$\widetilde F_p(\theta)=F_p(\theta)+\delta F(\theta_0)$.
It has the exact effective force at the fork without fitting future motion.
Its same-state defect is $\delta F(\theta)-\delta F(\theta_0)$.
Differentiating the projector formula, using (1) and the uniform inverse
Gram bound, gives $\|D\delta F\|\le C_DW^{-\alpha}$.
For completeness, differentiate $K^{-1}$ as
$D(K^{-1})[v]=-K^{-1}DK[v]K^{-1}$. After subtracting the two projector
derivatives, every term contains either $J_C-J_{C,p}$ or
$D(J_C-J_{C,p})$, while all other factors have width-independent bounds.
Hence $D\Pi-D\Pi_p=O(W^{-\alpha})$; applying the product rule to
$\Pi J_H^Te_H-\Pi_pJ_{H,p}^Te_{H,p}$ proves the asserted order. Thus

$$
\|F(\theta)-\widetilde F_p(\theta)\|
\le C_DW^{-\alpha}\|\theta-\theta_0\|.
$$

If exact effective speed is bounded by $V/W$, the approximation contribution
in (3) is at most
$C_DV e^{Lt/W}t^2/(2W^{\alpha+1})$, where $t=\eta N$.
The unanchored bound scales as $t/W^\alpha$. This improves the short-time
coefficient; it does not add a width power when $t=O(W)$. Ordinary tracking
also affects the distance from the initial state and must be included there.
Away from the fork, the anchored field does not exactly preserve coarse
balance; its actual coarse leakage is
$J_C(\theta)[\delta F(\theta_0)-\delta F(\theta)]$.

A sharper speed bound is available under a stronger target condition. If
$y_H$ is orthogonal to every empirical polynomial of degree at most three,
then $J_{H,3}^Ty_H=0$. Consequently

$$
\|F\|\le\|J_H\|\|P_Hf\|
        +\|J_H-J_{H,3}\|\|y_H\|
\le\frac{L_HA_cU^3/3+C_{J,3}Y_H}{W^2}.
$$

Here $C_{J,3}$ is the cubic remainder constant defined above. This retains
the force from generated lower-mode error in the first term. For
effective-only motion it improves the anchored error to
$O(t^2/W^{\alpha+2})$, hence $O(W^{-\alpha})$ on $t=O(W)$, while the same
conditioning and containment requirements still apply. Pure fifth and ninth
fine targets satisfy this orthogonality condition when defined in the exact
empirical polynomial basis. Small lower-mode contamination in a different
target construction must be charged separately.

Finally, these are **absolute** error bounds. For a pure fifth-mode target,
orthogonality removes its loading from the cubic model. The omitted
fifth-order tanh term can generate a per-neuron force of order $W^{-5/2}$,
the same order as the cubic force-error bound. Generated lower-mode error
can also contribute at that order. Cubic relative error need not vanish.
A relative-accuracy theorem requires a lower bound on the force or motion
being predicted; neither small tracking nor the absolute width orders
provide that lower bound.

## 4. A slower conditional clock when the target has no quadratic or cubic load

**Example.** A target confined to empirical degree five or nine cannot load
the cubic feature Jacobian directly. The network can still generate
lower-degree error, and correcting that error still moves its geometry.
Both sources of effective force are smaller than the target-general bound
in Section 8 of the mechanism note. The following refinement retains both.

Assume the small-parameter bounds, $\kappa$-conditioned coarse projection and
exact empirical orthogonality through degree three used above. Define

$$
B_f=\frac{A_cU^3}{3},\qquad
G_*=L_HB_f+C_{J,3}Y_H,\qquad
D_*=L_2B_f+C_{2,3}Y_H+L_H^2,
\tag{5}
$$

where $L_2=4A_cU+\sqrt2U^2$ bounds $W\|DJ_H\|$, and $C_{J,3},C_{2,3}$
are the cubic remainder constants in (1). Let $M_C$ uniformly bound
$\|DJ_C\|$; one sufficient choice is
$M_C=\sqrt2+8A_c/(3\sqrt3)$ for $W\ge1$.
All derivative norms here are induced Euclidean operator norms, with sample
outputs normalized by the empirical mean.

### Force and response bounds

Put $r=J_H^Te_H$ before the coarse projection. Exact target orthogonality
gives $J_{H,3}^Ty_H=0$ at every state and also
$(DJ_{H,3}[v])^Ty_H=0$ for every direction $v$. Therefore

$$
\begin{aligned}
\|r\|&\le\frac{G_*}{W^2},\\
Dr[v]&=(DJ_H[v])^TP_Hf
       -(D(J_H-J_{H,3})[v])^Ty_H+J_H^TJ_Hv,\\
\|Dr\|&\le\frac{D_*}{W^2}.
\end{aligned}
$$

Since $F=\Pi r$, $\|\Pi\|=1$ and
$\|D\Pi\|\le2M_C/\sqrt\kappa$, this proves

$$
\boxed{\ \|F\|\le\frac{G_*}{W^2},\qquad
\|DF\|\le\frac{D_*+2M_CG_*/\sqrt\kappa}{W^2}.\ }
\tag{6}
$$

Thus both force magnitude and local feedback are smaller. The derivative
claim is uniform on the stated conditioned domain, not a statement about a
measured Jacobian at one checkpoint. It becomes a Lipschitz bound between
states only when their connecting segment lies in that domain.

The individual neuron coordinates admit the sharper bound needed to preserve
the parameter regime. For $\ell=a,b,c$, set

$$
\begin{array}{c|ccc}
\ell&a&b&c\\ \hline
d_\ell&A_cU^2&A_cU^2&U^3/3\\
q_\ell&A_c&A_c&U\\
s_\ell&A_cc_{1,3}U^4&A_cc_{1,3}U^4&c_{0,3}U^5
\end{array}
\qquad
K_\ell^*=d_\ell B_f+s_\ell Y_H+
                  \frac{q_\ell G_*}{\sqrt\kappa}.
$$

Here $c_{r,3}$ denotes the Taylor constant $c_r$ for $p=3$.
The exact fine Jacobian column, its cubic remainder and the coarse column
are bounded respectively by $d_\ell/W^{3/2}$,
$s_\ell/W^{5/2}$ and $q_\ell/\sqrt W$.
Also $\|(J_CJ_C^T)^{-1}J_Cr\|\le G_*/(\sqrt\kappa W^2)$.
Combining these three estimates proves

$$
|F_{\ell,j}|\le K_\ell^*/W^{5/2},\qquad \ell=a,b,c.
\tag{7}
$$

### A window closed by initial margins

Suppose ordinary GD tracking obeys
$|(r_C)_{\ell,j}|\le\delta_\ell^*/W^{5/2}$ on a proposed enclosing
neighborhood. If the initial parameter constants are $A_{\ell,0}<A_\ell$,
then the inequalities

$$
\frac{\eta N}{W^2}(K_\ell^*+\delta_\ell^*)
\le A_\ell-A_{\ell,0},\qquad \ell=a,b,c,
\tag{8}
$$

close the parameter bounds by the same first-exit induction as before.
On a symmetric grid, the additional sufficient condition

$$
\sqrt v\left(\|c_0\|-
       \frac{\eta N}{W^2}(K_c^*+\delta_c^*)\right)
       -\frac{L_H}{W}\ge\sqrt\kappa
\tag{9}
$$

closes coarse conditioning at the same time. The readout displacement in
parentheses is a full Euclidean readout-block bound, not a single-coordinate
bound. A comparison with a polynomial trajectory also needs its corresponding
readout-motion bound and the extra polynomial Jacobian remainder in (4).
The chosen readout ball and coordinate margins can define a convex domain
on which (6) applies. Persistent tracking smallness is still a hypothesis;
it has not been inferred from a small initial value.

These tests permit $\eta N$ proportional to $W^2$ when their constants and
initial margins are uniform and the proportionality constant is small enough.
At every update, including sign crossings,

$$
|\lambda_{j,n+1}-\lambda_{j,n}|
\le\frac{\eta h(K_a^*+\delta_a^*)}{W^{5/2}}.
$$

With $h$ proportional to $W^{-1}$ the normalized rate is
$O(\eta W^{-7/2})$. If $hA_a/\sqrt W<\lambda_*$, (8)–(9) exclude all
threshold hits during that window. These are conditional analytic tests;
their constants have not been numerically certified for the width experiment.

### How a lower-mode target load restores the faster terms

Write $y_H=y_{2:3}+y_{\ge4}$ with $y_{\ge4}$ orthogonal to degree at most
three, and let $\varepsilon=\|y_{2:3}\|$. Repeating the proof adds
$L_H\varepsilon/W$ to the full-force bound and
$(L_2+2M_CL_H/\sqrt\kappa)\varepsilon/W$ to the derivative bound.
For coordinate $\ell$, it adds
$(d_\ell+q_\ell L_H/\sqrt\kappa)\varepsilon/W^{3/2}$.
Thus an order-one quadratic or cubic load restores the target-general
orders. To retain the sharper orders it suffices to have
$\varepsilon=O(W^{-1})$, with its constant included in the bounds.
Exact orthogonality is one sufficient condition, not a universal property of
the function collection. None of these absolute estimates proves relative
forecast precision or a common barrier for all targets.
