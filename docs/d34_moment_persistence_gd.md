# Ordinary GD: a computable moment and support persistence certificate

The purpose of this result is to remove two assumptions that would otherwise
make a persistence claim circular: that coarse conditioning survives and that
tracking remains small. A scalar recurrence bounds both from the initial
state. The result applies to the exact tanh network, with mixed readout signs
and nonzero hidden biases. Its constants are deliberately conservative;
whether the resulting interval is informative is a separate numerical question.

The starting example is a checkpoint with small rescaled parameters and a
small, but nonzero, coarse disequilibrium. The recurrence below allows this
disequilibrium to decay or initially grow. It does not require it to have a
particular power of the width. Unlike effective gradient flow, ordinary GD
need not decrease the fine loss. We therefore prove descent of the **total**
loss and use it to bound the residual on every update segment.

## 1. Model, initial quantities, and regional constants

Let $f(x)=\sum_{j=1}^W c_j\tanh(a_jx+b_j)+d$ on a symmetric empirical
grid with $|x|\le1$ and $v=\langle x^2\rangle_m>0$. All sample norms use
the empirical mean. The loss is $L=\|f-y\|_m^2/2$, and GD uses the ordinary
Euclidean parameter gradient and a constant step $\eta$.

Write $X_j=\sqrt W(a_j,b_j,c_j)=(\alpha_j,\beta_j,\zeta_j)$ and

$$
r_0=\max_j|X_{j,0}|,\qquad M_0=\mathbb E_W|X_{j,0}|^2,
\qquad E_{s,0}=\mathbb E_W(\alpha_{j,0}^2+\zeta_{j,0}^2)>0.
$$

Use the exact coarse/fine decomposition from the
[state-dependent persistence note](d34_state_dependent_persistence.md):
$g=F+J_C^Tz$, $K=J_CJ_C^T$, $\ell=K^{-1}J_CJ_H^Te_H$, and
$z=e_C+\ell$. The coarse sample coordinates are $1,x/\sqrt v$; the fine
space is their entire orthogonal complement. In particular, $F$ is the
effective fine component of the gradient, so its learning velocity is $-F$.

| Symbol | Meaning |
|---|---|
| $M$, $E_s$ | Rescaled total particle second moment and slope–readout second moment. |
| $r_0$, $R$ | Initial maximum particle radius and fixed outer radius. |
| $m_n$, $s_n$, $r_n$ | Upper bound on $\sqrt M$, lower bound on $\sqrt{E_s}$, and upper support bound. |
| $\overline z_n$ | Upper bound on the norm of coarse disequilibrium $z_n$. |
| $u_n$ | Upper bound on the physical effective-gradient norm $\|F_n\|$; the effective-flow note's rescaled speed is $W\|F\|$. |
| $\eta$, $W$, $h$ | GD step size, physical neuron count, and construction spacing in $\lambda=h|a|$. |

Choose explicit regional bounds

$$
R>r_0,\qquad \overline M>M_0,\qquad 0<\underline E_s<E_{s,0},
$$

and calculate

$$
\delta=\sqrt{\frac{v\underline E_s}{1+2\overline M}}
              -\frac{3R^3}{W}>0,\qquad \kappa=\delta^2.
\tag{1}
$$

The positivity test in (1) is essential; squaring a negative bracket is not
allowed. For a specified multiplier $\mu>1$, the deterministic evaluation
choice is $R=\mu r_0$, $\overline M=\mu^2M_0$,
$\underline E_s=E_{s,0}/\mu^2$. The theorem also permits other explicit regions.

Define the following constants, entirely from the initial loss and region:

$$
\begin{aligned}
\overline Y&=\sqrt{2L_0},& Y&=2\overline Y,& c_*&=4Y,\\
J&=\sqrt{1+2\overline M},&
H_C&=\sqrt2+\frac{4\sqrt2 R^2}{W},\\
J_F&=3R^3,& H_F&=(8+4\sqrt2)R^2,\\
f_*&=J_FY,\\
D_\ell&=\frac{2H_Cf_*}{\kappa}
 +\frac{H_FY+J_F^2/W}{\sqrt\kappa},\\
B_*&=c_*R^3\left(1+\sqrt2\sqrt{\overline M/\kappa}\right),\\
L_F&=H_FY+\frac{J_F^2}{W}+\frac{2H_Cf_*}{\sqrt\kappa}.
\end{aligned}
\tag{2}
$$

Require

$$
\eta\left(J^2+2\overline YH_C\right)\le1,
\qquad a_*:=\frac{\eta c_*R^2}{W}\le1.
\tag{3}
$$

These are sufficient step-size tests, not claims that larger steps fail.

## 2. The scalar certificate and its conclusion

Initialize $\overline z_0=\|z_0\|$, $m_0=\sqrt{M_0}$,
$s_0=\sqrt{E_{s,0}}$, and $u_0=\|F_0\|$, with $r_0$ as above. Iterate

$$
\begin{aligned}
\overline z_{n+1}&=(1-\eta\kappa)\overline z_n
 +\frac{\eta D_\ell}{W}\left(\frac{f_*}{W}+J\overline z_n\right)
 +\frac{\eta^2H_C}{2}\left(\frac{f_*}{W}+J\overline z_n\right)^2,\\
m_{n+1}&=(1+a_*)m_n+\eta J\overline z_n,\\
s_{n+1}&=(1-a_*)s_n-\eta J\overline z_n,\\
r_{n+1}&=r_n+\eta\left(\frac{B_*}{W}+\sqrt2R\overline z_n\right),\\
u_{n+1}&=\left(1+\frac{\eta L_F}{W}\right)u_n
             +\frac{\eta L_FJ}{W}\overline z_n.
\end{aligned}
\tag{4}
$$

**Theorem.** Suppose (1)–(3) hold and, through the proposed endpoint $N$,

$$
r_n<R,\qquad m_n<\sqrt{\overline M},\qquad
s_n>\sqrt{\underline E_s}\qquad (0\le n\le N).
\tag{5}
$$

Then every GD state and every straight update segment through step $N$
remains in the specified support/moment region. At the states,

$$
\begin{gathered}
L_n\le L_0,\quad K_n\succeq\kappa I,\quad
\|z_n\|\le\overline z_n,\quad \|F_n\|\le u_n,\\
\sqrt{M_n}\le m_n,\quad \sqrt{E_{s,n}}\ge s_n,
\quad \max_j|X_{j,n}|\le r_n.
\end{gathered}
\tag{6}
$$

Thus the fine force cannot amplify faster than the explicit last recurrence
in (4). Tracking is an evolving correction with its own recurrence, not an
assumed negligible future term. No alignment or sign restriction is imposed.

For $\lambda_j=h|a_j|$, choose $\lambda_*>\lambda_0\ge0$. The fraction
of particle labels that ever cross $\lambda_*$ by step $N$ obeys

$$
\operatorname{fraction}_{\rm ever}(\lambda_*,N)
\le \operatorname{fraction}_0(\lambda>\lambda_0)
 +\frac{h^2(m_N-m_0)^2}{W(\lambda_*-\lambda_0)^2}.
\tag{7}
$$

The bound may be capped at one. The pointwise support conclusion also gives
$\max_{j,n\le N}\lambda_{j,n}\le hr_N/\sqrt W$.

## 3. Proof: analytic bounds before the discrete induction

Consider the open region $\max_j|X_j|<R$, $M<\overline M$,
$E_s>\underline E_s$. The affine coarse Jacobian has Gram matrix

$$
K_{\rm aff}=\begin{pmatrix}
1+\mathbb E(\zeta^2+\beta^2)&\sqrt v\,\mathbb E\alpha\beta\\
\sqrt v\,\mathbb E\alpha\beta&v\mathbb E(\zeta^2+\alpha^2)
\end{pmatrix}.
$$

Cauchy–Schwarz gives $\det K_{\rm aff}\ge vE_s$ and
$\operatorname{tr}K_{\rm aff}\le1+2M$. Also

$$
\|J_C-J_{C,\rm aff}\|\le3R^3/W.
\tag{8}
$$

To verify (8), use $|u|=|ax+b|\le\sqrt2R/\sqrt W$,
$|\tanh u-u|\le|u|^3/3$, and $|1-\tanh'(u)|\le u^2$.
The sum of empirical squared differences over all parameter columns is
at most $(80/9)R^6/W^2$; orthogonal projection to the coarse coordinates
cannot increase this norm. The output-bias column has zero difference.
Singular-value perturbation now gives (1) throughout the region.

The full output Jacobian has operator norm at most $J$: its squared
Hilbert–Schmidt bound is $1+\mathbb E(2\zeta^2+v\alpha^2+\beta^2)$.
The output Hessian, as a bilinear map to empirical sample space, has norm
at most $H_C$. Indeed its neuron block has norm at most
$|c\tanh''u|(1+x^2)+|\tanh'u|\sqrt{1+x^2}$.
The bounds $|\tanh''u|\le2|u|$ and $|\tanh'u|\le1$ prove the stated
constant. In particular $\|DJ_C\|\le H_C$.

Projecting away the affine output and differentiating gives

$$
\|J_H\|\le J_F/W,\qquad \|DJ_H\|\le H_F/W.
\tag{9}
$$

The first follows from the same affine-Jacobian difference used in (8).
For the second, the fine projection removes the constant and linear parts
of the output Hessian; the tanh derivative bounds give
$4A_c(A_a+A_b)+\sqrt2(A_a+A_b)^2$ with
$A_a=A_b=A_c=R$. This is $H_F$; the detailed fine-Hessian estimate is in
[the fine-Hessian proof](d34_mechanism_rate_theorems.md#8-a-width-dependent-rate-without-freezing-the-effective-map).

Whenever $\|e_H\|\le Y$, orthogonality of the parameter projection yields
$\|F\|\le f_*/W$ and $\|\ell\|\le f_*/(W\sqrt\kappa)$. Differentiating
$\ell=K^{-1}J_CJ_H^Te_H$ gives the exact identity

$$
D\ell[v]=K^{-1}\left\{
 (DJ_C[v])F+
 J_C\left[(DJ_H[v])^Te_H-(DJ_C[v])^T\ell+J_H^TJ_Hv\right]
\right\}.
$$

For the sharper constant in (2), write $P=K^{-1}J_C$ and
$\Pi=I-J_C^TP$. Direct differentiation gives
$DP[v]=K^{-1}(DJ_C[v])\Pi-P(DJ_C[v])^TP$, hence
$\|DP\|\le2H_C/\kappa$. Combining this with
$\|P\|\le1/\sqrt\kappa$ and (9) proves
$\|D\ell\|\le D_\ell/W$.
Similarly, differentiating the orthogonal projector defining $F$ gives
$\|DF\|\le L_F/W$. These constants bound the whole parameter vector,
including its output-bias coordinate.

A sharper raw-force bound controls the moments. In the label-RMS plus
output-bias metric and the clock $\tau=t/W$, the raw fine gradient has
neuron components

$$
G_\alpha=W\zeta\langle e_H,x\tanh'(ax+b)\rangle_m,\quad
G_\beta=W\zeta\langle e_H,\tanh'(ax+b)\rangle_m,\quad
G_\zeta=W^{3/2}\langle e_H,\tanh(ax+b)\rangle_m.
$$

Orthogonality of $e_H$ to $1,x$ allows Taylor subtraction about $b$.
Using $|\tanh'''|\le2$ gives component bounds
$\sqrt2YR^2|\alpha|$, $YR^2|\alpha|/2$, and
$\sqrt2YR^2|\alpha|$, respectively. Their sum is at most
$c_*R^2|\alpha|$. The coarse projection is contractive, so

$$
\|F\|\le\frac{c_*R^2}{W}\sqrt{\mathbb E\alpha^2}.
\tag{10}
$$

The rescaled coarse-gradient block for any neuron has norm at most
$\sqrt2R$. Its compensation multiplier is bounded by the raw RMS force
divided by $\sqrt\kappa$. Consequently

$$
\sqrt W|F_{a,b,c,j}|\le B_*/W,
\qquad \sqrt W|(J_C^Tz)_{a,b,c,j}|\le\sqrt2R\|z\|.
\tag{11}
$$

## 4. Proof: every update segment and tracking are controlled

Assume the conclusions hold at step $n$. Formula (10), together with
$\mathbb E\alpha^2\le M$ and $\mathbb E\alpha^2\le E_s$, gives for every
fraction $t\in[0,1]$ of the outgoing step

$$
\begin{aligned}
\sqrt{M(\theta_n-t\eta g_n)}
 &\le(1+ta_*)\sqrt{M_n}+t\eta J\overline z_n\le m_{n+1},\\
\sqrt{E_s(\theta_n-t\eta g_n)}
 &\ge(1-ta_*)\sqrt{E_{s,n}}-t\eta J\overline z_n\ge s_{n+1}.
\end{aligned}
$$

These are the triangle and reverse-triangle inequalities in the neuron
label norm. Their coefficients are nonnegative by (3). Equation (11)
also bounds every segment's support by $r_{n+1}$. Thus (5) places the
**entire** next segment in the region before any derivative estimate on
that segment is used.

At the initial endpoint $\|g_n\|\le J\overline Y$ follows from
$L_n\le L_0$. On the segment, the output Jacobian bound therefore gives
$\|f-y\|_m\le\overline Y+\eta J^2\overline Y\le2\overline Y=Y$.
The total-loss Hessian is bounded by $J^2+2\overline YH_C$ there.
Taylor's theorem and (3) prove

$$
L_{n+1}\le L_n-\frac\eta2\|g_n\|^2\le L_n.
$$

This establishes the residual bound needed for the next update without
assuming fine-loss descent.

Since $J_CF=0$, coarse Taylor expansion gives

$$
z_{n+1}=(I-\eta K_n)z_n+(\ell_{n+1}-\ell_n)+\xi_n,
\qquad \|\xi_n\|\le\frac{\eta^2H_C}{2}\|g_n\|^2.
$$

The segment derivative estimate gives
$\|\ell_{n+1}-\ell_n\|\le\eta D_\ell\|g_n\|/W$.
By (3), $\|I-\eta K_n\|\le1-\eta\kappa$, and
$\|g_n\|\le f_*/W+J\overline z_n$. This proves the first recurrence in (4).
The segment bound on $DF$ similarly proves its last recurrence. Induction
establishes (6).

Finally the label-RMS movement in update $n$ is at most
$\eta(c_*R^2m_n/W+J\overline z_n)=m_{n+1}-m_n$. Minkowski's inequality bounds
the label-RMS of the largest displacement over all iterates through $N$ by
$m_N-m_0$. Any newly acquired particle must travel at least
$\sqrt W(\lambda_*-\lambda_0)/h$ in its rescaled slope coordinate.
Markov's inequality proves (7), counting each label only once.

## 5. What this certificate does and does not establish

Every input to (4) is computable from the initial checkpoint and specified
regional radii. Passing (5) proves structural persistence for ordinary GD;
observing a later trajectory is unnecessary. Failure of a regional test
means only that this conservative certificate stops there.

The step-size condition, the Gram perturbation margin, and the accumulated
tracking correction may make the interval short. They must be evaluated
before claiming a useful training horizon. This theorem uses the general
$\eta n/W$ clock. Stronger cancellations in special targets are not used,
so it does not establish a $W^2$ clock or permanent saturation. Numerically
evaluated scalar recurrences require outward rounding for a certified
instance; ordinary floating-point evaluation is an applicability estimate.
