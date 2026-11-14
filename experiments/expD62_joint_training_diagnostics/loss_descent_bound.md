# A finite-horizon movement bound for joint GD

This is a proposed complementary bound for the scale-acquisition statement. It does not replace the paper's target-specific output-error floor, and it is not a claim of novelty: its main inequality follows from the standard GD descent inequality and Cauchy–Schwarz.

Let all parameters, including output bias, be collected in theta. Use the sampled half-MSE loss

\[
L(\theta)=\tfrac12\|q_\theta-f\|_Z^2,
\qquad \theta_{n+1}=\theta_n-\eta\nabla L(\theta_n).
\]

Number the selected starting checkpoint as n=0. Assume, for the moment, sufficient decrease at every update:

\[
L(\theta_{n+1})\le L(\theta_n)-\frac\eta2\|\nabla L(\theta_n)\|^2.
\tag{1}
\]

Summing (1), using nonnegativity of L, and applying Cauchy–Schwarz to the updates gives

\[
\begin{aligned}
\sum_{j<n}\eta\|\nabla L(\theta_j)\|^2&\le2L(\theta_0),\\
\|\theta_n-\theta_0\|
&\le\sum_{j<n}\eta\|\nabla L(\theta_j)\|\\
&\le\sqrt{\eta n}\sqrt{\sum_{j<n}\eta\|\nabla L(\theta_j)\|^2}
\le\sqrt{2\eta nL(\theta_0)}.
\end{aligned}
\tag{2}
\]

Projection onto the hidden-parameter block or slope block cannot increase norm. Therefore

\[
\sqrt{M_{h,n}}\le\sqrt{M_{h,0}}+\sqrt{2\eta nL(\theta_0)},
\qquad
\gamma_{\mathrm{RMS},n}\le\gamma_{\mathrm{RMS},0}
+\sqrt{\frac{2\eta nL(\theta_0)}{W}}.
\tag{3}
\]

The starting loss is the full half-MSE; it is not the earlier nonaffine-residual allowance E0. This ceiling grows as sqrt(n) and does not feed its own value into a cubic growth allowance.

## A sufficient step-size condition from the starting state

For the tanh network q(x)=b_out+sum_j w_j tanh(gamma_j x+beta_j), with |x|<=1, write E=||q(theta_0)-f||_Z, c=||w_0||_2, and m=||w_0||_infinity. Fix a horizon H and define

\[
D=E\sqrt{\eta H},\quad
A=\sqrt{W+1+2(c+D)^2},\quad
r=D+\eta AE,
\]
\[
J_*^2=W+1+2(c+r)^2,\quad
E_*=E(1+\eta J_*^2),\quad
\widehat L=J_*^2+E_*\bigl(4(m+r)+\sqrt2\bigr).
\tag{4}
\]

If eta times L_hat <= 1, then (1)–(3) hold for all n<=H for exact-arithmetic GD.

To prove this without assuming future trajectory bounds, use induction. Prior sufficient-decrease steps imply L(theta_n)<=L(theta_0) and ||theta_n-theta_0||<=D. The sample-normalized output Jacobian satisfies

\[
\|Dq_\theta\|_{\mathrm{op},Z}^2\le\|Dq_\theta\|_{\mathrm{HS},Z}^2
\le W+1+2\|w\|_2^2.
\tag{5}
\]

Thus the next proposed step has length at most eta A E. Its whole line segment lies within radius r of theta_0, where (5) is bounded by J_*^2. Along this segment the residual RMS is at most E+J_* eta A E <= E_*. The network-output Hessian has independent three-parameter neuron blocks. Using |tanh'|<=1, |tanh''|<=2, and ||(x,1)||<=sqrt(2), each block has operator norm at most 4|w_j|+sqrt(2). Consequently the loss Hessian on the step segment satisfies

\[
\|\nabla^2 L\|_{\mathrm{op}}
\le J_*^2+E_*\bigl(4(m+r)+\sqrt2\bigr)=\widehat L.
\tag{6}
\]

Taylor's descent inequality with eta L_hat<=1 now gives (1) for this step, closing the induction. The condition uses only the starting parameters, starting residual, width, step size, and horizon. It does not inspect future losses or assume concentration remains constant.

## Numerical evaluation and scope

`loss_descent_bound.py` evaluates (4) at a chosen saved checkpoint and compares (2)–(3) with all subsequent saved states. These are floating-point evaluations of analytic inequalities, not outward-rounded certificates for the implemented floating-point optimizer. Observing monotone losses only at sparse checkpoints would not by itself establish (1); the starting-state step-size check is supplied for that reason.

This proves a finite-budget bound on RMS slope and total movement under its sufficient step-size condition. It does not imply that every individual slope remains below the RMS ceiling, that the target cannot be fitted accurately, or that the bound holds for arbitrary longer horizons without rechecking the condition. The original output-error and spectral arguments address different parts of that question.
