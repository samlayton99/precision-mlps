# Singular values of neighboring tanh differences

**For uniformly spaced centers and a shared dimensionless slope, neighboring tanh differences have singular-value bounds independent of width after the common column scale is removed.** The corresponding physical tanh representation has a condition number that grows linearly with width on its zero-total-readout subspace. This proves that differencing removes a cumulative-coordinate penalty. It does not remove the exponential suppression of fine patterns when lambda is small, and it is not a theorem about the full finite-window training matrix with halos, bias, unequal learned slopes, and cumulative envelope scales.

**Table 1. Symbols and conventions for the proof.**

| Symbol | Meaning |
|---|---|
| $h>0$ | Physical spacing of adjacent centers. |
| $\lambda>0$ | Common dimensionless slope, with physical slope $\gamma=\lambda/h$. |
| $m$ | Number of adjacent differences; these use $m+1$ tanh centers. |
| $q\in\mathbb R^m$ | Unscaled coefficients of the neighboring differences. |
| $B_m$ | Map from $q$ to its function, measured in $L^2(\mathbb R,dx)$. |
| $T_m$ | Physical tanh map, restricted to weights whose sum is zero. |
| $\kappa$ | Largest singular value divided by the smallest, with Euclidean coefficient norms. The Gram/Hessian condition number is its square. |
| $\omega$ | Angular frequency in units of the center spacing; $\omega=\pi$ represents alternating adjacent coefficients. |

This is a derivation for an idealized uniform dictionary, not a fit to the training curves. The use of $dx$ explains a factor of two relative to the report's uniform probability measure $dx/2$; condition numbers are unchanged by that common factor.

## A finite-slope bound independent of width

Place centers at $jh$ and define

$$
\phi_j(x)=\tanh\!\left(\lambda\left(\frac{x}{h}-j\right)\right),
\qquad
\psi_j(x)=\phi_j(x)-\phi_{j+1}(x),
\qquad
B_mq=\sum_{j=0}^{m-1}q_j\psi_j.
$$

**Proposition 1.** For every $m\ge1$, $h>0$, $\lambda>0$, and coefficient vector $q$,

$$
\boxed{
h\,a_\lambda\|q\|_2^2
\ \le\ \|B_mq\|_{L^2(\mathbb R)}^2
\ \le\ 4h\|q\|_2^2,
\qquad
a_\lambda=
\frac{4\pi^2}{\lambda^2\sinh^2\!\left(\frac{\pi^2}{2\lambda}\right)}.
}
$$

Consequently every singular value of $B_m$ lies between $\sqrt{ha_\lambda}$ and $2\sqrt h$, and

$$
\boxed{
\kappa(B_m)\le C_\lambda,
\qquad
C_\lambda=\frac{2}{\sqrt{a_\lambda}}
=\frac{\lambda}{\pi}\sinh\!\left(\frac{\pi^2}{2\lambda}\right).
}
$$

The bound depends on lambda but not on $m$ or $h$. Absolute singular values shrink like $\sqrt h$ under grid refinement; it is the ratio, or the spectrum after dividing by $\sqrt h$, that is controlled.

**Proof.** Set $t=x/h$ and introduce the probability density and piecewise-constant function

$$
\rho_\lambda(t)=\frac{\lambda}{2}\operatorname{sech}^2(\lambda t),
\qquad
p_q(t)=\sum_{j=0}^{m-1}q_j\mathbf1_{[j,j+1]}(t).
$$

Integrating the derivative of tanh gives the exact convolution identity

$$
(B_mq)(ht)=2(\rho_\lambda*p_q)(t).
$$

Since $\|\rho_\lambda\|_1=1$ and $\|p_q\|_2=\|q\|_2$, the convolution inequality gives the upper bound immediately.

For the lower bound use the Fourier convention $\widehat f(\omega)=\int_{\mathbb R}f(t)e^{-i\omega t}\,dt$. The sech-squared transform yields

$$
\widehat\rho_\lambda(\omega)=M_\lambda(\omega)
=\frac{\pi\omega/(2\lambda)}{\sinh(\pi\omega/(2\lambda))},
\qquad M_\lambda(0)=1.
$$

This transform follows from the standard logistic-derivative integral; see [Bilge and Ozdemir, equation (3)](https://arxiv.org/pdf/1502.07182), with the Fourier normalization changed as specified above. Also,

$$
\widehat p_q(\omega)
=e^{-i\omega/2}\operatorname{sinc}(\omega/2)Q(\omega),
\qquad
Q(\omega)=\sum_{j=0}^{m-1}q_je^{-ij\omega},
\qquad
\operatorname{sinc}(u)=\frac{\sin u}{u}.
$$

On $[-\pi,\pi]$, $\operatorname{sinc}(\omega/2)\ge2/\pi$ and $M_\lambda(\omega)\ge M_\lambda(\pi)>0$. The latter follows because $v/\sinh v$ decreases for $v\ge0$. Restricting the nonnegative Fourier energy integral to this interval and using Parseval for $Q$ gives

$$
\begin{aligned}
\frac1h\|B_mq\|_2^2
&=\frac{4}{2\pi}\int_{\mathbb R}
M_\lambda(\omega)^2\operatorname{sinc}^2(\omega/2)
|Q(\omega)|^2\,d\omega\\
&\ge\frac{16}{\pi^2}M_\lambda(\pi)^2
\frac1{2\pi}\int_{-\pi}^{\pi}|Q(\omega)|^2\,d\omega\\
&=a_\lambda\|q\|_2^2.
\end{aligned}
$$

This proves the result for every finite coefficient vector. In the language of stable translated bases, it is a Riesz bound. The general Fourier-Gramian framework is described in [Aldroubi et al., Section 2](https://mate.dm.uba.ar/~hafg/papers/determining03.pdf); the explicit constants above come from the present tanh calculation.

## What improves relative to physical tanh coefficients

A single tanh is not in $L^2(\mathbb R)$ because of its constant tails. To compare the same function space with the same output norm, restrict the physical coefficients to

$$
\mathcal W_m=\left\{w\in\mathbb R^{m+1}:\sum_{j=0}^{m}w_j=0\right\},
\qquad
T_mw=\sum_{j=0}^{m}w_j\phi_j.
$$

Define the rectangular difference matrix $E_m:\mathbb R^m\to\mathcal W_m$ by

$$
E_mq=(q_0,\ q_1-q_0,\ldots,\ q_{m-1}-q_{m-2},\ -q_{m-1})^T.
$$

It is a bijection onto $\mathcal W_m$, and $B_m=T_mE_m$. It includes both end differences; it is distinct from the square matrix that retains the final tanh anchor in the implemented full model.

**Proposition 2.** For fixed $\lambda>0$,

$$
\frac{\kappa(E_m)}{C_\lambda}
\le\kappa(T_m)\le C_\lambda\kappa(E_m),
\qquad
\kappa(E_m)=\cot\!\left(\frac{\pi}{2(m+1)}\right).
$$

Thus $\kappa(T_m)=\Theta_\lambda(m)$ while $\kappa(B_m)$ remains bounded independently of $m$. Differencing removes a width-dependent conditioning penalty in this precise comparison. These conservative bounds do not assert that the difference coordinates have a smaller condition number for every individual finite $m,\lambda$, or that every singular value increases.

**Proof.** The matrix $E_m^TE_m$ is tridiagonal with diagonal 2 and adjacent entries $-1$. Its eigenvectors have entries $\sin(jk\pi/(m+1))$, $j,k=1,\ldots,m$, giving singular values

$$
\sigma_k(E_m)=2\sin\!\left(\frac{k\pi}{2(m+1)}\right).
$$

From Proposition 1, for every nonzero $q$,

$$
ha_\lambda\frac{\|q\|^2}{\|E_mq\|^2}
\le
\frac{\|T_mE_mq\|^2}{\|E_mq\|^2}
\le
4h\frac{\|q\|^2}{\|E_mq\|^2}.
$$

Taking maxima and minima over $q$ bounds the extreme singular values of $T_m$. Dividing those bounds gives the stated condition-number inequalities.

In the sharp-step limit, the result becomes exact: $B_m^*B_m=4hI$, so

$$
\kappa(B_m)=1,
\qquad
\kappa(T_m)=\cot\!\left(\frac{\pi}{2(m+1)}\right)
\sim\frac{2(m+1)}{\pi}.
$$

The Gram/Hessian condition number therefore changes from order $m^2$ to one in this limit. This proof compares coefficient norms for the same represented functions; no target or optimizer enters it.

## The remaining difficulty at lambda 0.25

Independence from width does not imply a small condition number. At the reference value $\lambda=0.25$, Proposition 1 gives

$$
a_{0.25}\simeq1.80834\times10^{-14},
\qquad
C_{0.25}\simeq1.48727\times10^7.
$$

The large bound reflects real attenuation, rather than solely a loose estimate. For the infinite uniform lattice, the Gramian multiplier of the unscaled bumps is

$$
G_\lambda(\omega)=
4\sum_{k\in\mathbb Z}
\operatorname{sinc}^2\!\left(\frac{\omega+2\pi k}{2}\right)
M_\lambda(\omega+2\pi k)^2.
$$

It determines the energy gain of coefficient modes, since the squared output norm is $h/(2\pi)$ times the integral of $G_\lambda|Q|^2$ over $[-\pi,\pi]$. At the constant and alternating coefficient modes,

$$
G_\lambda(0)=4,
\qquad
G_\lambda(\pi)=\frac{8\pi^2}{\lambda^2}
\sum_{r=0}^{\infty}
\operatorname{csch}^2\!\left((2r+1)\frac{\pi^2}{2\lambda}\right).
$$

At $\lambda=0.25$, their singular-gain ratio is

$$
\sqrt{\frac{G_{0.25}(0)}{G_{0.25}(\pi)}}
\simeq1.05166\times10^7.
$$

Consequently the infinite-lattice condition number is at least this large, and is at most $C_{0.25}$. Constant and alternating infinite sequences are understood as Fourier modes; long finite packets approach these gains. This is not a claim that a specific 512- or 1024-width training matrix has exactly this condition number.

More generally, as $\lambda\downarrow0$, the alternating-mode gain ratio and the upper bound both grow in proportion to $\lambda\exp(\pi^2/(2\lambda))$, with different constant factors. Differencing removes the cumulative-coordinate penalty while leaving this exponential smoothing penalty. The distinction is relevant to the gamma hypothesis, but it does not establish that the measured training residual is concentrated in the worst alternating mode or prove an exponential training slowdown.

## Scope for the implemented experiment

The theorem applies directly to unscaled $q$ coefficients of equal-slope, uniformly spaced differences measured over the whole real line. Four features of the implemented experiment require additional estimates:

- **Finite observation window and halos.** Restricting the integral to $[-1,1]$ can reduce the norm of a halo bump almost to zero. The whole-line lower bound therefore does not transfer automatically. The bias and final tanh anchor also require a separate analysis.
- **Cumulative envelope scales.** The optimizer trains $\theta$ with $q=S\theta$. On the theorem's block, $\kappa(B_mS)\le C_\lambda\,S_{\max}/S_{\min}$. This does not prove those scales are optimal, and a width-dependent scale ratio can introduce its own width dependence.
- **Discrete sampling.** The continuous Gram matrix must be compared with the sampled one. A quadrature error estimate in operator norm is needed for a rigorous sampled lower bound, especially when $a_\lambda$ is small.
- **Unequal learned slopes.** The convolution and translated-basis arguments use a common slope. For unequal slopes, neighboring differences still cancel their asymptotic tails but need not form the same positive localized bumps.

Accordingly, the theorem explains a structural benefit and separates it from the remaining geometry-dependent smoothing. It does not prove that the full experimental matrix is well-conditioned, that lambda 0.25 is optimal for optimization, or that Adam has a particular convergence rate. The empirical comparisons remain in the [experiment report](../results/checkpoint_D_optimizers/expD06_fixed_center_scales/relative_rate_results.md#neighbor-difference-readouts-improve-adam-across-uniform-targets).

## Combining the reference scales with differences

The original scaled training uses $w=D_w a$, $b=d_ba_b$, and $\gamma=\lambda/h$, where $D_w=\operatorname{diag}(\sqrt{\alpha_j})$ contains the hidden-readout scales from the fixed reference construction. The following extension derives scales for cumulative readouts from the same allowances. It changes the readout metric deliberately; it is not an identity preserving the original optimizer or a guarantee that training reaches the reference bandwidth.

Let $C$ be the square cumulative-sum matrix and $L=C^{-1}$ its lower bidiagonal inverse. For all $W$ neurons, including the final anchor, set

$$
q=Cw,\qquad
s_j^2=\sum_{k=1}^{j}\alpha_k,\qquad
S=\operatorname{diag}(s_j),\qquad
\boxed{w=LS\theta,\quad b=d_b\theta_b,\quad\gamma=\lambda/h.}
$$

There are two direct justifications for $S$. First, the construction bounds $|w_k|\le\alpha_k$ imply $|q_j|\le\sum_{k\le j}\alpha_k=s_j^2$. Applying the original square-root-envelope prescription to $q$ therefore yields $S$. This transfers reference bounds; it does not impose bounds on trained coefficients.

Second, suppose the original scaled coefficients $a_j$ have independent Xavier draws with common variance $\sigma_a^2=2/(W+1)$. Then

$$
\operatorname{Cov}(q)=\sigma_a^2 CD_w^2C^T,
\qquad
\operatorname{Var}(q_j)=\sigma_a^2s_j^2,
\qquad
\operatorname{Var}(\theta_j)=\sigma_a^2.
$$

Thus the new trained coefficients retain the same marginal initialization variance. They are correlated: independent Xavier draws of $\theta$ would produce a different physical network. A paired comparison must draw $a$ and physical $\gamma$ once, form $w=D_wa$ and $\lambda=h\gamma$, and initialize the difference arm with $\theta=S^{-1}Cw$. Bias starts at zero. Use physical-slope Xavier with tanh gain $5/3$; absorbing initial slope signs into readouts preserves the signed-draw function. This is the existing `xavier_a_reference` initialization in [core.py](../experiments/expD06_fixed_center_scales/core.py), distinct from the historical signed-envelope initialization.

Apply $S$ **before** differencing. For $j<W$, the column multiplying $\theta_j$ is then $s_j(\phi_j-\phi_{j+1})$; the final column is $s_W\phi_W$. Applying unequal neuron scales after differencing would instead produce $d_j\phi_j-d_{j+1}\phi_{j+1}$, whose tails do not cancel when $d_j\ne d_{j+1}$. Do not multiply by $D_w$ again. Keep this map fixed during training, including after slope sign crossings; changing signs or scales online would define another algorithm. Once slopes have different signs, the feature differences need not retain tail cancellation, which should be diagnosed rather than silently corrected.

With one shared GD rate $\eta$, gradient descent on $(\theta_b,\theta,\lambda)$ gives exactly

$$
\boxed{
\Delta w=-\eta LS^2L^T\nabla_w\mathcal L,
\qquad
\Delta b=-\eta d_b^2\partial_b\mathcal L,
\qquad
\Delta\gamma=-\eta h^{-2}\nabla_\gamma\mathcal L.
}
$$

This combines the prescribed slope scale with a coupled readout preconditioner. There is one scalar rate, but no longer a separate effective scalar rate for each physical readout: neighboring gradient entries contribute to its update. Keeping the full correlated metric $CD_w^2C^T$ on $q$ would reproduce the original physical mobility $D_w^2$ exactly. Replacing that correlated metric by its diagonal $S^2$ is the additional preconditioning choice; the variance/envelope argument motivates it without proving it optimal. Adam can use the same coordinates, but these GD update and step-size guarantees do not become Adam guarantees.

### Constant shared learning-rate ablation

The coordinate maps prescribe the relative physical updates. The absolute shared scalar $\eta$ is an experimental ablation: keep it constant for all 100,000 updates of each run, with exactly the same value for readouts and slopes. Compare the original scaled coordinates and scaled neighbor differences at each matched scalar rate and physical initialization. The [experiment protocol](../experiments/expD06_fixed_center_scales/README.md#constant-shared-rate-sweep-with-neighbor-differences) specifies the first grid. There is no backtracking, rate decay, gradient clipping, or independently tuned block rate in this comparison.

The frozen-readout calculation below provides a stability diagnostic. It does not select or adapt the joint-training rate. Learning slopes makes the loss nonlinear, so the fixed-dictionary bound alone does not certify joint stability; the constant-rate ablation measures which shared rates are stable and effective.

For a **frozen** geometry, let $B$ be the complete sampled feature matrix in the chosen readout coordinates, including bias, anchor, and halos, divided by $\sqrt M$. The half-MSE Hessian is exactly $B^TB$. Any positive step below $2/\|B\|_2^2$ is stable on its nonzero modes; a conservative, fully prescribed choice is

$$
\eta=\frac1U,
\qquad
U\ge\|B\|_2^2,
\qquad
U=\|B\|_F^2\ \text{is one directly computable choice}.
$$

It guarantees $\mathcal L(z-\eta g)\le\mathcal L(z)-\eta\|g\|^2/2$ for the readout gradient $g$. This prescription uses the actual scaled features and needs no inverse, least-squares training update, or target-based LR search. It can be conservative. Proposition 1 also gives $U=4hS_{\max}^2$ for its whole-line difference block, but that expression omits the full experiment's bias and anchor and cannot be used as their bound.

### Conditioning is a rate barrier, not an error floor

For a fixed dictionary and ordinary GD, each attainable residual singular component satisfies

$$
e_i(t)=(1-\eta\sigma_i^2)^t e_i(0).
$$

A positive but tiny singular value therefore causes slow convergence; it does not create a nonzero optimization floor in exact arithmetic. Residual components outside the feature span remain as representation error. With a step of order $1/\sigma_{\max}^2$, a singular-value ratio of order $10^7$ permits a worst-direction time scale of order $10^{14}$ updates. That worst direction matters only insofar as the target and current residual occupy it.

This is not a universal bound for every first-order algorithm. Acceleration and conjugate-gradient methods have different condition-number dependence, and changing coordinates changes the condition number itself; see [Shewchuk, Sections 6 and 9](https://www.cs.cmu.edu/~quake-papers/painless-conjugate-gradient.pdf) for the quadratic comparison. The first paired experiment should distinguish measured training error, remaining target-loaded weak modes, numerical stagnation, and the diagnostic refit error, rather than label a finite-budget plateau a fundamental floor.

## Numerical cross-checks

The proof is analytic. Independent numerical checks confirm its normalizations and matrix identities: the sech-squared Fourier transform by quadrature; the finite-slope singular-value bounds for $m=8,32,64$ at $\lambda=0.25,1$; and the exact sharp-step condition numbers for $m=8,32,128$. Finite-slope checks use midpoint quadrature with 64 points per cell and padding of $20/\lambda$ cells on each side. The alternating-mode constants were evaluated with 70-digit arithmetic, eight positive series terms, and a geometric bound on the remaining tail. Values and numerical settings are saved in [the theorem check artifact](../results/checkpoint_D_optimizers/expD06_fixed_center_scales/ratio_uniform_reference/neighbor_difference_theorem_checks.json). These are local algebra/quadrature checks, with no new training runs.

The [combined-coordinate check](../results/checkpoint_D_optimizers/expD06_fixed_center_scales/ratio_uniform_reference/combined_coordinate_checks.json) verifies the matched physical initialization, cumulative variances, full nonlinear gradient transformations, and one certified frozen-readout step at $N=512$, seed 0. This is an implementation-level identity check, not a training comparison or rate-selection result.
