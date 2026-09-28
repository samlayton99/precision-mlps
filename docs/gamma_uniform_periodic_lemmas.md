# Exact gamma-dependent spectra on uniform periodic grids

The periodic model gives an exact, interpretable answer: gamma attenuates known frequencies, and uniform geometry lets us explicitly identify the resulting learning modes. It is a starting theorem, not a theorem about ordinary tanh on a finite interval. We retain finite-grid aliasing below instead of identifying Fourier waves with eigenvectors without checking the geometry.

| Symbol | Meaning |
| --- | --- |
| $L$ | Period length |
| $W,m=qW$ | Number of hidden features and samples, with integer $q\ge1$ |
| $\gamma>0$ | Common slope |
| $\omega_n=2\pi n/L$ | Angular frequency |
| $M_\gamma(\omega)$ | Logistic smoothing multiplier |
| $a_n,\alpha_k$ | Continuous Fourier coefficient and sampled alias sum |
| $\mu_r$ | Kernel eigenvalue; not dimensionless bandwidth |
| $\eta$ | Gradient descent step size |

## 1. Smoothing is the exact entry point for gamma

Let $\rho_\gamma(x)=\gamma\operatorname{sech}^2(\gamma x)/2$. It is nonnegative and integrates to one. Convolving the real-line sign function with this density gives

$$
(\rho_\gamma*\operatorname{sign})(x)
=2\int_{-\infty}^x\rho_\gamma(t)\,dt-1=\tanh(\gamma x).
$$

Its Fourier transform is

$$
M_\gamma(\omega)=\frac{z}{\sinh z},\qquad
z=\frac{\pi|\omega|}{2\gamma},\qquad M_\gamma(0)=1.
$$

To verify this, substitute $t=e^{2\gamma x}$ into the Fourier integral. The integral becomes $\int_0^\infty t^{-is}(1+t)^{-2}\,dt$ with $s=\omega/(2\gamma)$. The beta integral gives $\Gamma(1-is)\Gamma(1+is)=\pi s/\sinh(\pi s)$, by the gamma reflection identity.

Periodize the density by summing its translates by $L$. For any periodic profile $g$ with Fourier coefficients $b_n$, its periodic convolution with this density has coefficients

$$
a_n(\gamma)=b_nM_\gamma(\omega_n).
$$

We use $g(x)=\operatorname{sign}(\sin(2\pi x/L))$, with value zero at its jumps. Its coefficients are $b_n=2/(i\pi n)$ for odd $n$, and zero otherwise, including $n=0$. Denote its smoothed version by $\phi_\gamma$. This function is periodic and has both positive and negative transitions. It is not ordinary tanh globally. Its smoothed Fourier series converges absolutely.

## 2. Uniform continuum centers: an exactly diagonal theorem

Equip periodic functions with inner product $L^{-1}\int_0^L\overline f g$. For uniform continuum centers, define

$$
K_\gamma(x,y)=\frac1L\int_0^L
\phi_\gamma(x-c)\overline{\phi_\gamma(y-c)}\,dc.
$$

The kernel integral operator also uses measure $dy/L$. Fourier waves are its eigenfunctions, with eigenvalues

$$
\mu_n(\gamma)=|b_n|^2M_\gamma(\omega_n)^2.
$$

**Proof.** Expand both features in their absolutely convergent Fourier series. Integrating over the center cancels every pair with distinct frequencies, leaving $K_\gamma(x,y)=\sum_n|a_n|^2e^{i\omega_n(x-y)}$. Applying the operator to one Fourier wave proves the claim. An independently trained constant feature adds eigenvalue one in the constant direction. $\square$

Thus the eigenvectors and target projections do not change with gamma in this model. Gamma changes the eigenvalues by an explicit squared multiplier. This is the standard convolution argument, stated here as the baseline for the more difficult finite-interval problem.

## 3. Aligned finite grids: exact alias blocks

Set $x_\ell=\ell L/m$, $c_j=jL/W$, and $m=qW$. Use the **raw readout convention**

$$
J_{\ell j}=\frac{\phi_\gamma(x_\ell-c_j)}{\sqrt m},
\qquad K=JJ^*.
$$

The loss is $\|J\theta-y/\sqrt m\|^2/2$. There is no $1/\sqrt W$ factor in a hidden feature. If that factor is introduced, divide the hidden-feature eigenvalues below by $W$; an unscaled bias still has eigenvalue one.

Let $U$ be the unitary discrete Fourier matrix, $U_{\ell k}=m^{-1/2}e^{2\pi i\ell k/m}$, with indices $k=0,\ldots,m-1$. Define the absolutely convergent sums

$$
\alpha_k=\sum_{h\in\mathbb Z}a_{k+hm}.
$$

For each residue $r=0,\ldots,W-1$, collect the components $\alpha_k$ with $k\equiv r\pmod W$ into $\alpha^{(r)}$.

**Theorem (finite periodic spectrum).** In Fourier coordinates, $K$ has independent blocks

$$
(U^*KU)_{r}=W\alpha^{(r)}\alpha^{(r)*}.
$$

Every nonzero block contributes one eigenvalue $\mu_r=W\|\alpha^{(r)}\|^2$, with Fourier-coordinate eigenvector $v_r=\alpha^{(r)}/\|\alpha^{(r)}\|$, supported in that block. All remaining directions have eigenvalue zero.

**Proof.** Sampling the Fourier expansion gives

$$
(U^*J)_{kj}
=\alpha_k e^{-2\pi i k j/W}.
$$

The alias phase agrees because $m/W$ is an integer. Multiplying by its adjoint gives $\alpha_k\overline{\alpha_{k'}}\sum_{j=0}^{W-1}e^{-2\pi i(k-k')j/W}$. The sum equals $W$ if $k\equiv k'\pmod W$ and zero otherwise. Each resulting outer product has the stated spectrum. $\square$

When $m=W$, blocks are scalar: the discrete Fourier eigenvectors are fixed and $\mu_k=W|\alpha_k|^2$. When $q>1$, their directions can change with gamma **inside the known alias blocks**. No general matrix diagonalization is needed, but it would be incorrect to identify each individual Fourier wave with an eigenvector.

The square-wave feature is odd, so its sampled mean vanishes on the aligned grid: $\alpha_0=0$. Consequently, an added bias column $\boldsymbol1/\sqrt m$ is orthogonal to all hidden columns and contributes its own eigenvector and eigenvalue one. This conclusion uses odd symmetry, not merely zero continuous mean.

**Parity limitation.** If $W$ is even, $m=qW$ is even. Even frequency bins have zero coefficients and therefore even residue blocks vanish. At most $W/2$ hidden eigenvalues are nonzero. A bias restores only the constant direction. These inaccessible modes are a property of the chosen periodic square wave, not a learning delay induced by small gamma. The continuum model likewise has zero eigenvalues at even frequencies. A target acquisition claim must exclude or explicitly include this floor.

## 4. Target weights and the exact training law

For any nonzero sampled target $y$, let $\widehat y=U^*y$. For each nonzero block define

$$
p_r=\frac{|v_r^*\widehat y|^2}{\|y\|^2}.
$$

Include the analogous bias weight if present, and let $p_\perp=1-\sum_r p_r$. Starting readout weights at zero, with $0\le\eta\mu_r\le1$, the squared relative residual after $n$ updates is exactly

$$
E(n)^2=p_\perp+\sum_r p_r(1-\eta\mu_r)^{2n}.
$$

**Proof.** The sample residual obeys $r_{n+1}=(I-\eta K)r_n$. Expand the initial residual in the orthonormal block eigenvectors and their orthogonal complement, apply the recurrence, and use Pythagoras. Scaling $y$ by $1/\sqrt m$ cancels in the relative residual. $\square$

If a collection of modes has target weight at least $p$ and rates at most $a<1$, then $E(n)\ge\sqrt p(1-a)^n$. Reaching relative residual $\varepsilon<\sqrt p$ requires

$$
n\ge\frac{\log(\sqrt p/\varepsilon)}{-\log(1-a)}.
$$

This bound is useful only when the spectral and target-weight statements supplying $a,p$ are useful. In the diagonal cases these quantities are explicit without a target-dependent eigensolve.

## 5. Gamma caps, width scaling, and alias errors

The derivative of $z/\sinh z$ is negative for $z>0$. Indeed, its numerator $\sinh z-z\cosh z$ starts at zero and has derivative $-z\sinh z<0$. Therefore $\gamma\le\bar\gamma$ implies $M_\gamma(\omega)\le M_{\bar\gamma}(\omega)$. For the continuum kernel this gives a mode-by-mode spectral upper bound immediately.

For the finite kernel, define

$$
U_k(\bar\gamma)=\sum_{h\in\mathbb Z}
|b_{k+hm}|M_{\bar\gamma}(\omega_{k+hm}).
$$

The exact block formula and triangle inequality give the explicit cap bound

$$
\mu_r(\gamma)\le W\sum_{k\equiv r\ (W)}U_k(\bar\gamma)^2.
$$

This upper bound is not a claim that the exact finite eigenvalues are monotone in gamma: signed alias coefficients can cancel. If aliases are small, retain the leading coefficient instead. For a signed principal representative $k$, put $\tau_k=\sum_{h\ne0}|a_{k+hm}|$. Then

$$
(|a_k|-\tau_k)_+\le|\alpha_k|\le|a_k|+\tau_k.
$$

Squaring and summing within each block gives two-sided eigenvalue bounds. The implementation retains all modes through a cutoff; the following tail controls every remaining alias.

**Absolute tail lemma.** Let $N\ge0$ be an integer cutoff, $n_0$ the smallest odd integer greater than $N$, and $a=\pi^2/(\gamma L)$. Then

$$
\sum_{|n|>N}|a_n|
\le T_N:=\frac{8a}{\pi}
\frac{e^{-a n_0}}{(1-e^{-2a})(1-e^{-2a n_0})}.
$$

**Proof.** For positive odd $n$, $|a_n|=2a/(\pi\sinh(an))\le(4a/\pi)e^{-an}/(1-e^{-2an_0})$. Sum the geometric series over $n=n_0,n_0+2,\ldots$, and double it for negative frequencies. $\square$

This bounds the uniform feature approximation error and the sum of absolute alias errors. For the truncated matrix $J_N$, including an unchanged bias if used,

$$
\|J-J_N\|\le\sqrt W T_N,
\qquad
\|K-K_N\|\le2\|J_N\|\sqrt W T_N+WT_N^2.
$$

The first bound follows from the Frobenius norm; the second expands the kernel difference. The ordered eigenvalues of the two Hermitian kernels differ by at most this last bound, by the variational characterization of eigenvalues. This does not by itself bound individual eigenvectors or target weights near repeated or tiny eigenvalues.

For a deliberately band-limited profile with $|n|<W/2$, neither kind of aliasing occurs. Its nonzero eigenvalues are exactly $W|b_n|^2M_\gamma(\omega_n)^2$ on either grid. This is an exact surrogate theorem; the tail lemma is necessary when returning to the full smoothed square wave.

At a grid-relative frequency $n/W\to\xi>0$, the multiplier argument is $z=\pi^2 n/(\gamma L)$. Taking $\gamma/W\to g>0$ retains the nonzero multiplier $(\pi^2\xi/(gL))/\sinh(\pi^2\xi/(gL))$. If $\gamma/W\to0$, the multiplier tends to zero with asymptotic $2ze^{-z}$. On the accessible modes of the alias-free band-limited surrogate, raw eigenvalues scale as

$$
\mu_n=\frac{4W}{\pi^2n^2}M_\gamma(\omega_n)^2.
$$

Thus proportional slope removes the extra exponential smoothing penalty, but an algebraic $1/W$ factor remains for $n\asymp W$; width-normalized features have $1/W^2$ instead. The update rate is $\eta\mu_n$, so a step-size rule may add another width or gamma dependence. This statement does not claim width-independent acquisition time or finite-alias monotonicity.

## 6. The separate ordinary-tanh interval result

Padding the period can place the extra negative transition far from the observed input-center differences. The periodic feature can then approximate ordinary tanh accurately there. However, the actual samples and centers occupy only part of that padded period. For compatible grids the matrix takes the form $J_{\mathrm{interval}}\approx R_xJ_{\mathrm{periodic}}E_c$, where $R_x$ selects samples and $E_c$ selects centers, with the appropriate sample normalization adjusted explicitly. Its kernel contains the center mask $E_cE_c^*$ as well as the sample restriction. These masks mix frequencies.

The sharp periodic theorem does not establish that these masks are a small perturbation. The [finite-interval theorem and full proof](gamma_uniform_grid_theorem.md) instead use an antiperiodic reference on the physical interval and retain explicit boundary corrections. That construction bounds the remaining error, encloses eigenvalues, and transfers target-energy and acquisition-time predictions to ordinary tanh. The present note supplies the periodic intermediary results; it does not replace the boundary analysis or the corrected finite spectral calculation.

## Verification

The implementation is [uniform_periodic.py](../experiments/expD36_frozen_gamma_probe/uniform_periodic.py), with [independent tests](../tests/test_expD36_uniform_periodic.py). Tests synthesize features directly as sine sums, construct their kernels independently, and compare full eigenpair reconstruction and GD residuals for $q=1,4$, odd and even widths, and with and without bias. They also test the parity floor, truncation bounds, and gamma/width multiplier scaling. The code computes exact target weights for the retained Fourier model, up to floating-point error; its analytic tail bounds do not turn those weights into interval-certified weights for the infinite model. No GPU training is involved.
