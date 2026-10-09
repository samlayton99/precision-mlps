# Activation-dependent construction certificates

This note supplies an independently derived scalar construction for the Radon scaling argument. It distinguishes what holds for a general derivative kernel from what has been proved for specific activations. There are no new numerical experiments or Lean developments here.

The main positive result below is a finite, explicit Gaussian-kernel/erf/GELU construction with logarithmic width under a global Fourier exponential-moment assumption, and an algebraic-width counterpart under a Fourier polynomial-moment assumption. Both use a finite number of halo centers and no readout solve. The global Fourier assumption is stronger than local analyticity; that difference must not disappear when this result is used in the main argument.

## 1. Objects, normalization, and the scalar problem

Let the target interval be `[-T,T]`, where `T>0`. Let `g` be the real scalar profile to approximate. For an integer `r>=0`, write `p=g^(r)`. We assume that `p` is specified on the real line, has an integrable Fourier transform, and satisfies Fourier inversion. For `r>0`, the values `g^(ell)(0)` for `0<=ell<r` are also supplied. The Fourier convention is

\[
\widehat p(\omega)=\int_{\mathbb R}p(s)e^{-i\omega s}\,ds,
\qquad
p(s)=\frac1{2\pi}\int_{\mathbb R}\widehat p(\omega)e^{i\omega s}\,d\omega.
\]

It suffices to start with a supplied transform satisfying these inversion conditions; the argument does not require an ordinary Fourier transform of `g` itself. For example, a sigmoid profile can have an integrable derivative even when the sigmoid is not integrable.

Let the real activation `phi` satisfy

\[
\phi^{(r)}(u)=mK(u),\qquad m\ne0,\qquad \int_{\mathbb R}K(u)\,du=1.
\]

For `r=0` this says `phi=mK`. In the master certificate, take `K` to be Schwartz, which is a sufficient, explicit condition for the sampling manipulations. Some less regular kernels can be handled by additional arguments; they are not silently covered by the Schwartz proof.

For a positive bandwidth `gamma`, define

\[
K_\gamma(u)=\gamma K(\gamma u),\qquad
\widehat K_\gamma(\omega)=\widehat K(\omega/\gamma).
\]

The normalization in the eventual activation readout is fixed by

\[
\frac{d^r}{ds^r}\phi(\gamma(s-c))
=m\gamma^{r-1}K_\gamma(s-c).
\]

## 2. A general computable error certificate

Choose a frequency cutoff `B>0`, a spacing `h>0`, and `gamma>0`. Require

\[
\widehat K(u)\ne0\quad\text{for }|u|\le B/\gamma.
\]

Continuity then gives a strictly positive minimum modulus on this compact band. Define the cutoff density by the finite integral

\[
\rho_B(c)=\frac1{2\pi}\int_{-B}^{B}
\frac{\widehat p(\omega)}{\widehat K(\omega/\gamma)}e^{i\omega c}\,d\omega.
\tag{2.1}
\]

Its Fourier transform is the quotient inside the integral, multiplied by the cutoff indicator. It is real when `p` and `K` are real, by conjugate symmetry. In particular,

\[
\|\rho_B\|_\infty\le A_\rho,
\qquad
A_\rho:=\frac1{2\pi}\int_{-B}^{B}
\frac{|\widehat p(\omega)|}{|\widehat K(\omega/\gamma)|}\,d\omega.
\tag{2.2}
\]

The sharp cutoff is permitted. It can produce a density with slowly decaying spatial tails, but all its derivatives are bounded, and its product with a Schwartz kernel is Schwartz. Absolute integrability of the unanchored activation integral is not assumed or used.

Set `Omega=2*pi/h`. First consider the infinite *kernel* sum

\[
D_h(s)=h\sum_{j\in\mathbb Z}\rho_B(jh)K_\gamma(s-jh).
\]

For each frequency `omega`, sampling the integrand at `jh` gives

\[
h\sum_{j\in\mathbb Z}e^{i\omega jh}K_\gamma(s-jh)
=\sum_{k\in\mathbb Z}
\widehat K\left(\frac{\omega+k\Omega}{\gamma}\right)
e^{i(\omega+k\Omega)s}.
\tag{2.3}
\]

This is the ordinary Poisson identity. It can also be derived by observing that the left side times `exp(-i omega s)` is `h`-periodic and integrating one period to compute each Fourier coefficient. Substituting (2.1) into `D_h`, and using (2.3), yields

\[
D_h(s)=\frac1{2\pi}\int_{-B}^{B}\widehat p(\omega)
\sum_{k\in\mathbb Z}
\frac{\widehat K((\omega+k\Omega)/\gamma)}{\widehat K(\omega/\gamma)}
e^{i(\omega+k\Omega)s}\,d\omega.
\tag{2.4}
\]

The `k=0` term is precisely the truncated target derivative. Every `k!=0` term is an alias. Absolute convergence follows from the Schwartz condition, the finite band, and the nonzero denominator. Therefore define

\[
E_{\rm cut}:=\frac1{2\pi}\int_{|\omega|>B}|\widehat p(\omega)|\,d\omega,
\tag{2.5}
\]

\[
E_{\rm alias}:=\frac1{2\pi}\int_{-B}^{B}|\widehat p(\omega)|
\sum_{k\ne0}
\left|\frac{\widehat K((\omega+k\Omega)/\gamma)}{\widehat K(\omega/\gamma)}\right|\,d\omega.
\tag{2.6}
\]

Then

\[
\|p-D_h\|_{L^\infty(\mathbb R)}\le E_{\rm cut}+E_{\rm alias}.
\tag{2.7}
\]

Choose a physical halo width `delta>0`, and retain exactly the grid indices

\[
J=\left\lfloor\frac{T+\delta}{h}\right\rfloor,
\qquad -J\le j\le J.
\]

The first omitted center on the right is `(J+1)h>T+delta`; on the left it is its negative. Thus for any `s` in the target interval, each omitted center lies at least distance `delta` away. The finite kernel sum is

\[
D_{h,J}(s)=h\sum_{j=-J}^{J}\rho_B(jh)K_\gamma(s-jh).
\]

An explicit tail certificate is

\[
E_{\rm halo}:=hA_\rho\sup_{|s|\le T}
\sum_{|j|>J}|K_\gamma(s-jh)|.
\tag{2.8}
\]

For an activation with a known decreasing tail envelope, the infinite positive sum in (2.8) is bounded by its first term plus an integral, as done below. This is a bound on the omitted centers, not an unspecified boundary oracle.

Define the raw activation sum

\[
S(s)=\frac{h}{m\gamma^{r-1}}
\sum_{j=-J}^{J}\rho_B(jh)\phi(\gamma(s-jh)).
\tag{2.9}
\]

Its `r`th derivative is exactly `D_{h,J}`. For `r>=1`, add the explicit anchor polynomial

\[
P(s)=\sum_{\ell=0}^{r-1}
\bigl[g^{(\ell)}(0)-S^{(\ell)}(0)\bigr]\frac{s^\ell}{\ell!},
\qquad q=S+P.
\tag{2.10}
\]

For `r=0`, simply use `q=S`. Repeated integration of `q^(r)-g^(r)`, with zero anchor discrepancies, proves

\[
\boxed{
\|q-g\|_{L^\infty([-T,T])}
\le H_r(T)\bigl(E_{\rm cut}+E_{\rm alias}+E_{\rm halo}\bigr),
\quad
H_r(T)=\begin{cases}1,&r=0,\\T^r/r!,&r\ge1.\end{cases}
}
\tag{2.11}
\]

For example, for `s>=0` the error is the integral of `(s-u)^(r-1)/(r-1)!` times the derivative discrepancy over `0<=u<=s`. The integral of the absolute kernel is `s^r/r!`. Reversing orientation proves the same bound for negative `s`.

For an arbitrary activation, the polynomial `P` must also be realized by the allowed architecture, or its separate approximation error counted. Writing it down is not permission to leave a polynomial side channel outside a purported pure MLP.

## 3. Concrete activations with a Gaussian certificate

Let

\[
\varphi(u)=\frac{e^{-u^2/2}}{\sqrt{2\pi}},
\qquad \Phi(u)=\int_{-\infty}^{u}\varphi(v)\,dv.
\]

The following choices have `m=1`:

| Activation `phi(u)` | Derivative order `r` | Kernel `K(u)` | Fourier transform `Khat(v)` |
|---|---:|---|---|
| Gaussian bump `varphi(u)` | 0 | `varphi(u)` | `exp(-v^2/2)` |
| Gaussian CDF `Phi(u)` | 1 | `varphi(u)` | `exp(-v^2/2)` |
| Exact GELU `u Phi(u)` | 2 | `(2-u^2)varphi(u)` | `(1+v^2)exp(-v^2/2)` |

The Gaussian CDF is a rescaled and shifted erf activation. For unnormalized `erf(u/sqrt(2))`, use `m=2`; the same kernel proof applies and the readouts divide by two.

For all three kernels,

\[
\frac1{|\widehat K(v)|}\le e^{v^2/2},
\tag{3.1}
\]

and

\[
|K(u)|\le C_K e^{-u^2/4},\qquad C_K=\frac4{\sqrt{2\pi}}.
\tag{3.2}
\]

For GELU, to verify (3.2), write

\[
|K(u)|\le\frac{(2+u^2)e^{-u^2/2}}{\sqrt{2\pi}}
\le\frac4{\sqrt{2\pi}}e^{-u^2/4}.
\]

The second inequality follows from `(2+z)exp(-z/4)<=4` for `z>=0`. The Gaussian kernel satisfies the same, looser bound.

## 4. One common finite design

Choose numbers `B>0` and `t>=1`, and set

\[
\gamma=\frac{B}{\sqrt t},\qquad
h=\frac{\pi}{2B},\qquad
\delta=\frac{2t}{B},\qquad
J=\left\lfloor\frac{T+\delta}{h}\right\rfloor.
\tag{4.1}
\]

These choices give

\[
\Omega=4B,\quad \gamma h=\frac\pi{2\sqrt t},\quad
\gamma\delta=2\sqrt t.
\tag{4.2}
\]

Suppose an explicit number `A` satisfies

\[
\frac1{2\pi}\int_{\mathbb R}|\widehat p(\omega)|\,d\omega\le A,
\qquad A_\rho\le A,
\qquad E_{\rm cut}\le Ae^{-t}.
\tag{4.3}
\]

The analytic and algebraic assumptions below each imply (4.3) with a computable choice of `A`, `B`, and `t`.

### 4.1 The entire alias sum, including GELU's polynomial factor

For `|omega|<=B` and nonzero integer `k`, let `n=|k|`. Then

\[
|\omega+4kB|\ge(4n-1)B,\qquad
|\omega+4kB|\le(4n+1)B.
\]

For GELU, the absolute kernel ratio in (2.6) is

\[
\frac{1+(\omega+4kB)^2/\gamma^2}{1+\omega^2/\gamma^2}
\exp\left[-\frac{(\omega+4kB)^2-\omega^2}{2\gamma^2}\right].
\]

Bound the denominator of the prefactor below by one. Since `gamma^2=B^2/t`, the ratio is at most

\[
\bigl[1+(4n+1)^2t\bigr]
\exp[-4tn(2n-1)].
\tag{4.4}
\]

The same bound applies to the Gaussian kernel, which has no polynomial prefactor. Because `t>=1` and `n>=1`,

\[
1+(4n+1)^2t\le26tn^2,
\qquad n(2n-1)\ge n.
\]

Sum over both signs of `k`, and put `z=exp(-4t)`:

\[
\sum_{k\ne0}|\text{kernel ratio}|
\le52t\sum_{n\ge1}n^2e^{-4tn}
=52t\frac{z(1+z)}{(1-z)^3}
\le60t e^{-4t}.
\tag{4.5}
\]

The last inequality uses `0<z<=exp(-4)`; the remaining multiplier `52(1+z)/(1-z)^3` is less than 60 throughout that interval. Substitution into (2.6) proves

\[
E_{\rm alias}\le60At e^{-4t}\le3Ae^{-t}.
\tag{4.6}
\]

For the last inequality, `60t exp(-3t)` is decreasing for `t>=1` and its value at one is less than three. Thus every alias and every target frequency in the cutoff band has been bounded; the estimate does not retain only the nearest alias.

### 4.2 The finite halo sum and its actual endpoint

The first omitted right center is `(J+1)h>T+delta`. For any `|s|<=T`, its distance from `s` exceeds `delta`. The subsequent omitted centers are at least `delta+lh` away, for `l=0,1,...`. The same bound holds on the left. Consequently (3.2) and (2.8) give

\[
E_{\rm halo}\le2AC_K h\gamma
\sum_{\ell\ge0}\exp\left[-\frac{(\gamma\delta+\ell\gamma h)^2}{4}\right].
\tag{4.7}
\]

For a decreasing nonnegative function, its left rectangle sum is at most its first rectangle plus the integral of the function. Here this gives

\[
h\gamma\sum_{\ell\ge0}e^{-(z+\ell\gamma h)^2/4}
\le\gamma h e^{-z^2/4}+\int_z^\infty e^{-u^2/4}\,du,
\quad z=\gamma\delta.
\]

For `z>0`,

\[
\int_z^\infty e^{-u^2/4}\,du
\le\int_z^\infty\frac u z e^{-u^2/4}\,du
=\frac2z e^{-z^2/4}.
\]

Use (4.2), so `z=2 sqrt(t)`, to obtain

\[
E_{\rm halo}
\le\frac{2C_K(\pi/2+1)}{\sqrt t}Ae^{-t}
\le9Ae^{-t}.
\tag{4.8}
\]

The numerical multiplier before `1/sqrt(t)` is less than 8.21. The bound nine is a convenient certified upper bound.

### 4.3 The scalar certificate and neuron count

Combining (4.3), (4.6), and (4.8) with (2.11) proves

\[
\boxed{\|q-g\|_{L^\infty([-T,T])}\le13H_r(T)A e^{-t}.}
\tag{4.9}
\]

There are `2J+1` grid neurons. Directly from (4.1),

\[
\boxed{
2J+1\le\frac4\pi(TB+2t)+1.
}
\tag{4.10}
\]

The number outside `[-T,T]` is at most `2 delta/h+2=8t/pi+2`. Therefore the halo cost is included, not discarded as an asymptotically negligible term.

The grid is symmetric and need not contain the target endpoints exactly. On `[-T,T]` the number of core centers is `2 floor(T/h)+1`. This is an explicit alternative finite-grid convention; it must not be described as exactly `N` endpoint-inclusive centers without adjusting the definitions.

## 5. Analytic Fourier class: logarithmic scalar width

Assume, for a supplied `a>0`,

\[
A_a:=\frac1{2\pi}\int_{\mathbb R}
e^{a|\omega|}|\widehat p(\omega)|\,d\omega<\infty.
\tag{5.1}
\]

Given `t>=1`, choose `B=t/a` in (4.1). Then

\[
\gamma=\frac{\sqrt t}{a},\quad h=\frac{\pi a}{2t},\quad \delta=2a.
\tag{5.2}
\]

For `|omega|<=B`,

\[
\frac{\omega^2}{2\gamma^2}
=\frac{a\omega^2}{2B}\le\frac a2|\omega|.
\]

Therefore (3.1) implies

\[
A_\rho\le\frac1{2\pi}\int_{-B}^{B}
|\widehat p(\omega)|e^{a|\omega|/2}\,d\omega\le A_a.
\]

Also

\[
E_{\rm cut}\le e^{-aB}\frac1{2\pi}\int_{|\omega|>B}
e^{a|\omega|}|\widehat p(\omega)|\,d\omega\le A_ae^{-t}.
\]

Thus (4.3) holds with `A=A_a`. For a requested scalar error `epsilon>0`, choose

\[
t=\max\left\{1,\log\frac{13H_r(T)A_a}{\epsilon}\right\}.
\tag{5.3}
\]

If `A_a=0`, the derivative is zero and only its anchor polynomial remains; this degenerate case should be handled directly rather than taking `log(0)`. For positive `A_a`, equations (4.9) and (4.10) give

\[
\|q-g\|_\infty\le\epsilon,
\qquad
W_{\rm grid}\le\frac4\pi\left(\frac T a+2\right)t+1.
\tag{5.4}
\]

For fixed target constants, this is `O(log(1/epsilon))` scalar width with an explicit finite halo. The physical halo width remains fixed while its number of centers grows linearly with `t`. The overlap parameter is not fixed: `lambda=gamma h=pi/(2 sqrt(t))`.

This is an exact-arithmetic representation theorem. It does not say that fixed double precision achieves arbitrarily small epsilon.

### Scope of the analytic assumption

Condition (5.1) implies a bounded analytic continuation of `p` to every closed strip narrower than width `a`. Its converse requires suitable decay and integrability, not merely local analyticity on the target interval. In particular, polynomial targets on an interval need not have a globally decaying analytic continuation agreeing with them there. A nonzero polynomial cannot be replaced by an arbitrary decaying analytic continuation without changing its values, by the identity theorem.

Therefore this theorem must be presented as a constructive global Fourier class. It is not a replacement for the paper's local analytic tanh theorem. If an extension or regularization is used to enter this class, its error, cost, and effect on `A_a` must be counted.

## 6. Finite regularity: algebraic scalar width

Instead of (5.1), assume that, for `s>0`,

\[
A_s:=\frac1{2\pi}\int_{\mathbb R}
(1+|\omega|)^s|\widehat p(\omega)|\,d\omega<\infty.
\tag{6.1}
\]

Choose `B>=1`, set

\[
t=s\log(1+B)\ge1,
\tag{6.2}
\]

and use (4.1). The scalar function `log(1+u)/u^2` decreases on `(0,infinity)`: its derivative has the sign of

\[
\frac u{1+u}-2\log(1+u)<0.
\]

For `0<=u<=B`, this implies

\[
\frac{u^2}{B^2}\log(1+B)\le\log(1+u).
\]

Hence the inverse Gaussian multiplier obeys

\[
e^{\omega^2/(2\gamma^2)}
=\exp\left[\frac{s\omega^2}{2B^2}\log(1+B)\right]
\le(1+|\omega|)^{s/2}
\quad (|\omega|\le B).
\]

Consequently `A_rho<=A_s`. Also,

\[
E_{\rm cut}\le(1+B)^{-s}A_s=A_se^{-t}.
\]

Equations (4.9) and (4.10) therefore give

\[
\boxed{
\|q-g\|_\infty\le13H_r(T)A_s(1+B)^{-s},
\quad
W_{\rm grid}\le\frac4\pi\bigl[TB+2s\log(1+B)\bigr]+1.
}
\tag{6.3}
\]

For example, take

\[
B\ge\max\left\{1,e^{1/s}-1,
\left(\frac{13H_r(T)A_s}{\epsilon}\right)^{1/s}-1\right\}.
\]

This yields `O(epsilon^(-1/s))` grid width for fixed positive `T,A_s,s`, with an additional `O(log(1/epsilon))` halo count already included in (6.3). The physical halo width is `2s log(1+B)/B` and tends to zero as resolution increases.

The regularity assumption is on `p=g^(r)`. It must not be restated as merely `g is C^s` with the same exponent. Fourier weighted integrability is a specific regularity class; ordinary derivative assumptions sufficient to imply it have to be stated separately.

## 7. Realizing anchors as a pure one-hidden-layer network

For the Gaussian bump (`r=0`) there is no anchor polynomial. For the Gaussian CDF or erf (`r=1`) there is one output constant.

For exact GELU (`r=2`), (2.10) is affine. The identity

\[
\phi(u)-\phi(-u)=u
\]

follows from `Phi(-u)=1-Phi(u)`. Thus an affine correction `a+b s` is realized by an output bias and two extra GELU neurons, for example `b[phi(s)-phi(-s)]`.

After the Radon profiles are combined, all affine anchors combine into one multidimensional affine function `a+b dot x`. Two GELU neurons `phi(b dot x)` and `phi(-b dot x)` with readouts `+1,-1`, and one output bias, realize it. One does not need two extra affine neurons for every direction. This is an exact constant-factor saving.

The count for the complete GELU network is therefore the sum of grid-neuron counts over directions plus at most two, provided arbitrary first-layer weights for those two anchor units are permitted. There is no added depth or external polynomial evaluation.

## 8. Numerical evaluation is a stated part of the construction

Equation (2.1) is a finite one-dimensional Fourier integral. Computing it does not require a least-squares or global readout solve. It does require access to the profile transform or a certified way to compute it. Neither an integral formula nor a finite error bound supplies this information from arbitrary unqualified point samples.

For an explicit quadrature certificate, define

\[
b(\omega)=\widehat p(\omega)/\widehat K(\omega/\gamma),
\qquad H_c(\omega)=b(\omega)e^{i\omega c}.
\]

If `b` is twice differentiable on `[-B,B]`, and a known number `M_2` bounds `|H_c''(omega)|` uniformly for every retained center, the composite midpoint rule with `Q` equal panels has density error at most

\[
\frac1{2\pi}\frac{(2B)^3M_2}{24Q^2}
=\frac{B^3M_2}{6\pi Q^2}.
\tag{8.1}
\]

One may bound

\[
|H_c''|\le|b''|+2(T+\delta)|b'|+(T+\delta)^2|b|.
\]

Split at known nonsmooth frequency points before using this rule. Analytic integrands permit faster certified quadratures. Equation (8.1) is a concrete sufficient procedure, not a claim that midpoint quadrature is optimal. If neither regularity information nor a certified transform evaluator is supplied, the representation theorem remains mathematically defined but an automatic finite-time certification from black-box values alone is not justified.

If every evaluated density has error at most `eta_rho`, the induced error in the `r`th derivative is at most

\[
h\eta_\rho\sup_{|s|\le T}\sum_{j=-J}^J|K_\gamma(s-jh)|.
\]

For the Gaussian envelope (3.2), split the lattice at the points closest to `s`, bound each one-sided sum by its first term plus an integral, and use (4.2). This gives

\[
h\sum_{j\in\mathbb Z}|K_\gamma(s-jh)|
\le C_K(2\gamma h+2\sqrt\pi)
\le C_K(\pi+2\sqrt\pi)<11.
\]

Thus density integration errors add at most `11 H_r(T) eta_rho` to the function error when anchoring is exact. Anchor-data errors `eta_ell` add `sum_(ell<r) eta_ell T^ell/ell!`. Final coefficient rounding, hidden-weight rounding, and activation evaluation errors require their own numerical allowance; they are not included in (4.9).

## 9. Entering the Radon construction

For normalized sphere measure, a direction `theta` has

\[
\widehat g_\theta(\omega)
=C_d|\omega|^{d-1}\widehat F(\omega\theta),
\qquad C_d=\frac{|\mathbb S^{d-1}|}{2(2\pi)^{d-1}}.
\]

Consequently

\[
\widehat p_\theta(\omega)
=(i\omega)^r C_d|\omega|^{d-1}\widehat F(\omega\theta).
\]

The analytic moment used here is therefore explicitly

\[
A_{a,\theta}
=\frac{C_d}{2\pi}\int_{\mathbb R}
e^{a|\omega|}|\omega|^{d-1+r}|\widehat F(\omega\theta)|\,d\omega.
\tag{9.1}
\]

This exhibits the dimension and frequency dependence hidden if `A` is merely called a constant. A uniform bound in `theta` allows a common scalar design; nonuniform bounds permit different widths and spacings per direction. Positive angular quadrature weights turn scalar errors into `sum alpha_m epsilon_m`, with no factor equal to the number of directions.

Angular analytic approximation still requires separate control of how the profiles depend on `theta`. Radial Fourier exponential moments alone do not prove exponential angular convergence. The angular construction and its error certificate belong in the main note.

## 10. A finite-sample regularization for locally smooth profiles

The following lemma supplies real polynomial approximants without a hidden regression or numerical-integral oracle. It can feed the paper's local analytic tanh encoder when `g` is only finitely smooth on an enlarged real interval.

Let the supplied real interval be `[-L0,L0]`, with `L0>T`, and define the even periodic function

\[
G(\theta)=g(L_0\cos\theta).
\]

For an integer `n>=1`, define the Fejer kernel by its finite Fourier series

\[
F_{n-1}(\theta)=\sum_{|k|<n}\left(1-\frac{|k|}{n}\right)e^{ik\theta}.
\]

It is nonnegative and its normalized integral is one. Define the de la Vallee Poussin kernel

\[
V_L(\theta)=2F_{2L-1}(\theta)-F_{L-1}(\theta),\qquad L\ge1.
\]

Its Fourier multiplier equals one for `|k|<=L`, decreases linearly to zero between `L` and `2L`, and vanishes at `|k|>=2L`. Its degree is `2L-1`.

Choose an integer `Q>=3L`, for example `Q=4L`, and sample

\[
\theta_j=2\pi j/Q,\qquad G_j=g(L_0\cos\theta_j),\quad0\le j<Q.
\]

Define the finite trigonometric polynomial

\[
(\mathcal V_{L,Q}G)(\theta)
=\frac1Q\sum_{j=0}^{Q-1}G_j V_L(\theta-\theta_j).
\tag{10.1}
\]

There are two exact facts about this sampled operator:

1. Each discrete Fejer average has norm at most one. Nonnegativity gives this after noting that its discrete kernel weights sum exactly to one; the latter follows because `Q` exceeds the kernel degree and every nonconstant Fourier mode has zero discrete average. Hence `||V_(L,Q)||_(infinity->infinity)<=3`.
2. The operator reproduces every trigonometric polynomial of degree at most `L`. To check this, apply it to `exp(i ell theta)` with `|ell|<=L`. In the kernel expansion, a discrete average can be nonzero only if `ell-k` is a multiple of `Q`. But `|ell-k|<=3L-1<Q`, so only `ell=k` survives. Its kernel multiplier is one.

If `P` is any trigonometric polynomial of degree at most `L`, then

\[
G-\mathcal V_{L,Q}G
=(G-P)-\mathcal V_{L,Q}(G-P),
\]

so

\[
\|G-\mathcal V_{L,Q}G\|_\infty\le4\|G-P\|_\infty.
\]

The sampled function is even because the grid, data, and kernel are even under index reversal. It therefore corresponds to a real algebraic polynomial `p_L(s)` of degree at most `2L-1` on `[-L0,L0]`, through `s=L0 cos(theta)`. Thus

\[
\boxed{
\|g-p_L\|_{L^\infty([-L_0,L_0])}\le4E_L(g),\qquad
\|p_L\|_{L^\infty([-L_0,L_0])}\le3\|g\|_\infty,
}
\tag{10.2}
\]

where `E_L(g)` is the best uniform degree-`L` algebraic-polynomial approximation error on that interval. The polynomial is computed from `4L` real samples with a Fourier/cosine transform and explicit coefficient multipliers. This is not least squares. No quadrature error is hidden in (10.2); its reproduction and norm bounds apply directly to the finite sampled operator.

If the real samples have uniform error `eta`, the resulting polynomial changes by at most `3 eta` on the real interval. On a Bernstein ellipse with parameter `rho>1`, elementary Chebyshev coefficient bounds give

\[
\sup_{\mathcal E_\rho([-L_0,L_0])}|p_L|
\le(4L-1)\|p_L\|_{[-L_0,L_0]}\rho^{2L-1}.
\tag{10.3}
\]

Indeed, the constant Chebyshev coefficient is bounded by the real sup norm, each other coefficient by twice that norm, and `|T_k(z/L0)|<=rho^k` on the ellipse. Summing at most `2L-1` nonconstant coefficients gives the displayed looser bound.

If a known real regularity bound implies `E_L(g)<=C L^(-s)`, choose `L=O(epsilon^(-1/s))`. The polynomial is entire; it can be passed to the local analytic tanh construction. Its complex norm grows at most exponentially in `L` on a fixed reserved ellipse. In a *joint* tanh parameter certificate that costs linearly in `log(B_poly/epsilon)`, this gives `n,R=O(L+log(1/epsilon))`. This final inference depends on that joint certificate, including its lambda and halo conditions; the fixed-lambda theorem alone does not prove it. Coefficient sizes and working precision may also grow with `L` and must be recorded.

## 11. Why there is no common geometric rate for every activation

A quantitative counterexample is a one-hidden-layer ReLU network approximating `g(x)=x^2` on `[-1,1]`. With `W` ReLU units, its restriction to this interval has at most `W` breakpoints and hence an affine segment of length at least `2/(W+1)`.

On any interval `[a,b]` of length `ell`, let `m0=(a+b)/2`. An affine function `l` satisfies `l(a)+l(b)-2l(m0)=0`, whereas

\[
a^2+b^2-2m_0^2=\ell^2/2.
\]

If `|x^2-l(x)|<=E` throughout the interval, the absolute value of the difference of those expressions is at most `4E`. Thus `E>=ell^2/8`. Apply this to the longest affine segment to obtain

\[
\boxed{
\|x^2-q_W\|_{L^\infty([-1,1])}
\ge\frac1{2(W+1)^2}.
}
\tag{11.1}
\]

The target is entire, but logarithmic width is impossible for this architecture and activation. The same restriction-to-a-line argument applies to a higher-dimensional one-hidden-layer ReLU network. Conversely, a uniform linear-spline interpolant gives error at most `||g''|| infinity h^2/8` and has explicit slope-jump ReLU coefficients, so algebraic approximation is constructive and useful.

A tent activation has three slope-break locations per neuron, so the same argument gives at least `1/[2(3W+1)^2]` error for `x^2`. A fixed-degree polynomial activation is still more restrictive: every width produces a polynomial of at most that degree, so some targets cannot converge at all.

Fourier zeros do not automatically disqualify a kernel. A finite-band construction can keep its used band below the first zero by choosing `gamma`; the resulting width and alias rate must then be derived. What is impossible is to divide through a required zero without another argument.

The appropriate general result is therefore the explicit certificate (2.11), followed by verified activation-specific rates. The Gaussian-family theorems prove geometric or algebraic rates for their stated target classes. The tanh paper supplies a different local analytic boundary construction. ReLU and tents have algebraic limits even for analytic targets. These cases cannot truthfully be collapsed into one activation-independent logarithmic theorem.

