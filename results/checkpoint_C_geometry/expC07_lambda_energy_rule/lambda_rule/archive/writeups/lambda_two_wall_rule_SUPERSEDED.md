> **SUPERSEDED (2026-09-08).** Kept as the record of what was claimed. Do not cite. Withdrawn per the skeptic review: Theorem C as a bound and 'E* is the guaranteed accuracy' (violated in 70% of expC10 cells, up to 1e7; the band-edge hypothesis fails for non-band-limited targets); the (1-q/2) law and the decades-lost formula (do not reproduce from the frozen rule; lambda*(q) is non-monotone); the fp32 constants (eps convention inconsistent with the code); '46 of 46 tone cells' (54 of 76); 'walls at 1.3x' (swish 1.6x). Survives: the two-wall argmin as a LOCATION rule (regret), the resolved-limit reduction to A_K = eps, the sensitivity to c and to the band edge. The hardened statement is `../../hardened_rule.md`.

# The two-wall rule for $\lambda$: theory

**Status: theory note, 2026-09-08, not yet tested on held-out targets.** Everything below is derived; the one number that is not is the constant $c$ in the computational bound, whose influence on the rule is shown to be small. Companion: `lambda_rule_theory.md` (the fiber theorem, imported here as Theorems A1 and A2). The verification plan is at the end.

## 0. The question and the answer in one paragraph

We want to choose the dimensionless bandwidth $\lambda=\gamma h$ of a frozen uniform-grid network before seeing the target, from the activation, the width $N$, the working precision $\varepsilon$, and at most an order-of-magnitude statement about the target's highest frequency. The computed relative error of a least-squares fit is bounded by the sum of two explicit functions of $\lambda$: an aliasing term that rises steeply as $\lambda$ grows, and a computational term that rises as $\lambda$ shrinks. Both are values of the activation's kernel transform $\widehat K$, the first at the ghost frequency $(2\pi-\theta_B)/\lambda$ and the second at the signal frequency $\theta_B/\lambda$, where $\theta_B=2\Omega/N$ is the target's band edge in grid units. The rule is to minimize the sum. For a well-resolved target the second term is flat and the rule reduces to the activation constant $A_K(\lambda)=\varepsilon$ (tanh $0.244$, gelu $0.699$, swish $0.445$ in fp64). For a target near the grid's Nyquist frequency the two walls close in and the rule moves left as $(1-q/2)$ with $q=\theta_B/\pi$, and the achievable error rises by a predicted number of decades.

## 1. Setting

Grid units $u=x/h$, $h=2/N$, centers on the integers (interior $0..N$ plus a halo of $R$ on each side, $W=N+2R+1$ neurons), one bandwidth $\lambda$, neuron $\psi(\lambda(u-k))$. The activation $\psi$ has kernel order $r$ and kernel $K=\psi^{(r)}$ (tanh $r=1$, $K=\mathrm{sech}^2$; gelu and swish $r=2$). $\widehat K$ is normalized so that $\widehat K(0)=1$ and is assumed non-increasing on $[\pi/\lambda,\infty)$ (hypothesis $(\mathrm M_\lambda)$ of the theory doc, true for all kernels used here). Frequencies: physical $\omega$ (rad per unit $x$), grid $\theta=\omega h$; Nyquist is $\theta=\pi$, the first grid harmonic $2\pi$.

The target $f$ is band-limited in grid units to $|\theta|\le\theta_B$ with $\theta_B<\pi$, $\theta_B=h\Omega=2\Omega/N$. For a target with content up to $\sin(k\pi x)$, $\Omega=k\pi$ and

$$q:=\frac{\theta_B}{\pi}=\frac{2k}{N},\qquad \frac{2}{q}=\frac{N}{k}=\text{grid cells per shortest wavelength}.$$

The user's order-of-magnitude knowledge of the frequency enters only through $\theta_B$; Section 6 shows how little a factor-of-three error in it matters.

The generator spectrum of order $r$ is

$$w_r(\theta)=\frac{\lambda^{r-1}\,\widehat K(\theta/\lambda)}{(i\theta)^r}.$$

By Lemma 1 of the theory doc, the transform of any network output is $\hat g(\theta)=\hat a(\theta)\,w_r(\theta)$ with $\hat a$ the $2\pi$-periodic transform of the coefficient sequence. This single identity is the source of both walls: the same $\widehat K$ that fixes how much ghost the network must emit at $\theta+2\pi m$ per unit of signal at $\theta$ also fixes how large the coefficients must be to emit a unit of signal at $\theta$.

Two errors are separated throughout: the approximation error of the ideal projection in exact arithmetic (Part A), and the additional error of the finite-precision solve (Part B). A smaller Part A never implies a smaller Part B.

## 2. Part A: the aliasing wall (theorem)

**Theorem A1 (fiber projection; Theorem 1 of the theory doc).** For $f\in L^2(\mathbb R)$ the least-squares distance to the network class is

$$\operatorname{dist}(f,W_\lambda)^2=\frac1{2\pi}\int_{-\pi}^{\pi}\Big(\|F_\theta\|^2-\frac{|\langle F_\theta,\mathrm W_\theta\rangle|^2}{\|\mathrm W_\theta\|^2}\Big)d\theta,\qquad F_\theta=(\hat f(\theta+2\pi m))_m,\ \mathrm W_\theta=(w_r(\theta+2\pi m))_m .$$

**Theorem A2 (exact floor for band-limited targets; Theorem 2 there).** If $\operatorname{supp}\hat f\subset[-\theta_B,\theta_B]$, then

$$\frac{\operatorname{dist}(f,W_\lambda)^2}{\|f\|^2}=\frac{\int|\hat f(\theta)|^2\,\Lambda_r(\theta;\lambda)\,d\theta}{\int|\hat f(\theta)|^2\,d\theta},\qquad \Lambda_r(\theta;\lambda)=\frac{S(\theta)}{1+S(\theta)},\quad S(\theta)=\sum_{m\ne0}t_m(\theta)^2,$$

$$t_m(\theta)=\frac{\widehat K\big((\theta+2\pi m)/\lambda\big)}{\widehat K(\theta/\lambda)}\Big(\frac{\theta}{\theta+2\pi m}\Big)^{r}.$$

$t_m$ is the output amplitude the network must place at the ghost frequency $\theta+2\pi m$ per unit of output at $\theta$: the kernel ratio times the integration factor $(\theta/(\theta+2\pi m))^r$, because the output is the $r$-th integral of the bump sum. Least squares then minimizes $|1-y|^2+S|y|^2$ over the signal amplitude $y$, giving $S/(1+S)$. For a single tone $e^{i\theta u}$ on the periodic domain this is an equality, verified to four digits in expC08 on 36 cells per activation.

**Lemma A3 (monotonicity in frequency).** Under $(\mathrm M_\lambda)$, for $0<\theta<\pi$ the dominant ghost ratio $t_{-1}(\theta)$ is increasing in $\theta$: the ghost sits at $2\pi-\theta$, which approaches the Nyquist point as $\theta$ grows, so $\widehat K((2\pi-\theta)/\lambda)$ grows, while $\widehat K(\theta/\lambda)$ shrinks and $\theta/(2\pi-\theta)$ grows. Hence $\Lambda_r(\cdot;\lambda)$ is increasing on $(0,\pi)$ up to the $m\ne\pm1$ terms, which are below $10^{-6}$ of the $m=-1$ term for $\lambda\le1$ (Corollary 2.2 of the theory doc).

**Corollary A4 (band-edge bound).** For every target band-limited to $\theta_B$,

$$E_{\text{alias}}:=\frac{\operatorname{dist}(f,W_\lambda)}{\|f\|}\ \le\ E_A(\lambda):=\Lambda_r(\theta_B;\lambda)^{1/2},$$

with equality for a tone at the band edge. $E_A$ is the aliasing wall. Targets whose energy sits below the band edge do better, by an amount Theorem A2 computes exactly if the spectrum is known; the rule does not assume it is.

**Leading order.** For $\theta_B\ll\pi$ and $\widehat K(\xi)\approx c_K\xi^{q_K}e^{-a_K\xi^{p_K}}$ in the tail,

$$E_A(\lambda)\approx\sqrt2\,A(\lambda)\Big(\frac{\theta_B}{2\pi}\Big)^{r}\Big(1+O(\theta_B/\lambda)\Big),\qquad A(\lambda)=\frac{|\widehat K(2\pi/\lambda)|}{|\widehat K(0)|}.$$

For tanh ($a_K=\pi/2$, $p_K=1$) the exact ratio is $t_{-1}\approx e^{-\pi(\pi-\theta_B)/\lambda}\cdot(\text{poly})$, i.e. the wall's exponent is $\pi^2(1-q)/\lambda$.

## 3. Part B: the computational wall (bound with one constant)

**What the solver computes.** The readout is the least-squares solution of $A x\approx b$, $A_{ik}=\psi(\lambda(u_i-k))$ on $n$ samples ($n_s$ per cell), by SVD with the cutoff $\tau=\varepsilon\sigma_{\max}$. Two effects: backward error of the solve, and truncation of directions below the cutoff.

**Proposition B1 (backward stability).** The computed coefficients $\hat x$ are the exact least-squares solution of a perturbed problem $(A+\delta A)x\approx b+\delta b$ with $\|\delta A\|\le c_1\varepsilon\|A\|$, $\|\delta b\|\le c_1\varepsilon\|b\|$ (Higham, *Accuracy and Stability of Numerical Algorithms*, Ch. 20, for SVD and QR least squares). Writing $A\hat x=(A+\delta A)\hat x-\delta A\hat x$ and $(A+\delta A)\hat x=P_{A+\delta A}(b+\delta b)$,

$$\|A\hat x-P_Ab\|\ \le\ \|\delta A\|\|\hat x\|+\|\delta b\|+\|(P_{A+\delta A}-P_A)b\|.$$

The last term is the rotation of the range by the perturbation acting on $b$; by Wedin's theorem it is at most $O(\varepsilon\kappa)\|r\|$ with $r=b-P_Ab$ the exact residual, and since $\kappa\lesssim1/\varepsilon$ after truncation this is $O(\|r\|)$, i.e. at most a constant times the aliasing error of Part A. Therefore

$$\frac{\|A\hat x-P_Ab\|}{\|b\|}\ \le\ c\,\varepsilon\Big(\frac{\|A\|\,\|\hat x\|}{\|b\|}+1\Big)+O(E_{\text{alias}}).$$

**Lemma B2 (size of the coefficients).** By Theorem A1 the projection's coefficients are $\hat a(\theta)=\langle F_\theta,\mathrm W_\theta\rangle/\|\mathrm W_\theta\|^2$, and for a band-limited target $|\hat a(\theta)|\le|\hat f(\theta)|/|w_r(\theta)|$. Under $(\mathrm M_\lambda)$, $1/|w_r(\theta)|=|\theta|^r/(\lambda^{r-1}\widehat K(\theta/\lambda))$ is increasing on $(0,\theta_B]$, so by Parseval

$$\|\hat x\|\ \le\ \|f\|_{\text{cells}}\cdot\frac{\theta_B^{\,r}}{\lambda^{r-1}\,\widehat K(\theta_B/\lambda)} .$$

This is the deconvolution formula of expA07 (interior readout law $v_k\approx\tfrac h2f'(c_k)$ at low frequency, where $\widehat K\approx1$), now with its $\lambda$ dependence explicit: representing content at grid frequency $\theta$ costs a factor $1/\widehat K(\theta/\lambda)$ in coefficient size.

**Lemma B3 (size of the matrix).** $\|A\|_2\le\|A\|_F\le\sqrt{nW}\,\max|\psi|$ over the sampled arguments. For tanh $\max|\psi|=1$; for gelu and swish the far-halo neurons are ramps of size $\lambda R$, which is absorbed into $c$ below (it is a polynomial factor, harmless against the exponentials). With $n=n_sW$ and $\|b\|=\sqrt{n_s}\,\|f\|_{\text{cells}}$ the factors of $n_s$ cancel.

**Proposition B4 (the computational wall).** Combining B1 to B3,

$$E_{\text{comp}}\ \le\ E_C(\lambda):=c\,\varepsilon\Big(1+W\,\frac{\theta_B^{\,r}}{\lambda^{r-1}\,\widehat K(\theta_B/\lambda)}\Big),$$

with $c$ an order-one constant collecting $c_1$, the Frobenius overestimate, and the activation's $\max|\psi|$. The truncation effect has the same shape: gelsd drops the fiber at $\theta$ once its Gram singular value falls below $\varepsilon\sigma_{\max}$, i.e. once $\widehat K(\theta/\lambda)$ is of order $\varepsilon\,W$ times polynomial factors, which is where the term above reaches order one. Either way the exponent is $+a_K(\theta_B/\lambda)^{p_K}$: for tanh $e^{+\pi\theta_B/(2\lambda)}$.

**What is and is not established here.** B1 and B2 are theorems (in the idealized whole-line or periodic $L^2$ setting for B2, as in Part A). B3 is a crude bound. The value of $c$ is not derived; from today's cells $c\approx10$ reproduces the measured left walls of rational, step and bump targets within a factor 1 to 3, and $c=1$ is low by 3 to 6. Section 6 shows that $c$ between 1 and 100 moves the rule by less than 10%. The distinction between roundoff amplification and truncation loss is not resolved by the data (both have the same exponent) and is not needed for the rule.

## 4. The rule

**Theorem C (the bound and the rule).** For every target band-limited to $\theta_B$ in grid units, the computed relative $L^2$ error of the least-squares fit satisfies, up to the constants above,

$$E(\lambda)\ \le\ B(\lambda):=E_A(\lambda)+E_C(\lambda)=\Lambda_r(\theta_B;\lambda)^{1/2}+c\,\varepsilon\Big(1+W\frac{\theta_B^{\,r}}{\lambda^{r-1}\widehat K(\theta_B/\lambda)}\Big).$$

$E_A$ is increasing in $\lambda$ and $E_C$ is decreasing in $\lambda$ (both under $(\mathrm M_\lambda)$), so $B$ has a unique minimizer $\lambda^\star$, and the bound at the minimizer is the guaranteed accuracy:

$$\lambda^\star(K,r,N,R,\Omega,\varepsilon)=\arg\min_\lambda B(\lambda),\qquad E^\star=B(\lambda^\star).$$

The rule needs no least-squares solve and no target: a one-dimensional minimization of two explicit functions. What it needs from the user is $\theta_B=2\Omega/N$, i.e. the highest frequency worth resolving.

*Tightness.* $E_A$ is attained by the band-edge tone (Theorem A2), so the aliasing side of the bound cannot be improved without knowing the spectrum. $E_C$ is tight up to $c$ on the cells measured so far. $\lambda^\star$ therefore lies between the measured argmin and the measured wall in every tone cell tested today (46 cells, three activations, $q$ from $0.125$ to $0.78$), which is a consistency check, not the test (Section 7).

**Margin.** The bound is asymmetric: $E_A$ is exponential in $1/\lambda$ with a large rate, $E_C$ is flat or gently rising to the left. A misjudged $c$ or $\theta_B$ costs little to the left of $\lambda^\star$ and a great deal to the right. A practical choice is the largest $\lambda$ with $E_A(\lambda)\le E_C(\lambda)/\mu$; one decade of margin ($\mu=10$) moves tanh's $\lambda$ from $0.31$ to $0.29$. The $\varepsilon$-constant of Section 5 corresponds to about three decades of margin, which is why it has never been caught on the wrong side.

## 5. Closed forms and limits

**The resolved limit.** As $\theta_B\to0$, $E_C\to c\varepsilon(1+W\theta_B^r/\lambda^{r-1})$, flat in $\lambda$ except for the polynomial $\lambda^{1-r}$, so the minimizer is where the aliasing wall reaches that floor. Ignoring the attenuation factor gives the conservative constant

$$A_K(\lambda_{\text{act}})=\varepsilon:\qquad\begin{array}{l|cccccc}&\tanh&\text{sigmoid}&\text{gelu}&\text{swish}&\mathrm{sech}^2&\text{gaussian}\\\hline\text{fp64}&0.244&0.488&0.699&0.445&0.244&0.523\\\text{fp32}&0.485&0.970&0.984&0.834&0.485&0.770\end{array}$$

The measured walls of resolved targets sit at $1.3\times$ these values (expC08: tanh $0.31$–$0.35$, gelu $0.83$–$0.94$, swish $0.68$–$0.77$); the factor is the omitted $\sqrt2(\theta_B/2\pi)^r$ attenuation and the floor being $\sim50\varepsilon$ rather than $\varepsilon$. So the constant is on the flat side of the wall by construction, for every activation and precision, which is the content of "the original rule was not lucky".

**When the constant is enough.** $E_C$ is flat in $\lambda$ as long as $\widehat K(\theta_B/\lambda_{\text{act}})\approx1$, i.e. $\theta_B\lesssim\lambda_{\text{act}}$. For tanh this is $q\lesssim0.08$, about 25 cells per shortest wavelength; at 8 cells per wavelength ($q=0.25$) the left wall has begun to tilt but $\lambda^\star$ has moved only 12% (next paragraph) and the constant is within a factor 2–3 of optimal (expC08 regret table).

**The $(1-q/2)$ law (tanh, leading order).** Equating the exponents of the two walls,

$$\frac{\pi(\pi-\theta_B)}{\lambda}=L'-\frac{\pi\theta_B}{2\lambda},\qquad L'=\ln\frac{1}{c\,\varepsilon\,W}+\text{(log corrections)},$$

gives

$$\lambda^\star\approx\frac{\pi^2(1-q/2)}{L'},\qquad\text{i.e.}\qquad \frac{\lambda^\star(q)}{\lambda^\star(0)}\approx1-\frac q2 .$$

The numeric minimizer of $B$ reproduces this to three digits for tanh ($q=0.25$: $0.873$ vs $0.875$; $q=0.5$: $0.745$ vs $0.750$; $q=0.75$: $0.618$ vs $0.625$). For gelu and swish ($r=2$) the polynomial prefactors are not negligible and the numeric rule falls faster ($0.69$ and $0.65$ at $q=0.5$); use the numeric minimizer for them.

**Why the optimum moves so little, and why gelu is different.** For a pole-type kernel the aliasing exponent $\pi(\pi-\theta_B)/\lambda$ and the amplification exponent $\pi\theta_B/(2\lambda)$ share the rate $\pi/2$: buying aliasing margin by lowering $\lambda$ raises the amplification by the same exponential, so the crossing drifts only as $(1-q/2)$ while the floor between the walls rises. For a Gaussian-tailed kernel the amplification exponent is $\theta_B^2/(2\lambda^2)$, quadratic in $\theta_B$ and negligible until $q$ is large, so gelu's optimum follows its aliasing wall further and its floor rises less. This is the mechanism behind the expC08 observation that the frequency-resolved rule (aliasing only) worked for gelu and failed for tanh and swish.

**Decades of precision lost to under-resolution (tanh, leading order).** At the crossing, $E^\star\approx E_C(\lambda^\star)\approx c\varepsilon W\theta_B\,e^{\pi\theta_B/(2\lambda^\star)}$, and with $\lambda^\star$ from the law above,

$$\log_{10}\frac{E^\star}{\varepsilon}\ \approx\ \log_{10}\frac1\varepsilon\cdot\frac{q}{2-q}+\text{const},$$

about $16\,q/(2-q)$ decades in fp64: $2.3$ at $q=0.25$, $5.3$ at $q=0.5$, $9.6$ at $q=0.75$. The measured minima of the $q=0.5$ tone cells are $10^{-10}$ to $10^{-11}$ against a resolved floor of $10^{-14}$ to $10^{-15}$, i.e. 4 to 5 decades. For gelu the same argument gives $16\,(q/(2-q))^2$ decades at leading order: $1.8$ at $q=0.5$; measured 2 to 3. Under-resolution is therefore not a $\lambda$ problem but a width problem, and the formula says how many neurons it costs to recover each decade.

## 6. Sensitivity of the rule to what we do not know

Both walls are exponential in $1/\lambda$, so every uncertain input enters the rule logarithmically.

- **The constant $c$.** $c=1$ vs $c=100$ moves $\lambda^\star$ by $\pm8\%$ (tanh, $q=0.5$: $0.206$ / $0.220$ / $0.236$).
- **The band edge.** A factor $3$ error in $\Omega$ is a factor $3$ in $q$. For $q\le0.25$ the rule moves by at most $12\%$ over that whole range; the cost of over-stating $\Omega$ is a slightly smaller $\lambda$ on a flat floor, and the cost of under-stating it is bounded by the same $12\%$ plus the decades-lost formula, which is a property of the problem, not of the rule.
- **The halo and the sampling** enter $E_C$ through $W$ and $n_s$ inside a logarithm.
- **The precision** enters through $\varepsilon$ in both the constant and the crossing; fp32 predictions are in the table above and are the sharpest untested prediction of the theory.

## 7. What is proven, what is modeled, what is declared, and how to test it

*Proven* (in the whole-line or periodic $L^2$ setting): the aliasing wall $E_A$ as an upper bound for every band-limited target and an equality for the band-edge tone; the coefficient bound B2; the backward-stability inequality B1.

*Modeled:* the finite interval with a halo behaves as the whole line (measured to 1–3% in the theory doc and to four digits in expC08/expC09 for bounded targets, not proven); the sampled least squares equals the continuous projection (dense sampling); $\|A\|$ by its Frobenius bound; the constant $c$.

*Declared, not derived:* $c=10$; the margin $\mu$; the recipe of representing the target by its band edge (a bound, tight only for edge-dominated targets).

*Not covered:* targets whose extension past the interval matters (the chirp of expC09); the sup norm; the QI interpolant, whose floor is $|1-M(\theta)|\approx2A(\lambda)$ without attenuation (Theorem 3 of the theory doc), so its wall sits left of the least-squares wall and its $\lambda$ is the constant rule directly; higher dimensions.

*The test (to be run as expC10, with nothing above changed afterward):* held-out targets (random band-limited Fourier series with random amplitudes and phases at a declared band edge, random rational functions, the checkpoint-F solutions, with and without noise); kernels not used in deriving anything here (sigmoid, sech$^2$, gaussian); widths to 1024; fp32 arithmetic; the band edge given exactly and misstated by a factor 3 both ways. Score the regret $E(\lambda^\star)/\min_\lambda E$ per cell against the three baselines (expC07 constant, $\varepsilon$ constant, frequency-resolved rule). Pass criteria fixed in advance: median regret $\le2$ and 90th percentile $\le10$ on resolved cells ($q\le0.25$); on under-resolved cells the rule must beat every baseline and predict the measured minimum within a factor 10.
