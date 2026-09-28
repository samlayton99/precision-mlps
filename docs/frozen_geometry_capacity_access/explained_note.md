**Why an accurate frozen tanh network can be difficult to train**

An explanatory rewrite of *Capacity versus accessibility in frozen tanh networks*, September 2026 revised note. Written for the frozen-geometry investigation on 21 September 2026.

The source is [the five-page note](../../papers/optimization_notes/frozen_geometry_capacity_access_note.pdf), also supplied in the attachment dated this conversation. This rewrite preserves its results and experimental specifications, expands the steps connecting them, and separates the main argument from the technical proof of the tanh approximation bound. Equation references such as “source (4)” refer to the original note. Experimental proposals below remain proposals in that source; this document does not report new measurements or authorize additional runs.

**The question and the four-part argument**

Freeze all hidden slopes and centers. Suppose a least-squares readout fits the target accurately, but ordinary gradient descent on the readout fits it slowly. What accounts for that difference?

1. A bounded-slope tanh feature can be approximated accurately by a polynomial. At a sufficiently high polynomial degree, very little of the feature remains outside that polynomial space.
2. Small feature remainders need not remove representational capacity. Readout coefficients can amplify them and cancel other contributions. An accurate prediction may therefore exist while requiring substantial coefficient movement.
3. If the target needs a correction in a direction where those feature remainders respond weakly, reaching the requested accuracy requires substantial coefficient movement. A gradient-flow theorem turns that requirement into a lower bound on time. Separate results address discrete GD.
4. Increasing gamma weakens this particular bounded-slope obstruction. Whether it actually makes the required corrections easier to learn is an empirical question, after controlling readout coordinates and initialization.

The word “tail” will mean the remainder after a polynomial approximation. It never means the spatial ends of the domain or a Fourier tail in this document. No polynomial is fitted as part of ordinary GD: polynomial fitting is a diagnostic for identifying difficult target directions.

This note addresses the difficulty of learning the readout at a fixed geometry. It does not prove that joint training keeps gamma small, or that increasing gamma guarantees fast training. Its polynomial argument operates directly on the finite sample grid; it does not require a whole-line Fourier objective or a residual whose low frequencies have already disappeared.

**Setup: exactly what is fixed and what is trained**

There are $m$ training points $x_i\in[-1,1]$ and $W$ tanh neurons. The physical network is

$$
f_c(x)=c_0+\sum_{j=1}^{W}c_j\tanh(a_jx+b_j).
$$

The real slopes $a_j$ and hidden biases $b_j$ are fixed. Assume $|a_j|\leq\Gamma$ for every neuron, where $\Gamma>0$ is a common upper bound. For a shared positive slope, $a_j=\gamma$ and $\Gamma=\gamma$. For heterogeneous slopes, $\Gamma$ bounds the maximum absolute slope, not their mean or median. A center representation is $b_j=-a_jt_j$, with fixed center $t_j$.

The vector $c=(c_0,c_1,\ldots,c_W)^\top$ contains the physical readout coefficients. A fixed invertible matrix $M\in\mathbb R^{(W+1)\times(W+1)}$ defines optimizer coordinates $\theta$ by $c=M\theta$. Raw readout training uses $M=I$.

Define the normalized physical feature matrix $A\in\mathbb R^{m\times(W+1)}$ and target vector $y\in\mathbb R^m$ by

$$
A_{i0}=\frac1{\sqrt m},\qquad
A_{ij}=\frac{\tanh(a_jx_i+b_j)}{\sqrt m},\qquad
y_i=\frac{f(x_i)}{\sqrt m}.
$$

The matrix seen by the optimizer is $B=AM$. Its normalized prediction is $B\theta$, its residual is $r=B\theta-y$, and its loss is

$$
L(\theta)=\frac12\|r\|^2
=\frac1{2m}\sum_{i=1}^{m}\bigl[f_c(x_i)-f(x_i)\bigr]^2.
\tag{source 1}
$$

Vector norms are Euclidean. A matrix norm without a subscript is its induced operator norm, equal to its largest singular value. The Frobenius norm, written $\|\cdot\|_F$, is the square root of the sum of squared entries. The normalization above makes $\|r\|$ the training RMS error and $\|r\|/\|y\|$ the training relative $L_2$ error. The theorems concern this sampled norm. Error on a separate evaluation grid is an additional measurement.

**Before the four parts: what least squares does and does not establish**

Let $\Pi=BB^\dagger$ be the orthogonal projector onto the column span of $B$, where $B^\dagger$ is the Moore–Penrose pseudoinverse. The vector $\Pi y$ is the best attainable prediction on these samples. It is unique even if the coefficients producing it are not.

Decompose the residual as

$$
B\theta-y=(B\theta-\Pi y)-(I-\Pi)y.
$$

The first term lies in the feature span and the second is perpendicular to it. Consequently,

$$
\|B\theta-y\|^2
=\underbrace{\|(I-\Pi)y\|^2}_{\text{unrestricted span error}}
+\underbrace{\|B\theta-\Pi y\|^2}_{\text{unfinished readout optimization}}.
\tag{source 2}
$$

Multiplying both terms by $1/2$ gives the familiar $L=F+G$ split: $F$ is the least-squares approximation loss of this geometry, and $G$ is the remaining readout optimization gap. In this frozen problem, $F$ is constant during training.

Since $B^\top(I-\Pi)y=0$, the gradient is

$$
\nabla_\theta L=B^\top(B\theta-y)=B^\top(B\theta-\Pi y).
$$

The unattainable part of the target contributes to the loss but cannot drive readout updates. This is the purpose of the identity $B^\top y=B^\top\Pi y$: the readout optimization dynamics are driven by the attainable part. It is not an additional obstruction.

Because $M$ is invertible, $\operatorname{range}(AM)=\operatorname{range}(A)$. Changing $M$ preserves the exact least-squares prediction but changes the dynamics. Finite-precision least-squares routines can nevertheless report different answers when coordinate changes interact with singular-value cutoffs. Exact span and numerically retained span must be distinguished.

**Part 1. Identify what each feature supplies beyond a polynomial**

**1.1. Polynomial projection is a comparison space, not the network's feature span**

For an integer $k\geq0$, form the sampled polynomial matrix

$$
V_k=\begin{pmatrix}
1&x_1&\cdots&x_1^k\\
\vdots&\vdots&&\vdots\\
1&x_m&\cdots&x_m^k
\end{pmatrix}.
$$

The matrix $P_k=V_kV_k^\dagger$ projects onto the sampled polynomials of degree at most $k$. The matrix $Q_k=I-P_k$ removes that polynomial component. Both are $m\times m$ matrices acting on vectors of values across the samples. Numerically, form an orthonormal polynomial basis $U_k$ and apply $P_kv=U_k(U_k^\top v)$; do not form an inverse of a monomial Gram matrix.

The polynomial projector $P_k$ and the feature-span projector $\Pi$ answer different questions. $P_k$ asks what a degree-$k$ polynomial can describe. $\Pi$ asks what this frozen network can describe with unrestricted readouts. Neither space need contain the other.

Apply the polynomial projection column by column:

$$
A=P_kA+Q_kA.
$$

For a tanh column, $P_kA_{:j}$ is its best sampled polynomial fit and $Q_kA_{:j}$ is its approximation remainder. The constant feature has zero remainder for every $k\geq0$.

The cutoff $k$ is not a network width or a training step. If the samples are distinct, $P_{m-1}=I$ and all higher-degree tails vanish. Thus only finitely many nonzero tails can matter on a fixed grid. With repeated sample locations, saturation can happen sooner.

**1.2. Why a small feature remainder produces a weak response**

Let $q\in\mathbb R^m$ be any unit vector orthogonal to sampled degree-$k$ polynomials. Equivalently, $P_kq=0$ and $Q_kq=q$. Then

$$
B^\top q=B^\top Q_kq=(Q_kB)^\top q.
$$

The polynomial part of every column has zero inner product with $q$. Only the remainders can respond to it. Cauchy–Schwarz therefore gives

$$
\|B^\top q\|^2\leq\|Q_kB\|^2.
$$

This statement applies to every direction outside the polynomial space, regardless of whether the target actually needs it. Target relevance enters in Part 3.

**1.3. The tanh-specific bound**

For a nonzero slope magnitude $\gamma$, define

$$
\rho_\gamma=\frac{\pi}{2\gamma}+\sqrt{1+\frac{\pi^2}{4\gamma^2}},
\qquad
\alpha_\gamma^*=\log\rho_\gamma
=\operatorname{arsinh}\!\left(\frac{\pi}{2\gamma}\right).
$$

The number $\rho_\gamma>1$ controls polynomial approximation decay. The exponent $\alpha_\gamma^*$ is unrelated to the readout allowances $\alpha_j$ used later in coordinate maps.

Define the explicit approximation bound

$$
U_k(\gamma)=\frac{4\rho_\gamma^{-k}}{\rho_\gamma-1}
\left[\frac1{\sqrt{\gamma^2+\pi^2/4}}+\frac1{\pi(k+1)}\right],
\qquad
e_k(\gamma)=\min\{\tanh\gamma,U_k(\gamma)\},\qquad e_k(0)=0.
\tag{source 5}
$$

For every real bias, a degree-$k$ polynomial approximates $\tanh(\gamma x+b)$ on all of $[-1,1]$ with error at most $e_k(\gamma)$. A uniform continuous bound implies the same bound on the normalized sampled column: there are $m$ entries, each at most $e_k/\sqrt m$, so its Euclidean norm is at most $e_k$. Orthogonal projection is the best sampled polynomial approximation and can only improve it.

It follows that

$$
\|Q_kA\|^2\leq\|Q_kA\|_F^2
\leq\sum_{j=1}^{W}e_k(|a_j|)^2
\leq W e_k(\Gamma)^2.
$$

Multiplying by the readout map gives Theorem 2:

$$
\boxed{
\|B^\top q\|^2
\leq\|Q_kB\|^2
\leq\|M\|^2W e_k(\Gamma)^2
=:\mathcal B_k,
\quad \|q\|=1,\ P_kq=0.
}
\tag{source 6}
$$

The quantity $\mathcal B_k$ is a structural upper bound on squared response, not its measured value. Its exponential factor is $e^{-2k\alpha_\Gamma^*}$; the explicit prefactors above must also be retained for numerical comparisons. The proof uses tanh's complex poles and is given in Appendix A.

Small slopes make this bound decay faster with polynomial degree. Biases and neuron locations are arbitrary, so the bound is uniform over them. Actual responses can be much smaller because of saturation, particular locations, target alignment, or cancellation. Increasing $\Gamma$ loosens this upper bound; it does not prove monotonic growth of actual response in every direction.

**Part 2. Explain why a good solution can exist despite weak response**

**2.1. What squared access measures**

For any unit correction direction $q$, define

$$
\mu(q)=\|B^\top q\|^2=q^\top Kq,\qquad K=BB^\top\in\mathbb R^{m\times m}.
$$

The matrix $K$ describes how a readout gradient step changes predictions. If the residual equals $q$, one GD step with learning rate $\eta$ changes the parameters and residual by

$$
\Delta\theta=-\eta B^\top q,\qquad \Delta r=-\eta Kq.
$$

The correction along the original unit error is $-q^\top\Delta r=\eta\mu(q)$. This is not generally the norm of the full prediction update; the update can also have components perpendicular to $q$.

For gradient flow, $\dot\theta=-B^\top r$ and $\dot r=-Kr$. Differentiating the squared residual norm gives

$$
\frac{d}{dt}\|r\|^2=2r^\top\dot r=-2\|B^\top r\|^2.
$$

At a time with $r\neq0$, set $q=r/\|r\|$. Then

$$
-\frac{d}{dt}\log\|r\|
=-\frac{1}{2\|r\|^2}\frac{d}{dt}\|r\|^2
=\frac{\|B^\top r\|^2}{\|r\|^2}
=\mu(q).
$$

Thus squared access is the instantaneous relative decay rate of the current residual norm under unit-rate flow. The residual direction generally changes, so a single measured $\mu(q)$ is not a constant rate for the entire trajectory. The time bound in Part 3 handles that issue without assuming the residual direction stays fixed.

**2.2. Small access can mean either missing capacity or weak sensitivity**

Let $\chi=\|\Pi q\|^2$, the fraction of a unit direction's squared norm lying in the feature span. If $\chi>0$, define $\widehat q=\Pi q/\sqrt\chi$, the unit vector in that attainable direction. Because $B^\top q=B^\top\Pi q$,

$$
\boxed{\mu(q)=\chi\,\widehat q^\top K\widehat q.}
\tag{source 3}
$$

The first factor measures how much is representable. The second measures sensitivity within that representable direction. If $\chi=0$, access is zero. Even when $\chi=1$, access can be tiny: the direction is present but weak.

To see what “weak within the span” means, let $u_i$ be an orthonormal eigenbasis of $K$ with eigenvalues $\nu_i\geq0$. The positive $\nu_i$ are the squared nonzero singular values of $B$. For an attainable unit vector $\widehat q$, its response is

$$
\widehat q^\top K\widehat q
=\sum_{\nu_i>0}\nu_i|u_i^\top\widehat q|^2.
$$

It is small when the direction's weight lies mainly on small positive eigenvalues. Near-dependent columns are one way this happens. Small column scales are another. A worst condition number alone does not say whether the target needs those weak directions.

**2.3. The precise coefficient cost**

For any coefficient change $\Delta\theta$,

$$
|q^\top B\Delta\theta|
=|(B^\top q)^\top\Delta\theta|
\leq\sqrt{\mu(q)}\,\|\Delta\theta\|.
$$

The left side is output movement along the chosen unit direction. Small access limits that output per unit coefficient movement. It does not forbid producing the output with sufficiently large coefficients.

The source illustrates this with two columns $p$ and $p+\varepsilon q$, where $q$ is outside the low-degree polynomial space and $\varepsilon$ is small. Their difference divided by $\varepsilon$ is exactly $q$. Thus their span contains $q$, even though each column has only a small remainder in that direction. The required coefficients have magnitude $1/\varepsilon$.

That example demonstrates why “nearly polynomial features” does not imply “a polynomial feature span.” It also does not prove that our least-squares solution has such coefficients; their necessary size must be assessed for the actual target.

**Part 3. Identify an unavoidable correction and derive its training cost**

**3.1. Which target remainder actually matters?**

First consider zero readout initialization, so the initial residual is $r_0=-y$. Define

$$
D_k(f)=\frac{\|Q_ky\|}{\|y\|}.
$$

This is the relative best degree-$k$ polynomial error on the training grid. It is a norm ratio; $D_k^2$ is the corresponding energy fraction. For a nonzero tail, define its unit direction $q_k^y=Q_ky/\|Q_ky\|$ and its squared access $\mu_k^y=\|B^\top q_k^y\|^2$.

The target has component $(q_k^y)^\top y=\|Q_ky\|$ along this direction. If a prediction has total error at most $\epsilon\|y\|$, its error along any unit direction is at most that amount. Hence it must produce at least $[D_k(f)-\epsilon]_+\|y\|$ along $q_k^y$, where $[z]_+=\max\{z,0\}$.

If $D_k\leq\epsilon$, leaving this tail unresolved does not itself prevent reaching tolerance. It might still be difficult to fit the rest of the target, but this tail supplies no necessary-correction certificate. If $D_k>\epsilon$, some correction in this direction is unavoidable.

**3.2. Capacity when coefficient size is limited**

Suppose optimizer coordinates must satisfy $\|\theta\|\leq R$, where $R$ is a specified norm budget. For each nonzero target tail,

$$
\begin{aligned}
\|B\theta-y\|
&\geq |(q_k^y)^\top(B\theta-y)|\\
&\geq \|Q_ky\|-|(B^\top q_k^y)^\top\theta|\\
&\geq \|Q_ky\|-R\sqrt{\mu_k^y}.
\end{aligned}
$$

Taking the positive part, dividing by $\|y\|$, and choosing the strongest degree gives

$$
\mathcal E_B(R):=\min_{\|\theta\|\leq R}\frac{\|B\theta-y\|}{\|y\|}
\geq\sup_k\left[D_k(f)-\frac{R\sqrt{\mu_k^y}}{\|y\|}\right]_+.
\tag{source 9}
$$

Reaching relative tolerance $\epsilon$ therefore requires

$$
R\geq\|y\|\sup_k\frac{[D_k(f)-\epsilon]_+}{\sqrt{\mu_k^y}}.
$$

This is a statement about capacity under a coefficient budget. It is not an assertion that the target is absent from the unrestricted span. These are budgets on $\theta$; equal numerical budgets in different maps $M$ represent different physical constraints. For a physical coefficient budget, use $M=I$ or explicitly impose $\|M\theta\|\leq R$. A maximum individual coefficient is not the Euclidean coefficient norm used here.

**3.3. Necessary movement from an arbitrary initialization**

For the time theorem, allow any initial readout $\theta_0$ with nonzero initial residual $r_0=B\theta_0-y$. Define

$$
\delta_k=\frac{\|Q_kr_0\|}{\|r_0\|},\qquad
q_k=\frac{Q_kr_0}{\|Q_kr_0\|},\qquad
\mu_k=\|B^\top q_k\|^2.
$$

Only degrees with nonzero $Q_kr_0$ enter. These are fixed probes computed at initialization, not changing projections of the evolving residual. At zero initialization, $\delta_k=D_k(f)$ and $q_k=-q_k^y$, which leaves squared access unchanged.

Suppose an iterate reaches $\|r(T)\|\leq\epsilon\|r_0\|$, with $0<\epsilon<1$. Because $q_k^\top r_0=\delta_k\|r_0\|$ and $|q_k^\top r(T)|\leq\epsilon\|r_0\|$, it must have changed its residual along this direction by at least $[\delta_k-\epsilon]_+\|r_0\|$. Also $r(T)-r_0=B[\theta(T)-\theta_0]$. Therefore,

$$
\boxed{
[\delta_k-\epsilon]_+\|r_0\|
\leq|q_k^\top B[\theta(T)-\theta_0]|
\leq\sqrt{\mu_k}\,\|\theta(T)-\theta_0\|.
}
\tag{source 10}
$$

This necessary movement statement applies to any successful iterate, whatever algorithm produced it. It does not assume that low-degree residuals were fitted first or that the residual remains polynomial-orthogonal during training.

More generally, the same argument works for any fixed unit probe $q$, with $\delta=|q^\top r_0|/\|r_0\|$ and $\mu=\|B^\top q\|^2$. Polynomial tails are selected because they give a known initial component and a tanh-specific upper bound on its access. A probe does not have to be an eigenvector, and the trajectory is free to rotate away from it.

**3.4. The extra fact needed to turn movement into a GD time bound**

Large required movement alone is not a convergence theorem: an arbitrary algorithm could make a very large jump. We now use the specific dynamics of unit-rate Euclidean gradient flow,

$$
\dot\theta=-B^\top r,\qquad \dot r=-Kr.
$$

Use the orthonormal eigenbasis $Ku_i=\nu_i u_i$ and expand $r_0=\sum_i\beta_i u_i$, where $\beta_i=u_i^\top r_0$. Each residual component then solves a scalar equation:

$$
r(t)=\sum_i\beta_i e^{-t\nu_i}u_i,
\qquad
\|\theta(t)-\theta_0\|^2
=\sum_{\nu_i>0}\beta_i^2\frac{(1-e^{-t\nu_i})^2}{\nu_i}.
\tag{source 11}
$$

The first formula follows from $\dot r=-Kr$. For the second, integrate $\dot\theta=-B^\top r(t)$; the vectors $B^\top u_i$ are mutually orthogonal and have squared norm $\nu_i$. Zero eigenvalues contribute a persistent residual but no coefficient movement.

Define the first hitting time

$$
T_\epsilon=\inf\{t\geq0:\|r(t)\|\leq\epsilon\|r_0\|\}.
$$

If it is finite, continuity gives equality at the first hit. A concavity argument applied to the spectral formula proves

$$
\boxed{
\|\theta(T_\epsilon)-\theta_0\|^2
\leq T_\epsilon\|r_0\|^2\frac{(1-\epsilon)^2}{\log(1/\epsilon)}.
}
\tag{source 12}
$$

Appendix B gives the full concavity calculation. This is the essential additional bound: it limits how much net coefficient displacement gradient flow can have accumulated when it first reaches the requested residual reduction.

Square source (10) and insert source (12). Cancel $\|r_0\|^2$ to obtain Theorem 1:

$$
\boxed{
T_\epsilon\geq
\underbrace{\frac{\log(1/\epsilon)}{(1-\epsilon)^2}}_{C_\epsilon}
\sup_k\frac{[\delta_k-\epsilon]_+^2}{\mu_k}.
}
\tag{source 4}
$$

The numerator is the squared necessary correction in an initial-tail direction, relative to initial residual size. The denominator is its squared readout sensitivity. The supremum chooses the degree giving the strongest lower bound; it does not choose the best polynomial fit, since $P_k$ already supplies the best fit for each degree.

As $k$ grows, $\delta_k$ decreases. Directional access $\mu_k$ need not be monotone because its unit direction changes with $k$. The bound does not automatically improve by taking the largest possible degree. A positive numerator divided by zero means infinite hitting time. A zero numerator contributes zero. If the tolerance is unattainable, the hitting time is infinite independently of whether the selected polynomial probes expose it.

The time is relative to $\|r_0\|$. It equals a target-relative error threshold only for zero readout or another initialization with the appropriate normalization. It is continuous flow time, not a number of optimizer updates. Adam is not covered.

**3.5. Where the tanh theorem enters the time result**

The time theorem is a general statement about a fixed linear least-squares problem. Polynomial projections are one useful choice of probes, and tanh is not required for its proof. The tanh-specific explanation comes from substituting Part 1's structural bound:

$$
\mu_k\leq\|Q_kB\|^2\leq\mathcal B_k
\quad\Longrightarrow\quad
T_\epsilon\geq C_\epsilon\sup_k\frac{[\delta_k-\epsilon]_+^2}{\mathcal B_k}.
$$

The lower bound using measured $\mu_k$ is at least as strong as the one using $\mathcal B_k$. A useful audit measures the entire chain. A large gap between $\mu_k$ and $\mathcal B_k$ means that the slope-only bound does not closely describe the actual response, even if both bounds are correct.

The optimality of $C_\epsilon$ is a separate statement from tightness on our experiments. For two samples $(-1,1)$, a constant feature and one tanh feature, raw coordinates, and $r_0=(-1,1)^\top/\sqrt2$, the residual is an eigenvector with eigenvalue $\mu=\tanh^2\gamma$. Taking $k=0$ gives $\delta_0=1$ and exact hitting time $\log(1/\epsilon)/\mu$, equal to the bound. No larger universal prefactor works. Other spectral distributions can make actual training much slower than the bound.

**3.6. Actual GD steps require their own calculation**

For constant-step GD,

$$
\theta_{n+1}=\theta_n-\eta B^\top r_n,
\qquad 0<\eta\leq\frac1{\|B\|^2},
$$

the residual recurrence is $r_{n+1}=(I-\eta K)r_n$. The chosen step range keeps all spectral factors in $[0,1]$. The exact sampled residual curve is

$$
\boxed{\|r_n\|^2=\sum_i\beta_i^2(1-\eta\nu_i)^{2n}.}
\tag{source 13a}
$$

Define the initial whole-residual access

$$
\mu_0=\frac{\|B^\top r_0\|^2}{\|r_0\|^2}.
$$

Convexity of $\nu\mapsto(1-\eta\nu)^{2n}$ and weights $\beta_i^2/\|r_0\|^2$ give $\|r_n\|^2/\|r_0\|^2\geq(1-\eta\mu_0)^{2n}$. Thus the first successful iteration $N_\epsilon$ obeys

$$
N_\epsilon\geq\left\lceil\frac{\log(1/\epsilon)}{-\log(1-\eta\mu_0)}\right\rceil,
\qquad 0<\eta\mu_0<1.
\tag{source 13b}
$$

Equality is attained for residuals in a single eigenspace, up to integer rounding. A whole-residual average can hide a weak necessary tail. The note therefore also gives a tail-based discrete condition. Summing the GD updates yields

$$
\|\theta_n-\theta_0\|^2
=\sum_{\nu_i>0}\beta_i^2\frac{[1-(1-\eta\nu_i)^n]^2}{\nu_i}.
$$

Bound this by $\|r_0\|^2$ times the largest factor and combine with source (10). Any successful step count must satisfy

$$
\boxed{
\max_{0\leq\nu\leq\|B\|^2}
\frac{[1-(1-\eta\nu)^n]^2}{\nu}
\geq\sup_k\frac{[\delta_k-\epsilon]_+^2}{\mu_k},
}
\tag{source 14}
$$

where the factor at $\nu=0$ is defined by its limit, zero. This is a computable necessary condition on $n$. It is not an exact hitting-time formula. The spectral curve is the exact reference for the frozen sampled GD problem, subject to numerical resolution of its singular modes. Simply dividing the flow bound by a learning rate is not the discrete theorem.

**Part 4. What increasing gamma can change**

**4.1. The bounded-slope obstruction weakens, but success is not guaranteed**

As $\Gamma$ increases, $\alpha_\Gamma^*=\operatorname{arsinh}(\pi/(2\Gamma))$ decreases. At a fixed degree, the structural bound permits larger feature remainders and larger access. It removes some theoretical restriction on progress. It does not guarantee that the actual geometry provides a large response in the target direction, that the best fit improves monotonically, or that other conditioning problems disappear.

For a fixed slope bound $\Gamma_0$ and raw coordinates, the envelope has the form $C(\Gamma_0)W\exp(-2\alpha_{\Gamma_0}^*k)$ with a degree-independent upper prefactor. If a relevant degree is $k=\kappa W$, where $\kappa>0$ is fixed, the response upper bound is exponentially small in width. A readout map whose norm grows only polynomially with width does not cancel that exponential factor.

If instead $\Gamma=cW$ with fixed $c>0$, then

$$
k\alpha_\Gamma^*=\kappa W\operatorname{arsinh}\!\left(\frac{\pi}{2cW}\right)
\longrightarrow\frac{\pi\kappa}{2c}.
\tag{source 15}
$$

The exponential factor no longer decays exponentially with width in this regime. Prefactors must still be checked. For example, the source derives

$$
k+1\geq\Gamma\quad\Longrightarrow\quad
U_k(\Gamma)\leq\frac8\pi\left(1+\frac1\pi\right)e^{-k\alpha_\Gamma^*}.
$$

For a uniform center grid, $\lambda=\gamma h$ is the dimensionless slope. Keeping $\lambda$ fixed while refining $h$ makes $\gamma=\lambda/h$ grow with resolution. It is the appropriate connection to the QI scale, but the theorem does not prove that $\lambda=0.25$ is the optimal training value.

**4.2. Why the target tail must be measured rather than presumed**

The small access denominator alone is not enough for a large time bound. The numerator can also be very small.

To state the source's conditional width argument, suppose the relevant target tails satisfy two-sided bounds $D_k(f)\asymp e^{-\beta k}$, where $\beta>0$ is a target approximation rate. Suppose also that the requested tolerance tightens with width as $\epsilon_W\asymp e^{-sW}$, with $s>0$. Here $\asymp$ means bounded above and below by fixed positive multiples in the regime under discussion. Choose a degree just below that needed for the tolerance, with $D_k\geq2\epsilon_W$. Then $k=(s/\beta)W+O(1)$, provided the sample grid supports these degrees.

At this degree, $[D_k-\epsilon_W]_+^2$ has the same exponential scale as $e^{-2\beta k}$. Dividing by the structural access envelope gives exponential factor

$$
e^{2(\alpha_\Gamma^*-\beta)k},
$$

besides the width, map, and tolerance prefactors. A fixed-$\Gamma$ exponential obstruction follows in this setting when $\alpha_\Gamma^*>\beta$.

A fixed tolerance need not require $k$ to grow with width. Entire targets such as a fixed-frequency sine can have faster-than-geometric polynomial tails, so the assumed two-sided rate need not apply. A quadratic has exactly zero sampled tail for $k\geq2$ in exact arithmetic. These are substantive limits on the explanation, not minor qualifications. The result does not establish failure for every $\Gamma=o(W)$ or necessity of linearly growing slopes for every target. Measure $D_k$, use the actual tolerance, and maximize the actual ratio.

**4.3. Remove a trivial global time rescaling before comparing maps**

Multiplying every feature by a scalar rescales all curvatures and the flow clock. To compare accessibility relative to the fastest direction, define

$$
\widehat\mu(q)=\frac{\mu(q)}{\|B\|^2},\qquad
\widehat{\mathcal B}_k=\frac{\mathcal B_k}{\|B\|^2}.
$$

Multiplying the flow bound by $\|B\|^2$ gives

$$
T_\epsilon\|B\|^2
\geq C_\epsilon\sup_k\frac{[\delta_k-\epsilon]_+^2}{\widehat\mu_k}.
$$

The normalized time is a product, $T\|B\|^2$, not a quotient. GD with $\eta=c/\|B\|^2$ similarly removes a uniform feature scaling, with $0<c\leq1$ for the discrete statements above. Nonuniform coordinate changes can still change the normalized spectrum and target access.

**A second diagnostic: how much coefficient freedom does recovery require?**

**5.1. Define exactly one damped readout step**

The damping probe asks whether a given correction can be made without allowing very large changes in optimizer coordinates. Starting from a fixed residual $r$, choose $\zeta>0$ and solve

$$
\Delta\theta_\zeta
=\arg\min_d\left\{\frac12\|r+Bd\|^2+\frac\zeta2\|d\|^2\right\}.
$$

The vector $d$ is a trial readout change; $\zeta$ is the penalty on its squared norm. Large $\zeta$ discourages coefficient movement. The residual is affine in the readout, so the exact Hessian and Gauss–Newton matrix both equal $B^\top B$. The normal equations give

$$
\Delta\theta_\zeta=-(B^\top B+\zeta I)^{-1}B^\top r,
\qquad
r_\zeta^+=r+B\Delta\theta_\zeta=\zeta(K+\zeta I)^{-1}r.
\tag{source 16}
$$

For a unit probe $q$, define the remaining fraction

$$
\mathcal R(q;\zeta)=\|\zeta(K+\zeta I)^{-1}q\|.
$$

Each damping value is a fresh solve from the same probe. This is not a training trajectory, a sequence of diminishing damping values applied to updated residuals, or a test that learns gamma. An exact undamped least-squares correction recovers the attainable component and leaves only the span error.

**5.2. What the remaining fraction measures**

Write $q=\sum_i\omega_i u_i$ in the eigenbasis of $K$, so $\sum_i\omega_i^2=1$. Then

$$
\mathcal R(q;\zeta)^2
=\sum_i\omega_i^2\left(\frac\zeta{\nu_i+\zeta}\right)^2.
$$

A component with curvature $\nu_i\gg\zeta$ is mostly removed. A component with $\nu_i\ll\zeta$ mostly remains. Damping is therefore a soft cutoff on curvature, not on polynomial degree or Fourier frequency.

The function $x\mapsto\zeta^2/(x+\zeta)^2$ is convex for $x\geq0$. Jensen's inequality gives

$$
\boxed{
\mathcal R(q;\zeta)\geq\frac\zeta{\mu(q)+\zeta}.
}
$$

For a polynomial-orthogonal probe, $\mu(q)\leq\mathcal B_k$, and hence

$$
\boxed{
P_kq=0\quad\Longrightarrow\quad
\mathcal R(q;\zeta)\geq\frac\zeta{\mathcal B_k+\zeta}.
}
\tag{source 17}
$$

Equality in the first bound holds when the probe lies in one eigenspace; it is sharp given only $\mu(q)$. To reduce a unit probe to fraction $\epsilon$, a necessary condition is $\zeta\leq\epsilon\mu(q)/(1-\epsilon)$. It is not sufficient when the probe occupies several curvature scales or has an unattainable component.

Use relative damping $\rho=\zeta/\|B\|^2$. The bound becomes $\mathcal R\geq\rho/(\widehat\mu+\rho)$. Recovery at larger relative damping means the correction can be made despite a stronger penalty relative to the fastest curvature.

As $\zeta\downarrow0$, all positive-curvature components disappear and zero-curvature components remain:

$$
\mathcal R(q;\zeta)\longrightarrow\|(I-\Pi)q\|.
$$

The limiting remainder is missing span. The damping scale required to approach it reflects weak positive modes. An isolated polynomial tail can lie partly outside the feature span even if the full target lies inside it, because polynomial projection need not preserve that span. Test common tail probes, the full target, and the attainable target $\Pi y$. When using $\Pi y$ as a target, state whether its error is normalized by $\|\Pi y\|$ or by the original $\|y\|$.

**Readout coordinate controls: what they change and why they matter**

**6.1. A different map preserves capacity but changes optimization**

For physical gradient $g_c=\nabla_cL$, the chain rule gives $\nabla_\theta L=M^\top g_c$. Ordinary GD in optimizer coordinates produces

$$
\Delta c=-\eta MM^\top g_c.
$$

Thus a diagonal prefactor is squared in the physical GD update. Adam's coordinatewise moment normalization changes this relationship; the displayed GD formula does not describe Adam. The damped probe also depends on the chosen map: $\zeta\|d\|^2$ in optimizer coordinates penalizes a physical step by $\zeta\|M^{-1}\Delta c\|^2$. Equal native damping under different maps is a deliberate change of metric.

These controls ask whether the gamma effect survives changes in how the same function space is parameterized. They do not isolate gamma by making every other spectral property identical; changing gamma itself can alter both capacity and conditioning.

**6.2. The five maps in the note**

Let $\alpha_j>0$ be fixed reference readout allowances from the PR's construction at $\lambda_{\rm ref}=0.25$, with $\alpha_0$ the separate output-bias allowance. They are unrelated to the slopes $a_j$. Set $D=\operatorname{diag}(\sqrt{\alpha_0},\ldots,\sqrt{\alpha_W})$.

For neighboring coordinates, order neurons by center and let $L$ be the $W\times W$ lower-bidiagonal difference matrix, with diagonal entries $1$ and subdiagonal entries $-1$. If $d$ contains cumulative physical hidden readouts, then $c_{1:W}=Ld$ and $d_j=\sum_{\ell\leq j}c_\ell$. Define $s_j=\sum_{\ell\leq j}\alpha_\ell$, excluding the bias.

| Map | Physical hidden readout | Physical output bias |
|---|---|---|
| Raw | $c_j=\theta_j$ | $c_0=\theta_0$ |
| Collective | $c_j=\sqrt{\alpha_j}\theta_j$ | $c_0=\sqrt{\alpha_0}\theta_0$ |
| Individual | $c_j=\alpha_j\theta_j$ | $c_0=\alpha_0\theta_0$ |
| Collective neighboring | $c_{1:W}=L\operatorname{diag}(\sqrt{s_j})\theta_{1:W}$ | $c_0=\sqrt{\alpha_0}\theta_0$ |
| Individual neighboring | $c_{1:W}=L\operatorname{diag}(s_j)\theta_{1:W}$ | $c_0=\alpha_0\theta_0$ |

Neighbor features are adjacent differences $\phi_j-\phi_{j+1}$, plus the final tanh anchor $\phi_W$, where $\phi_j(x)=\tanh(a_jx+b_j)$. The anchor preserves invertibility and the full function space. Scale the cumulative coefficients before differencing. Scaling the two tanhs unequally after differencing would destroy their tail cancellation. Do not apply the individual-neuron scales again on top of the cumulative scales.

Ordinary allowances are of order $h$ in the fixed reference construction; corrected outer halos and the bias have different allowances. Their exact recipe is in the implementation supplement below. These are fixed reference scales, not guaranteed optimal training scales or bounds on every target's learned coefficients. The map is kept fixed while gamma is swept.

**6.3. Initialization is a separate ablation**

The note proposes three physical hidden-readout starts, with output bias zero:

$$
c_{j,0}=0,\qquad
c_{j,0}=\sqrt{\alpha_j}\sigma_X\xi_j,\qquad
c_{j,0}=\alpha_j\sigma_X\xi_j,
\qquad \sigma_X=\sqrt{\frac2{W+1}},\quad \xi_j\sim N(0,1).
$$

Reuse the same Gaussian draws and the same physical readout across maps, setting $\theta_0=M^{-1}c_0$. Drawing independent optimizer coordinates in each map changes the physical initial network and confounds the comparison. The PR's individual-coordinate comparison re-encoded its collective physical start; the independent alpha-Xavier start is a distinct proposed experiment. A signed-envelope start $\alpha_j\operatorname{sign}(\xi_j)$ is another optional start and is not Gaussian Xavier.

For nonzero starts, the time certificate uses $r_0$, not $y$. For zero starts, the required target tails are shared across gamma and maps. Selected bias and halo scaling controls test whether those special columns dominate the chosen metric.

**What must be tested for the explanation to matter**

**7.1. The quantities to compute on each frozen problem**

Fix a target, samples, geometry, map, initial physical readout, and accuracy tolerance. For each nonzero initial-residual tail, compute

$$
\delta_k,\quad q_k,\quad
\mu_k=\|B^\top q_k\|^2,\quad
\|Q_kB\|^2,\quad
\mathcal B_k=\|M\|^2We_k(\Gamma)^2.
$$

Report their curvature-normalized counterparts where appropriate. For zero starts, also report $D_k(f)=\delta_k$. The mathematical chain to check is

$$
\mu_k\leq\|Q_kB\|^2\leq\mathcal B_k.
$$

The first inequality can be loose because the target direction need not be the strongest direction of the remainder matrix. The second can be loose because the uniform slope bound covers arbitrary dictionaries and samples. Measuring both gaps shows where the explanatory loss occurs.

| Claim to assess | Measurement | What would limit the explanation? |
|---|---|---|
| The target needs the difficult correction | $[\delta_k-\epsilon]_+$ at the actual tolerance | The supposedly difficult tails are already below tolerance. |
| Bounded slopes restrict useful response | Directional access, operator remainder norm, structural envelope across gamma | The structural bound is much too large to explain the measured weakness, or the chosen tails do not show a relevant restriction. |
| The weakness imposes a substantial training delay | Measured-access and structural time bounds versus the exact flow hitting time; discrete conditions versus GD | Valid lower bounds lie far below the actual delay. |
| The effect is optimization within a useful span | Span error, coefficient-budget fits, full-target and projected-target checks | A tested correction is principally unattainable, so the result is capacity failure rather than slow attainable recovery. |
| Larger gamma improves practical fitting after controls | Independent-grid GD/Adam error at fixed budgets across maps and starts | The gamma advantage vanishes after reasonable coordinate/rate controls, or occurs only in isolated unstable states. |
| Recovery becomes easier, not merely larger in absolute units | Fixed-probe recovery versus relative damping, with step norms and span limits | Apparent recovery changes are explained by a global scaling, changing probes, or changing numerical cutoffs. |

One can confirm the inequalities without establishing that this mechanism dominates our optimization gap. Conversely, a weak certificate limits what this argument demonstrates; it does not by itself disprove a true theorem or identify another mechanism. The purpose is to measure the size of the explanation.

**7.2. Figures that expose the important comparisons**

**Necessary correction versus access.** Put polynomial degree $k$ on the horizontal axis. Plot $[D_k-\epsilon]_+^2$ and normalized measured access for several frozen gammas, with normalized structural envelopes. The quantities are dimensionless but have different meanings. Their ratio, multiplied by $C_\epsilon$, is the time certificate. Plot the ratio separately or identify its maximizing degree; a logarithmic vertical gap represents a ratio, not an area or a training stage. Include the tolerance and target in the caption. Do not turn numerically unresolved values into zeros.

**Does the bound account for the delay?** Put frozen gamma on the horizontal axis and normalized flow hitting time $T_\epsilon\|B\|^2$ on the vertical axis. Compare the exact spectral hitting time with the measured-direction lower bound and structural lower bound at the same tolerance and initialization. Show unattainable or unresolved cases explicitly. Separately compare exact discrete GD curves with actual GD and source (14). Do not put flow time on an axis labeled “steps.”

**Damping recovery.** Put $\rho=\zeta/\|B\|^2$ on the horizontal axis and unit-probe remainder $\mathcal R$ on the vertical axis. Every point comes from one solve from the same probe. Moving toward smaller damping permits larger coefficient changes and reduces the remainder. Curves that recover at larger damping indicate easier normalized access. Plot the limiting span error and record step norms. Identify the probe, including its degree if it is a polynomial tail.

**Practical optimizer comparison.** Plot independent-grid relative $L_2$ error against gamma or lambda at equal update budgets, together with error-versus-step trajectories. Separate optimizer and coordinate-map effects, retain final and late-window summaries, and use least-squares fits to diagnose capacity. Distance to the numerical least-squares floor is not the sole success criterion: a reproducible gain in attainable digits at a practical budget is meaningful even when neither run reaches that floor.

**7.3. The original note's proposed experimental specification**

The following records the source protocol for reproducibility. It is not a replacement for later choices Sam made in the actual campaign, including the smaller initial width, sample counts, 24-halo convention, and 20k-step first phase. State deviations whenever comparing results with this specification.

| Item | Source proposal |
|---|---|
| Initial grid | $N=512$, $h=2/N$, $H=\lceil\sqrt N\rceil=23$ halo neurons per side, $W=N+1+2H=559$ tanh neurons. |
| Centers and slopes | $t_j=-1+jh$ for grid slots $j=-H,\ldots,N+H$; freeze $a_j=\gamma$, $b_j=-\gamma t_j$. Grid-slot labels here are distinct from neuron indices $1,\ldots,W$. |
| Gamma sweep | $\{1,2,4,8,16,32,50,64\}$; report $N,W,h,\lambda=\gamma h$. At this width $\lambda=0.25$ means $\gamma=64$. |
| Training/evaluation samples | $16N+1$ endpoint-inclusive training points; a separate grid of 32,768 midpoints for evaluation. |
| Targets | $\sqrt2\sin(2\pi x)$; $\sqrt5\,x^2$; $[\sin(2\pi x)+0.1\sin(20\pi x)]/\sqrt{0.505}$; add Runge, conventionally $1/(1+25x^2)$ in this repo. |
| Width follow-up | $N=256,1024$, comparing constant gamma with constant lambda before accuracy floors obscure trends. |
| GD | Full-batch FP64, no momentum; constant $\eta=c/\|B\|^2$ with $c=1$ primary and $c=1/2$ control. |
| Adam | Ordinary Adam, $(\beta_1,\beta_2)=(0.9,0.999)$, no weight decay; prescribed constant/decayed schedules and equal-budget learning-rate/epsilon calibration at small and large gamma. |
| Budgets and reporting | $10^4$, $5\times10^4$, $2\times10^5$ updates; final error plus late-window mean, median, and excursions; paired seeds for nonzero starts. |
| Training restrictions | No Armijo, fallback directions, or in-training readout solves. Geometry remains frozen. |
| Coordinate/start controls | The five maps and three physical starts above, fixed reference allowances, with selected unscaled-bias/ordinary-halo controls. Log native and physical coefficient norms. |
| Polynomial diagnostics | Orthonormal polynomial QR up to $k=256$; common unit polynomial probes at degrees 16, 32, 64, 128, and necessary target tails. A unit polynomial probe should specify the orthogonal polynomial basis and normalization. |
| Damping | One step from the same residual at each $\rho=1,10^{-2},\ldots,10^{-24}$; refine numerically resolved transitions; no adaptive damping or repeated convergence. |

The quadratic is a control: its target tails vanish for $k\geq2$. Slow training on it can still occur, but cannot be attributed to necessary higher-degree target tails. That distinction makes it useful rather than a failed test.

**7.4. Numerical controls are necessary for interpreting tiny quantities**

Factor the normalized feature matrix $B$ directly. Forming a rounded Gram matrix can destroy the small singular modes whose behavior is being investigated. Compute damped solves by SVD or augmented QR, for example from the stacked least-squares system $[B;\sqrt\zeta I]d\approx[-r;0]$.

For difficult cells, recompute from feature evaluation onward at 80 and then 120 decimal digits, and vary numerical cutoffs. Increasing precision only after constructing a rounded FP64 matrix does not recover lost information. A discarded singular mode is not a proof of exact nullspace. A tail below reliable arithmetic resolution is not proven zero. Unresolved damping transitions and hitting times should be labeled unresolved rather than assigned a spurious floor or finite value.

Verify map identities, raw-versus-neighbor function evaluation, matched physical initial states, polynomial orthogonality, the capacity split, and the exact single-eigenspace cases of the flow and damping bounds. A least-squares residual computed with a truncated SVD is a numerical reference, not automatically the exact unrestricted span error. Report budget-limited plateaus as such. The source does not call for an LSMR comparison or a general optimizer competition.

**What would constitute an explanation of the observed gamma effect?**

At the requested accuracy, identify a target correction that must be made. Show that small-gamma features respond weakly to it. Verify that this weakness survives relevant readout-coordinate controls and yields a substantial, numerically resolved time certificate. Then show that larger gamma changes access and recovery of that same correction in the direction predicted, alongside better finite-budget fitting. The unrestricted least-squares error must be tracked so capacity changes are not confused with optimization changes.

None of these frozen-readout results establishes why gamma itself does or does not move under joint training. That remains a separate mechanism to analyze once the cost of each fixed geometry is understood.

**Appendix A. Why bounded-slope tanh has the stated polynomial approximation rate**

This appendix retains the source's complex-analysis argument. Its role is to prove the feature-remainder envelope used in Part 1; it is not needed to define the empirical access measurement $\|B^\top q\|^2$.

**A.1. Locate the poles and approximate each pole contribution**

The classical product for $\cosh z$ gives, by logarithmic differentiation,

$$
\tanh z=\sum_{\ell\geq0}\left(\frac1{z-it_\ell}+\frac1{z+it_\ell}\right),
\qquad t_\ell=\pi(\ell+\tfrac12).
$$

The terms are paired; the paired series converges locally uniformly away from poles. For $\phi(x)=\tanh(\gamma x+b)$ with $\gamma>0$, the poles in the complex $x$ plane are $z_\ell^\pm=(-b\pm it_\ell)/\gamma$. Each contributes $-\gamma^{-1}(z_\ell^\pm-x)^{-1}$. Changing the real bias shifts the poles horizontally, while their imaginary distances are $t_\ell/\gamma$.

For any complex $z\notin[-1,1]$, choose $s=\sqrt{z^2-1}$ so $w=z+s$ has $|w|>1$. Let $T_n$ be the degree-$n$ Chebyshev polynomial, characterized by $T_n(\cos\vartheta)=\cos(n\vartheta)$. The resolvent expansion is

$$
\frac1{z-x}=\frac1s\left(1+2\sum_{n\geq1}w^{-n}T_n(x)\right).
$$

Truncate after degree $k$. Since $|T_n(x)|\leq1$ on $[-1,1]$, its remainder obeys

$$
\left\|\frac1{z-\cdot}-p_k\right\|_\infty
\leq\frac2{|s|}\sum_{n>k}|w|^{-n}
=\frac{2|w|^{-k}}{|s|(|w|-1)}.
\tag{source 7}
$$

Here $p_k$ denotes this truncated polynomial, not the sampled projector $P_k$.

**A.2. Make the pole bound uniform over the real bias**

Put $r=|w|=e^u>1$ and $v=|\operatorname{Im}z|$. The relation $z=(w+w^{-1})/2$ gives the ellipse parameterization. Writing $w=e^{u+i\vartheta}$ gives $v=\sinh u\,|\sin\vartheta|\leq\sinh u$. Also $s=(w-w^{-1})/2$, so

$$
|s|^2(r-1)^2
=(r-1)^2\sinh^2u+v^2\left(\frac{2r}{r+1}\right)^2.
$$

At fixed $v$, both terms increase with $r$. The least admissible value is $r(v)=v+\sqrt{1+v^2}$, attained at $|\sin\vartheta|=1$, where $|s|=\sqrt{1+v^2}$. The numerator $r^{-k}$ also decreases with $r$. Therefore the bound depends only on the imaginary pole distance and is uniform over its real part, hence over $b$.

Define

$$
d_\ell=\sqrt{1+(t_\ell/\gamma)^2},\qquad
\rho_\ell=t_\ell/\gamma+d_\ell.
$$

Combining the two conjugate poles at each index yields

$$
\inf_{\deg p\leq k}\|\phi-p\|_\infty
\leq R_k(\gamma):=\frac4\gamma\sum_{\ell\geq0}
\frac{\rho_\ell^{-k}}{d_\ell(\rho_\ell-1)}.
\tag{source 8a}
$$

The remainders are $O(\ell^{-k-2})$, so their sum converges. Together with the paired pole expansion, this justifies combining the truncated approximants into a degree-$k$ approximation.

**A.3. Bound the pole sum explicitly**

The summand decreases with $v=t_\ell/\gamma$, whose spacing is $\pi/\gamma$. Bound the sum by its first term plus the corresponding integral. The first pole has $\rho_0=\rho_\gamma$ and gives

$$
\frac{4\rho_\gamma^{-k}}{(\rho_\gamma-1)\sqrt{\gamma^2+\pi^2/4}}.
$$

For the integral, substitute $v=\sinh u$, so $\rho=e^u$ and $dv/\sqrt{1+v^2}=du$. The contribution is at most

$$
\frac4\pi\int_{\log\rho_\gamma}^{\infty}\frac{e^{-ku}}{e^u-1}\,du.
$$

Expanding $1/(e^u-1)=\sum_{n\geq1}e^{-nu}$ and integrating positive terms gives

$$
\int_{\log\rho_\gamma}^{\infty}\frac{e^{-ku}}{e^u-1}\,du
=\sum_{n\geq k+1}\frac{\rho_\gamma^{-n}}n
\leq\frac{\rho_\gamma^{-k}}{(k+1)(\rho_\gamma-1)}.
$$

Adding the first term proves $R_k(\gamma)\leq U_k(\gamma)$ in source (5).

There is also a constant approximant: choose the midpoint of the endpoint values of the monotone function $\tanh(\gamma x+b)$. Its uniform error is half the endpoint range, at most $\tanh\gamma$. Taking the smaller of this bound and $U_k$ gives $e_k$. Negative slopes reduce to positive slopes by oddness; zero slopes produce constant features.

The function $e_k(\gamma)$ increases with $\gamma$. For the first term of $U_k$, use

$$
\frac{4\rho_\gamma^{-k}}{(\rho_\gamma-1)\sqrt{\gamma^2+\pi^2/4}}
=\frac8\pi\frac{\rho_\gamma+1}{\rho_\gamma^2+1}\rho_\gamma^{-k}.
$$

Both terms decrease with $\rho_\gamma>1$, while $\rho_\gamma$ decreases with $\gamma$. The constant-approximation bound also increases with $\gamma$. This justifies replacing all actual slope magnitudes by their common upper bound $\Gamma$ in Theorem 2.

For the prefactor bound used in Part 4, the first term is at most $(8/\pi)\rho_\Gamma^{-k}$. Also $\rho_\Gamma-1\geq\pi/(2\Gamma)$; if $k+1\geq\Gamma$, the second term is at most $(8/\pi^2)\rho_\Gamma^{-k}$. Their sum gives the claimed constant.

**A.4. What sharpness means here**

For the centered feature $\tanh(\Gamma x)=\sum_n c_nT_n(x)$, the same pole expansion gives, for odd $n$,

$$
|c_n|=\frac4\Gamma\sum_{\ell\geq0}\frac{\rho_\ell^{-n}}{d_\ell}
\geq\frac4{\sqrt{\Gamma^2+\pi^2/4}}\rho_\Gamma^{-n}.
$$

Under the normalized Chebyshev measure $dx/(\pi\sqrt{1-x^2})$, different Chebyshev degrees are orthogonal and $\|T_n\|=1/\sqrt2$ for $n\geq1$. If $n_k$ is the first odd degree greater than $k$, any degree-$k$ approximation leaves at least $|c_{n_k}|/\sqrt2$ of error in that orthogonal component. Together with the upper bound, this gives the exponential rate $\rho_\Gamma^{-k}$: the $k$th-root limit of the best error is $\rho_\Gamma^{-1}$.

Fine equal-weight Chebyshev grids reproduce each fixed-degree approximation error in the limit. Thus a faster exponential rate cannot hold uniformly over arbitrary sample grids and biases. This is not an asymptotic claim on one fixed finite grid, where polynomial interpolation eventually makes the remainder zero.

The raw width factor is also worst-case sharp. Repeat the same nonzero projected feature column $z$ across $W$ neurons. The resulting remainder matrix has squared operator norm $W\|z\|^2$. Therefore a uniform theorem cannot replace the linear width factor with a smaller-order one without additional assumptions. Neither this example nor exponential sharpness asserts that our particular center grid saturates the bound. Uniform numerical prefactors are not claimed optimal.

**Appendix B. The sharp displacement bound for gradient flow**

At a finite first hitting time $T=T_\epsilon$, define spectral weights $p_i=\beta_i^2/\|r_0\|^2$ and remaining squared-mode fractions $z_i=e^{-2T\nu_i}$. Then $\sum_i p_i=1$ and $\sum_i p_iz_i=\epsilon^2$.

Define a scalar function

$$
H(z)=\frac{2(1-\sqrt z)^2}{-\log z},\qquad 0<z<1,
\qquad H(0)=H(1)=0.
$$

This function is called $F(z)$ in the source proof; it is renamed here to avoid confusing it with the approximation loss in $L=F+G$. The displacement formula becomes

$$
\frac{\|\theta(T)-\theta_0\|^2}{T\|r_0\|^2}
=\sum_i p_iH(z_i).
$$

To check concavity, write $s=-\tfrac12\log z\geq0$ and $g(s)=(1-e^{-s})^2/s$, so $H(z)=g(s)$. Differentiation gives

$$
H''(z)=\frac{g''(s)+2g'(s)}{4z^2},
\qquad
g''(s)+2g'(s)
=\frac{2e^{-2s}}{s^3}(s+1-e^s)\bigl((s-1)e^s+1\bigr)\leq0.
$$

The first bracket is nonpositive because $e^s\geq1+s$. The second is nonnegative: it is zero at $s=0$ and its derivative is $se^s\geq0$. Continuity handles the endpoint values. Thus $H$ is concave on $[0,1]$.

Jensen's inequality now gives

$$
\sum_i p_iH(z_i)\leq H\left(\sum_i p_iz_i\right)
=H(\epsilon^2)=\frac{(1-\epsilon)^2}{\log(1/\epsilon)},
$$

which proves source (12). Zero eigenvalues correspond to $z_i=1$ and contribute zero displacement, so no full-rank assumption was used. If the tolerance is never reached, the time lower bound is automatically satisfied with infinite hitting time.

**Implementation supplement: where the PR's readout allowances come from**

This supplement records implementation details needed to reproduce the coordinate controls; they are not part of the proof of the access theorem. The code is [core.py](../../experiments/expD06_fixed_center_scales/core.py), with the map audit in [pr_conditioning_review.md](pr_conditioning_review.md).

Given $h=2/N$, reference bandwidth $\lambda_{\rm ref}=0.25$, and the implementation constant $\delta_{\rm ref}=0.25$, every neuron first receives

$$
\alpha_{\rm ordinary}=\frac{h}{2(\delta_{\rm ref}-\pi h/(2\lambda_{\rm ref}))}.
$$

The denominator must be positive; this particular allowance recipe is not defined at arbitrarily coarse widths. It is a reference construction envelope, not a target-specific coefficient solve. The source explaining the derivation of these envelopes is referenced by the PR but absent from this checkout; the implemented formula and its reference choices can be verified, while the necessity of the choice $\delta_{\rm ref}=0.25$ is not established here.

For $H$ halo neurons per side, let $R=\lceil H/2\rceil$ be the number of corrected outer slots. Set $q=e^{-2\lambda_{\rm ref}}$ and $p_n=\prod_{r=1}^n(1-q^r)$ with $p_0=1$. This local scalar $q$ is unrelated to a sample-space probe. Number corrected halo slots $i=1,\ldots,R$ from the far outside toward the domain. The added allowance on both corresponding outer slots is

$$
\Delta\alpha_i
=\frac{h}{2\delta_{\rm ref}}
\left(\frac\pi{2\lambda_{\rm ref}}+\frac{4\log2}{\pi}\right)
\frac{q^{[i(i+1)-1]/2}}{p_{i-1}p_{R-i}}
\prod_{\substack{r=1\\r\neq i}}^R(1+q^{r-1/2}).
$$

Those slots receive $\alpha_{\rm ordinary}+\Delta\alpha_i$; every other hidden neuron retains the ordinary allowance. The bias receives $\alpha_0=1+\sum_{j=1}^{W}\alpha_j$. With fixed $H$ and reference bandwidth, the added corrections are exactly proportional to $h$. If $H=\lceil\sqrt N\rceil$ changes with width, recompute the correction weights too. These formulas create the halo values; they are not a manual ramp toward one.

Using $\alpha_j$ versus $\sqrt{\alpha_j}$ is a separate parameterization choice. The former measures each coefficient in units of its allowance. The latter gives $\|\theta\|^2=\sum_jc_j^2/\alpha_j\leq\sum_j\alpha_j$ whenever the physical coefficients satisfy $|c_j|\leq\alpha_j$. For cumulative coefficients, summing the allowances gives $|\sum_{\ell\leq j}c_\ell|\leq s_j$, motivating the corresponding neighboring scales. These arguments motivate the maps; they do not establish optimality for GD or Adam.

**Source coverage and references**

| Original content | Location in this rewrite |
|---|---|
| Section 1; equations (1)–(3) | Setup, capacity split, and Part 2. |
| Theorem 1; equation (4) | Part 3, with the displacement proof completed in Appendix B. |
| Section 2; equations (5)–(8), rate/width sharpness | Part 1 and Appendix A. |
| Section 3; equations (9)–(14), optimal time prefactor | Part 3 and Appendix B. |
| Section 4; equation (15), conditional width argument | Part 4. |
| Section 4; Proposition 3 and equations (16)–(17) | Damping diagnostic. |
| Section 5; maps, starts, protocol, decision rule, numerical controls | Coordinate controls and experimental tests. |
| PR allowance details discussed in this conversation | Implementation supplement; distinguished from the source theorem. |

The source cites the preceding project draft *Sharp readout access bounds and frozen-geometry audits* (19 September 2026), the NIST DLMF infinite product for cosh (4.36.2), PR #2 at commit `b71a03396990`, and Kingma and Ba's *Adam: A Method for Stochastic Optimization* (ICLR 2015). Those are attributions carried from the source. This rewrite does not introduce an Adam convergence theorem or new experimental evidence.
