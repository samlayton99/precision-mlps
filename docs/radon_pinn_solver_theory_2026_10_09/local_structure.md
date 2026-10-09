# Exact profile coordinates and nonlinear PDE structure

Date: 2026-10-09. Scope: worker report for the investigation described in [STATUS.md](STATUS.md). These are algebraic results and conditional cost bounds, not a completed fast nonlinear solver. No experiments or parameter sweeps were performed.

The useful transformation is exact: cumulative readouts turn a tanh ridge profile into a convolution of a piecewise-constant density with the logistic derivative, plus one anchor. This preserves every readout and makes values, derivatives, boundary traces, and local nonlinear linearizations share the same radial locality. It establishes inexpensive **individual-direction** operators. It does not establish inexpensive aggregation across all directions or a width-independent iteration count.

## 1. Representation, conventions, and admissibility

Let the scalar network be

\[
q(x)=b+\sum_{m=1}^M\sum_{j=1}^N
w_{jm}\tanh(\gamma(\theta_m\cdot x-c_j)),\qquad
\|\theta_m\|_2=1,
\]

with strictly increasing centers. Geometry is fixed. In the uniform case, \(c_{j+1}-c_j=h\) and the regime of interest is approximately fixed \(\lambda=\gamma h\). The proof also permits distinct ordered center grids per direction, provided their local occupancy is counted.

There are \(K\) physical sample sites. A declared PDE, boundary, or data row requests a finite collection \(\mathcal A\) of multi-indices, of cardinality \(H\), with maximum order \(r_*\). Write \(R=r_*+1\). Complexity below assumes a bounded number of scalar equations and fields per site; the corresponding multiplicities must otherwise be included. \(K\) is not assumed independent of \(M\) or \(N\).

Everything through Section 3 is exact in real arithmetic. The truncation in Section 4 approximates the same operator with an explicit error budget. No coefficient penalty, smaller function family, domain replacement, or external PDE solve is introduced.

## 2. Exact cumulative-coordinate proposition

For one direction put \(t=\theta_m\cdot x\), suppress \(m\), and define

\[
a_j=\sum_{\ell=1}^j w_\ell,\qquad a_0=0.
\]

The inverse is \(w_j=a_j-a_{j-1}\). Define the interior basis functions and anchor by

\[
\beta_j(t)=\tanh(\gamma(t-c_j))-
\tanh(\gamma(t-c_{j+1})),\quad j<N,
\qquad \tau_N(t)=\tanh(\gamma(t-c_N)).
\]

**Proposition 1.** Every original profile has the exact representation

\[
g(t)=a_N\tau_N(t)+\sum_{j=1}^{N-1}a_j\beta_j(t).
\tag{1}
\]

**Proof.** Substitute \(w_j=a_j-a_{j-1}\) into the original sum and collect coefficients of each \(a_j\). Interior coefficients are adjacent differences; the last coefficient is the anchor. This is a triangular invertible coordinate change, not a restriction on readouts. \(\square\)

The stronger integral interpretation follows by integrating the derivative of tanh:

\[
\beta_j(t)=\int_{c_j}^{c_{j+1}}
\gamma\operatorname{sech}^2(\gamma(t-c))\,dc.
\tag{2}
\]

Thus the interior part is

\[
\int_{c_1}^{c_N} a(c)\,\gamma\operatorname{sech}^2(\gamma(t-c))\,dc,
\qquad
a(c)=a_j\text{ on }[c_j,c_{j+1}).
\tag{3}
\]

The kernel has integral 2. This is a smoothed piecewise-constant density, with **all** original degrees of freedom retained. In particular,

\[
\beta_j\ge0,\qquad
\sum_{j<N}\beta_j(t)=\tanh(\gamma(t-c_1))-
\tanh(\gamma(t-c_N))\le2.
\tag{4}
\]

These bounds explain why locality can be controlled by the largest cumulative coefficient, without multiplying an individual-entry error by \(N\).

The anchor remains global and must be kept; the bias is also global. Dropping either would change the function class. Their presence leads to bordered rather than purely banded blocks.

### Derivatives and stable evaluation

All multivariate derivatives retain the original direction factors:

\[
\partial^\alpha q(x)
=\sum_m\theta_m^\alpha
\left[a_{Nm}\tau_N^{(|\alpha|)}(t_m)
+\sum_{j<N}a_{jm}\beta_j^{(|\alpha|)}(t_m)\right],
\quad t_m=\theta_m\cdot x,
\tag{5}
\]

with the bias added for \(|\alpha|=0\). Here the superscript includes physical \(t\)-derivatives and hence \(\gamma^{|\alpha|}\).

Evaluating \(\beta_j\) by subtracting two rounded saturated tanh values is not acceptable at high accuracy. If \(u=\gamma(t-c_j)\), \(v=\gamma(t-c_{j+1})\), and \(\delta=\gamma(c_{j+1}-c_j)>0\), then

\[
\beta_j=\frac{\sinh\delta}{\cosh u\cosh v}.
\tag{6}
\]

Scaled exponential or log-cosh evaluation avoids the subtraction and overflow. Derivatives can share this accurately evaluated factor:

\[
\beta_j^{(r)}=\gamma^r\beta_j Q_r(\tanh u,\tanh v),
\]

where \(Q_0=1\) and

\[
Q_{r+1}(U,V)=-(U+V)Q_r+
(1-U^2)\partial_UQ_r+(1-V^2)\partial_VQ_r.
\tag{7}
\]

This recurrence follows by the product and chain rules. For fixed derivative order the degree and number of operations are fixed. It supplies one explicit cancellation-avoiding route; it is not a completed floating-point error proof. For example, \(Q_1=-(U+V)\), and \(Q_2=2(U^2+UV+V^2)-2\).

The cumulative map has \(\|T\|_\infty=N\), \(\|T^{-1}\|_\infty=2\) for \(N>1\), so its condition number in these coefficient norms is \(2N\). Exact invertibility alone does not certify numerical stability when converting arbitrary supplied readouts. Solving directly in \(a\) avoids repeated prefix conversion, but recovering tiny differences of large \(a_j\) can still lose readout accuracy. Physical field accuracy and individual readout accuracy must be distinguished without silently relaxing the former.

### Exact partition of unity, including both endpoint tails

There is a particularly useful normalization when the profile has its own additive constant \(b_m\). Write \(T_j(t)=\tanh(\gamma(t-c_j))\), and set

\[
B_0=(1-T_1)/2,\qquad B_j=(T_j-T_{j+1})/2\ (1\le j<N),
\qquad B_N=(1+T_N)/2.
\tag{7a}
\]

All \(B_j\) are nonnegative, and their sum is exactly one on the whole real line. Let \(S=\sum_jw_j\) and define

\[
d_j=b_m-S+2\sum_{\ell=1}^{j}w_\ell,\quad j=0,\ldots,N,
\tag{7b}
\]

with an empty sum for \(j=0\). Then

\[
b_m+\sum_jw_jT_j=\sum_{j=0}^Nd_jB_j,
\qquad
w_j=(d_j-d_{j-1})/2,\quad b_m=(d_0+d_N)/2.
\tag{7c}
\]

Collecting each \(T_j\)'s coefficient verifies the identity. In these coordinates the exact synthesis map satisfies

\[
\left\|\sum_jd_jB_j\right\|_{L^\infty(\mathbb R)}
\le\|d\|_\infty.
\tag{7d}
\]

This is a width-independent **upper** stability bound. It neither supplies a lower bound on an observed domain nor prevents coefficients from cancelling after synthesis.

For the original single-bias network, independently varying all \(b_m\) would introduce \(M-1\) redundant constant coordinates. One exact bookkeeping choice is \(b_m=0\) for \(m<M\), leaving \(b_M=b\); in (7c) this means \(d_{0m}+d_{Nm}=0\) for the first \(M-1\) directions. Alternatively the redundant representation needs an explicit constant gauge. The endpoint tails cannot be dropped. Their derivatives are localized near the corresponding endpoint, while their values are global; local normal blocks now have at most a two-column border per direction.

### A genuine width-uniform radial stability theorem, with its limits

For the normalized interior basis on a uniform grid, use the dimensionless coordinate \(s=t/h\) and define

\[
\beta_\lambda(s)=\tfrac12[\tanh(\lambda s)-\tanh(\lambda(s-1))]
=\kappa_\lambda*\mathbf1_{[0,1]}(s),
\qquad \kappa_\lambda(s)=\tfrac\lambda2\operatorname{sech}^2(\lambda s).
\]

For any finitely supported sequence of coefficients \(d_j\), with no restriction on its values, one has

\[
h c_\lambda\sum_j|d_j|^2
\le
\left\|\sum_jd_j\beta_\lambda(t/h-j)\right\|_{L^2(\mathbb R)}^2
\le h\sum_j|d_j|^2,
\qquad
c_\lambda=\left[\frac2\pi\frac{\pi^2/(2\lambda)}{\sinh(\pi^2/(2\lambda))}\right]^2>0.
\tag{7e}
\]

**Proof.** Integer translates of \(\beta_\lambda\) form a nonnegative partition of unity. Jensen's inequality gives
\(|\sum_jd_j\beta_\lambda(s-j)|^2\le\sum_j|d_j|^2\beta_\lambda(s-j)\).
Each translate integrates to one, proving the upper bound after scaling by \(h\).

For the lower bound use the Fourier transform \(\widehat f(\omega)=\int f(s)e^{-i\omega s}\,ds\). Direct transformation of \(\kappa_\lambda\) gives

\[
\widehat\beta_\lambda(\omega)
=e^{-i\omega/2}\operatorname{sinc}(\omega/2)m_\lambda(\omega),
\quad m_\lambda(\omega)=\frac{\pi\omega/(2\lambda)}{\sinh(\pi\omega/(2\lambda))},
\]

where \(\operatorname{sinc}(u)=\sin(u)/u\), with the continuous value at zero. To check the transform constant, substitute \(v=e^{2\lambda s}\) into the kernel integral: it becomes \(\int_0^\infty v^{-i\omega/(2\lambda)}(1+v)^{-2}\,dv\), whose beta-integral value is \(\Gamma(1-ia)\Gamma(1+ia)=\pi a/\sinh(\pi a)\), with \(a=\omega/(2\lambda)\). Plancherel and periodicity of \(\sum_jd_je^{-ij\omega}\) give the Gram symbol

\[
P_\lambda(\omega)=\sum_{\ell\in\mathbb Z}
|\widehat\beta_\lambda(\omega+2\pi\ell)|^2,
\quad -\pi\le\omega\le\pi.
\]

The \(\ell=0\) term is at least \([(2/\pi)m_\lambda(\pi)]^2=c_\lambda\), since sinc and \(m_\lambda\) have the stated lower bounds on this interval. Fourier-series Parseval then yields the lower bound. \(\square\)

This theorem applies to arbitrary finite sets of interior coefficients and has constants independent of their number. It concerns full-line profile \(L^2\), not an empirical point norm, a bounded physical-domain norm, the endpoint tail functions (which are not individually in \(L^2(\mathbb R)\)), a sum over directions, or a PDE residual norm. None of those follows automatically.

At \(\lambda=0.25\), \(c_\lambda\approx4.52086\times10^{-15}\). The bound on the synthesis condition number is therefore about \(1.48727\times10^7\), and on its Gram matrix about \(2.21197\times10^{14}\). These are width independent but numerically severe. The severity is not just an artifact of the lower estimate: \(P_\lambda(0)=1\), whereas

\[
P_\lambda(\pi)=2c_\lambda+\sum_{|2\ell+1|\ge3}
\frac4{\pi^2(2\ell+1)^2}m_\lambda((2\ell+1)\pi)^2.
\]

For \(\lambda=0.25\) the remaining positive terms are exponentially smaller. Long alternating coefficient sequences, normalized in \(\ell^2\), concentrate their Fourier energy near \(\pi\) and approach this small Rayleigh quotient. Thus highly oscillatory cell coefficients are strongly smoothed even as the width grows. A proposed inverse of this smoothing must count its arithmetic amplification. The theorem shows a real radial structure, not machine-epsilon recovery of arbitrary coefficient patterns.

## 3. General local nonlinearities preserve the factorization

Let \(A_{\alpha,m}\) map direction \(m\)'s cumulative coordinates to the values of \(\partial^\alpha q\) at all requested sites. Equation (5) is the cumulative-coordinate version of \(\Phi^{(r)}D\). Stack the requested derivative values into the sampled jet

\[
U=\mathcal A a=\sum_m\mathcal A_m a_m,
\qquad U_{\alpha,k}=\partial^\alpha q(x_k).
\]

The term “jet” here means only the finite list of values and derivatives needed by the declared equations. A local nonlinear PDE has residual

\[
\mathcal R_k(a,p)=F_k(U_{\cdot,k},p),
\tag{8}
\]

where \(p\) contains any finite-dimensional physical parameters. Locality is in physical space, not in direction index. Each sample first aggregates all direction contributions, then applies \(F_k\). This does not enumerate pairwise direction products.

At a current candidate \(a,p\), define \(C_k=D_UF_k(U_{\cdot,k},p)\), and let \(C\) be the block diagonal collection of these small matrices. The exact Jacobian is

\[
J_a=C\mathcal A=[B_1\;\cdots\;B_M],
\qquad B_m=C\mathcal A_m.
\tag{9}
\]

Consequently,

\[
J_av=\sum_m B_mv_m,\qquad
(J_a^\top z)_m=B_m^\top z.
\tag{10}
\]

For a scalar residual row this reads

\[
B_m(k,j)=\sum_{\alpha\in\mathcal A}
\frac{\partial F_k}{\partial U_\alpha}
\theta_m^\alpha\beta_j^{(|\alpha|)}(t_{km}),
\tag{11}
\]

with the anchor column treated the same way. All nonlinear coefficients multiply radial derivative banks; no dense readout Hessian is required for these actions.

Equation (9) covers local nonlinear boundary conditions, observation losses, and inverse coefficient fields after stacking the jets of all represented fields. A few unknown physical parameters add global columns \(D_pF\). An unknown field adds another profile family, whose own sample-to-direction cost must be counted. Integral or other nonlocal observations need their actual observation operators; they do not inherit pointwise locality merely by being called data.

### Exact lifted formulation

For whatever residual objective and hard constraints the application actually specifies, introduce \(U\) as an independent variable and impose the exact linear equality

\[
U=\mathcal A a.
\tag{12}
\]

For example, a prescribed least-squares problem becomes

\[
\min_{a,U,p}\ \ell(F(U,p))
\quad\text{subject to }U=\mathcal A a
\quad\text{and the original hard constraints}.
\tag{13}
\]

Eliminating \(U\) returns the original problem exactly. Nonlinearity now resides in small physical-point blocks, while the global link is a linear synthesis map with known derivative structure. This is useful for designing primal-dual or block iterative methods, but is not a complexity theorem by itself.

In particular, an augmented term \(\|U-\mathcal A a\|^2\) does **not** make the \(a\)-subproblem independent by direction: it still produces \(\mathcal A_m^\top\mathcal A_\ell\) cross blocks. Introducing separate \(U_m=\mathcal A_m a_m\) and enforcing \(U=\sum_mU_m\) exposes independent local subproblems, but explicitly storing every \(U_m\) costs \(O(KHM)\), and the consensus iteration count is unproved. No such splitting is adopted here as a completed solution.

## 4. Certified locality at any fixed derivative order

For every integer \(p\ge1\), there is an explicit finite constant \(C_p\) such that

\[
|\tanh^{(p)}u|\le C_p e^{-2|u|}.
\tag{14}
\]

One choice is \(C_p=4\max_{|v|\le1}|P_{p-1}(v)|\), where
\(\tanh^{(p)}u=(1-\tanh^2u)P_{p-1}(\tanh u)\). The polynomials are obtained recursively by differentiation. In particular \(C_1=4\).

At a query \(t\), retain every center interval \([c_j,c_{j+1}]\) intersecting \([t-T/\gamma,t+T/\gamma]\). Keep the anchor exactly. Write \(\widetilde g^{(r)}\) for the resulting truncated sum of the exact derivative basis functions.

**Proposition 2.** For each fixed \(r\ge0\),

\[
|g^{(r)}(t)-\widetilde g^{(r)}(t)|
\le C_{r+1}\gamma^r e^{-2T}
\max_{j<N}|a_j|.
\tag{15}
\]

**Proof.** Differentiate (2) under its finite integral. The absolute sum over discarded intervals is bounded by the maximum coefficient times

\[
\gamma^{r+1}\int_{|c-t|>T/\gamma}
|\tanh^{(r+1)}(\gamma(t-c))|\,dc.
\]

Use (14) and substitute \(u=\gamma(t-c)\). The two tails have total integral \(\int_{|u|>T}e^{-2|u|}\,du=e^{-2T}\), giving (15). Discarded intervals are disjoint, which is the reason no factor \(N\) appears. \(\square\)

For a multivariate derivative, summing over directions gives the explicit bound

\[
|\partial^\alpha q-\widetilde{\partial^\alpha q}|
\le C_{r+1}\gamma^r e^{-2T}
\sum_m|\theta_m^\alpha|\max_{j<N}|a_{jm}|,
\quad r=|\alpha|.
\tag{16}
\]

For nonuniform slopes use the corresponding \(\gamma_m^r\) and \(T_m\) term by term. There is no assumption here that cumulative amplitudes remain bounded during a solve.

On a uniform grid, the number of retained intervals per query is at most

\[
s\le\min\left(N-1,\;\left\lceil\frac{2T}{\lambda}\right\rceil+2\right).
\tag{17}
\]

To make a particular derivative error at most \(\varepsilon_\alpha\), a sufficient radius is

\[
T\ge\frac12\log\frac{
C_{r+1}\gamma^r\sum_m|\theta_m^\alpha|\|a_{m,\mathrm{int}}\|_\infty
}{\varepsilon_\alpha},
\tag{18}
\]

with zero radius sufficient if the logarithm is negative. At fixed physical center span and fixed \(\lambda\), \(\gamma\) grows proportionally to \(N\). Thus fixed absolute derivative accuracy requires a radius containing at least a term \((r/2)\log N\), as well as any growth in coefficient amplitudes and direction sums. The stencil is not unconditionally constant at machine precision. In the worst case it saturates at the full \(N\).

On irregular center grids, (15) survives unchanged, but the number of intersecting cells replaces (17). Dense center clustering can make that number large. Irregular **physical** sample locations do not by themselves cause this problem.

These are approximations of the jets of the exact network. A hard truncation mask as a function of \(x\) can be discontinuous; one must not differentiate the masked value evaluator and identify its derivative with the intended PDE jet. The derivatives are computed analytically first, then their sums are truncated under the separate bounds.

### Nonlinear propagation of evaluation errors

Suppose the exact and approximate jets remain in a convex set on which \(\|DF\|\le L\). In compatible finite-dimensional norms, with \(E=\mathcal A-\widetilde{\mathcal A}\),

\[
\|F(\mathcal A a,p)-F(\widetilde{\mathcal A}a,p)\|
\le L\|Ea\|.
\tag{19}
\]

This follows from the line-segment integral of \(DF\). For a least-squares residual perturbation \(e\), the objective changes by at most \(\|R\|\|e\|+\tfrac12\|e\|^2\). If \(DF\) is also \(L_D\)-Lipschitz on that set, then

\[
\|J-\widetilde J\|
\le L\|E\|+L_D\|Ea\|\,\|\widetilde{\mathcal A}\|.
\tag{20}
\]

The norms of the nonlinear coefficients therefore belong in the tolerance budget. No universal fixed tanh cutoff certifies every nonlinear PDE or every iterate. Bounds on a selected norm ball are numerical certification hypotheses, not permission to impose that ball as a new prior.

## 5. What becomes banded, and what does not

Use one common window radius sufficient for all declared derivatives. At any sample, the nonanchor columns of a truncated \(B_m\) are nonzero only in a consecutive interval of at most \(s\) center cells. This holds for arbitrary sample locations and arbitrary values of the current nonlinear coefficients in (11).

**Proposition 3.** For diagonal nonnegative row weights, the nonanchor part of
\(\widetilde B_m^\top W\widetilde B_m\) is banded, with index distance at most \(s-1\). Including the anchor produces a one-column/row border. Several coupled fields produce a block-banded matrix with one anchor per field.

**Proof.** A row contributes an outer product supported on its active center interval. Two columns separated by at least \(s\) cannot occur in that interval together. Sum the row outer products. Anchors are retained at every point and couple to all columns. \(\square\)

The bias is one global column across directions, and finitely many physical inverse parameters are additional global columns. Correlated observational noise or nonlocal row weighting can destroy the local Gram sparsity unless its whitening operator has suitable structure.

For positive definite local blocks, banded factorization costs \(O(Ns^2)\), storage \(O(Ns)\), and triangular solves \(O(Ns)\). The fixed-size border has the same leading bound. Local rank deficiency invalidates ordinary Cholesky; a rank-revealing or constrained treatment and its cost must be supplied. Artificial diagonal penalties would change the problem unless they arise only inside an exactly converged equivalent algorithm.

The **cross-direction** block

\[
\widetilde B_m^\top W\widetilde B_\ell,\qquad m\ne\ell,
\tag{21}
\]

is generally not banded in either pair of center indices. A physical point can project to unrelated indices in the two directions. Proposition 3 does not prove that independent local solves solve the global system.

For the same fixed nonlinear coefficients and sample weights, write \(E_m=B_m-\widetilde B_m\). The Gram approximation has the useful certification bound

\[
\|B_m^\top WB_m-\widetilde B_m^\top W\widetilde B_m\|
\le2\|W^{1/2}\widetilde B_m\|\|W^{1/2}E_m\|
+\|W^{1/2}E_m\|^2.
\]

This follows by expanding the difference into two cross terms and one quadratic error term. It controls an operator approximation; inverting that approximation still amplifies perturbations according to the relevant smallest singular values.

## 6. Honest complexity ledger

The following costs are for a fixed number of fields/equations per site and fixed maximum derivative order; explicit \(R,H\) remain visible. All active coordinates are retained. A common regular grid permits constant-time active-window location; irregular grids add binary searches unless locations are precomputed.

| Operation | Work | Memory / qualification |
|---|---:|---|
| Form arbitrary physical projections \(\theta_m\cdot x_k\) | \(O(KMd)\) | \(O(KM)\) if cached; streaming trades memory for recomputation |
| Convert existing readouts to cumulative coordinates | \(O(MN)\) | \(O(MN)\); numerical prefix errors must be controlled |
| Form requested sampled jets, or apply their transpose | \(O(KM(Rs+H))\) | \(O(KH+MN)\) when streamed, plus projection storage if used |
| Evaluate local PDE nonlinearity and its jet derivative | Actual \(K\)-site cost of \(F,DF\) | Includes inverse fields, coupled equations, and requested derivative count |
| Apply nonlinear Jacobian or transpose | \(O(KM(Rs+H))\) plus local \(F\) derivative work | No \(M^2\) matrix needs storage |
| Assemble all within-direction Gram blocks | \(O(KM(H+Rs+s^2))\) | \(O(MNs)\) plus borders; recomputed when nonlinear coefficients change |
| Factor all positive definite local Gram blocks | \(O(MNs^2)\) | \(O(MNs)\) |
| Apply all local block inverses once | \(O(MNs)\) | Does not alone apply the inverse of the full coupled operator |
| Explicit direction-wise lifted jets | At least \(O(KHM)\) initialization | \(O(KHM)\) storage unless an equivalent streamed algorithm is proved |

For jet evaluation, compute \(g_m^{(r)}(t_{km})\) once per derivative order using the radial stencil, then multiply by the \(H\) requested direction factors. This gives \(Rs+H\), rather than a pessimistic \(Hs\). A PDE such as the Laplacian can contract its direction factors analytically and further reduce \(H\). For general PDE declarations \(H\) itself may grow with dimension; it cannot be declared constant without inspecting the declaration.

There is no factorization required for a plain Jacobian action. Factorization costs enter only if a proposed block method uses those blocks. In a nonlinear method, the coefficients \(DF\) change; reusing an old factorization may be useful but needs a convergence theorem.

If \(K=\Theta(MN)\), direct aggregate evaluation becomes

\[
O(M^2N(Rs+H)),
\]

not \(O(MN^2)\) in general. Even if one pass fits the target in a particular relation between \(M\) and \(N\), the total solve still multiplies by the number of passes. Extensive held-out validation pays the same synthesis cost at its own additional sample count. Neither sampling nor iteration count is suppressed in these statements.

## 7. Conclusions and unresolved admissibility-sensitive steps

**Established here:** an invertible change of coordinates gives a common localized basis for values and all fixed-order derivatives; its tail error has an explicit bound without a factor \(N\); general local nonlinear PDEs factor through a finite sampled jet; the per-direction linearized normal blocks are certified approximations to bordered banded matrices; BC/IC and point data fit the same algebra.

**Not established:** an aggregation algorithm meeting the target when \(K\) grows with the full readout count; a width-uniform stable global decomposition; an iteration bound for nonlinear/coupled/inverse problems; an end-to-end floating-point accuracy theorem. Lifting exposes the local nonlinearity but does not remove the coupling in its linear consistency constraints.

The transformation is useful because it preserves the original function space while putting **all** required local operators into one radial locality pattern. Merely writing the Jacobian as \(C\mathcal A\), or introducing additional jet variables, would be a restatement without this locality and a fast stable aggregation mechanism.

Restricting coefficients to a smooth/low-rank family, ignoring anchor modes, assuming a bounded-domain transform from a whole-space identity, or stopping a splitting method before its constraints are accurately satisfied would each require an admissibility decision. None has been adopted. A future global theorem must retain the full consistency equations and account for cross-direction cancellation rather than declaring each profile an independent PDE solution.
