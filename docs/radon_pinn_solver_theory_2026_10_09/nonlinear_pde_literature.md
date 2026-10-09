# Established nonlinear PDE frameworks and the exact frozen-network connection

Date: 2026-10-09. Primary mathematical sections were inspected, rather than relying on abstracts. This note separates existing formulations from our finite-dimensional deductions. No implementation, benchmark, norm prior, domain decomposition, or changed network has been adopted.

The immediate conclusion is that fixed-feature PDE collocation, nonlinear residuals of linear derivative observations, and selective elimination of linear inverse parameters already have established formulations. The open part for this project is the cost and stability of its particular structured operators, not inventing a general nonlinear PDE framework.

## 1. Five primary references and precisely what transfers

### PIELM: the direct linear precursor

V. Dwivedi and B. Srinivasan, *Physics Informed Extreme Learning Machine (PIELM)—A rapid method for the numerical solution of partial differential equations*, Neurocomputing 391 (2020), 96–118. [Author preprint](https://arxiv.org/pdf/1907.03507).

Read Section 3, especially equations (15)–(24), PDF pages 5–6. A single-hidden-layer tanh network has fixed random hidden coefficients and unknown output weights. Equations (16)–(17) differentiate the activation and multiply by powers of its fixed input weights: this is already the elementary chain-rule structure underlying our \(\Phi^{(r)}D\). PDE, boundary, and initial rows are stacked into a linear system, solved using a pseudoinverse.

Its stated method here is for linear PDEs. The paper's observation that a square system has a unique solution from equation counting alone is insufficient: invertibility still requires rank. Its later distributed representation introduces separate local networks. Neither random geometry nor that piecewise architecture is needed for our algebra, and the latter is not our required single global MLP. This reference supports the fixed-feature collocation starting point; it does not establish our width scaling or nonlinear convergence.

### locELM: nonlinear fixed-readout equations already solved directly

S. Dong and Z. Li, *Local extreme learning machines and domain decomposition for solving linear and nonlinear partial differential equations*, CMAME 387 (2021), 114129. [Author-hosted paper](https://www.math.purdue.edu/~sdong/PDF/locelm_CMAME2021.pdf).

Read Section 2.1, equation (2); Section 2.3, equation (22); and Appendix A, equations (45)–(47), printed pages 5, 12–16, and 55. Fixed hidden features make the solution linear in output weights. Nonlinear PDEs produce nonlinear least squares; Appendix A derives a Newton increment followed by linear least squares. The displayed PDE class has linear highest-order terms and nonlinear lower-order terms. Initial/boundary constraints and inter-subdomain continuity are included.

This is a direct precedent for applying nonlinear PDE solvers to frozen features. Its domain decomposition and separate local networks are architectural differences, and its time blocks need not assemble into our prescribed global space–time MLP. Those features cannot be imported as an invisible acceleration. The paper does not give the sought direction-count-linear complexity theorem. The transferable item is the nonlinear residual/Jacobian formulation and incremental solve, not a guarantee for arbitrary nonlinear principal parts, high-dimensional Navier–Stokes, or million-neuron accuracy.

### Nonlinear Gaussian-process collocation: the finite-jet formulation

Y. Chen, B. Hosseini, H. Owhadi, and A. M. Stuart, *Solving and learning nonlinear PDEs with Gaussian processes*, JCP 447 (2021), 110668. [Author preprint](https://arxiv.org/pdf/2103.12959).

Read Proposition 2.3; Assumption 3.1 and Corollary 3.2; Theorem 3.3; Section 3.4.3; Corollary 4.4 (PDF pages 13, 14–16, 21, and 29). Nonlinear PDE/BC operators are functions of linear observations, including derivatives. A minimum-Hilbert-norm formulation reduces to auxiliary observation values with nonlinear constraints. Inverse fields and observations use a product space; scattered sites are permitted. Theorem 3.3 assumes a unique classical solution in the chosen space, suitable compact/continuous embeddings, and vanishing interior/boundary fill distances. It concerns minimizers, not guaranteed convergence of a nonlinear optimizer.

The minimum norm is substantive. Section 3.4.3 uses dense cubic factorization in the experiments. Its then-cited fast-factorization guarantees concern Dirac observations and elliptic Green kernels. The later sparse-Cholesky paper below extends the theory to derivative observations under specific hypotheses. Neither the prior nor those kernel hypotheses are automatic for our frozen network.

### Variable projection: eliminate genuinely linear blocks

D. P. O'Leary and B. W. Rust, *Variable Projection for Nonlinear Least Squares Problems*, Computational Optimization and Applications 54 (2013), 579–593. [Author-hosted manuscript](https://www.cs.umd.edu/users/oleary/software/varpro.pdf).

Read Section 1, equations (2)–(3), and Sections 2.2–2.3 (PDF pages 2, 5–8). This develops the Golub–Pereyra variable-projection method: optimize nonlinear variables while solving their conditional linear least-squares problem exactly, with the reduced Jacobian obtained from the original derivatives. The manuscript discusses rank deficiency, QR/SVD implementations, and complications when constraints on the linear parameters become active. Smooth reduced derivatives require suitable local rank/active-set behavior; global convergence is not guaranteed.

The transferable point is exact elimination of a truly linear conditional block. A neural output being linear in its weights does not make a nonlinear PDE residual linear in those weights. Thus variable projection cannot eliminate all readouts of general \(F(Aw)\) by an ordinary linear solve. It can eliminate small physical-parameter blocks, pressure blocks in suitable formulations, or readouts of a linear PDE conditional on unknown physical parameters. Each inner solve still costs work; the reduction is not a width-scaling result.

### Sparse Cholesky: a newer fast theorem that includes derivative observations

Y. Chen, H. Owhadi, and F. Schäfer, *Sparse Cholesky Factorization for Solving Nonlinear PDEs via Gaussian Processes*, [March 2024 revision](https://arxiv.org/pdf/2304.01294).

Read Section 4.1, Theorems 4.1–4.2, and Section 5 (PDF pages 15–21). The theorem assumes a bounded convex Lipschitz domain, a Green kernel of a symmetric positive local isomorphism \(\mathcal L:H_0^s\to H^{-s}\), and controlled scattered-point homogeneity. Every interior site has a Dirac observation; additional independent derivatives have order \(J<s-d/2\). Diracs are ordered first by boundary-conditioned maximin ordering.

With \(L\) observations, Theorem 4.2 bounds inverse/covariance Frobenius errors by \(\varepsilon\) using sparsity radius \(O(\log(L/\varepsilon))\). The resulting storage/work scale as \(O(L\log^d(L/\varepsilon))\) and \(O(L\log^{2d}(L/\varepsilon))\), with dimension, kernel, smoothness, and point-regularity dependent constants. Kernel entry access must also be priced for our representation.

The paper explicitly says this guarantee need not apply to the reduced Newton matrix \(DF\,G\,DF^\top\), which can lack enough Diracs. Nearly constant inner iterations and nonlinear convergence there are empirical/assumed. This is a real fast derivative-kernel theorem, not an automatic theorem for arbitrary finite neural kernels or all nonlinear solves.

## 2. Our finite-dimensional mapping, proved directly

Let \(P=MN\), optionally including a bias as one additional feature, and define

\[
q_w(x)=\phi(x)^\top w,\qquad
\phi_{jm}(x)=\tanh(\gamma(\theta_m\cdot x-c_j)).
\]

For a multi-index \(\alpha\), its derivative feature is exactly

\[
\phi_{jm,\alpha}(x)=\gamma^{|\alpha|}\theta_m^\alpha
\tanh^{(|\alpha|)}(\gamma(\theta_m\cdot x-c_j)).
\tag{1}
\]

The present construction specifies geometry rather than drawing it randomly. That difference changes approximation and structure, not the finite linear-algebra identity.

Let \(L\) count all required scalar jet observations at all sites, and collect their feature rows into \(A\in\mathbb R^{L\times P}\). Typically \(L\) is comparable to the site count times the number of requested derivative channels; it need not equal either count. Then every declared local PDE, boundary, and point-data residual has the form

\[
R(w,p)=F(Aw,p)-y.
\tag{2}
\]

Multiple fields use block copies of the feature map; unknown represented coefficient fields add their own jet blocks. Unknown finite physical parameters can remain in \(p\). Auxiliary variables give the **unconditionally exact** reformulation

\[
z=Aw,\qquad R=F(z,p)-y.
\tag{3}
\]

This lift changes no objective, constraints, or hypothesis space. The nonlinear derivatives are

\[
J_w=F_z(Aw,p)A,\qquad J_p=F_p(Aw,p).
\tag{4}
\]

Only the coefficients in \(F_z\) change with the candidate solution; \(A\) stays fixed at fixed sample sites and network geometry. No probabilistic interpretation or kernel is required for (1)–(4).

### The exact finite neural kernel

Define the deterministic finite-rank kernel

\[
\mathcal K(x,y)=\phi(x)^\top\phi(y).
\tag{5}
\]

Its derivative blocks are

\[
\partial_x^\alpha\partial_y^\beta\mathcal K(x,y)
=\sum_{m,j}\gamma^{|\alpha|+|\beta|}\theta_m^{\alpha+\beta}
\tanh^{(|\alpha|)}(s_{jm}(x))
\tanh^{(|\beta|)}(s_{jm}(y)).
\tag{6}
\]

Consequently the kernel matrix for our jet observations is simply

\[
G=AA^\top.
\tag{7}
\]

This is an exact dual expression for the same finite network, not an infinite-width limit, a trained neural tangent kernel, or a replacement radial-basis model. Kernel representers formed from derivatives of (5) remain linear combinations of the original hidden features.

For any achievable jet vector \(z\in\operatorname{range}A\), elementary orthogonal decomposition gives

\[
w=A^\top G^\dagger z+n,\qquad n\in\ker A,
\tag{8}
\]

and

\[
\min_{Aw=z}\|w\|_2^2=z^\top G^\dagger z.
\tag{9}
\]

**Proof.** The vector \(A^\top G^\dagger z\) belongs to \(\operatorname{range}A^\top\), satisfies the constraint since \(GG^\dagger z=z\), and is orthogonal to \(\ker A\). Pythagoras shows that any nonzero nullspace component increases its squared Euclidean norm. Substitution gives (9).

Thus, **if** the original problem actually asks for the least coefficient norm subject to its nonlinear constraints, it becomes

\[
\min_{z\in\operatorname{range}A,\,p}
z^\top G^\dagger z
\quad\text{subject to }F(z,p)=y,
\tag{10}
\]

with any originally declared data losses or parameter terms retained. Equation (10) is our explicit finite-feature counterpart of a conditional representer formulation. Its kernel-side output is reconstructed as the same one-hidden-layer network by (8).

But (3) is exact for every objective, whereas (10) introduces a selection rule unless that norm was already prescribed. All vectors in (8) have the same finite training jets; they can differ between the sites. Removing \(n\) therefore preserves sampled residuals but need not preserve the physically correct solution outside them. A regularizing nugget \(G+\varepsilon I\), changed coefficient metric, or reduced-rank approximation also needs explicit justification; none is adopted here.

The finite-rank qualification matters: \(\operatorname{rank}G\le P\), so if \(L>P\) then \(G\) is singular. Even when \(L\le P\), duplicate/independent derivative constraints and feature redundancies determine its rank. Replacing the range condition in (10) by unconstrained \(z\in\mathbb R^L\) can create jets not realizable by the network. The corresponding RKHS norm is the minimum coefficient norm over all representations of a function, which also accounts for globally redundant features. An invertible cumulative-coordinate change does not preserve Euclidean coefficient norms automatically; the metric must transform with it if preserving an existing norm objective.

## 3. Where variable projection actually helps our inverse problems

These are direct algebraic applications, with the original residual objective unchanged.

For a linear PDE conditional on physical parameters \(p\), write the stacked weighted residual as \(B(p)w-y\). For each \(p\), an exact conditional least-squares solve gives \(w_*(p)\); then minimize \(\|B(p)w_*(p)-y\|^2\). This does not require hidden-feature training. Its inner problem is still the large structured linear problem, whose cost is not eliminated.

For nonlinear Navier–Stokes with constant unknown viscosity, conditional on velocity and pressure the momentum residual is

\[
r_0(u,p)-\nu L(u),\qquad L(u)=\Delta u.
\]

With a prescribed positive semidefinite sample weight \(W\), no additional viscosity-dependent observations/constraints, and \(L^\top WL>0\), the exact unconstrained conditional minimizer is

\[
\nu_*(u,p)=\frac{L^\top Wr_0}{L^\top WL}.
\tag{11}
\]

The sums include velocity components and sample sites. This removes one scalar from the nonlinear search without changing the neural velocity representation. If positivity, bounds, or viscosity data are prescribed, solve that actual conditional problem instead; (11) no longer describes all cases. A small denominator exposes lack of identifiability rather than an algorithmic nuisance.

Pressure readouts and constant viscosity can also form one conditional linear block given velocity, provided all relevant BC/data/gauge equations are included. With a variable viscosity field, the symmetric-stress operator is linear in that field and its gradient conditional on velocity, so elimination is algebraically possible but may itself be large and poorly determined. No claim that such a field should be eliminated follows solely from linearity.

For a generic nonlinear PDE residual, the solution readouts remain nonlinear through products such as \((A_0w)\odot(A_\Delta w)\). Variable projection applies only to identified conditional linear blocks, or to each linearized step, not to an imagined linearity of the entire problem.

## 4. Costs: duality exchanges width for jet count

The following are elementary dense-linear-algebra counts, not costs claimed by the cited methods. They apply to the explicit finite-feature matrices above; more structure may reduce them only with a separate algorithm and bound.

| Operation | Generic dense cost |
|---|---:|
| Evaluate/store the jet feature matrix \(A\) | \(O(LP)\) entries/work apart from dimension and derivative evaluation factors |
| Form \(G=AA^\top\) naively | \(O(L^2P)\) work and \(O(L^2)\) output storage |
| Factor full-rank \(G\) | \(O(L^3)\) work, \(O(L^2)\) storage |
| Reconstruct one readout from a kernel dual vector | \(O(LP)\) work |
| Direct QR-type least squares when \(L\ge P\) | \(O(LP^2)\) work |
| Apply \(G\) without forming it | One application each of \(A^\top\) and \(A\), plus vector storage |

A generic analytic kernel can sometimes evaluate one entry without a feature sum. Our **exact frozen-neural** kernel (5) contains \(P\) terms under direct summation. Exact tanh center-sum identities can reduce the center-count dependence after preprocessing, but the remaining angular sum and preprocessing must still be counted. If each requested entry costs \(O(M)\) and the sparse factorization requests \(Q\) distinct entries, their evaluation costs \(O(MQ)\) in addition to factor arithmetic. Local dense submatrices, not only the nonzero positions of the output factor, are used by the sparse algorithm. If \(L\ll P\), the dual problem can be attractive; if \(L\) grows with \(P\), it does not remove the large problem. Derivative channels increase \(L\) even before adding extensive validation. The nonlinear optimizer still uses repeated objective/Jacobian applications and has no automatic bounded iteration count.

For our local/profile evaluator, applying \(A\) and \(A^\top\) can use the already derived structure. Kernel duality then inherits that implementation rather than bypassing it. In particular, this is not a justification for forming either \(P\times P\) normal matrices or \(L\times L\) kernel matrices at million-scale sizes.

### What would transfer a fast reference-kernel inverse to the neural matrix?

Here is a precise finite-matrix statement, separate from the hypotheses of the sparse-Cholesky theorem. Let \(G_0\) be a positive definite reference derivative Gram matrix and let the actual neural matrix \(G_n\) satisfy

\[
cG_0\preceq G_n\preceq CG_0,\qquad 0<c\le C.
\tag{12}
\]

Suppose an available positive definite preconditioner satisfies

\[
(1-\eta)G_0^{-1}\preceq B\preceq(1+\eta)G_0^{-1},\quad 0\le\eta<1.
\]

Then

\[
\kappa(B^{1/2}G_nB^{1/2})
\le\frac Cc\frac{1+\eta}{1-\eta}.
\tag{13}
\]

**Proof.** Congruence preserves the Loewner inequalities. The eigenvalues of \(B^{1/2}G_0B^{1/2}\) equal those of \(G_0^{1/2}BG_0^{1/2}\), which lie in \([1-\eta,1+\eta]\). Apply (12) to bound the smallest and largest eigenvalues of the preconditioned neural matrix.

This is a preconditioner for the original neural linear algebra and does not replace the candidate network. But (12) is a substantive requirement, especially as samples and derivative orders increase. A smooth kernel, pointwise kernel approximation, or a small unscaled entry error does not imply it. A sufficient relative approximation is

\[
\|G_0^{-1/2}(G_n-G_0)G_0^{-1/2}\|_2\le\delta<1,
\]

giving \(c=1-\delta\), \(C=1+\delta\). The reference theorem's absolute inverse error must also be small enough to imply the relative \(\eta\) bound; this may require tolerances depending on \(\|G_0\|\). If \(G_n\) is singular, a positive definite comparison on all observation coordinates is impossible; a justified common observable subspace is needed.

More generally a positive definite coefficient metric \(Q\) produces

\[
G_Q=AQ^{-1}A^\top,\qquad
\mathcal K_Q(x,y)=\phi(x)^\top Q^{-1}\phi(y).
\]

Every such kernel still represents only functions in the same finite network span. Choosing \(Q\) as an algorithmic preconditioning device can preserve the original equations, but using it to define the minimum-norm solution changes the selection rule unless that rule was already declared. Neither an arbitrary \(Q\) nor an invertible reparameterization makes the finite kernel a Green kernel of a local isomorphism on an infinite-dimensional Sobolev space. The source theorem's zero-boundary function space and geometric conditions also cannot silently replace arbitrary BCs and the user's domain.

Thus an existing fast reference inverse offers a concrete, established conditional route to preconditioning. To claim it here one must establish the discrete relative comparison, price kernel/metric application, and retain the actual neural operator and nonlinear convergence requirements. The source does not itself establish those comparisons for this Radon construction.

## 5. What this changes in the research framing

The established starting point is fixed-feature collocation with analytic derivative features. The next exact operation is nonlinear composition through sampled jets; its Jacobian is already a sum of point multipliers, derivative banks, and direction multipliers. Selective variable projection is available when a physical unknown enters linearly. Kernel/representer coordinates are an optional equivalent algebraic description only after preserving range constraints and any declared objective; a minimum-norm prior is a separate modeling decision.

The useful contribution to pursue is therefore a fast, stable implementation and inversion strategy for these **specific** derivative-feature operators under the construction's radial/angular geometry, with total site count and conditioning visible. The literature establishes the surrounding formulation. It does not supply an off-the-shelf \(O(MN^2)\), machine-accuracy, general nonlinear-PDE solver for this frozen MLP, and no such guarantee is asserted here.

For the separate question of physical error, the supporting [Batlle–Chen–Hosseini–Owhadi–Stuart error analysis](https://arxiv.org/pdf/2305.04962), Theorem 3.1 and Theorem 3.8, combines inverse PDE stability, a sampling inequality, forward continuity, function-space regularity, and a bound on the candidate's norm. Its optimal-recovery formulation supplies that last bound through norm minimization. Theorem 3.8 includes nontrivial boundaries and interior/boundary fill distances. These conditions explain why accurate finite residuals alone are insufficient; they are not a proof that our unregularized neural candidates already satisfy the required norm or stability bounds.
