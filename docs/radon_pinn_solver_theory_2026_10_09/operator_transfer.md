# Native-neural operator transfer through certified encoding

The objective here is a standard one-hidden-layer tanh output that is solved using its **own** PDE, boundary, initial, and data residuals. A reference representation may organize coordinates and supply a preconditioner. It must not replace the neural residual by a reference PDE solve followed by encoding.

The useful theorem is an operator perturbation result: if the encoding preserves the required differential operators in the right norm, a stable reference coordinate system transfers its local conditioning to the actual network. The theorem does not provide a fast stable reference system for arbitrary domains or difficult nonlinear PDEs. It isolates exactly what such a construction must prove.

## 1. Objects, output, and the admissible function space

Let \(X\) be a normed physical solution space and \(Y\) a Hilbert residual space. Components of \(X\) may include velocity, pressure, temperature, an unknown material field, and finite-dimensional parameters. Components of \(Y\) contain **all** declared PDE, boundary/initial, data, and gauge residuals. Their norms and weights are specified as part of the formulation, not adjusted to hide a poorly satisfied equation.

Write

\[
\mathcal F:X\supset\mathcal U\longrightarrow Y
\]

for the declared nonlinear residual map. We require it to be continuously differentiable on the neighborhood used in a local theorem. This hypothesis is made in function spaces where the actual products, traces, and observations are defined. For instance, point evaluation is not a bounded functional on every Sobolev space, and a strong second derivative is not controlled by an energy norm containing only one derivative. These choices cannot be omitted.

For each reference resolution \(p\), let

\[
V_p a=\sum_{\ell=1}^{p}a_\ell v_{p,\ell},\qquad a\in\mathbb R^p,
\]

be a stable physical coordinate family. A complete/refinable family has

\[
\overline{\bigcup_p\operatorname{range}V_p}^{\,X}=X
\]

or the declared solution subspace. Boundary layers, increasing oscillation, and other difficult solutions require increasing \(p\); no fixed polynomial degree, rank, sparsity, smoothness radius, or known-solution family is imposed.

Let \(S_W\) synthesize the actual fixed-geometry network with \(W=MN\) tanh neurons plus its bias. Suppose a known **linear** encoder \(E_p\) maps reference coefficients to readouts, and define

\[
Q_p=S_WE_p,\qquad q_a=Q_pa.
\tag{1}
\]

The entire optimization uses \(\mathcal F(Q_pa)\). The exported coefficients are \(w=E_pa\); the output remains precisely the flat tanh network \(S_Ww\). No reference basis function or external solver is evaluated as part of the exported field.

There are two distinct function-space contracts:

1. **Preserve every function in the current finite network.** Require \(\operatorname{range}Q_p=\operatorname{range}S_W\). After exact duplicate/antipodal removal, this generally requires \(p\ge\dim\operatorname{range}S_W\), with an encoder of full row rank. A square invertible encoder is sufficient.
2. **Use an encoded, complete sequence of trial spaces.** A rectangular \(E_p\) with \(p<W\) may still approximate every target as \(p,W\to\infty\). It does not impose a fixed target family, but it generally omits functions from the particular width-\(W\) network. Density of the union does not prove equality at finite width.

The transfer theorem covers either contract but does not conflate them. If the task requires the first, a lower-dimensional encoder cannot silently replace it. Conversely, a reference space that grows to resolve each target is not the same assertion as a fixed low-degree prior. Establishing which finite-space contract the eventual method satisfies is a mathematical obligation, not a performance detail.

## 2. Certified neural jets are the input certificate

Set \(D_p=Q_p-V_p\). For each derivative needed by the PDE, require an operator estimate, for example

\[
\left\|\partial^\alpha D_p a\right\|_{Z_\alpha}
\le\delta_{p,\alpha}\|a\|_{A_p},
\qquad |\alpha|\le r,
\tag{2}
\]

and the analogous estimate for every boundary trace, observation, or parameter-field quantity. The coefficient norm \(A_p\) must be specified; it is ideally the reference physical norm, not an unscaled coefficient norm whose constants deteriorate with \(p\).

The spaces \(Z_\alpha\) must control the actual operator. A product \(q\,\Delta q\), for example, can be bounded from suitable \(L^\infty\) control of \(q\) and \(L^2\) control of \(\Delta q\), together with corresponding local bounds. Uniform errors of values alone do not control it. For a general smooth local PDE declaration, one can collect all required jets into a Banach space \(Z\), prove the jet map \(X\to Z\) bounded, and bound the declared finite-dimensional nonlinearities on the relevant jet neighborhood. This is an interface condition, not a claim that every PDE has the same graph norm.

**Simple counterexample to a value-only certificate.** On \([-1,1]\), let

\[
d_\varepsilon(x)=\varepsilon\tanh(x/\varepsilon^2).
\]

This is already one standard tanh neuron. Its uniform value error from zero is at most \(\varepsilon\), whereas

\[
|d_\varepsilon'(0)|=\varepsilon^{-1},\qquad
\|d_\varepsilon''\|_\infty\asymp\varepsilon^{-3}.
\]

Thus arbitrarily accurate uniform field encoding can have arbitrarily inaccurate differential residuals. The relevant certificate must be for actual neural derivatives and traces, not an analytic derivative of the underlying reference function.

In particular, increasing \(\gamma\) in the fixed-\(\lambda\) regime moves the finite tanh poles toward the real domain. A fixed analytic tube bound on an underlying target does not become a fixed analytic tube bound on the finite neural output. The needed statement is the real-domain jet/operator defect (2), with the appropriate encoding constants.

## 3. Linear operator transfer without replacing the residual

First let \(\mathcal F(q)=Lq-f\), where \(L:X\to Y\) includes the full linear constraint stack. Define

\[
A_p^0=LV_p,\qquad A_p=LQ_p.
\]

Suppose the reference residual Gram matrix

\[
H_p^0=(A_p^0)^*A_p^0
\]

is positive definite after genuine physical gauges have been fixed. Its ideal right preconditioner is \(P_p=(H_p^0)^{-1/2}\), for which \(A_p^0P_p\) has all singular values equal to one. An implementable approximate preconditioner may instead satisfy

\[
\mu\|z\|_2\le\|A_p^0P_pz\|_Y\le\nu\|z\|_2.
\tag{3}
\]

**Transfer proposition.** If

\[
\eta_p=\|L(Q_p-V_p)P_p\|_{\ell^2\to Y}<\mu,
\tag{4}
\]

then

\[
(\mu-\eta_p)\|z\|_2
\le\|A_pP_pz\|_Y
\le(\nu+\eta_p)\|z\|_2.
\tag{5}
\]

**Proof.** Write \(A_pP_p=A_p^0P_p+LD_pP_p\) and use the triangle and reverse triangle inequalities. This also proves injectivity of the encoded linearized problem.

For an exact reference isometry and \(\eta_p\le\eta<1\), the preconditioned normal matrix has eigenvalues in

\[
[(1-\eta)^2,(1+\eta)^2],
\]

uniformly in width. These are ordinary operator perturbation inequalities; no novelty is claimed.

The solve remains

\[
\min_a\|LQ_pa-f\|_Y^2,
\]

with every residual component present. Reference solves may apply \(P_p\) during this optimization. Solving \(LV_pa=f\) and returning \(Q_pa\) is a different procedure and is not licensed by the proposition.

A reference left inverse can also help construct updates, but replacing the objective by the norm of the left-inverted residual can annihilate residual components. In an overdetermined system, a left inverse is not injective on the whole residual space. Therefore such a replacement is not an exact reformulation of the original constraint stack without an additional proof.

## 4. Local nonlinear transfer, including inverse fields

At the same coefficient vector \(a\), compare

\[
J_p^0(a)=\mathcal F'(V_pa)V_p,
\qquad
J_p(a)=\mathcal F'(Q_pa)Q_p.
\]

The difference has the exact decomposition

\[
J_p(a)-J_p^0(a)
=\mathcal F'(Q_pa)D_p
+\left[\mathcal F'(Q_pa)-\mathcal F'(V_pa)\right]V_p.
\tag{6}
\]

Assume in a stated neighborhood that

\[
\|\mathcal F'(u)\|_{X\to Y}\le C_F,
\qquad
\|\mathcal F'(u)-\mathcal F'(v)\|_{X\to Y}
\le L_F\|u-v\|_X.
\]

For a reference right preconditioner \(P_p(a)\), (6) gives

\[
\|[J_p(a)-J_p^0(a)]P_p(a)\|
\le
C_F\|D_pP_p(a)\|
+L_F\|D_pa\|_X\|V_pP_p(a)\|.
\tag{7}
\]

The norms on the right are the indicated coefficient-to-physical-space norms. A sharper proof may use individual jet norms rather than a single \(X\) bound.

If the right side is at most \(\eta<\mu\), and the reference Jacobian satisfies (3) throughout the neighborhood with width-independent \(\mu,\nu\), the **actual neural Jacobian** satisfies (5) there. This transfers local least-squares conditioning. It does not establish global convergence, existence of a solution within the finite trial space, or width-independent iteration counts for an arbitrary globalization strategy.

At a consistent neural root, the least-squares Hessian is \(J_p^*J_p\), so the bound gives local curvature in the preconditioned coordinates. At a nonzero-residual stationary point, the Hessian includes

\[
\langle \mathcal F(Q_pa),\,\mathcal F''(Q_pa)[Q_ph,Q_ph]\rangle_Y.
\]

A small-residual/second-derivative bound is required to control this term. It cannot be dropped when claiming convergence near an inexactly representable solution.

For inverse problems, take \(X\) to be the product of state, unknown coefficient fields, and finite parameters. Encode each represented field; finite parameters can use an exact identity encoding. Equation (6) remains valid. Identifiability is then a lower bound on the **joint** physical Jacobian, not merely the state PDE operator. No encoding or preconditioner repairs missing experimental information.

For Navier–Stokes-type systems, the velocity-pressure pair, divergence equation, initial/boundary constraints, and pressure gauge belong in these product spaces. At regimes where a linearization is singular or nearly singular, the reference stability constant can deteriorate for physical reasons. The theorem permits difficult solutions if they are resolved and locally stable in the chosen formulation; it does not promise globally unique steady solutions or width-independent conditioning across every flow regime. Treating a time-dependent problem requires the corresponding space-time or time-step stability assumptions, rather than an unannounced switch to a simpler steady equation.

## 5. Per-basis-function accuracy is not an operator norm bound for free

Suppose each preconditioned reference coordinate has a defect bounded by

\[
\|LD_pP_pe_\ell\|_Y\le\epsilon_\ell.
\]

Then Cauchy–Schwarz gives

\[
\eta_p\le
\left(\sum_{\ell=1}^p\epsilon_\ell^2\right)^{1/2}.
\tag{8}
\]

Equal per-column error \(\epsilon\) only yields \(\eta_p\le\sqrt p\,\epsilon\). This is sharp when all error columns point in the same residual direction. A dimension-independent encoding certificate cannot be inferred from excellent accuracy on one reference function.

Possible valid ways to avoid this accumulation are a direct relative operator estimate, bounded-overlap error supports with quantified interaction constants, or orthogonality/cancellation proved for the full encoder. These are structural theorems to establish, not assumptions supplied by the word “Radon.” Measuring errors in raw columns before preconditioning adds the factor \(\|P_p\|\), unless the coefficient norm has already absorbed it.

Similarly, \(\|D_pa\|\le\delta_p\|a\|\) requires a bound on the reference coordinates of the relevant nonlinear neighborhood. Stable physical reference coordinates make that bound meaningful. A large hidden coordinate norm can negate a small entrywise encoding error.

## 6. Sampling, all constraints, and actual solution accuracy

For a quadrature/observation map \(T_K:Y\to\mathbb R^K\), the sampled neural residual is

\[
r_{p,K}(a)=T_K\mathcal F(Q_pa).
\]

All preceding perturbation bounds can be stated directly in the sampled norm. To infer the continuous PDE residual, one also needs norm equivalence or a separate validated error bound on the relevant residual family. In particular,

\[
c_s\|y\|_Y^2\le\|T_Ky\|_2^2\le C_s\|y\|_Y^2
\]

must be justified on the linearized residual range, and residual validation must cover the nonlinear residuals encountered. A linear tangent-space certificate alone does not certify every nonlinear residual globally.

Scattered points and nonuniform grids are allowed. Their weights, fill distances or other norming properties, and the total \(K\) must be counted. With unweighted sampling, a per-point error \(\epsilon\) can grow to \(\sqrt K\epsilon\) in the sampled Euclidean norm. Physical quadrature normalization can remove that purely normalization-induced growth; it cannot fix missing spatial coverage. For \(p\) locally identifiable independent coefficients, the sampled Jacobian needs at least \(p\) independent scalar constraints. The number of scalar equations per physical sample must therefore be explicit.

If the declared continuous nonlinear PDE has a local stability estimate

\[
\|q-u\|_X\le C_{\rm phys}\|\mathcal F(q)\|_Y
\tag{9}
\]

near its solution \(u\), then an actually evaluated neural residual bounds field error through (9). Total error includes neural approximation, nonlinear/linear iteration, sampling validation, activation/derivative evaluation, and compilation rounding. A tiny training-point loss is not this certificate.

A convenient approximation argument is: if \(V_pa_p\) approximates \(u\), and \(D_pa_p\) is small in the necessary graph norm, then \(Q_pa_p\) provides a good candidate in the actual neural space. The nonlinear solve still optimizes its own residual to find such a candidate; the argument does not assume the unknown \(a_p\) is supplied.

Completeness transfers as follows. If, for every \(u\), there are coefficients \(a_p\) with \(V_pa_p\to u\) and \(D_pa_p\to0\) in \(X\), then \(Q_pa_p\to u\). A uniform operator defect together with uniformly bounded coefficient norms is one sufficient condition. Density of reference spaces by itself is insufficient if encoding constants grow faster than refinement improves approximation.

## 7. Compilation and the cost ledger

Every application of the actual neural Jacobian or its transpose can exploit the given \(\Phi^{(r)}D\) structure. The reference transform never removes the need to calculate or certify those actual neural quantities. The encoded coordinate operations require \(E_p\) and \(E_p^*\) as well.

A dense stored encoder has \(Wp\) entries; at \(p\asymp W\) it alone destroys the target cost. A useful theorem must provide an explicitly structured encoder, transpose, and compiler. The same obligation applies to the reference inverse: invoking a generic finite-element, spectral, or Green-function inverse is not a cost bound and can hide the entire hard PDE solve.

The cost ledger includes:

- reference basis construction and its physical stability constants;
- encoder construction and operator-defect certification;
- \(E_p,E_p^*\) application, coefficient storage, and final compilation;
- actual neural jet evaluation and transpose aggregation over \(K\) samples and \(M\) directions;
- reference preconditioner setup and each application;
- local nonlinear iterations, linear iterations, validation, and precision requirements.

For example, the rotationally symmetric angular Gram factorization in `stability.md` has \(O(MN^2)\)-type applications but generic \(O(MN^3)\) setup. It is a diagnostic exact result, not a proved fast reference system for arbitrary domains. Fast angular aggregation is still required when \(K\) grows with the number of independent coordinates; retaining all scalar constraints must not quietly reintroduce a hidden \(M^2\) cost.

The flat output requirement adds a precision obligation. If \(\widehat w=E_pa+\delta w\) is the compiled readout, the relevant error is \(\|S_W\delta w\|_X\), including required derivatives. Highly cancelling readouts may need more internal precision or a certified evaluation scheme. An abstract stable reference coordinate norm does not establish accurate float64 compilation and ordinary summation automatically. No output layer or activation is changed by higher-precision coefficient construction, but its cost and output precision must be stated.

## 8. What this establishes, and the missing result

The transfer framework establishes a concrete admissible role for an auxiliary representation: it can supply **coordinates and preconditioning for a solve of the actual tanh network**. Differential-operator defects, not only function-value approximation, are the bridge. If those defects are small in a relative operator norm and the reference physical problem is locally stable, the neural Jacobian inherits stable conditioning. Boundary conditions, observations, inverse fields, scattered sampling, and all nonlinear coupling remain in the actual residual.

What is still missing is a constructive family for the stated general use cases with all three properties:

1. Complete/refinable physical coordinates, or exact coverage of the full finite network if that stronger contract is retained, with width-uniform stability in the relevant graph norm.
2. A structured encoder with relative neural-jet defect control, no uncounted dimension factor, and stable compilation to the flat network.
3. Fast reference operations and sample-to-direction aggregation on the actual geometry, with setup, sampling, nonlinear iterations, and precision included in an approximately linear-in-\(M\) cost bound.

Neither a fixed low-degree space nor an external PDE solve followed by encoding satisfies this missing theorem. The perturbation argument explains how a genuine construction would work; it does not claim that the fast angular/general-domain construction has been found.
