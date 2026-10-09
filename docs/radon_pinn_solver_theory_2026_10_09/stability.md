# Stability of the finite Radon network without sacrificing its function space

This report addresses the actual finite network, including all independent readouts. The main result is a precise separation: physical PDE stability plus a stable coordinate map would yield width-independent conditioning, but canonical continuous Radon coordinates do not automatically supply such a finite map. Direction-by-direction normalization alone provably cannot do so. An angular Fourier decomposition gives an exact constructive result in a symmetric geometry, with explicit costs and limitations.

No experiment was run and no admissible function was removed. All propositions below are mathematical derivations for this investigation; the references identify established surrounding theory, not a claim that this particular finite-network method is already implemented.

## 1. The finite synthesis map and the correct stability target

Let

\[
(Sw)(x)=b+\sum_{m=1}^M\sum_{j=1}^N
w_{jm}\tanh\!\left(\gamma(\theta_m\cdot x-c_j)\right),
\qquad V=\operatorname{range}S.
\]

The domain has nonempty interior. Let \(X\) be a Hilbert norm measuring the physical solution, including sufficient derivatives or traces for the declared residual. Let \(\mathcal F(q)\) include PDE, initial/boundary, observation, and gauge residuals. At a candidate field \(q\), write \(L_q=\mathcal F'(q)\).

**Conditional coordinate theorem.** Suppose

\[
a\|v\|_X\le\|L_qv\|_Y\le b\|v\|_X,
\qquad v\in V,
\tag{1}
\]

with \(a>0\), and all sampled residuals define a norm equivalent to the continuous residual norm on \(L_qV\), with constants \(c_s,C_s>0\). Let

\[
G=S^*S,\qquad w=G^{-1/2}z,
\]

where \(S^*\) uses the \(X\) inner product. After removing only exact coefficient redundancies, \(G\) is positive definite. Then

\[
\|SG^{-1/2}z\|_X=\|z\|_2,
\]

and the singular values of the sampled Jacobian in \(z\) coordinates belong to

\[
[\sqrt{c_s}\,a,\;\sqrt{C_s}\,b].
\tag{2}
\]

**Proof.** The isometry is the identity \(G^{-1/2}GG^{-1/2}=I\). Apply (1) and the sampling norm bounds to \(v=SG^{-1/2}z\). No approximation or discarded function is involved.

More generally a computable change \(w=Pz\) with

\[
\alpha I\preceq P^*GP\preceq\beta I
\tag{3}
\]

produces the condition bound \(\kappa\le(b/a)\sqrt{C_s\beta/(c_s\alpha)}\). This is the exact theorem a proposed fast transform must satisfy. For nonlinear problems it is a local Jacobian statement, not global Newton convergence. A uniform neighborhood, derivative regularity, and globalization conditions remain necessary.

Three distinct failures can occur: \(a\) small is physical ill-conditioning; \(c_s\) small is inadequate sampled constraint control; \(\alpha/\beta\) small is basis/coordinate ill-conditioning. Treating all three as readout conditioning hides the relevant mathematical problem.

Dense computation of \(G^{-1/2}\) takes quadratic storage and cubic generic setup in \(P=MN+1\), so the existence of the isometry is **not** a complexity result. Matrix-free evaluation of \(G\) does not by itself make its square root or inverse cheap.

## 2. Exact redundancy is smaller than continuous Radon redundancy

**Finite tanh independence.** For a common positive \(\gamma\), distinct centers within each direction, and direction/center pairs distinct modulo

\[
(\theta,c)\sim(-\theta,-c),
\]

the finite tanh features and the constant are linearly independent on every open real domain.

**Proof sketch with the singularity argument.** Extend the features meromorphically to \(\mathbb C^d\). A feature has simple pole hyperplanes

\[
\theta\cdot z=c+\frac{i\pi(k+1/2)}\gamma,\qquad k\in\mathbb Z.
\]

Nonparallel real directions give different pole hyperplanes. For the same oriented unit direction, distinct real centers give disjoint parallel pole hyperplanes. Antipodal pairs have already been identified. Choose a pole hyperplane of one feature, and a point on it that is not on any pole hyperplane of the finitely many other features (their countable intersections cannot cover it). Equality on a real open domain analytically continues to a meromorphic identity, so the residue at that point forces this feature's coefficient to be zero. Repeat for every feature; the remaining constant vanishes. In one dimension distinct centers have disjoint poles and the same argument is immediate.

Thus the common symmetric center/direction construction has exact identities from \(\tanh(-s)=-\tanh(s)\), and literal duplicates can be merged. Apart from these identities, there is generally no exact coefficient gauge to quotient away.

This matters for canonical Radon coordinates. A continuous canonical analysis map may synthesize the same function through a unique selected continuous profile family, but it generally does not return profiles in the original finite tanh span. The original globally defined ridge network is not usually a decaying whole-space function either; canonical analysis requires an explicit domain extension and its costs.

If a finite projection \(P_c\) purports to enforce canonical profile relations while preserving all represented functions, it must satisfy

\[
SP_c=S.
\]

When \(S\) is injective, this implies \(P_c=I\). A nontrivial canonical restriction therefore needs either a proved exact redundancy, a larger exact coordinate representation, or an approximation/restriction. It must not silently turn into a lower-dimensional function family.

## 3. A precise obstruction to independent-direction preconditioning

Consider two features on a ball centered at zero,

\[
f_0(x)=\tanh(\gamma x_1),\qquad
f_\varepsilon(x)=\tanh\!\left(\gamma(\cos\varepsilon\,x_1+\sin\varepsilon\,x_2)\right).
\]

In any fixed finite-order Sobolev norm on the ball, smooth parameter dependence gives

\[
f_\varepsilon=f_0+\varepsilon h+O(\varepsilon^2),
\qquad h(x)=\gamma x_2\operatorname{sech}^2(\gamma x_1)\ne0.
\tag{4}
\]

For a rotationally invariant inner product, \(\|f_\varepsilon\|=\|f_0\|\) and \(\langle f_0,h\rangle=0\). The normalized correlation satisfies

\[
\rho_\varepsilon=1-\frac{\|h\|^2}{2\|f_0\|^2}\varepsilon^2+O(\varepsilon^3).
\]

The two-feature normalized Gram matrix has eigenvalues \(1\pm\rho_\varepsilon\), hence condition number \(\Theta(\varepsilon^{-2})\). Independent diagonal rescaling cannot remove the correlation: minimizing the condition number of a positive two-by-two matrix over such scalings gives equal normalized diagonal entries and the same \((1+|\rho|)/(1-|\rho|)\) bound.

This is already a counterexample to any universal direction-only block preconditioner, by taking one center per direction. Dense angular refinement supplies arbitrarily close directions. The obstruction occurs even for the perfectly stable identity observation operator; it is a basis defect, not a difficult PDE.

The normalized small mode

\[
\frac{f_\varepsilon-f_0}{\varepsilon}\longrightarrow h
\]

is a genuine nonzero physical function. Removing it because the original Gram eigenvalue is small restricts the function space. An allowed remedy is to represent the difference explicitly and retain it.

For a pair, the exact invertible coordinates consisting of a mean and a divided difference do this. Their values can be evaluated without subtracting almost equal rounded activations, using

\[
\tanh b-\tanh a=\frac{\sinh(b-a)}{\cosh a\cosh b},
\]

with an overflow-safe implementation. This handles the elementary cancellation without changing the network. It is not yet a uniform transform for arbitrarily many directions: higher angular moments and many simultaneous nearly dependent combinations must also be controlled. Ordinary pairwise/Haar differences do not establish (3).

## 4. An exact positive result: angular symmetry can supply the desired M scaling

Assume **for this proposition only** that the physical domain and \(X\) inner product are rotation-invariant in two dimensions, the center grid is identical in every direction, and

\[
\theta_m=(\cos(2\pi m/M),\sin(2\pi m/M)).
\]

Let \(\phi_{jm}\) be the corresponding tanh feature. Rotation invariance gives

\[
G_{(j,m),(k,\ell)}=H_{jk}(m-\ell\pmod M),
\tag{5}
\]

so \(G\) is block-circulant in the angular index. The exact discrete Fourier transform in that index yields

\[
(F_M\otimes I_N)G(F_M^*\otimes I_N)
=\operatorname{diag}(\widehat H_0,\ldots,\widehat H_{M-1}),
\tag{6}
\]

where each \(\widehat H_\nu\) is an \(N\times N\) Hermitian block. The bias couples only to angular frequency zero. Exact antipodal dependencies appear as exact null modes and may be quotiented, or an independent geometry may be chosen from the start. Small nonzero eigenvalues must remain.

Consequently a physical isometry can be formed by angular Fourier transformation followed by block inverse square roots. After the blocks have been factored, its application costs

\[
O(NM\log M+MN^2),
\]

and stores \(O(MN^2)\) values. This is an exact finite-network coordinate change, preserving every independent readout combination. It remains useful for non-rotation-invariant nonlinear PDE coefficients and observations if the physical stability and sampling hypotheses of Section 1 hold: the **reference norm**, not the PDE Jacobian, is what is diagonalized.

The honest setup ledger is:

- \(MN^2\) distinct Gram entries before the angular transform; calculating them to certified precision has its own integration cost.
- \(O(N^2M\log M)\) angular transformation of those entries.
- Generic \(O(MN^3)\) radial block factorizations.
- Potentially growing arithmetic precision to resolve small nonzero eigenvalues and apply the transform accurately.

There is no established \(O(MN^2)\) setup theorem here. Nevertheless, this proves that angular coupling need not force \(M^2\) complexity for the exact physical Gram matrix in a suitable geometry. It is a more substantive starting point than independent-direction normalization.

**Scope boundary.** One may not silently replace an arbitrary physical domain with a ball, impose periodic directions on an arbitrary supplied grid, or truncate angular modes. A ball norm outside the observed domain is not automatically uniformly equivalent to the physical norm on that domain as the network grows. The rotation-invariant proposition is a special-case proof and possible building block, not a proposed restriction of Sam's requested general setting.

**Explicit failure of a larger-ball reference norm.** Let the physical domain be the left half of the unit ball, and suppose the bank contains direction \(e_1\), center \(c=1/2\), and the bias. The exact represented function

\[
v_\gamma(x)=1+\tanh\!\left(\gamma(x_1-1/2)\right)
\]

has \(\|v_\gamma\|_{H^r(\Omega)}\le C_r(1+\gamma^r)e^{-\gamma}\), while its norm on the full ball stays bounded below because it approaches 2 on a fixed cap with \(x_1>3/4\). Thus the larger-ball norm and physical-domain norm have no width-uniform equivalence when \(\gamma\propto N\), precisely the fixed-\(\lambda\) regime. Even identity observation on the physical domain is exponentially small relative to the ball-normalized mode. This counterexample needs no claim about exotic inverse problems or arbitrary analytic continuation.

Using the ball isometry still preserves all network functions, so the change itself is admissible. What fails is its proposed uniform conditioning guarantee. Assuming a uniform extension bound would remove this counterexample, but an unproved bound cannot be inserted into the general theorem. Requiring all admissible readouts to satisfy such a bound would be an additional restriction; knowing that the unknown target alone has a bounded analytic extension does not prevent the unrestricted optimization space from containing these poorly observed perturbations.

In higher dimensions, spherical harmonics diagonalize the continuous rotation-invariant angular kernel, but a finite spherical quadrature grid does not automatically diagonalize this nonpolynomial kernel exactly. Exactness for a bounded polynomial degree cannot be applied to all angular modes of the tanh features. A certified *relative* approximation to the Gram matrix would be sufficient; a small absolute kernel error alone is insufficient near tiny Gram eigenvalues. General group-based transforms may also incur irreducible-representation block sizes and multiplicities; these are not a free extension of the planar FFT cost.

## 5. The constructive general theorem still needed

One route is an exact multilevel angular/radial decomposition whose synthesis blocks are \(T_\ell\). It must cover the same finite space and satisfy, on its exact redundancy quotient,

\[
A\sum_\ell\|z_\ell\|^2
\le\left\|\sum_\ell T_\ell z_\ell\right\|_X^2
\le B\sum_\ell\|z_\ell\|^2,
\tag{7}
\]

with \(A,B\) independent of \(M,N\), and with both analysis and synthesis having a demonstrated near-linear-in-\(M\) cost. For redundant frame coordinates the correct condition is instead a bounded frame operator/invertible quotient metric; (7) cannot hold on redundant coefficient vectors themselves.

A related subspace-correction formulation requires uniformly stable decompositions and a bounded interaction constant, plus inexpensive subspace inverses. Small per-direction bandwidth only addresses the last item. The near-direction example shows why an angular hierarchy is necessary; it does not prove that one is sufficient.

The established subspace-correction framework gives the right conditional convergence machinery ([Xu, 1992](https://epubs.siam.org/doi/10.1137/1034116)). Continuous ridgelet frames supply stable representations under their own domain, scale, and measure assumptions ([Candès, *Ridgelets: Estimating with Ridge Functions*](https://candes.su.domains/publications/downloads/Ridgelet_Stat.pdf)). Neither result alone certifies (7) for this finite fixed-slope tanh bank on arbitrary domains. A proof must explicitly connect the candidate frame to the **same finite span**, rather than replace the span with a wavelet/Fourier truncation.

## 6. Data, inverse problems, and machine precision

Let \(p=\dim V\) after exact redundancy removal. If there are \(K<p\) independent scalar sampled residuals, their linearization \(J:\mathbb R^p\to\mathbb R^K\) has a nonzero nullspace. No positive sampled lower stability constant is possible. This is a statement about stable local identification, not a claim that a nonlinear map with singular derivative can never be injective. A differentiable map with a locally Lipschitz inverse must have an injective derivative. Additional unknown coefficient fields add dimensions and must be counted.

Even unlimited accurate boundary/PDE samples do not remove structural inverse nonidentifiability. For example, conductivity \(a(x)\) observed only through one constant solution of \(-\nabla\cdot(a\nabla q)=0\) remains undetermined for every positive \(a\). No coordinate change can create missing experimental information. Multiple excitations, extra observations, or a prior would alter that inference problem and need Sam's decision if not already part of its declaration.

For an identifiable inverse problem, small singular values of the *physical* derivative still limit data-to-solution precision. Whitening readout coordinates does not change that information limit. PDE and observation noise must be distinguished from rounding errors.

For floating-point work the necessary target is a bound such as

\[
\|\delta q\|_X\lesssim
\frac1a(\text{model/discretization error}+\text{sampling error}+\text{operator arithmetic error}),
\]

plus nonlinear local terms. Machine epsilon alone is not a physical error bound. Exact divided differences, compensated or higher-precision internal arithmetic, and certified approximations can preserve the same function space; discarding tiny nonzero modes cannot. If raw readouts are required to be returned in fixed precision and subsequently evaluated by ordinary independent tanh summation, that additional output contract may itself defeat stable transformed coordinates. It has not been assumed here.

**Analytic target bounds do not transfer literally to the finite tanh output.** A tanh neuron has complex poles at normal distance \(\pi/(2\gamma)\) from its real transition hyperplane. Distinct nonduplicate real features cannot cancel those pole hyperplanes, by the independence argument above. In the fixed-\(\lambda\) regime, \(\gamma\) grows with \(N\), so a width-independent complex tube containing those transition locations eventually contains actual poles of every active such neuron. Imposing a fixed analytic tube bound directly on the finite neural output can therefore exclude all nonconstant outputs whose active pole hyperplanes intersect that tube. An analytic certificate belongs to an underlying target or reference profile, together with a quantified finite-network encoding defect. It must not be silently imposed on the actual finite tanh field. Real Sobolev or real extension bounds are a different possible condition and need their own explicit statement.

## 7. Admissibility classification and next decision

| Route | Status |
|---|---|
| Quotient literal duplicate/antipodal features | Exact and admissible; only modest dimension reduction |
| Cumulative radial coordinates and exact angular divided differences | Invertible and admissible; no complete uniform stability proof yet |
| Physical Gram isometry or a certified relative approximation | Admissible in principle; fast general construction is unproved |
| Angular Fourier block factorization under actual rotational symmetry | Exact special-case theorem; application has requested M scaling; setup/precision costs remain |
| Use a symmetric auxiliary geometry without a uniform norm-equivalence proof | Not justified for the original general problem |
| Retain only canonical low-degree/bandlimited Radon profiles | Changes the function family unless equivalence is proved; do not adopt silently |
| Remove small nonzero Gram singular modes | Changes admissible functions; do not adopt silently |
| Add smoothness regularization to cure inverse nonidentifiability | Changes inference problem; requires an explicit decision |
| Claim all generic nonlinear PDEs converge to machine precision | Unsupported; local well-posedness and arithmetic assumptions are indispensable |

The useful next mathematical question is whether the **actual supplied direction grid and physical domain** admit a fast, relatively accurate physical Gram factorization or a uniformly stable multilevel substitute. The present report does not prove that theorem. Restricting to a symmetric domain would permit progress on the exact Fourier construction, but it must be labeled a special-case theory study rather than substituted for the required general PINN capability.
