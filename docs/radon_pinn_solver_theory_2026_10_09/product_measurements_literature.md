# Products of two linear measurements of the same neural readout

This is the first nonlinear rung, before general nonlinear PDEs. The equations are

\[
a_i=\langle A_i,W\rangle_F,\qquad
b_i=\langle B_i,W\rangle_F,\qquad a_i b_i=k_i,
\quad i=1,\ldots,K.
\]

Here \(W\in\mathbb R^{N\times M}\) is unrestricted. No sparse or low-rank assumption is made on \(W\). A common constant right side is the special case \(k_i=k\). Boundary, initial, and point-data equations may additionally be affine-linear in this same readout. All discussion keeps the original flat tanh MLP.

The relevant literature is **generalized phase retrieval / quadratic sensing**, with bilinear recovery as a related, but not identical, model. The precise matching is useful: this problem has a rank-one lift for every possible \(W\), but storing that lift is unnecessary and prohibitive at the intended width.

## 1. Exact quadratic structure, with no large tensor

Let \(p=MN\), \(w=\operatorname{vec}W\), \(\alpha_i=\operatorname{vec}A_i\), and \(\beta_i=\operatorname{vec}B_i\). Define operators \(A,B:\mathbb R^p\to\mathbb R^K\) with these row vectors. Then

\[
F(w)=(Aw)\odot(Bw),\qquad
F_i(w)=w^TQ_iw,
\quad Q_i=\tfrac12(\alpha_i\beta_i^T+\beta_i\alpha_i^T).
\tag{1}
\]

Each \(Q_i\) has rank at most two. Its possible nonzero eigenvalues are

\[
\tfrac12\bigl(\alpha_i^T\beta_i\pm\|\alpha_i\|\|\beta_i\|\bigr).
\]

Thus an independent pair of factors gives one positive and one negative eigenvalue. Standard intensity phase retrieval, where \(A=B\), is the positive rank-one special case. Replacing a general product by squared magnitudes would change the equations.

The rank-two matrices need never be assembled. Write \(u=Aw\), \(v=Bw\), and \(r=u\odot v-k\). The exact actions are

\[
J(w)h=v\odot Ah+u\odot Bh,
\qquad
J(w)^Tz=A^T(v\odot z)+B^T(u\odot z).
\tag{2}
\]

For

\[
\ell(w)=\tfrac12\|r(w)\|_2^2+	frac12\|Cw-d\|_2^2,
\]

where \(Cw=d\) is the linear constraint stack, the objective is quartic plus quadratic, and

\[
\nabla^2\ell(w)h
=J^TJh+A^T(r\odot Bh)+B^T(r\odot Ah)+C^TCh.
\tag{3}
\]

The Hessian of one product residual is rank at most two; the Hessian of its squared residual also has range in \(\operatorname{span}\{\alpha_i,\beta_i\}\), hence rank at most two. At a zero-residual solution, the quadratic-product part reduces to \(J^TJ\), with rank at most \(K\). Away from a solution, the exact Hessian can be indefinite. Across all products, its range lies in the span of the rows of \(A\) and \(B\), so its rank is at most \(\min(p,2K)\), before adding \(C\).

For the Radon construction, \(A\) and \(B\) are precisely flattened derivative banks \(\Phi^{(r)}D\), possibly different orders. Equations (1)–(3) preserve that structure. Nonlinearity adds pointwise products and a fixed number of structured forward/transpose operations, rather than an explicit \(M^2\) sum.

This proves cheap *operator actions conditional on cheap A/B actions*. It does not prove a cheap globally convergent solve or permit hiding sample count \(K\) in the complexity.

## 2. Exact symmetry, zero branches, and affine constraints

The homogeneous product map always satisfies \(F(-w)=F(w)\). If at least one \(k_i\ne0\), rescaling a solution by a real scalar preserves the product equations only for scalar \(\pm1\). Unlike a two-independent-factor problem, there is no general continuous ambiguity \((h,m)\mapsto(th,t^{-1}m)\), because both factors here come from the same \(w\).

For a nonzero \(k_i\), neither factor can vanish and the factor signs obey \(\operatorname{sign}(u_i)\operatorname{sign}(v_i)=\operatorname{sign}(k_i)\). The signs of the individual factors generally remain unknown. Locally they cannot change without passing through a zero or leaving feasibility.

When every \(k_i=0\), the exact feasible set is

\[
\bigcup_{I\subseteq\{1,\ldots,K\}}
\left(\ker A_I\cap\ker B_{I^c}\right).
\tag{4}
\]

This union includes potentially many branch subspaces and the zero readout. At \(w=0\), \(J(w)=0\); a Newton correction based only on this Jacobian cannot recover a nonzero branch. When exactly one factor vanishes, the local row equation selects the derivative of that factor. If both vanish, its first-order constraint is zero and second-order structure matters. Affine data intersect these sets and can remove branches or the zero solution.

Nonzero affine data \(Cw=d\) break global sign symmetry when \(d\ne0\); homogeneous data do not. If exact constraints are eliminated using \(w=w_0+Zz\), the products acquire known linear and constant terms in \(z\). This is exact, but a dense nullspace matrix \(Z\) can destroy fast Radon operations and cannot be treated as free.

There is a useful exact midpoint identity:

\[
F(v+h)-F(v-h)=2J(v)h.
\tag{5}
\]

Therefore, on an affine space \(\mathcal A=\{w:Cw=d\}\) with \(d\ne0\), the product measurements are globally injective if and only if

\[
\ker J(v)\cap\ker C=\{0\}
\quad\text{for every }v\in\mathcal A.
\tag{6}
\]

Indeed, every indistinguishable pair has a feasible midpoint and a difference in \(\ker C\); conversely such a null direction produces the pair \(v\pm h\). This is an algebraic characterization, not an efficient test over all feasible \(v\). At one candidate solution, full rank of the stacked \([J(w);C]\) gives local identification only. In the homogeneous unconstrained case, the unavoidable \(\pm\) pair corresponds to midpoint zero and must be factored out.

## 3. What lifting buys—and what it does not

Set \(X=ww^T\). Then

\[
F_i(w)=\langle Q_i,X\rangle_F,
\qquad X\succeq0,\quad\operatorname{rank}X=1.
\tag{7}
\]

This rank-one condition is automatic for **every** readout matrix \(W\), including a full-rank \(N\times M\) matrix. It is not a low-rank prior on the neural weights. It changes a quadratic map into linear measurements of a rank-one positive matrix.

A literal lift contains \(p(p+1)/2\) scalars. At a million readouts that is roughly half a trillion unknown entries. Removing the rank-one constraint gives a relaxation, not an exact reformulation unless exact recovery is proved. Returning a higher-rank matrix or its best rank-one approximation does not automatically solve the original equations.

Affine constraints can be kept exactly by augmenting \(z=(1,w)\) and lifting \(Z=zz^T\), requiring \(Z_{00}=1\). Then each data row is linear in \(Z_{0,1:p}\), and products use its lower-right block. The rank-one requirement remains essential. This augmentation explains how boundary data fit the algebra; it does not make the large SDP viable.

Factoring the lift as \(X=ww^T\) from the outset removes quadratic storage, but returns exactly the original quartic optimization. A factor with several columns enlarges the lifted feasible class; its output is not one MLP unless a rank-one recovery certificate is subsequently proved.

## 4. Primary-source literature and actual hypotheses

The theorem statements and surrounding definitions were read in the linked full PDFs, rather than inferred from abstracts. Page references below are printed paper pages where present. Each source summary is deliberately short; the algebra above and applicability analysis below are our derivations.

**Wang and Xu, *Generalized phase retrieval: measurement number, matrix recovery and beyond*.** Theorem 2.1, p. 8, equates injectivity up to global sign for real symmetric quadratic measurements with full-rank Jacobian at every nonzero signal, equivalently spanning by the vectors \(Q_iw\). Theorem 4.1, pp. 19–20, gives injectivity for generic real symmetric sensing matrices of prescribed ranks when \(K\ge2p-1\). These are identifiability results, not a convergence or operation-count theorem. Generic indefinite rank-two matrices are relevant to (1), but a fixed PDE derivative bank is not shown generic by this theorem. [Full text, Theorems 2.1 and 4.1](https://arxiv.org/pdf/1605.08034)

**Candès, Strohmer, and Voroninski, *PhaseLift*.** Equations (1.4)–(1.5), pp. 2–3, give rank-constrained PSD lifting and its trace-minimization relaxation. Theorem 1.1, pp. 3–4 in the inspected version, proves exact recovery for independent uniformly distributed sphere measurement vectors with \(K\ge c p\log p\), with high probability. Theorem 1.2, p. 6, supplies noisy stability under its stated normalization. This is rank-one positive sensing, unlike the generally indefinite product matrices here. [Full text, Theorems 1.1–1.2](https://arxiv.org/pdf/1109.4499)

**Candès, Li, and Soltanolkotabi, *Phase Retrieval via Wirtinger Flow*.** Theorem 3.3, p. 5, proves spectral initialization and geometric convergence for Gaussian magnitude-squared measurements with \(K\ge c p\log p\). The stated step restriction is of order \(1/p\); the following discussion, p. 6, gives \(O(p\log(1/\varepsilon))\) iterations and two sensing-operator actions per step. Thus even a nonlifted convergence theorem must be read with its iteration dependence. It does not directly cover correlated indefinite PDE products. [Full text, Theorem 3.3 and following complexity discussion](https://arxiv.org/pdf/1407.1065)

**Bhojanapalli, Neyshabur, and Srebro, *Global Optimality of Local Search for Low Rank Matrix Recovery*.** Definition 2.1, p. 3, specifies restricted isometry of the lifted sensing operator. Theorem 4.1, p. 5, shows that all local minima of the factorized noiseless PSD recovery objective are global when the rank-\(2r\) restricted-isometry constant is below \(1/5\). Theorem 3.2 uses the stronger \(1/10\) condition for its strict-saddle bound. For our algebra, the lifted target has \(r=1\) automatically; the substantive unproved condition is the sensing geometry, not low-rank neural weights. [Full text, Definition 2.1 and Theorems 3.2/4.1](https://arxiv.org/pdf/1605.07221)

**Aghasi, Ahmed, Hand, and Joshi, *BranchHull*.** Its program (BH), p. 4, replaces each product equality by a convex hull of a known-sign hyperbola branch and minimizes squared coefficient norms. Theorem 1, p. 4, assumes independent iid Gaussian subspace matrices, fixed nonzero target factors, and known factor signs. With factor dimensions \(p_1,p_2\), it gives success probability at least \(1-\exp[-(K-(2p_1+2p_2-3))^2/(2(K-1))]\) for \(K>2p_1+2p_2-3\). Lemma 1, p. 7, proves branch convexity. This is a nonlifted convex recovery result under additional information, not unrestricted product solving. [Full text, program BH, Theorem 1, Lemma 1](https://arxiv.org/pdf/1702.04342)

### Applicability to the actual derivative banks

1. **Linear measurement count is conceivable.** The generic quadratic-identifiability result does not require \(p^2\) observations. The indefinite rank-two form itself is compatible with an information count proportional to \(p\). It says nothing about fast use of all those observations or machine-precision conditioning for deterministic \(\Phi^{(r)}D\).
2. **Random physical points are not iid Gaussian feature entries.** All columns are evaluations of correlated tanh features at a common point, and \(A_i,B_i\) may be derivatives of the same features. Randomizing point locations does not meet the cited Gaussian sensing hypotheses automatically.
3. **RIP is not implied by rank-two sensing.** It must be proved for the actual operator and relevant norm. The matrix-sensing paper states RIP on all low-rank matrices. The symmetrized \(Q_i\) annihilate skew-symmetric matrices, so they cannot satisfy that full-matrix version literally. One would need a suitable symmetric-space theorem or a justified nonsymmetric extension \(X\mapsto\alpha_i^TX\beta_i\). No such verification has been obtained here.
4. **BranchHull can retain the same unknown algebraically.** Set its two target coefficient vectors equal to \(w\) and add the linear constraint \(h=m\). With independent Gaussian sensing matrices and known signs, the original theorem's uniquely recovered equal-norm target remains feasible and optimal in the restricted convex program. Thus “same unknown” is not itself an obstruction. For PDE features, the random-matrix hypotheses and sign information are the missing ingredients.
5. **Unknown signs and convex relaxation matter.** For known factor signs, each condition \(\operatorname{sign}(k_i)u_iv_i\ge|k_i|\) on the chosen branch is convex. Equality remains nonconvex. Minimizing a norm over the relaxation can prefer another point unless exactness is established. A positive forcing does not imply a known sign of each factor. The sparse BranchHull variants add sparsity assumptions and are not appropriate here.
6. **Boundary/data information can help.** It can fix the global sign and eliminate tangent null modes through \(C\), but it does not make all quadratic equations convex. The complete actual residual and its constraints must remain in any proposed method.

## 5. A scalar PDE diagnostic: u u_x = f

For a continuously differentiable scalar field on an interval,

\[
u u_x=f
\quad\Longleftrightarrow\quad
(u^2)'=2f.
\]

With \(u(x_0)=g\), any solution must satisfy

\[
u(x)^2=g^2+2\int_{x_0}^x f(t)\,dt=:H(x).
\tag{8}
\]

Where \(H>0\), the sign branch is constant on each connected component and a nonzero boundary value fixes it. At zeros, \(H\ge0\) alone does not ensure a differentiable square root. For example, \(H=x^2\) permits the smooth signed root \(u=x\), whereas the nonnegative root \(|x|\) is not differentiable at zero. A classical solution also requires \(f=0\) at any zero of \(u\). A simple zero of \(H\) cannot produce a differentiable real solution through that point.

For \(f=k\ne0\), (8) gives a square root of an affine function wherever it remains positive. For \(f=0\) on a connected interval, \(u^2\) is constant, so continuous \(u\) is constant. In contrast, finitely sampled equations \(u(x_i)u_x(x_i)=0\) have the branch structure (4) and need not control the equation between samples.

This diagnostic cleanly exposes branch choice, zero degeneracy, and boundary compatibility. Computing the analytic square root and encoding it would be solve-then-encode, so it is not adopted as the neural method. Instead it supplies an independent mathematical reference for the first nonlinear rung, while the unknown remains the neural readout and the residual uses the actual neural derivative.

## 6. The next justified step

The exact nonlinear system can be handled through \(A,B,A^T,B^T\), retaining the \(\Phi^{(r)}D\) structure without a quadratic-size lift. Existing theory identifies the next proof target: establish quantitative identification and a favorable local or global geometry for these **actual deterministic derivative measurement pairs**, including boundary/data rows.

For a local method, the immediate quantity is the smallest singular value of \([J(w);C]\) in an appropriate physical coordinate norm. For a global quadratic-sensing route, one needs a deterministic analogue of the relevant injectivity/stability or landscape conditions, not merely many samples. For a convex branch route, one needs justified factor signs and a relaxation-exactness theorem; neither may be silently added.

This gives an incremental ladder: understand one product equation and its branches; understand multiple products of one shared readout; add affine constraints; exploit the structured operator actions; then prove a quantitative deterministic recovery condition. No sparse weights, low-rank W, changed activation, omitted constraints, or relaxed accuracy target has been introduced.
