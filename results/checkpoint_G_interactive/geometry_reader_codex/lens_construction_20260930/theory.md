# A reusable finite-defect encoder from an explicit reference geometry

Mathematical derivation and audit, September 30, 2026. This note concerns fixed geometries and solved readouts. It contains no training-dynamics claim. The construction uses an explicit uniform reference encoder and a small geometry-dependent update; a full solve on the new geometry is reserved for independent validation.

## 1. Setup, including the constrained periodic implementation

Let H be the observation Hilbert space with the actual fitting inner product. It can be continuous function L2, a weighted finite sample space, or a Fourier representation of that same norm. Independently fitted baseline components are removed consistently from the features and target; f below denotes the resulting target. Let X be the allowed coefficient space, with its ordinary Euclidean inner product. Usually \(X=\mathbb R^N\). In the periodic implementation here,

\[
X=\{v\in\mathbb R^N:\mathbf1^Tv=0\}.
\]

This is an explicit model constraint on derivative masses. A free output mean is fitted separately. Removing the constant coefficient mode is not, in general, the same as removing the function's DC mode: that coefficient mode can also produce nonzero alias harmonics. The constraint and the free output mean must therefore both be stated. For a periodic derivative representation, zero total mass also prevents an artificial constant background from appearing when one uses zero-mean periodic Green features.

Let \(A_0:X\to H\) be the reference synthesis map. Assume it is injective on X and that its inverse Gram operator is known explicitly. Write

\[
G=A_0^*A_0\quad\text{on }X,
\qquad Q=G^{-1}\quad\text{on }X,
\]

and extend Q by zero on \(X^\perp\) if X is a proper subspace. Adjoint products below include the appropriate projection onto X. For a uniform periodic dictionary, G is diagonal in coefficient Fourier coordinates. The implementation obtains Q by dividing by its strictly positive, nonconstant Fourier eigenvalues; it does not discover a reference basis with a generic SVD. The constrained constant mode has inverse multiplier zero by definition, not by an unreported numerical cutoff.

Select p edited neurons with the N-by-p selector S, so \(S^Tv\) extracts their coefficients. Assume \(S^T:X\to\mathbb R^p\) is onto. For unconstrained X this holds for distinct selected indices. For \(X=\mathbf1^\perp\), it holds whenever \(p<N\): arbitrary values on those p indices can be balanced by an unselected coefficient.

Let U have p columns, each equal to the new feature minus its old reference feature, in the same coefficient normalization and fitting norm. Then the edited dictionary is exactly

\[
A=A_0+US^T.
\tag{1}
\]

The edit may be a finite center or width change; no small-displacement approximation has been made.

The analysis is written over real spaces for readability. Fourier implementations use Hermitian adjoints in place of transposes.

## 2. Separate transport inside the reference space from leakage outside it

The reference projection and target decomposition are

\[
P_0=A_0QA_0^*,\qquad
a=QA_0^*f,\qquad r=f-A_0a,
\]

so \(r\perp\operatorname{ran}A_0\). Decompose the edit columns in the same way:

\[
B=QA_0^*U,\qquad W=U-A_0B=(I-P_0)U.
\tag{2}
\]

Each column of B says how the corresponding feature difference is encoded by the reference dictionary. Each column of W is the genuinely new part of that difference, orthogonal to every reference feature. Define

\[
T=I_X+BS^T.
\]

Substitution into (1) gives

\[
\boxed{A=A_0T+WS^T.}
\tag{3}
\]

For any v in X, put \(\alpha=S^Tv\). Orthogonality now gives an exact decomposition of the fitting loss:

\[
\boxed{\|f-Av\|_H^2
=\|a-Tv\|_G^2+\|r-W\alpha\|_H^2,}
\qquad \|x\|_G^2=x^TGx.
\tag{4}
\]

To verify it, write the residual as \(A_0(a-Tv)+(r-W\alpha)\). The first term lies in the reference range and the second is orthogonal to it, so their cross term vanishes. Also \(\|A_0x\|_H^2=x^TGx\).

This is the exact meaning of a geometry-dependent lens in this construction. Part of a geometry change remixes reference coordinates through T. Another part changes the attainable function directions through W. If \(r=0\), the second term is a positive leakage penalty \(\alpha^T W^*W\alpha\). For a general target,

\[
\|r-W\alpha\|^2=\|r\|^2-2\operatorname{Re}\langle W^*r,\alpha\rangle+\alpha^T W^*W\alpha.
\tag{5}
\]

Leakage can therefore improve an old nonzero residual. It is not always a penalty relative to the reference fit.

Another exact identity is \(f-Av=(f-U\alpha)-A_0v\). The effective target has the **minus** sign \(f-U\alpha\), and it depends on the unknown edited coefficients. Treating it as a fixed modified target, or treating the edit as a scalar reweighting of observations, is generally incorrect.

## 3. Eliminate all unchanged coefficients explicitly

Set

\[
C=S^TQS,\qquad H_J=C^{-1},\qquad
D=QS H_J,\qquad M=I_p+S^TB.
\tag{6}
\]

C is positive definite. Indeed, for a nonzero p-vector z, surjectivity of \(S^T|_X\) implies that the projection of Sz onto X is nonzero; Q is positive definite there. Thus \(z^TCz>0\). Also \(S^TD=I_p\).

Hold \(\alpha=S^Tv\) fixed and introduce \(w=Tv=v+B\alpha\). Its selected coordinates obey

\[
S^Tw=S^Tv+S^TB\alpha=M\alpha.
\]

For this fixed alpha, minimize \(\|a-w\|_G^2\) subject to \(S^Tw=M\alpha\). A Lagrange multiplier lambda gives

\[
G(w-a)=\Pi_XS\lambda,
\quad w-a=QS\lambda,
\quad C\lambda=M\alpha-a_J,
\qquad a_J=S^Ta.
\]

Here \(\Pi_X\) is Euclidean projection onto X, and \(Q\Pi_X=Q\). Therefore

\[
w=a+D(M\alpha-a_J),
\qquad
\boxed{v=a-B\alpha+D(M\alpha-a_J).}
\tag{7}
\]

The selected coordinates are indeed alpha:

\[
S^Tv=a_J-(M-I)\alpha+M\alpha-a_J=\alpha.
\]

Every term in (7) lies in X, including in the constrained periodic implementation. Moreover,

\[
\|a-w\|_G^2
=(M\alpha-a_J)^T H_J(M\alpha-a_J),
\]

because \(D^*GD=H_J\). Consequently the full new-geometry optimization reduces exactly to p variables:

\[
\boxed{\min_{\alpha\in\mathbb R^p}
\|a_J-M\alpha\|_{H_J}^2+\|r-W\alpha\|_H^2.}
\tag{8}
\]

Expanding this quadratic defines

\[
K=M^*H_JM+L,\qquad L=W^*W,\qquad
q=M^*H_Ja_J+W^*r.
\tag{9}
\]

When K is positive definite,

\[
\boxed{\alpha=K^{-1}q,\qquad
v=a-B\alpha+D(M\alpha-a_J).}
\tag{10}
\]

This uses a p-by-p solve, not a solve over all N readouts. The target residual enters through only p extra measurements. Since \(W\perp\operatorname{ran}A_0\),

\[
W^*r=W^*f=U^*r.
\]

The optimal squared error is also predicted without synthesizing every sample:

\[
E_{\rm new}^2=\|r\|^2+a_J^*H_Ja_J-q^*K^{-1}q.
\tag{11}
\]

Near cancellation, evaluate error from the residual terms in (8), rather than subtracting nearly equal scalars in (11).

For a fixed edited geometry, all of B, W, C, D, M, K, and a factorization of K can be precomputed. Every new target needs only its explicit reference encoding a, the p residual measurements \(W^*f\), the same small factorization, and the linear recovery in (7). No new-geometry full least-squares solve is part of prediction.

## 4. A stable small-system implementation

Equation (9) is useful for theory, but solving its normal equations squares the condition number of the small residualized system. Whiten with a Cholesky factor

\[
C=RR^*,\qquad
\alpha=R\beta,\qquad z=R^{-1}a_J,
\]

and define

\[
M_w=R^{-1}MR,\qquad W_w=WR.
\]

Then (8) is

\[
\min_\beta\|z-M_w\beta\|^2+\|r-W_w\beta\|^2,
\tag{12}
\]

with small Gram

\[
K_w=M_w^*M_w+W_w^*W_w=R^*KR.
\]

The unchanged reference has \(M_w=I\), \(W_w=0\), and \(K_w=I\). A thin QR of the p leakage columns, \(W_w=Q_wR_w\), reduces (12) to the small stacked problem

\[
\min_\beta
\left\|\begin{pmatrix}M_w\\R_w\end{pmatrix}\beta
-\begin{pmatrix}z\\Q_w^*r\end{pmatrix}\right\|^2.
\tag{13}
\]

The omitted part of r contributes an additive constant. The stacked matrix has at most 2p rows and p columns. A QR solve avoids forming its normal equations. All its geometry-dependent parts can be reused across targets.

Compute W explicitly as the difference \(U-A_0B\) in the observation representation. The identity

\[
W^*W=U^*U-B^*GB
\]

is exact algebraically but subtracts positive matrices that may be nearly equal. It can manufacture inaccurate or negative leakage eigenvalues when leakage is tiny. Explicit residual columns are preferable, with orthogonality checked numerically.

## 5. Exact geometry tests and quantitative failure conditions

The residualized new geometry is identifiable if and only if

\[
\operatorname{ker}M\cap\operatorname{ker}W=\{0\},
\]

equivalently K is positive definite. In whitened coordinates the diagnostic is

\[
\sigma_{\min}\!\left(\begin{pmatrix}M_w\\W_w\end{pmatrix}\right)>0.
\tag{14}
\]

This measures whether a combination of edited coefficients becomes almost reproducible by the unchanged dictionary. A very small value means large coefficient sensitivity is expected even if the fitted function remains accurate. It is a relative diagnostic against the reference's own residualized edited directions; it does not replace reporting the reference geometry's conditioning.

Simple sufficient bounds follow from positivity:

\[
\lambda_{\min}(K_w)\ge\sigma_{\min}(M_w)^2.
\]

If \(\|M_w-I\|\le\eta<1\), then

\[
\lambda_{\min}(K_w)\ge(1-\eta)^2,
\qquad
\lambda_{\max}(K_w)\le(1+\eta)^2+\|W_w\|^2.
\tag{15}
\]

Leakage is positive in this Gram and can rescue a singular transport map. For example, take \(A_0=[e_1,e_2]\), replace its second feature by \(e_3\), and select that index. Then \(B=-e_2\), \(M=0\), and \(W=e_3\). T is singular, but \(K=1\) and the new dictionary \([e_1,e_3]\) is perfectly independent. The geometry loses one old direction and gains a new one; an invertible change of old coordinates cannot describe that event.

If K is singular, (8) still describes all optimal functions, but \(K^\dagger q\) does not automatically select the minimum norm of the **entire** readout vector in (7). It selects a convention on alpha. Since \(v=v_{\rm const}+(DM-B)\alpha\), a full-vector minimum-norm convention needs a further minimization along \(\ker K\). Likewise, separate singular cutoffs in the small and full systems can select different coefficients. Full-rank comparisons avoid this ambiguity; explicit deletions are handled separately below.

## 6. When an inverse transport really is an accurate decoder

If M is invertible, so is T on X. Direct multiplication verifies

\[
T^{-1}=I_X-BM^{-1}S^T,\qquad S^TT^{-1}=M^{-1}S^T.
\tag{16}
\]

Use canonical reference coordinates \(w=Tv\). Then (4) becomes

\[
\|a-w\|_G^2+\|r-WM^{-1}S^Tw\|^2.
\tag{17}
\]

In whitened reference coordinates \(x=G^{1/2}w\), \(z_0=G^{1/2}a\), and with

\[
F=WM^{-1}S^TG^{-1/2},
\]

the problem is \(\min_x\|z_0-x\|^2+\|r-Fx\|^2\), so

\[
x=(I+F^*F)^{-1}(z_0+F^*r).
\tag{18}
\]

For a target exactly in the reference space, r=0. The canonical target coordinates are then modified by a positive finite-rank contraction, not an arbitrary new encoding. If \(\|F\|\le\varepsilon\),

\[
\|Tv-a\|_G\le\frac{\varepsilon^2}{1+\varepsilon^2}\|a\|_G.
\tag{19}
\]

This follows by diagonalizing the positive operator \(F^*F\): its correction multiplier is \(s^2/(1+s^2)\). For an arbitrary target,

\[
\|Tv-a\|_G
\le\frac{\varepsilon^2}{1+\varepsilon^2}\|a\|_G
+\min(\varepsilon,1/2)\|r\|.
\tag{20}
\]

The second bound uses \(s/(1+s^2)\le\min(s,1/2)\). Converting this bound to raw coefficient error multiplies it by \(\|T^{-1}G^{-1/2}\|\); unstable residue coordinates can still amplify a small functional discrepancy.

Thus the simple transport-only prediction \(v_{\rm transport}=T^{-1}a\) has a controlled regime: small whitened leakage, bounded inverse transport, and a small reference residual. The p-variable encoder in (10) retains the leakage and residual terms exactly instead of discarding them.

For small feature edits U, B and W are first order in U. When r=0 and transport remains uniformly invertible, the correction to the canonical coordinates Tv is second order. Expanding \(T^{-1}\) then recovers the earlier first-order coefficient law

\[
v=a-Ba_J+O(\|U\|^2).
\]

For nonzero r the additional first-order term is \(QS U^*r\), giving

\[
v=a-Ba_J+QS U^*r+O(\|U\|^2).
\tag{21}
\]

This agrees with differentiating the ordinary normal equations. Bounds on the remainder depend on the reference and new geometry conditioning.

## 7. Deletion is a constraint, not a singular replacement solve

Replacing an edited feature by zero gives \(U=-A_0S\) in the unconstrained formulation, hence B=−S, M=0, and W=0. The zero column's coefficient is unidentifiable. The actual deletion problem fixes its coefficient to zero instead of asking a singular solve to assign it a value.

In either the unconstrained or constrained coefficient space, impose \(S^Tv=0\). The unchanged dictionary then gives

\[
\boxed{v_{\rm del}=a-QS(S^TQS)^{-1}a_J.}
\tag{22}
\]

Its selected entries are zero, and its extra optimal squared error is

\[
\boxed{E_{\rm del}^2-E_0^2=a_J^*(S^TQS)^{-1}a_J.}
\tag{23}
\]

In the constrained case Q must be the inverse on X, so the deletion also preserves the prescribed coefficient constraint. The unrestricted intermediate identity B=−S is not asserted there, since S need not map into X; the constraint derivation gives (22) directly.

For a mixture of p edited indices and deleted indices, retain the general objective (8), fix alpha's deleted coordinates to zero, and solve only for its remaining coordinates. If R injects the free edited coordinates, set \(\alpha=R\zeta\) and use \(R^*KR\zeta=R^*q\). Formula (7) then recovers every surviving coefficient.

## 8. Relation to the physical derivative and activation normalization

For

\[
u(x)=p(x)+\sum_jv_j\sigma(\gamma_j(x-c_j)),\qquad\deg p<r,
\]

the exact rth derivative is

\[
u^{(r)}(x)=\sum_jv_j\gamma_j^rK(\gamma_j(x-c_j)),\qquad K=\sigma^{(r)}.
\]

If \(M_K=\int K\ne0\), define a unit-mass kernel \(\kappa_\gamma(x)=\gamma K(\gamma x)/M_K\) and derivative mass \(m_j=M_K\gamma_j^{r-1}v_j\). Then

\[
u^{(r)}(x)=\sum_jm_j\kappa_{\gamma_j}(x-c_j).
\tag{24}
\]

Tanh has r=1 and \(m=2v\). GELU has r=2 and \(m=\gamma v\). ReLU has r=2 distributionally, \(K=\delta\), and again \(m=\gamma v\). Its normalized curvature kernel is a point mass independent of gamma: changing gamma while holding m fixed changes no represented function apart from baseline conventions. Raw ReLU v must rescale when converting back from m.

All matrices in the encoder must use one consistent coordinate convention. A mass-coordinate inverse and a raw-readout inverse are related by known diagonal factors, but coefficient penalties and singular cutoffs are generally not invariant under that rescaling.

Function fitting does not become ordinary derivative-L2 fitting after differentiation. Let \(J_r\) be the r-fold primitive whose polynomial component of degree less than r is orthogonal to that baseline space in the fitting norm. Then, after optimally fitting the baseline,

\[
\|f-u\|_{\rm function}^2
=\|J_r(f^{(r)}-u^{(r)})\|_{\rm function}^2.
\tag{25}
\]

Consequently all projections, Gram products, B, and W must inherit this integrated-error norm if they are computed in derivative coordinates. On a periodic domain each nonzero Fourier component has weight \(|\omega|^{-2r}\). Fitting raw derivative L2 instead answers a different question.

For the periodic normalized Green features, their nonzero Fourier amplitudes are proportional to

\[
\frac{\widehat\kappa_\gamma(\omega)e^{-i\omega c}}{(i\omega)^r}.
\tag{26}
\]

The unit-mass spectra are

\[
\widehat\kappa^{\rm tanh}_\gamma(\omega)
=\frac{\pi\omega/(2\gamma)}{\sinh(\pi\omega/(2\gamma))},
\qquad
\widehat\kappa^{\rm GELU}_\gamma(\omega)
=\left(1+\frac{\omega^2}{\gamma^2}\right)e^{-\omega^2/(2\gamma^2)},
\qquad
\widehat\kappa^{\rm ReLU}(\omega)=1.
\]

For GELU, \(K(x)=(2-x^2)\varphi(x)\), and transforming \(x^2\varphi\) as minus the second derivative of \(e^{-\omega^2/2}\) yields the stated spectrum. The circle length and Fourier normalization factors must be used consistently in A0, U, and the observation inner product; the operator derivation above is independent of that bookkeeping choice.

## 9. What makes this a prediction rather than a relabeled full solve

The finite-rank elimination itself is established least-squares algebra. Its useful application here requires all of the following:

1. An explicit reference geometry with a target-independent analytic encoder, such as the uniform Fourier Gram inverse, rather than a newly computed full SVD.
2. A bounded number p of specified center/width edits, with U obtained directly from those feature changes.
3. Geometry-only preparation of B, W, C, M, and the small update factorization, reused unchanged for new targets.
4. Predictions of all N coefficients and the new error floor before comparing to an independent full-geometry solve.
5. Reported residuals and condition diagnostics when the reference or edited geometry is weakly identifiable.

This does not yet produce a universal closed-form encoder for an arbitrary learned irregular geometry. It constructs and tests a reusable lens for a controlled class of finite deviations from QI-like reference geometries, including simultaneous finite center and width changes, general targets with reference residual, and deletion constraints. The exact structural distinction is transport inside the reference space plus explicitly measured leakage into new function directions.

Replacing the reference inverse by a truncated pseudoinverse without retaining its coefficient-domain constraints is invalid in general: an edit can activate old null directions. The present periodic construction avoids that ambiguity by specifying X in advance and retaining every strictly positive reference mode on X. Any additional numerical cutoff or regularization must be declared and propagated through both prediction and validation.
