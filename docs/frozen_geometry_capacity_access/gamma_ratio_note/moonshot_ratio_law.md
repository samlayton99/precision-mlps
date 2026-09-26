# What general gamma law can actually be proved?

23 September 2026. Mathematical investigation requested by Sam. No training or new parameter campaign was run. The scalar comparison figure reuses the existing saved spectra and positive target masses.

**Result.** Universal monotonicity of the actual normalized eigenvalues is false, even with symmetric uniform samples and centers. There is, however, a general scalar **upper envelope**, valid for the actual finite tanh network:

\[
\boxed{\quad
\frac{\lambda_i(K_\gamma)}{\lambda_1(K_\gamma)}
\le \min\left\{1,\;16\exp\!\left[
-\frac{2\pi^2(i-3)}{4\gamma X+\log16}\right]\right\},
\qquad i\ge4.
\quad}
\tag{1}
\]

Here all sample locations lie in an interval of radius \(X>0\). Centers can be arbitrary, and width does not appear. Features and the bias retain their original Euclidean coordinates. The bound increases with gamma and decreases exponentially with rank. It is an absolute bound at one gamma, with no reference slope or input eigenvalues.

This new bound is substantially less tight than the existing Fourier-integral bound, but already gives a necessary delay of **2.78 billion updates** at gamma 4 for the saved mixed-sine example. A refinement using the actual center interval gives **4.73 billion**. The existing integral bound gives **453 billion**; the actual spectral crossing is approximately **1.008 trillion**. Thus the scalar formula proves a useful obstruction while sacrificing roughly two orders of magnitude in this example.

Equation (1) comes from the exact Cauchy structure of tanh and a rational approximation theorem. It is not a polynomial approximation argument. It is also **not a consequence of pointwise positivity of \(M_\gamma\) alone**. The tight result derived directly from \(M_\gamma\) still retains the integral spectrum. These are complementary results, and the distinction should remain explicit.

## 1. The stronger proposed monotonicity law is false

Use two uniformly spaced samples \((-1,1)\), three uniformly spaced centers \((-3/2,0,3/2)\), and the bias. Define

\[
B_{aj}=m^{-1/2}\tanh\bigl(\gamma(x_a-c_j)\bigr),
\qquad B_{a0}=m^{-1/2},\qquad K_\gamma=B_\gamma B_\gamma^\top.
\]

The constant and odd sample vectors are eigenvectors. With
\(a=\tanh(5\gamma/2)\), \(b=\tanh(\gamma/2)\), their eigenvalues are

\[
\lambda_+=1+\frac{(a+b)^2}{2},\qquad
\lambda_-=\tanh^2\gamma+\frac{(a-b)^2}{2}.
\]

The larger eigenvalue is \(\lambda_+\): their difference is
\(1-\tanh^2\gamma+2ab>0\). Writing \(t=e^{-\gamma}\), the ratio is analytic near \(t=0\), and

\[
\frac{\lambda_-}{\lambda_+}
=\frac13+\frac49e^{-\gamma}+O(e^{-2\gamma}).
\]

Its derivative is negative for sufficiently large gamma. This example preserves symmetry, uniformity, bias, fixed geometry, and raw readout coordinates. The failure is not an eigenvector-tracking ambiguity. The existing default-geometry calculations also contain resolved decreasing far-tail ratios; simply requiring many uniform samples does not restore a theorem for every rank and every gamma.

Pointwise increase of \(M_\gamma\) proves an order statement for the **unnormalized whole-center-line operator**. Finite-center corrections and division by the changing leading eigenvalue are additional operations. Neither preserves the claimed normalized monotonicity in general.

An absolute rank law must also contain a spatial scale. Replacing every sample and center by \(s x_a,s c_j\) replaces gamma by \(s\gamma\). A nontrivial formula in gamma and rank alone, uniform over unconstrained geometries, cannot account for this rescaling.

## 2. A finite-network scalar theorem

Assume \(x_a\in[-X,X]\) and \(c_j\in[-C,C]\), after a common translation, with \(X,C>0\). Samples and centers need not be uniform, symmetric as sets, or commensurate. Let the eigenvalues of \(K_\gamma\) be in decreasing order. Define

\[
\chi_\gamma
=\frac{\cosh^2\!\bigl(\gamma(X+C)\bigr)}
{\cosh^2\!\bigl(\gamma(X-C)\bigr)},\qquad
L_\gamma=\log(16\chi_\gamma).
\]

**Theorem.** For every \(i\ge4\) within the sample dimension,

\[
\boxed{
r_i(\gamma):=\frac{\lambda_i(K_\gamma)}{\lambda_1(K_\gamma)}
\le U_i^{\rm interval}(\gamma)
:=\min\{1,16e^{-2\pi^2(i-3)/L_\gamma}\}.
}
\tag{2}
\]

Indices beyond the feature rank have zero eigenvalues and satisfy the same statement. Moreover \(\chi_\gamma\le e^{4\gamma X}\); substituting this upper bound gives (1), which consequently permits arbitrary real centers.

Both envelopes are nondecreasing in gamma. For (2),

\[
\frac{d}{d\gamma}\log\chi_\gamma
=2(X+C)\tanh\bigl(\gamma(X+C)\bigr)
-2|X-C|\tanh\bigl(\gamma|X-C|\bigr)>0.
\]

The inequality follows because \(t\mapsto t\tanh(\gamma t)\) is strictly increasing for \(t>0\). This proves monotonicity of a valid bound, without claiming monotonicity of the quantity bounded.

### Proof, including the raw-coordinate and bias factors

Put \(s_a=e^{2\gamma x_a}\), \(t_j=e^{2\gamma c_j}\), and define the rectangular matrix

\[
D_{aj}=\frac{t_j}{\sqrt m(s_a+t_j)}.
\]

The identity \(\tanh\bigl(\gamma(x_a-c_j)\bigr)=1-2t_j/(s_a+t_j)\) gives

\[
B_\gamma=[0,-2D]+R,
\qquad R=m^{-1/2}\mathbf1_m\mathbf1_{W+1}^\top,
\qquad \operatorname{rank}R=1.
\tag{3}
\]

This is an exact identity for the original feature matrix. No change of readout metric or assumption of orthogonality is being made.

We use the following established Cauchy-matrix inequality. If the positive row nodes lie in \(E=[a,b]\), negative column nodes in \(F=[c,d]\), with disjoint intervals, and
\(\chi=|(c-a)(d-b)/((c-b)(d-a))|\), then

\[
\sigma_{j+k}(D)\le
4e^{-\pi^2 k/\log(16\chi)}\sigma_j(D).
\tag{4}
\]

This is [Beckermann–Townsend, Corollary 4.2](https://arxiv.org/pdf/1609.09494), including its elementary logarithmic relaxation. Their theorem permits separate row and column weights, so the factor \(t_j/\sqrt m\) is covered. It follows from rank-one Sylvester displacement and a Zolotarev rational bound for separated intervals. This is rational approximation of the separated Cauchy problem, not polynomial approximation of tanh on the sample interval.

For our matrix the intervals are
\(E=[e^{-2\gamma X},e^{2\gamma X}]\) and
\(F=[-e^{2\gamma C},-e^{-2\gamma C}]\). Substitution in the cross-ratio gives exactly \(\chi_\gamma\).

Rank-one singular-value interlacing applied to (3), once in each direction, yields

\[
\sigma_i(B_\gamma)\le2\sigma_{i-1}(D),
\qquad
2\sigma_2(D)\le\sigma_1(B_\gamma).
\]

If \(\sigma_2(D)>0\), apply (4) with \(j=2\), \(k=i-3\), then square:

\[
r_i(\gamma)
\le\left(\frac{\sigma_{i-1}(D)}{\sigma_2(D)}\right)^2
\le16e^{-2\pi^2(i-3)/L_\gamma}.
\]

If \(\sigma_2(D)=0\), equation (3) gives \(\operatorname{rank}B_\gamma\le2\), and the conclusion is immediate. Finally,
\(\cosh(u+v)\le e^v\cosh u\) for \(u,v\ge0\) proves
\(\chi_\gamma\le e^{4\gamma\min(X,C)}\le e^{4\gamma X}\). This completes the proof.

For large \(\gamma X\), (1) has exponential factor approximately
\(e^{-\pi^2(i-3)/(2\gamma X)}\). Its exponent resembles the Fourier attenuation \(e^{-\pi|\omega|/\gamma}\) at the nominal frequency \(|\omega|\approx\pi(i-3)/(2X)\). This resemblance is not a rank-to-frequency theorem and was not used in the proof.

The simple logarithmic relaxation does not tend to zero as gamma tends to zero at fixed rank. The sharper elliptic-function version of the cited rational theorem does, because the separated intervals contract. That improvement is unnecessary for the tested gamma range and is not used in the numerical claims here.

## 3. Necessary training time and quantitative usefulness

With zero initial readout and \(\eta=1/(2\lambda_1)\), the exact relative squared residual is

\[
E_\gamma(n)^2=\sum_jp_j(\gamma)(1-r_j(\gamma)/2)^{2n}.
\]

For a cutoff \(i\), let \(P_i\) be the target's squared norm fraction in positive eigenmodes with indices at least \(i\). Any upper bound \(r_i\le U_i\) gives

\[
E_\gamma(n)^2\ge P_i(1-U_i/2)^{2n},\qquad
n_\epsilon\ge
\frac{\log(\sqrt{P_i}/\epsilon)}{-\log(1-U_i/2)}
\quad\text{if }P_i>\epsilon^2.
\tag{5}
\]

Apply a ceiling for integer updates. The scalar theorem does not remove the target-mass condition. Nor does it prove representability; the existing explicit readout check supplies that separate fact at gamma 4.

Here \(X=1\), \(C=76/64\), \(m=263\), and the target is
\(\sin(2\pi x)+\tfrac12\sin(6\pi x)+\tfrac14\sin(10\pi x)\).
All values below reuse the earlier saved data. No actual eigenvalue enters either scalar bound.

| Gamma | Actual rank-26 ratio | Integral upper bound | Scalar (2), finite interval | Scalar (1), arbitrary centers |
|---:|---:|---:|---:|---:|
| 4 | \(2.088\times10^{-12}\) | \(3.088\times10^{-12}\) | \(2.956\times10^{-10}\) | \(5.024\times10^{-10}\) |
| 8 | \(1.487\times10^{-7}\) | \(1.623\times10^{-7}\) | \(3.296\times10^{-5}\) | \(3.418\times10^{-5}\) |
| 16 | \(3.715\times10^{-5}\) | \(4.039\times10^{-5}\) | \(1.783\times10^{-2}\) | \(1.783\times10^{-2}\) |

| Gamma | Necessary steps, scalar (2) | Necessary steps, scalar (1) | Necessary steps, integral | Actual spectral crossing |
|---:|---:|---:|---:|---:|
| 4 | \(4.732\times10^9\) | \(2.784\times10^9\) | \(4.530\times10^{11}\) | \(1.008\times10^{12}\) |
| 8 | 56,629.9 | 54,595.2 | \(1.150\times10^7\) | 19,697,563 |
| 16 | 82.83 | 82.79 | 46,249.3 | 66,209 |

The scalar maxima use the four saved cutoffs \(12,20,26,32\); rank 26 wins at these three points. The integral maxima use the source's full cutoff search. Counts in the table are unrounded bounds. At rank 26, (2) is approximately 142, 222, and 480 times the actual ratio. The simpler (1) is approximately 241, 230, and 480 times the actual ratio. These factors should not be interchanged.

The scalar result proves severe low-gamma delay, but is much too loose to replace the integral calculation when the aim is quantitatively accurate necessary times across gamma 4–16. Monotonicity of \(U_i\) also does not make (5) monotone if the target mass \(P_i\) changes. A common positive lower bound on that mass is needed for a monotone training-time envelope.

![Scalar and integral ratio bounds](../../../results/checkpoint_D_optimizers/expD37_capacity_access_figures/gamma_solve/spectrum_mechanism/moonshot_scalar_ratio/scalar_ratio_comparison.png)

Code: `experiments/expD37_capacity_access_figures/gamma_solve/spectrum_mechanism/moonshot_scalar_ratio.py`. Data and image: `results/checkpoint_D_optimizers/expD37_capacity_access_figures/gamma_solve/spectrum_mechanism/moonshot_scalar_ratio/`. The driver evaluates the formulas, checks both scalar bounds against every imported ratio, and renders the plot. These are checked FP64 evaluations, not interval-certified counts.

## 4. The strongest useful law directly from the multiplier

For zero-mean sample vectors \(u\), the existing Fourier identity is

\[
u^\top H_\gamma u=
\frac2{\pi hm}\int_{\mathbb R}
\frac{M_\gamma(\omega)^2}{\omega^2}
\left|\sum_a u_a e^{-i\omega x_a}\right|^2\,d\omega,
\qquad
M_\gamma(\omega)=\frac{\pi|\omega|/(2\gamma)}{\sinh(\pi|\omega|/(2\gamma))}.
\]

Pointwise growth of the multiplier proves \(H_{\gamma_1}\preceq H_{\gamma_2}\) for \(\gamma_1\le\gamma_2\). Thus every ordered integral eigenvalue \(\alpha_j(\gamma)\) is nondecreasing. For finite lattice centers, the already proved analytic sampling allowance gives

\[
r_i(\gamma)\le
\min\left\{1,\frac{\alpha_{i-1}(\gamma)+\delta_\gamma}{\ell_\gamma}\right\}.
\tag{6}
\]

This is the stronger practical theorem. It does not require fixed eigenvectors or periodic sample eigenfunctions.

The coordinator also supplied a useful normalization refinement. For endpoint-uniform samples on \([-1,1]\), define

\[
a_{\gamma_0}(c)=
\frac{\log\cosh(\gamma_0(1+|c|))-
\log\cosh(\gamma_0(1-|c|))}{2\gamma_0},
\qquad
\ell_*=1+\sum_j[a_{\gamma_0}(c_j)-2/m]_+^2.
\]

The absolute continuous feature mean is \(a_\gamma(c)\) and increases with gamma. Its empirical mean differs by at most \(2/m\): a monotone function's endpoint-sample average lies within its total variation divided by \(m\) of its integral average. Therefore \(\lambda_1(K_\gamma)\ge\ell_*\) for every \(\gamma\ge\gamma_0\).

For \(\gamma_0=4\) in the current geometry, \(\ell_*=62.1800396266\), close to the pointwise mean bound 63.196 at gamma 4. Fixing the strip angle in \(\delta_\gamma\) makes both numerator terms nondecreasing. Hence

\[
\boxed{
r_i(\gamma)\le
\min\left\{1,\frac{\alpha_{i-1}(\Gamma)+\delta_\Gamma}{62.1800396266}\right\},
\qquad 4\le\gamma\le\Gamma.
}
\tag{7}
\]

The decimal is a numerical evaluation of the exact expression for \(\ell_*\), not a certified enclosure. In a formal theorem use that exact expression. Equation (7) is a useful monotone cap envelope derived directly from the multiplier. Unlike the bias-only denominator 1, its normalization incurs little loss. The remaining computation is the integral spectrum; it is not a closed scalar rank formula.

## 5. Why the attempted direct scalar Fourier simplification was not selected

An independent mathematical subagent derived a periodized Fourier comparison with explicit boundary-image corrections. It is a legitimate extension of the multiplier route, not an assumption that the finite network has Fourier eigenvectors. Briefly, write

\[
-2d\coth(\gamma d)=-2|d|-4\sum_{j\ge1}|d|e^{-2\gamma j|d|}.
\]

Periodization yields a known Fourier kernel, a rank-one quadratic correction after zero-mean compression, and boundary-image matrices. Each retained exponential image term costs at most two positive ranks. The omitted terms have a geometric norm bound. A larger period improves the boundary remainder but weakens the retained Fourier decay; a smaller period requires more boundary ranks.

For a valid upper bound on the \(\alpha_{25}\) needed at finite rank 26, bounded scalar optimization gave \(1.150\times10^{-6}\), \(2.162\times10^{-4}\), and \(7.295\times10^{-3}\) at gamma 4, 8, and 16. These exceed the numerical integral eigenvalues by factors approximately 5893, 19.9, and 2.66. The center-lattice allowance must still be added before transfer to the finite network. This route is too loose at the most consequential small-gamma point. It is not recommended as the main theorem, and its exploratory numerical optimization is not an interval certificate.

The single-frequency-cutoff route was already found too loose in the earlier review. Neither failed simplification disproves the possibility of a much sharper scalar theorem. They identify the specific difficulty: bounding finite-window frequency leakage and boundary effects without consuming the ranks that carry the exponentially small eigenvalues.

## Scope for the paper

The precise general statement to prove is an upper envelope for the normalized rates, followed by a target-conditioned necessary-time inequality. A universal monotonicity theorem for actual ratios should be dropped. Equation (1) supplies a simple, width-independent finite-network obstruction; equations (6)–(7) give substantially sharper quantitative results through the positive multiplier. The corrected two-sided intervals remain the right tool for establishing large increases between specific gamma values.

A tight closed scalar formula derived only through \(M_\gamma\), matching the useful integral bound across the studied range, remains unproved here. The available results already prove the requested low-gamma optimization obstruction; they do not prove fast attainment at higher gamma or for every target.
