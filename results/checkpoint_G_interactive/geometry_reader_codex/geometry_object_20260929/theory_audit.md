# Fixed-solution geometry: mathematical audit

Status: derivations and qualifications, 2026-09-29. This note studies fixed one-layer solutions. It contains no new training experiment and makes no claim that arbitrary Adam readouts are minimum-norm readouts.

## 1. A concrete geometric object for common-width tanh

Let

\[
u(x)=b+\sum_{j=1}^n v_j\tanh\bigl(\gamma(x-c_j)\bigr),\qquad \gamma>0,
\]

with distinct real centers. Introduce the monotone coordinate and positive pole parameters

\[
t=e^{2\gamma x},\qquad a_j=e^{2\gamma c_j}>0.
\]

The elementary identity

\[
\tanh\bigl(\gamma(x-c_j)\bigr)
=\frac{t-a_j}{t+a_j}
=1-\frac{2a_j}{t+a_j}
\]

gives a fixed-denominator rational representation. Set

\[
Q(t)=\prod_{j=1}^n(t+a_j),\qquad B=b+\sum_jv_j.
\]

Then

\[
u(x)=\frac{P(t)}{Q(t)},\qquad
P(t)=BQ(t)-\sum_j2a_jv_j\frac{Q(t)}{t+a_j}.
\tag{1}
\]

Every term on the right is a polynomial and \(\deg P\le n\). Conversely, evaluate at \(t=-a_j\). All terms vanish except the jth summand, so

\[
P(-a_j)=-2a_jv_j\prod_{k\ne j}(a_k-a_j)
=-2a_jv_jQ'(-a_j).
\]

Thus

\[
\boxed{v_j=-\frac{P(-a_j)}{2a_jQ'(-a_j)}}.
\tag{2}
\]

Since Q is monic, B is the coefficient of \(t^n\) in P, and \(b=B-\sum_jv_j\). These formulas give a bijection between all degree-at-most-n polynomials P and the n+1 network coefficients. The function family is precisely

\[
\left\{P(e^{2\gamma x})/Q(e^{2\gamma x}):\deg P\le n\right\}.
\]

This is more informative than renaming a design matrix. Geometry selects a rational denominator; readouts are its residues, expressed in tanh coordinates. The factors \(Q'(-a_j)\) describe coefficient amplification when centers nearly coincide. They do not, by themselves, establish that every close-center fit must have large coefficients: the numerator may vanish correspondingly.

For a target f, ordinary function-L2 fitting on \([A,B]\) becomes

\[
\int_A^B|f(x)-u(x)|^2dx
=\frac1{2\gamma}\int_{e^{2\gamma A}}^{e^{2\gamma B}}
\left|f\!\left(\frac{\log t}{2\gamma}\right)-\frac{P(t)}{Q(t)}\right|^2\frac{dt}{t}.
\tag{3}
\]

Equivalently, it is polynomial approximation to \(Q(t)f(\log t/(2\gamma))\) in the positive weight \(1/(2\gamma tQ(t)^2)\). This is an exact identification of the approximation problem, not a claim that its solution requires no computation.

There is also a direct interpolation encoder. Choose n+1 distinct sample coordinates \(t_i>0\), let \(y_i=f(\log t_i/(2\gamma))\), and let

\[
L_i(t)=\prod_{k\ne i}\frac{t-t_k}{t_i-t_k}.
\]

The unique network interpolant has

\[
P(t)=\sum_i y_iQ(t_i)L_i(t),\qquad
v_j=-\frac{\sum_i y_iQ(t_i)L_i(-a_j)}{2a_jQ'(-a_j)}.
\tag{4}
\]

This uses only target samples and explicit geometry factors. It is interpolation, not the continuous least-squares solution. It can be severely unstable: P is inferred on a positive t interval and evaluated at negative pole locations. Product formulas also need numerically stable evaluation. Its value here is the exact structure it exposes, not a blanket recommendation to implement it in ordinary precision.

### Exact deletion as removal of a polynomial evaluation

The rational coordinates also give a concrete form to the earlier finite-rank deletion theory. Keep gamma fixed and let \(F(t)=f(\log t/(2\gamma))\), \(g(t)=Q(t)F(t)\), and

\[
w(t)=\frac{1}{2\gamma tQ(t)^2}.
\]

The integration interval is \([e^{2\gamma A},e^{2\gamma B}]\), where w is continuous and strictly positive. Let \(P_0\) be the orthogonal projection of g onto the degree-at-most-n polynomial space \(\mathcal P_n\) with this weight. Its rational function \(P_0/Q\) is the original best fit.

Delete neuron j and write \(z_j=-a_j\). A rational function in the retained dictionary has denominator \(Q/(t+a_j)\) and a numerator of degree at most n−1. Written with the old denominator Q, its numerator is therefore exactly a polynomial P in \(\mathcal P_n\) satisfying \(P(z_j)=0\). Conversely, the factor theorem shows that every such P belongs to the retained dictionary.

Choose any orthonormal polynomial basis \(p_0,\ldots,p_n\) in this weighted space and define its reproducing kernel

\[
K_n(t,z)=\sum_{\ell=0}^n p_\ell(t)p_\ell(z).
\]

For \(P=\sum b_\ell p_\ell\), orthonormality immediately gives \(\langle P,K_n(\cdot,z)\rangle_w=\sum b_\ell p_\ell(z)=P(z)\), including external real evaluation locations z. Thus the constraint \(P(z_j)=0\) is the hyperplane perpendicular to \(K_n(\cdot,z_j)\). Projecting \(P_0\) onto that hyperplane gives

\[
\boxed{P_{\rm del}(t)=P_0(t)-\frac{P_0(z_j)}{K_n(z_j,z_j)}K_n(t,z_j).}
\tag{4a}
\]

Because \(g-P_0\) is orthogonal to all of \(\mathcal P_n\), Pythagoras separates the old fitting error from the cost of the constraint:

\[
\|g-P_{\rm del}\|_w^2-\|g-P_0\|_w^2
=\|P_0-P_{\rm del}\|_w^2
=\frac{|P_0(z_j)|^2}{K_n(z_j,z_j)}.
\tag{4b}
\]

These weighted errors equal the original squared function-L2 errors by (3). Formula (2) converts the corrected polynomial back to every retained readout. In particular, for any original index k,

\[
v_{k,\rm del}-v_k
=\frac{P_0(z_j)K_n(z_k,z_j)}{2a_kQ'(z_k)K_n(z_j,z_j)}.
\tag{4c}
\]

For k=j this is \(-v_j\), as required. For the others, it predicts the entire compensation pattern from a geometry-only reproducing kernel and the deleted neuron's original coefficient, since \(P_0(z_j)=-2a_jv_jQ'(z_j)\).

For several deleted indices \(J\), define \(\mathsf K_{ij}=K_n(z_i,z_j)\) and \(p_i=P_0(z_i)\), for i,j in J. Distinct evaluation locations, with \(|J|\le n\), give independent evaluation functionals on \(\mathcal P_n\), so \(\mathsf K\) is positive definite. The constrained projection is

\[
P_{\rm del}(t)=P_0(t)-\sum_{j\in J}K_n(t,z_j)(\mathsf K^{-1}p)_j,
\qquad
\Delta E^2=p^T\mathsf K^{-1}p.
\tag{4d}
\]

This is the same finite-rank projection mechanism as the general deletion formulas, now with an explicit algebraic meaning: deleting a neuron cancels an allowed pole, so the numerator must acquire a zero there. Constructing \(K_n\) still requires geometry-dependent work and can be difficult for severe clustering. The formula is not automatically a cheaper solver. Its theoretical value is identifying the representer governing the deletion and relating coefficient compensation to fixed-pole rational approximation.

## 2. Unequal widths: constrained complex pole families

For unequal positive \(\gamma_j\), each feature has simple complex poles at

\[
z_{j,k}=c_j+\frac{i\pi(k+1/2)}{\gamma_j},\qquad k\in\mathbb Z.
\tag{5}
\]

At a zero of \(\cosh(\gamma_j(z-c_j))\), numerator \(\sinh\) and the derivative of the denominator differ by a factor \(\gamma_j\). Consequently

\[
\operatorname{Res}_{z=z_{j,k}}v_j\tanh\bigl(\gamma_j(z-c_j)\bigr)=v_j/\gamma_j.
\tag{6}
\]

Centers specify the real locations of vertical pole families; widths specify their vertical spacing and distance from the real axis. One readout fixes all residues in its pole family. Coincident poles from different neurons have the sum of their residues. This is a constrained meromorphic representation, not an arbitrary choice of rational poles and residues.

If \(\gamma_j=p_j\gamma_0\) with positive integers \(p_j\), one can again set \(t=e^{2\gamma_0x}\) and write

\[
\tanh\bigl(\gamma_j(x-c_j)\bigr)
=\frac{t^{p_j}-a_j}{t^{p_j}+a_j},\qquad a_j=e^{2\gamma_jc_j}.
\]

This yields a rational representation with denominator \(\prod_j(t^{p_j}+a_j)\), but only an n+1-dimensional numerator subspace, generally much smaller than the space of all polynomials of that denominator's degree. The unrestricted numerator theorem in Section 1 is specific to a common gamma. Incommensurate gammas have no such common finite polynomial coordinate in general.

### Exact independence versus practical redundancy

There is a useful caution for the language of equivalence classes. A finite collection of real tanh features with distinct pairs \((c_j,\gamma_j>0)\), together with a constant, is linearly independent as functions on any open real interval.

To prove this, suppose a finite linear combination vanishes on an interval. Analytic continuation makes the corresponding meromorphic combination identically zero. Pick a center c appearing in the combination, and select the largest gamma among features at c. Its nearest upper pole \(c+i\pi/(2\gamma)\) cannot be a pole of a feature at a different real center. It also cannot be a pole of a smaller-gamma feature at c, whose nearest upper pole is farther from the real axis. Its residue therefore forces that feature's coefficient to zero. Repeat, then the constant must vanish too.

Thus generic finite tanh networks do not have exact continuous-function coefficient nullspaces. Two different finite geometries cannot generally be exchanged while preserving exactly the same nonconstant analytic function. The important redundancy in an actual low-rank numerical fit is approximate, sample-dependent, or defined after a singular cutoff. Duplicate features, signed-gamma redundancies before orientation, and zero readouts are exceptions. Infinite and periodized dictionaries require their own analysis.

### Connection to the kernel lens

For tanh define the unit-mass derivative kernel

\[
\kappa_\gamma(x)=\frac\gamma2\operatorname{sech}^2(\gamma x).
\]

Then

\[
u'(x)=2\sum_jv_j\kappa_{\gamma_j}(x-c_j).
\tag{7}
\]

Its Fourier transform, with \(\widehat g(\omega)=\int g(x)e^{-i\omega x}dx\), is

\[
k_\gamma(\omega)=\frac{\pi\omega/(2\gamma)}{\sinh(\pi\omega/(2\gamma))},\qquad k_\gamma(0)=1.
\tag{8}
\]

For large real frequency its exponential decay is controlled by \(\pi/(2\gamma)\), the distance of the nearest complex poles. The pole description and the derivative-kernel description therefore concern the same object in two coordinates. The former exposes analytic structure and clustered residues; the latter exposes smoothing, overlap, and resolvable frequencies.

## 3. What the general frame description does and does not add

Fix an observation Hilbert space H, its norm, a geometry, and a coefficient norm. The norm could be continuous function L2, weighted sample error, or derivative L2; these are different choices. Let \(A:\mathbb R^n\to H\) synthesize the non-baseline features after handling the baseline consistently.

The output subspace is \(V=\operatorname{ran}A\). In case of redundancy, the effective coefficient space is \(\mathbb R^n/\ker A\), equipped with

\[
\|u\|_{\mathcal G}^2=\min_{Av=u}\|v\|_2^2,\qquad u\in V.
\tag{9}
\]

The frame operator is

\[
S_{\mathcal G}=AA^*,\qquad
(S_{\mathcal G}q)(x)=\sum_j\phi_j(x)\langle\phi_j,q\rangle.
\tag{10}
\]

For \(u\in V\), the minimum-norm preimage is \(v=A^*S_{\mathcal G}^{\dagger}u\), and

\[
\|u\|_{\mathcal G}^2=\langle u,S_{\mathcal G}^{\dagger}u\rangle.
\tag{11}
\]

Indeed, decompose coefficient space orthogonally into \(\ker A\) and its complement. Adding a null component does not change u and increases coefficient norm. On the complement, the proposed formula is the inverse, because \(AA^*S^\dagger u=SS^\dagger u=u\). Substitution gives (11).

The pair (V, this norm) is a geometric object independent of neuron ordering. It is not independent of arbitrary changes of basis: replacing A by AT preserves V when T is invertible but generally changes S and the induced norm. Orthogonal T preserves both. Raw readout coordinates also require the actual dictionary, not just V and S.

This framework is exact but, by itself, not a new computational prediction. Its explanatory content must come from additional structure: the rational denominator above, translation symmetry below, local density, pole clusters, or explicit perturbation response formulas. Computing a fresh generic pseudoinverse and calling it an inverse lens adds no predictive mechanism.

## 4. Several uniform families: explicit Fourier allocation

Start with a continuum envelope approximation, then state the exact alias version separately. Suppose population s has nearly uniform spacing \(h_s\), common gamma \(\gamma_s\), and a smooth readout envelope \(v_s(c)\). Neglecting boundaries and lattice aliases,

\[
u'(x)\approx\sum_s\frac2{h_s}(\kappa_{\gamma_s}*v_s)(x).
\tag{12}
\]

The factor \(1/h_s\) is sampling density; it cannot be omitted when populations have different numbers of neurons. Let \(d=f'\). At one frequency, set \(a_s=2\widehat v_s/h_s\). Exact matching in this approximation requires

\[
\sum_s k_s a_s=\widehat d.
\tag{13}
\]

Meanwhile \(\sum_j|v_{s,j}|^2\approx h_s^{-1}\int|v_s(c)|^2dc\), so the frequency-wise coefficient-norm penalty is proportional to \(\sum_sh_s|a_s|^2/4\). A Lagrange multiplier for (13), or weighted Cauchy-Schwarz, gives

\[
a_s=\frac{\overline{k_s}/h_s}{\sum_t|k_t|^2/h_t}\widehat d,
\qquad
\boxed{\widehat v_s=\frac{\overline{k_s}}{2\sum_t|k_t|^2/h_t}\widehat d.}
\tag{14}
\]

For a single population, this is \(\widehat v=(h/2)\widehat d/k\). At low frequency relative to every gamma, \(k_s\approx1\), and all populations' envelopes approach \(d/(2\sum_t1/h_t)\). At higher frequency, broader kernels are suppressed more strongly; the narrower families receive more of the minimum-norm allocation. The derivative contribution assigned to family s is

\[
\widehat d_s=\frac{|k_s|^2/h_s}{\sum_t|k_t|^2/h_t}\widehat d.
\tag{15}
\]

Thus multiple scale families generally represent complementary filtered portions of the derivative. Individually undoing each kernel's blur does not automatically reveal a full copy of the target derivative in every family.

These formulas describe an exact matching, minimum-norm continuum model. They are not automatically the least-squares readouts of a finite irregular network, nor of arbitrary Adam coefficients. Small aliases alone are insufficient to justify approximating raw coefficients by (14), as Section 6 proves.

## 5. Exact periodic alias fibers, including the fitted quantity

For an exact setting, use a circle of length \(L=Nh\). Each population has N equally spaced centers \(jh+\delta_s\), common gamma within that population, and the zero-mean periodic primitive of its periodized derivative kernel as activation. A separate constant fits DC. This is a precise translation-invariant model; it is not literally the finite-interval, unperiodized tanh model with halos.

Fix a nonzero discrete base frequency \(\theta=2\pi p/N\) and coefficient pattern

\[
v_{s,j}=c_s e^{i\theta j},\qquad z_s=2c_s/h.
\]

The only output frequencies are

\[
\omega_\ell=(\theta+2\pi\ell)/h,\qquad\ell\in\mathbb Z.
\]

To see this, the jth translation contributes \(e^{-i\omega jh}\). Its sum against \(e^{i\theta j}\) is N if \(\omega h\equiv\theta\pmod{2\pi}\), and zero otherwise. Periodization divides the whole-line Fourier transform by L, so N/L supplies the factor \(1/h\).

Define the explicit geometry array

\[
a_{\ell s}=k_{\gamma_s}(\omega_\ell)e^{-i\omega_\ell\delta_s}.
\]

Then the output derivative and function Fourier amplitudes are

\[
\widehat {u'}_\ell=\sum_sa_{\ell s}z_s,
\qquad
\widehat u_\ell=\frac{\sum_sa_{\ell s}z_s}{i\omega_\ell}.
\tag{16}
\]

Write target derivative amplitudes \(d_\ell=i\omega_\ell\widehat f_\ell\). Function-L2 least squares in this frequency class is exactly

\[
\min_z\sum_\ell\frac{|a_\ell z-d_\ell|^2}{\omega_\ell^2}.
\tag{17}
\]

Derivative-L2 fitting instead minimizes the same sum without \(1/\omega_\ell^2\). This weighting distinction matters whenever aliases or other approximation errors are present.

Set

\[
H=\sum_\ell\frac{a_\ell^*a_\ell}{\omega_\ell^2},
\qquad b=\sum_\ell\frac{a_\ell^*d_\ell}{\omega_\ell^2}.
\]

Expansion of (17) yields \(z^*Hz-2\operatorname{Re}(z^*b)+\mathrm{constant}\); hence its minimum-norm solution is \(z=H^\dagger b\). The coefficient norm in this class is \(Nh^2\|z\|^2/4\), so ordinary Euclidean minimum norm is the correct convention for a common h.

This is an analytic block reduction: an arbitrarily large uniform network becomes one number-of-populations matrix per frequency, with entries determined by explicit kernel spectra. It does not replace an arbitrary irregular geometry by an equally expensive renamed system. For two populations and nonsingular H,

\[
z_1=\frac{H_{22}b_1-H_{12}b_2}{H_{11}H_{22}-|H_{12}|^2},\qquad
z_2=\frac{-\overline{H_{12}}b_1+H_{11}b_2}{H_{11}H_{22}-|H_{12}|^2}.
\tag{18}
\]

For a pure target tone, only \(d_0\) is nonzero. The entries of H still sum all aliases. Omitting them leaves a rank-one problem and produces the common-h case of (14); retaining them can select a substantially different coefficient allocation.

Population offsets can create meaningful alias cancellation. Two identical populations shifted by h/2 have opposite relative phases at alternating aliases. Their union is a uniform h/2 grid, and a suitable equal-envelope combination cancels the odd aliases of the coarser grid. Two distinct widths can also cancel a selected alias with opposite-sign readouts. For real two-channel principal amplitudes \((a,b)\) and alias amplitudes \((c,d)\), forcing desired amplitude D and zero alias gives

\[
z_1=\frac{dD}{ad-bc},\qquad z_2=-\frac{cD}{ad-bc}.
\tag{19}
\]

A small determinant produces large compensating coefficients. Additional aliases prevent this two-equation construction from being the complete solution in general, but the mechanism is explicit.

The zero residue class needs separate treatment: fit DC with the free baseline, remove its component from feature columns, and analyze the remaining nonzero harmonics in that class. Do not divide by zero or silently drop this class on a finite circle.

## 6. Why tiny aliases and a small output error do not identify raw coefficients

Take the explicit two-channel synthesis matrix

\[
B_\varepsilon=\begin{pmatrix}1&1\\\varepsilon&2\varepsilon\end{pmatrix},
\qquad y=\begin{pmatrix}1\\0\end{pmatrix},\qquad\varepsilon>0.
\]

The first row is the wanted frequency and the second a weak alias. Exact fitting imposes \(z_1+z_2=1\) and \(z_1+2z_2=0\), so

\[
z=(2,-1)^T\quad\text{for every }\varepsilon>0.
\tag{20}
\]

At \(\varepsilon=0\), the minimum-norm solution is instead \((1/2,1/2)^T\). For small positive epsilon, truncating the small singular direction converges to that latter solution. Its remaining alias has amplitude \(3\varepsilon/2\), tending to zero despite the order-one difference in coefficients.

Therefore, removing nearly invisible output directions need not make a small change in readouts. The limits of vanishing aliases and vanishing singular cutoff do not commute. Retaining tiny modes can select opposite-sign branches through alias cancellation even when their functional contribution is almost unobservable.

For the exact bank in Section 5, form \(B_{\ell s}=a_{\ell s}/(i\omega_\ell)\). The singular values of the full continuous synthesis map in this coefficient class are \(2/\sqrt h\) times the singular values of B, under ordinary coefficient Euclidean norm and unnormalized function L2. A relative cutoff applied to the complete dictionary must therefore use the largest singular value across all frequency classes (and the baseline if it is part of the penalized dictionary). Independently applying a relative cutoff in each fiber changes the problem. Normalizing the output norm changes a common factor, not this requirement.

A reported rank such as 81 retained columns out of 462 describes an observation norm and cutoff convention. It is not evidence of 381 exact analytic dependencies.

## 7. The coefficient field and an exact cross-scale constraint

If a readout is a minimum-norm least-squares solution, then it has the form \(v=A^*q\). The same statement holds for a specified truncated singular expansion with its corresponding q. For tanh this means its readouts sample one field

\[
\mathcal V(c,\gamma)=\int q(x)\tanh\bigl(\gamma(x-c)\bigr)dx.
\tag{21}
\]

For sampled fitting replace the integral by the correctly weighted sample sum. Differentiate with respect to c:

\[
\partial_c\mathcal V(c,\gamma)=-2(\kappa_\gamma*q)(c).
\tag{22}
\]

Extend finite-interval q by zero. If q annihilates constants, as with a consistently residualized free output bias, the field decays at both ends. Its center Fourier transform then satisfies

\[
i\omega\widehat{\mathcal V}_\gamma(\omega)
=-2k_\gamma(\omega)\widehat q(\omega).
\tag{23}
\]

Thus, wherever the ratio is defined,

\[
\widehat{\mathcal V}_\gamma/\widehat{\mathcal V}_\beta=k_\gamma/k_\beta.
\tag{24}
\]

This is a nontrivial restriction on whole coefficient envelopes: width-specific branches are related by known filters. The common latent object q is an inverse-frame dual field; it is not generally the target derivative. Actual readouts at finitely many irregular centers do not directly reveal the continuous envelope, so testing this identity requires sufficient within-cohort sampling or an explicit controlled model. Choosing cohorts after viewing coefficients can bias the conclusion.

Existence of this field alone is weaker than it may sound. With a mathematically full-column-rank finite dictionary, every coefficient vector can be written \(v=A^*q\), by taking \(q=A(A^*A)^{-1}v\). This includes arbitrary learned readouts. Their interpolating q can be extremely large or oscillatory in weakly observed directions. For a solved target, the specific relation is \(q=S^\dagger f\), or its explicitly truncated counterpart. With an effective-rank restriction, some actual readouts can lie outside the retained analysis range. A useful empirical statement therefore needs more than existence: a low-complexity, stable common field that predicts held-out coefficients, or agreement with the target-determined dual under a stated cutoff. Small output error does not establish those properties.

## 8. Connection to the existing derivative dual and perturbation theory

With a free polynomial baseline of degree less than r, the ordinary least-squares dual features annihilate that polynomial space. Repeated integration by parts gives a readout as an exact measurement \(v_j=\int f^{(r)}D_j\). This identifies the fitted information, while the frame, rational, and alias descriptions identify how geometry encodes it. Local derivative sampling is an additional approximation about the shape of \(D_j\), not a consequence of integration by parts alone.

For a full-column-rank dictionary A, solved readouts v, residual \(e=f-Av\), and infinitesimal geometry change dA, differentiating \(A^*(Av-f)=0\) gives

\[
A^*A\,dv=(dA)^*e-A^*(dA)v.
\tag{25}
\]

At exact fit, \(dv=-A^\dagger(dA)v\). If \(dA=AB\) is a genuine change of coordinates within the same space, then \(dv=-Bv\) and the represented function is unchanged to first order. Ordinary center and width changes generally have a component outside the old span. The new geometry therefore changes both the coordinate encoding and the available functions; it is not merely a gauge change.

The earlier explicit lattice response formulas remain useful precisely because translation symmetry made the inverse overlap filter known in advance. Adding defects then perturbed a known operator. An unconstrained irregular network requires identifying analogous structure before claiming an equally simple predictor.

## 9. Finite, falsifiable next analyses

1. **Actual solution anatomy:** separate the saved function, its exact derivative, signed derivative contributions by predefined gamma bins, and cancellation between bins. Report actual readouts separately from any canonical refit. A visually smooth readout band is insufficient evidence of a quasiuniform kernel family.
2. **Pole geometry:** map nearest poles \(c_j\pm i\pi/(2\gamma_j)\), with color/size representing residues \(v_j/\gamma_j\). Test whether apparent bands correspond to pole depths, spatial density, or mutually cancelling contributions. This plot displays an exact structure but does not itself establish a prediction.
3. **Common-gamma controlled encoder:** use (4) to predict interpolation coefficients from target samples, then verify against an independently evaluated function. Clearly distinguish interpolation and least squares, and monitor precision and pole-cluster cancellation.
4. **Two-family controlled prediction:** calculate (17)–(18) from spectra before fitting; verify coefficients and alias amplitudes, first retaining all modes and then with a globally specified cutoff. This tests the stronger multi-family hypothesis and the allocation caveat, not merely output reconstruction.
5. **Learned-family test:** define candidate families using centers/gammas alone, quantify how closely each matches a translated grid, and test analytic predictions on held-out targets or held-out neurons. A failed prediction is informative about boundaries, irregularity, alias cancellation, or unsupported minimum-norm assumptions. Do not tune family membership using the desired readouts.

The theoretical object can be described coherently at two levels: a constrained analytic function family selected by geometry, and an observation-dependent metric that determines how target information is encoded in its coordinates. QI is a particularly transparent translation-invariant regime of this object. Whether a particular learned solution is a superposition of such regimes remains a testable claim, not a consequence of the general framework.
