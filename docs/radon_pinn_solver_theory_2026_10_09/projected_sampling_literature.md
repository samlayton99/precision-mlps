# Projected sampling, actual observations, and a near-linear sampling theorem

Date: 2026-10-09. Scope: Sam's proposal to exploit the M one-dimensional profiles when choosing data/PDE samples. No lower-dimensional target prior, alternate domain, tensor sampling grid, or weaker physical norm is adopted.

## Result

The useful positive conclusion is stronger than merely saying that observations are coupled: **the literature already gives stable sampling with approximately p log p physical samples for an arbitrary fixed p-dimensional function space**, when the samples are drawn using the space's joint leverage/Christoffel function and weighted to preserve the intended physical norm. There is no inherent p² sample requirement in that theorem.

Here p is the dimension of distinguishable physical functions or linearized constraint responses, after genuine redundancies are removed. It is not automatically MN, but it is also not automatically N merely because the model has M one-dimensional profiles.

A concrete bridge to the Radon construction is derived below: sums of one-dimensional block leverage functions give a valid joint sampling distribution if the profile decomposition has a controlled lower stability constant. Thus the angular/radial coordinate work can improve **both numerical solution and sampling design**. The remaining obstacle is a provable stable decomposition and cheap sampling from its envelope, not the necessity of one independent dense data set for every direction.

## 1. Three different meanings of sampling a spoke

Write

\[
q(x)=\sum_{m=1}^M g_m(\theta_m\cdot x),\qquad
g_m(t)=\sum_{j=1}^Nw_{mj}\psi(\gamma(t-c_j)).
\]

1. **Project existing coordinates:** x_k produces t_{km}=θ_m·x_k for every m. These are M useful coordinates of the same observation y_k=q(x_k). They are not M observed values g_m(t_{km}).
2. **Measure the actual profile:** observations z_{mj}=g_m(t_j), if available, directly constrain each one-dimensional problem. O(N) suitably placed noiseless measurements per profile is plausible for an N-dimensional profile space with a stable sampling inequality. This is O(MN) scalar observations in total, not O(N) total.
3. **Measure a Radon projection:** Rq(θ,t)=∫_{θ·x=t}q(x)dσ(x) is a hyperplane integral. Canonical filtered profiles can be obtained from such data under the relevant inversion/filter assumptions. Projecting a point coordinate is not evaluating this integral, and an arbitrary readout profile g_m need not equal the canonical filtered Radon profile of q.

Sampling q along a physical spoke also mixes profiles:

\[
q(t\theta_m)=\sum_\ell g_\ell(t\theta_\ell\cdot\theta_m).
\tag{1}
\]

For θ_1=(1,0) and θ_2=(1,1)/√2, the first spoke returns g_1(t)+g_2(t/√2). Even infinitely many observations on that spoke do not directly label either summand. This identity holds inside the original network, without any approximation.

Actual Radon data also satisfy consistency conditions. For integrable compactly supported q,

\[
\int t^k Rq(\theta,t)dt=\int(\theta\cdot x)^kq(x)dx,
\]

which is a homogeneous polynomial of degree k in θ. The Fourier transform in t gives q's Fourier transform along the corresponding frequency line. These identities explain why genuine tomographic profiles contain useful structured information; they do not create those measurements from a list of point values. [Natterer, *The Mathematics of Computerized Tomography*, chapter 2](https://epubs.siam.org/doi/abs/10.1137/1.9780898719284.ch2)

## 2. Shared physical points can cover all projections cheaply—but coverage is not identifiability

Suppose every relevant one-dimensional profile range is partitioned into N bins. If a physical sampling distribution gives each of the MN projected bins probability at least π_min, K independent physical points miss any particular bin with probability at most exp(−Kπ_min). A union bound gives

\[
\Pr(\text{some projected bin is empty})\le MN e^{-K\pi_{\min}}.
\]

Thus

\[
K\ge\pi_{\min}^{-1}\log(MN/\delta)
\tag{2}
\]

suffices for simultaneous marginal coverage with probability at least 1−δ. If π_min is of order 1/N, this is O(N log(MN/δ)) points—not MN separately acquired point sets. Projection densities can become very small near boundary/extreme bins, so the stated probability condition cannot be presumed on a high-dimensional domain.

This is a real benefit for local interpolation, lookup structures, and some sampling diagnostics. It is not a stable recovery theorem. On the square [−1,1]², points (t_k,t_k) cover both x and y projections arbitrarily densely, yet the admissible two-neuron function

\[
q(x,y)=\tanh(\gamma x)-\tanh(\gamma y)
\]

vanishes at all of them while being nonzero elsewhere. The sampled joint feature space contains a null direction despite complete one-dimensional marginal coverage.

Consequently the right additional condition is a sampling inequality for the **sum space** (or the actual PDE constraint image), rather than coverage of every projected interval separately.

## 3. What additive-model theory contributes

With known directions, the linear readout fit is a generalized additive model in the projected covariates t_m=θ_m·x. Those covariates are generally dependent; for M>d their joint distribution is confined to a d-dimensional linear image. This dependence does not itself make the function fit impossible, but independent one-dimensional regression theorems cannot simply be applied M times.

Buja, Hastie and Tibshirani's full paper shows that backfitting is block Gauss–Seidel for the coupled normal equations. Their §3.4 identifies the finite-basis model with ordinary least squares using the concatenated design matrix. Theorem 9 proves convergence for symmetric smoothers with spectrum in [0,1], while their discussion of concurvity explains nonunique component decompositions. These results support shared data and one-dimensional block operations, but supply no width-uniform iteration count or independent-profile labels. [Buja–Hastie–Tibshirani, 1989, full text](https://www.stat.cmu.edu/~rnugent/PCMI2016/papers/LinearSmoothersBuja.pdf)

The connection is therefore useful computationally, with a specific unresolved issue: does the construction's exact multiscale coordinate system control the dependence between profile subspaces in the original physical norm?

## 4. A positive sampling theorem: approximately p log p, not p²

Let V be the actual p-dimensional physical function space to be sampled, with a specified probability measure ρ defining the intended L² norm. Assume point evaluations are defined for V; this is automatic for the finite smooth neural span. Choose any L²(ρ)-orthonormal basis b_1,…,b_p and define

\[
\kappa(x)=\sum_{j=1}^p|b_j(x)|^2,
\qquad d\mu(x)=\frac{\kappa(x)}p d\rho(x),
\qquad \omega(x)=\frac p{\kappa(x)}.
\tag{3}
\]

Points where κ=0 carry no μ mass and no function in V is observable there. The weighting preserves the original norm:

\[
\mathbb E_{x\sim\mu}\,\omega(x)|v(x)|^2=\|v\|_{L^2(\rho)}^2.
\]

For K independent μ samples, the normalized Gram matrix has expectation I and each unaveraged rank-one summand has operator norm p. A matrix Chernoff bound therefore gives, for 0<η<1,

\[
\Pr(\|G_K-I\|>\eta)
\le2p\exp(-K\eta^2/(3p)).
\]

In particular,

\[
K\ge3p\eta^{-2}\log(2p/\delta)
\tag{4}
\]

ensures with probability at least 1−δ that

\[
(1-\eta)\|v\|^2\le\frac1K\sum_k\omega(x_k)|v(x_k)|^2
\le(1+\eta)\|v\|^2\qquad\text{for every }v\in V.
\tag{5}
\]

This is the stability mechanism in optimal weighted least squares. Cohen–Migliorati's Theorem 2 and Corollary 1 establish corresponding Gram and approximation results for general spaces; their sampling pair is (3). Their bounds distinguish noiseless/noisy cases and estimator variants. The sample complexity has no additional ambient-dimension factor beyond p. It does not prove a supremum-norm machine-epsilon error or a cheap construction of κ. [Cohen–Migliorati, *Optimal weighted least-squares methods*, §2](https://arxiv.org/html/1608.00512)

For comparison, unweighted uniform sampling can need a much larger count because sup κ may greatly exceed p. Cohen–Davenport–Leviatan compute κ's supremum as p² for degree-(p−1) algebraic polynomials under the uniform measure; their stability criterion then yields quadratic rather than linear dimension dependence, up to logarithms. This is an example about sampling design, not a lower bound for all neural spaces. [Cohen–Davenport–Leviatan, §3](https://arxiv.org/pdf/1111.4422)

Two cautions matter operationally. First, stable sampling is not an efficient solve: densely evaluating or orthogonalizing p features at K proportional to p still costs quadratic work. Second, noisy observations do not become noiseless through the coordinate transform. Their propagation depends on the inverse problem and sample weighting, and machine-epsilon solution accuracy cannot be inferred from a high-probability norm equivalence alone.

## 5. A direct bridge from one-dimensional profiles to joint leverage sampling

The following derivation identifies a concrete role for the proposed structured decomposition.

Let a(x) be the joint feature column, partitioned into profile or multiscale blocks a_m(x). Let

\[
G=\int a(x)a(x)^*d\rho(x),\quad
G_{mm}=\int a_m(x)a_m(x)^*d\rho(x),\quad
B=\operatorname{diag}(G_{mm}).
\]

Work after removing genuine exact redundancies, or restrict all statements to an explicit identifiable quotient where the inverses below exist. Suppose the chosen decomposition satisfies

\[
G\succeq a_* B,\qquad a_*>0.
\tag{6}
\]

Define the sum of block leverage functions

\[
\ell_{\rm block}(x)=\sum_m a_m(x)^*G_{mm}^{-1}a_m(x),
\quad Q=\sum_m\dim(a_m),
\quad\int\ell_{\rm block}\,d\rho=Q.
\]

Since inversion reverses positive-definite order,

\[
\ell_{\rm joint}(x)=a(x)^*G^{-1}a(x)
\le a_*^{-1}\ell_{\rm block}(x).
\tag{7}
\]

Sample using dμ_block=(ℓ_block/Q)dρ and reciprocal importance weight. The whitened weighted row norm is bounded by Q/a_*, so the same concentration proof yields

\[
K\ge3\frac Q{a_*}\eta^{-2}\log(2p/\delta).
\tag{8}
\]

For ordinary independent radial blocks Q is approximately MN. If a_* stays bounded below, this is near-linear-width sampling built from block quantities. For value data, each block feature depends only on θ_m·x, so its Gram and leverage can be expressed using a one-dimensional marginal measure. However, physically sampling μ_block requires sampling from the corresponding density on the original domain; drawing projected coordinates alone does not provide the requisite conditional transverse distribution. Arbitrary domain geometry and mixed boundary/interior measures must be counted.

If a_* deteriorates because neighboring profiles cancel, (8) becomes correspondingly expensive. A small a_* does not prove optimal sampling itself is expensive: joint leverage may still achieve (4). It says the **cheap block envelope** is too loose. This is precisely where angular moment/detail coordinates could help. The task is to prove (6) or a comparably useful envelope in coordinates that preserve the physical solution space, without an MN-by-MN whitening operation.

Cohen–Dolbeault show that approximate Christoffel functions within a uniform constant factor retain near-optimal sample counts. Their Theorem 3.1 gives the factor explicitly, while their offline construction itself needs a sufficient sampling budget. Thus approximate leverage is an admissible route, but the cost of discovering the measure cannot be omitted. [Cohen–Dolbeault, §§3.1–3.2](https://arxiv.org/html/2010.11040)

## 6. Generalized sampling: O(N) per profile requires a stable measurement map

Suppose actual profile measurements are available. Counting N unknown coefficients and N scalar measurements is only a rank condition. To reach high accuracy the sampling map must also have a lower bound in the desired function norm. This is the generalized-sampling viewpoint: the stable sampling rate depends on the angle between the measurement and reconstruction spaces.

Adcock–Hansen–Poon prove in Theorem 4.1 that compactly supported multiresolution wavelet reconstructions from Fourier measurements have a linear stable sampling rate; Theorems 4.3 and 4.5 quantify instability below critical ratios and sharper Daubechies constants. Their conclusion is specific to those measurement/reconstruction spaces. It does not establish a linear stable sampling rate for arbitrary finite tanh translates, but it gives the correct theorem to seek rather than relying on dimension counting alone. [Adcock–Hansen–Poon, §4](https://arxiv.org/pdf/1208.5959)

For the proposed method, legitimate O(N)-per-profile information can therefore arise from genuine profile/tomographic measurements, or from internal profile interpolation during a **coupled** PDE solve. The second is computational state, not newly observed experimental data.

## 7. PDE constraints change which unknowns data must determine

There is no universal theorem saying a well-posed continuum PDE problem requires MN physical observations. For example,

\[
\partial_{x_1}u=0
\]

forces u(x_1,x′)=h(x′). Boundary data on x_1=0 determine h; with prescribed h, the continuum PDE and boundary condition determine u regardless of how redundantly u is represented. Many forward PDE problems need no interior solution measurements at all: the forcing, coefficients, geometry and boundary/initial values define the task.

However, enforcing a PDE at finitely many collocation points supplies finitely many scalar constraints on the neural coefficients. At a regular locally identifiable solution, the combined sampled PDE/boundary/data Jacobian must have rank p on the physical tangent space being recovered. K scalar constraints cannot provide rank greater than K. Counting physical points instead of scalar equations requires including equation components and independently measured derivatives. Singular isolated roots can exist with rank deficiency, but do not supply a stable inverse from this rank argument.

For a linear PDE, apply (3)–(5) to the actual constraint image space, including boundary/data rows on a disjoint union of their physical domains and their intended measures. Its dimension is the identifiable rank, not automatically the number of all neuron coefficients. For a nonlinear PDE, the same result applies to a **fixed** linearized residual space. The Jacobian depends on the state; a frozen sampling design is not automatically valid through a nonlinear solve. Uniform control along a region or an adaptive certified design needs an additional theorem. General nonlinear residuals are not a fixed p-dimensional linear space merely because their parameter vector has p entries.

Known forcing evaluations and internally evaluated residuals are also different from noisy externally acquired data. Their respective acquisition, validation, differentiation and solve costs should not be conflated.

If external observations already exist at fixed locations, optimal-design theory does not authorize replacing them with a newly drawn data set or dropping their constraints. It can guide additional measurements when acquisition is controllable, or the choice of additional PDE collocation/validation points. Existing mandatory observations remain part of the declared problem.

## 8. Admissible next step

A practical theory target is now specific: derive a computable leverage envelope for the full PDE/data constraint image using the exact radial/angular coordinate decomposition, and prove a controlled domination factor as in (7). This would turn the one-dimensional structure into near-linear physical sampling without pretending that projecting one observation produces M independent labels.

The important separation is:

- Projected-bin coverage can be cheap and shared, as (2) shows.
- Stable joint sampling can require only approximately p log p, as (4) shows.
- Achieving that sampling with cheap structured design requires a stable-decomposition/envelope theorem, exemplified by (6)–(8).
- Evaluating those samples at near-linear cost still requires the aggregation theorem in `aggregation.md` and `angular_resolution.md`.

No extra smoothness prior, profile sparsity, tensor grid, or altered boundary geometry is adopted in these statements. They support the research direction while identifying exactly which data and computation are being counted.

## Reading and verification

Read the full text relevant sections of Cohen–Migliorati, Cohen–Davenport–Leviatan, Cohen–Dolbeault, Buja–Hastie–Tibshirani, and Adcock–Hansen–Poon. The projected-bin bound, counterexamples, and block-leverage domination are direct derivations in this note. The Natterer chapter was available through its publisher description rather than full text; the two Radon identities used here follow directly from Fubini/change of variables. No experiments were run and no existing sampling or solver implementation was changed.
