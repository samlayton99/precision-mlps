# The Radon PDE problem one operation at a time

This is the main entry point for the investigation after Sam's October 9 request to proceed incrementally and ground each step in the literature. The representation remains a standard single-hidden-layer MLP. We add one mathematical operation at a time, retain boundary/data/inverse constraints, and distinguish exact algebra from inference guarantees and computational cost. No solver benchmark was tuned and no easier physical solution family was adopted.

The main literature connections are now specific: fixed-feature collocation for linear equations; quadratic sensing for products of linear derivative measurements; nonlinear kernel/optimal-recovery formulations for functions of sampled derivatives; variable projection for genuinely conditional-linear physical unknowns; and leverage-weighted sampling for the joint physical constraint space. None alone supplies the full desired complexity theorem, but they replace an undirected search with concrete existing formulations and explicit assumptions to verify.

## 1. Shared notation

Let W have N center readouts in each of M directions. Write w=vec(W), P=MN, and include any output bias as a separate feature when needed. For each physical point x_i and derivative multi-index α, define

\[
\ell_{\alpha,i}(W)=\gamma^{|\alpha|}
\langle\Phi^{(|\alpha|)}(x_i)D_\alpha,W\rangle_F.
\]

Stack these scalar observations into A_αw. Thus A_0w is the vector of field values, A_xw is its x derivative, and A_Δw is its Laplacian. All are fixed linear maps at fixed geometry and sample locations. The field is linear in w even when its PDE residual is not.

K counts physical sites; the number of scalar equations and the number of requested derivative observations can both exceed K. They must be counted separately. Inverse coefficient fields add their own unknown readout vectors.

## 2. A single derivative: u_x=0

At the sampled sites the equation is simply A_xw=0. This is a linear nullspace problem. Boundary or value observations add rows Cw=d. The relevant operator is the combined stack [A_x;C], not the PDE rows alone.

In the continuum, u is constant along each connected x-directed segment of the domain. On a product domain, u(x,x′)=h(x′), and data on x=0 specify h. This reduction is imposed by the PDE, not by an assumed low-dimensional target. If boundary values conflict along a characteristic, there is no exact classical solution; a numerical least-squares answer does not remove that incompatibility.

The geometry already matters at this first rung. Directions with zero x component contribute nothing to A_x. They remain available to express variation transverse to x. Other directions can cancel at finitely many sample points without satisfying the equation between them. In a generic finite tanh bank with no x-perpendicular direction, exact u_x=0 on an open set leaves only a constant, by the finite-feature pole-independence argument in `stability.md`. A finite neural discretization therefore needs an approximation/derivative-error statement; exact continuum identities cannot automatically be imposed on a finite space without checking compatibility.

This rung asks three separate questions: what the PDE's nullspace is, how the representation resolves it, and whether the PDE plus data samples stably identify the intended function.

## 3. Higher derivatives and sums

Replacing u_x by any mixed derivative changes only the activation derivative order and the diagonal direction factors. Summing derivatives gives a sum of the fixed operators:

\[
u_{xx}+u_{yy}=0
\quad\longrightarrow\quad
(A_{xx}+A_{yy})w=0.
\]

For unit purely spatial directions, the Laplacian collapses to one second-derivative bank since Σ_i θ_mi²=1. For space–time directions the spatial norm replaces one; the time component cannot be included in physical diffusion.

Higher-order homogeneous equations have larger continuum nullspaces and need the appropriate boundary/initial information. They do not become separate independent equations on every ridge profile: contributions from directions can cancel in the physical field.

## 4. Constant coefficients, known variable coefficients and forcing

For constant coefficients c_α,

\[
\sum_\alpha c_\alpha\partial^\alpha u=f
\quad\longrightarrow\quad
\left(\sum_\alpha c_\alpha A_\alpha\right)w=f_X.
\]

For known spatial coefficients a_α(x),

\[
\left(\sum_\alpha\operatorname{diag}(a_\alpha(x_i))A_\alpha\right)w=f_X.
\]

The system remains linear in the unknown readout. Known variable coefficients do not make it a nonlinear coefficient-recovery problem. Forcing changes the right-hand side, not the hidden features. Derivatives applied to products must still be expanded correctly: ∂_x(a u_x)=a u_xx+a_x u_x is different from a u_xx.

Known linear boundary operators, initial values, gauge conditions and point/derivative data append corresponding rows. Variable coefficients and normal vectors can vary independently at every point. This algebra supports scattered samples without a tensor grid.

This is already the fixed-feature collocation formulation in [Dwivedi–Srinivasan's PIELM, Section 3](https://arxiv.org/pdf/1907.03507). That paper explicitly differentiates tanh features and multiplies by fixed input-weight powers. Its pseudoinverse implementation and random hidden geometry do not provide our scale or conditioning guarantee. The useful starting point is the established operator formulation, with the Radon construction supplying additional structure.

## 5. An unknown coefficient is the first bilinear inverse problem

If a(x) is unknown and represented by a second readout v, then a u_x=f gives

\[
(A_0^{(a)}v)\odot(A_x^{(u)}w)=f_X.
\]

It is linear in w with v fixed, and linear in v with w fixed, but jointly bilinear. Forcing, coefficient/state observations, boundary conditions and independent experimental excitations determine identifiability. Multiplying a by a constant and dividing u appropriately may preserve some interior equations; state data can remove that ambiguity, but the complete declared problem must be checked.

This connects to bilinear inverse problems and separable nonlinear least squares. [O'Leary–Rust's variable-projection framework](https://www.cs.umd.edu/users/oleary/software/varpro.pdf) is applicable when a block really is conditionally linear. For example, a constant unknown viscosity can be eliminated exactly conditional on the velocity/pressure fields when its weighted sensitivity has nonzero norm and the declared constraints permit the unconstrained minimizer. This is not permission to call all readouts linear when the PDE contains u u_x.

An alternating sequence of conditional linear solves is an algorithm, not a global convergence theorem. Dense inner solves can also erase the desired scaling; their costs remain explicit.

## 6. One product with the same readout: the user's example

Let a_i=⟨A_i,W⟩_F, b_i=⟨B_i,W⟩_F and require a_i b_i=k_i. Put α_i=vec(A_i), β_i=vec(B_i). Then

\[
a_i b_i=w^TQ_iw,\qquad
Q_i=\tfrac12(\alpha_i\beta_i^T+\beta_i\alpha_i^T),\qquad
\operatorname{rank}Q_i\le2.
\]

This is generalized quadratic sensing, also studied under generalized phase retrieval. Q_i is generally indefinite. It is not the same as squared-magnitude sensing unless the two linear forms coincide suitably. No low-rank assumption is made on the original N-by-M readout matrix.

The residual and its exact linearization are

\[
R(w)=(Aw)\odot(Bw)-k,
\qquad
J(w)h=(Bw)\odot Ah+(Aw)\odot Bh.
\]

Only structured A/B applications and pointwise multiplication are needed. Boundary and data constraints Cw=d add linear rows. Squaring the product residual gives a quartic objective, not a convex quadratic objective. At zero readout the product Jacobian vanishes, although the added linear PDE/data rows need not vanish.

For k_i=0 each sample permits either factor to vanish. For nonzero k_i the two factors lie on sign branches. Products alone have the symmetry w↦−w; nonzero affine data can break it. These are physical/algebraic issues to examine before choosing an optimizer.

[Wang–Xu, Theorems 2.1 and 4.1](https://arxiv.org/pdf/1605.08034) give an exact Jacobian characterization and generic linear-in-P measurement counts for quadratic identification up to sign. Their generic sensing conditions have not been proved for these deterministic, correlated derivative banks. PhaseLift and nonlifted phase-retrieval methods provide additional viewpoints, but their usual positive/random measurement assumptions are not inherited by the PDE. The detailed source audit is in [product measurements](product_measurements_literature.md).

The formal lift X=ww^T makes the equations linear in X, but adds a rank-one PSD constraint. It is exact and does not imply W itself is low rank. Storing X would cost O(P²), defeating the task. Dropping rank one or introducing sparsity would change the problem unless a recovery theorem justified it. We retain the structured operator formulation instead.

## 7. The first nonlinear PDE has an exact conservation diagnostic

For a classical scalar solution along an x-directed interval,

\[
u u_x=f\quad\Longleftrightarrow\quad
\partial_x(u^2/2)=f.
\]

With u(x_0)=g, this implies u(x)^2=g²+2∫_{x_0}^x f(t)dt. For f=k, u is a suitable signed square root of an affine function wherever a classical real solution exists. Nonzero boundary values fix a sign on a connected positive branch; crossing zeros requires differentiability and compatibility. For f=0, continuous u is constant along the connected interval. Finite sampled products have more possibilities than this continuum solution set.

This identity is useful as an analytical diagnostic of information, branches and constraints. Replacing u by a separate network for u² and then taking a square root would change the final architecture, and solving the formula first then encoding would be a different workflow. Neither is adopted. The actual unknown remains the original readout and the residual uses its true derivative bank.

## 8. Sums of products and more complex nonlinearities

A finite sum of derivative products gives

\[
R(w)=Lw+\sum_{\ell=1}^s(A_\ell w)\odot(B_\ell w)-f.
\]

Each scalar equation is an affine-quadratic measurement; its quadratic matrix has rank at most 2s. Its Jacobian is a sum of the same structured forward maps with changing pointwise coefficients. Constant-viscosity incompressible Navier–Stokes has this form in the stacked velocity/pressure readouts: diffusion, time derivatives, pressure and divergence are linear, while convection is quadratic. Complicated flow does not increase the polynomial degree of these equations. Unknown viscosity contributes additional bilinear terms. The Navier–Stokes compatibility note keeps the required stress, boundary and gauge details.

A general local constitutive law instead gives

\[
R(w,p)=F(A_0w,A_{\alpha_1}w,\ldots,p)-y,
\qquad J_w=F_z(z,p)A,
\quad z=Aw.
\]

The nonlinearity acts on the finite collection of values/derivatives at each physical point. Introducing z and retaining z=Aw is an exact reformulation with all neural consistency constraints. Derivative banks and their direction factors remain the same throughout the nonlinear solve.

This is a close match to [Chen–Hosseini–Owhadi–Stuart's nonlinear PDE/GP formulation](https://arxiv.org/pdf/2103.12959). Their representer reduction additionally minimizes a chosen Hilbert norm. The auxiliary-variable formulation transfers exactly; the norm choice is an extra modeling/selection decision, not a consequence of the chain rule. Their convergence theorem concerns minimizers under solution-space and sampling assumptions, not automatic optimizer convergence for our neural representation.

## 9. Kernel viewpoint and a construction-specific exact identity

The frozen network has the deterministic finite kernel

\[
K(x,y)=\phi(x)^T\phi(y),\qquad
\partial_x^\alpha\partial_y^\beta K(x,y)
=\phi_\alpha(x)^T\phi_\beta(y).
\]

Stacking all derivative observations gives G=AA^T. This is the dual Gram matrix of the same MLP, not replacement by another activation or an infinite-width model. If z is realizable, w=A^T G^†z+n with n in ker A. A declared minimum coefficient norm sets n=0; doing so without such a criterion changes how unsampled behavior is selected. When G is singular, z must remain in range A.

For a single tanh direction, define the known unweighted center sum

\[
S_N(t)=\sum_{j=1}^N\tanh(\gamma(t-c_j)),\qquad
k_N(t,s)=\sum_j\tanh(\gamma(t-c_j))\tanh(\gamma(s-c_j)).
\]

Then, for t≠s, the exact identity

\[
\boxed{k_N(t,s)=N-\coth(\gamma(t-s))[S_N(t)-S_N(s)]}
\]

follows from tanh(u)tanh(v)=1−coth(u−v)(tanh(u)−tanh(v)). The removable diagonal value is k_N(t,t)=N−S_N′(t)/γ. The centers need not be uniform; only the common slope is used. Differentiating gives derivative-kernel blocks. For example,

\[
\partial_t\partial_s k_N(t,s)\big|_{s=t}
=\frac{2\gamma}{3}S_N'(t)-\frac1{6\gamma}S_N'''(t).
\]

Mixed derivatives at coincidence must use the bivariate limit, not an ordinary derivative of the diagonal formula. Near coincidence, stable divided-difference/Hermite formulas are needed; direct subtraction and coth evaluation can lose accuracy. These identities are independently checked algebraically, not benchmarked as a machine-precision implementation.

This turns the center sum in a kernel entry into differences of a shared one-dimensional known function. After preprocessing the required S_N jets at projected sample coordinates, a pair entry needs O(1) radial algebra per direction, rather than a direct N-term sum. Summing directions still costs O(M). A dense sample kernel consequently still costs O(MK²) pair work, and a sparse factorization requesting Q entries pays O(MQ) unless angular sharing improves it. This is a specific construction-to-kernel bridge with an explicit remaining cost, rather than a generic instruction to use a kernel trick.

The follow-up [Chen–Owhadi–Schäfer sparse-Cholesky paper](https://arxiv.org/pdf/2304.01294) proves near-linear-in-observation-count factorization for a class of derivative kernel matrices. The hypotheses concern sufficiently regular local elliptic Green kernels, controlled scattered samples, and required value measurements. They are not established by our tanh kernel merely being smooth. Its log factors also carry dimension dependence; factor arithmetic and entry-query costs are distinct. The precise assumptions, nonlinear reduced-matrix caveat and a conditional preconditioning-transfer statement are recorded in [the PDE literature audit](nonlinear_pde_literature.md).

## 10. What projecting samples really provides

One physical observation q(x_i) provides one scalar equation. Its M projections θ_m·x_i are useful coordinates of that equation, not M observed profile values. Likewise,

\[
q(t\theta_m)=\sum_\ell g_\ell(t\theta_\ell\cdot\theta_m)
\]

still mixes profiles. Actual filtered Radon data are different observations involving hyperplane integrals and filtering; they cannot be manufactured from coordinate projection alone.

There is nevertheless a useful positive statement. If each of the M N projected bins has probability at least c/N under a physical sampling measure, O(N log(MN/δ)) shared physical points suffice to hit every bin with probability at least 1−δ. This supports local profile coverage, not joint identification. The exact network tanh(γx)−tanh(γy) vanishes on every diagonal sample (t,t), despite covering both projections densely.

For stable joint sampling, [Cohen–Migliorati's optimal weighted least-squares theorem](https://arxiv.org/html/1608.00512) supplies a better target: approximately p log p physical samples for a fixed p-dimensional function space, using its joint leverage function and reciprocal importance weights. The weights preserve the intended L² norm. p is the identifiable joint dimension, not automatically N. Constructing the sampling distribution can itself be expensive; the theorem is not a solver-cost or uniform machine-epsilon certificate.

A direct bridge to our structure is available: if the joint Gram dominates the block-diagonal profile Grams by a fixed positive factor a_*, then the sum of one-dimensional block leverage functions gives a valid sampling envelope with a count of order (MN/a_*) log p. Thus a stable directional decomposition could improve sampling and inference together. The bound deteriorates if directions nearly cancel; marginal coverage cannot repair that. For nonlinear PDEs, this applies first to a fixed linearized constraint space, and uniform/adaptive control must be proved separately. Existing observations cannot be discarded or relocated to fit the sampling theorem. Full derivations and source checks are in [projected sampling](projected_sampling_literature.md).

## 11. The next incremental proof targets

The next stage is no longer a search for one universal transformation. It has three concrete tests, each with an established literature counterpart:

1. For the linear PDE/data stack, derive a computable structured physical norm and joint sampling envelope. This must price the center and angular operations and retain boundary/initial/data rows.
2. For a single same-readout product, establish identification and conditioning of [J(w);C] for the actual derivative-bank measurements. Generic phase-retrieval or Gaussian recovery assumptions must be proved or replaced, not borrowed by name.
3. For the kernel formulation, characterize the actual finite derivative Gram matrix well enough to prove a relative comparison with a fast reference kernel/operator, including rank and boundary effects. A small unscaled entry error is insufficient near ill-conditioned modes, and a reference minimum norm must not silently become a physical prior.

Only after these survive should sums of products and general local nonlinear laws inherit the result. The activation-derivative and diagonal-direction factors remain visible at every step. This progression preserves the standard one-hidden-layer MLP and general PDE ambition while making failures, information requirements and possible literature transfers identifiable at a much smaller mathematical scale.
