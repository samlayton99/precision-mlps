# Native ridge-network Navier–Stokes operators

Date: 2026-10-09. This note derives exact residual and derivative actions for a standard one-hidden-layer network. It retains nonlinear advection, pressure, scattered samples, initial/boundary observations, and inverse viscosity. It establishes operator compatibility and channel reductions, not a convergent high-Reynolds-number solver or a total complexity theorem. No numerical benchmarks or optimizer tuning were performed.

## 1. The network stays a standard MLP

Let \(x\in\mathbb R^d\), let \(t\) denote time, and use fixed space–time directions
\(\theta_m=(a_m,\tau_m)\in\mathbb R^{d+1}\), where \(a_m\in\mathbb R^d\) is the spatial part. For physical coordinates define

\[
s_{jm}(x,t)=\gamma(a_m\cdot x+\tau_m t-c_j),
\qquad \|a_m\|^2+\tau_m^2=1.
\]

The hidden activation is \(\psi=\tanh\); the exact chain-rule identities below also hold for any sufficiently differentiable activation. The velocity and pressure share hidden features but have independent readouts:

\[
u=b_u+\sum_{m,j}v_{jm}\psi(s_{jm}),\qquad
p=b_p+\sum_{m,j}\pi_{jm}\psi(s_{jm}),
\tag{1}
\]

where \(v_{jm}\in\mathbb R^d\) and \(\pi_{jm}\in\mathbb R\). There are \(MN\) hidden units and \((d+1)MN+d+1\) output parameters. A represented viscosity field adds one scalar output head. This is one affine hidden layer, an elementwise activation, and one affine output layer; the evaluation formulas do not add learned layers or change the represented function.

If inputs are rescaled, the physical derivative factors must include the inverse coordinate scales. Then the effective spatial and time derivative vectors need not satisfy the displayed unit-norm identity. In particular, spatial diffusion must never be replaced by a space–time Laplacian.

For a fixed direction, write the activation-derivative profiles

\[
U_m^{[r]}=\sum_jv_{jm}\psi^{(r)}(s_{jm}),\qquad
P_m^{[r]}=\sum_j\pi_{jm}\psi^{(r)}(s_{jm}).
\]

The brackets distinguish activation derivatives from physical derivatives: physical derivatives introduce powers of \(\gamma\). In cumulative/profile coordinates from [local_structure.md](local_structure.md), the identical physical profiles can be evaluated by localized differences with endpoint bookkeeping.

## 2. Exact contraction before assembling derivative tensors

Let \(G=\nabla_xu\), with \(G_{i\ell}=\partial_\ell u_i\). Direct differentiation gives

\[
u_t=\gamma\sum_m\tau_m U_m^{[1]},\quad
G=\gamma\sum_mU_m^{[1]}a_m^\top,
\]
\[
\nabla\cdot u=\gamma\sum_ma_m\cdot U_m^{[1]},\quad
\Delta_xu=\gamma^2\sum_m\|a_m\|^2U_m^{[2]},\quad
\nabla_xp=\gamma\sum_ma_mP_m^{[1]}.
\tag{2}
\]

For constant kinematic viscosity \(\nu\), use the declared momentum and continuity residuals

\[
R=u_t+Gu+\nabla p-\nu\Delta_xu-f,
\qquad D=\nabla\cdot u.
\tag{3}
\]

**Directional contraction proposition.** The exact same momentum residual is

\[
\boxed{
R=\gamma\sum_m(\tau_m+a_m\cdot u)U_m^{[1]}
-\nu\gamma^2\sum_m\|a_m\|^2U_m^{[2]}
+\gamma\sum_ma_mP_m^{[1]}-f.
}
\tag{4}
\]

**Proof.** Apply the rank-one matrices in (2) to \(u\):
\(Gu=\gamma\sum_mU_m^{[1]}(a_m^\top u)\). Trace the same matrices for divergence and trace the spatial Hessian for diffusion. Substitute into (3). \(\square\)

Thus nonlinear advection needs a projected-velocity coefficient \(a_m\cdot u\) multiplying the first derivative bank. Spatial diffusion needs the single second derivative bank multiplied by \(\|a_m\|^2\), which equals \(1-\tau_m^2\), **not one** for general space–time directions. Neither all \(d^2\) velocity-gradient entries nor all \(d^3\) velocity-Hessian entries need to be materialized for the residual. This remains true for large velocities, strong gradients, and arbitrary readouts; it is not a low-Reynolds-number approximation.

Equation (4) still depends on the aggregate velocity from every direction. It does not make directions independent or remove the physical nonlinearity.

### Exact low-rank modulation, without a low-rank flow assumption

At the physical sample sites form the point–direction modulation

\[
C_{km}=\tau_m+u(x_k,t_k)\cdot a_m.
\]

Let \(\mathsf U\in\mathbb R^{K\times d}\) have rows \(u_k^\top\), and \(\mathsf A\in\mathbb R^{M\times d}\) have rows \(a_m^\top\). Then exactly

\[
\boxed{\mathsf C=\mathbf1\tau^\top+\mathsf U\mathsf A^\top,
\qquad \operatorname{rank}\mathsf C\le d+1.}
\tag{4a}
\]

There is no assumption about the complexity, turbulence, smoothness scale, or rank of the velocity field beyond its evaluability at these sites. The rank bound comes from the first-order differential contraction. Store \(\mathsf U,\mathsf A,\tau\), not an independently parameterized \(K\times M\) array. The diffusion modulation \(\nu_k\|a_m\|^2\) is likewise a rank-one point–direction product, including variable scalar \(\nu_k\) when that is the declared operator coefficient.

The full feature/Jacobian matrix is **not** low rank as a consequence: it still contains the jointly varying radial factors \(\psi^{(r)}(s_{kjm})\). What is gained is an exact small sum of point multipliers and direction multipliers around fixed radial derivative evaluation maps. Geometry-dependent projection, window, or any future certified transform plans can therefore be reused across nonlinear iterations at fixed sites and geometry; only the physical-point coefficient factors change. Adaptive changes to sites or geometry require their own plan update and cost. This exact modulation structure is compatible with highly complicated flows and does not truncate their angular representation.

## 3. Exact Jacobian and transpose actions

Let \(\delta v,\delta\pi\) be arbitrary readout changes, with corresponding profile variations \(\delta U_m^{[r]},\delta P_m^{[r]}\), and let \(\delta\nu\) be a scalar viscosity variation if viscosity is unknown. Put \(L=\Delta_xu\). Differentiating (4), rather than dropping either occurrence of velocity in advection, gives

\[
\boxed{
\begin{aligned}
\delta R={}&\gamma\sum_m\left[
(\tau_m+a_m\cdot u)\delta U_m^{[1]}
+(a_m\cdot\delta u)U_m^{[1]}\right]\\
&-\nu\gamma^2\sum_m\|a_m\|^2\delta U_m^{[2]}
-\delta\nu L+\gamma\sum_ma_m\delta P_m^{[1]},\\
\delta D={}&\gamma\sum_ma_m\cdot\delta U_m^{[1]}.
\end{aligned}}
\tag{5}
\]

The term \((a_m\cdot\delta u)U_m^{[1]}\) is the often-important \(G\delta u\) part of the full Newton linearization. Omitting it produces a different linearization, whose behavior needs separate analysis.

For transpose actions, let \(z_k\in\mathbb R^d\) be a momentum-residual dual vector and \(\zeta_k\in\mathbb R\) a divergence dual at sample \(k\). Sample weights can be absorbed into these vectors. At that point define

\[
h_k=G_k^\top z_k
=\gamma\sum_ma_m(U_m^{[1]}\cdot z_k).
\tag{6}
\]

The contributions to the gradients of the original readouts are

\[
\boxed{
\begin{aligned}
(J^\top[z,\zeta])_{v_{jm}}
={}&\sum_k\bigl\{
\psi(s_{kjm})h_k\\
&+\gamma\psi'(s_{kjm})[(\tau_m+a_m\cdot u_k)z_k+\zeta_ka_m]\\
&-\nu\gamma^2\psi''(s_{kjm})\|a_m\|^2z_k\bigr\},\\
(J^\top[z,\zeta])_{\pi_{jm}}
={}&\sum_k\gamma\psi'(s_{kjm})(a_m\cdot z_k),\\
(J^\top[z,\zeta])_\nu
={}&-\sum_kz_k\cdot L_k.
\end{aligned}}
\tag{7}
\]

Velocity bias contributions are \(\sum_kh_k\); the pressure bias receives no interior contribution. Boundary/data rows add their own derivatives. These identities follow by substituting profile variations into \(\sum_k(z_k\cdot\delta R_k+\zeta_k\delta D_k)\) and collecting each readout variation. Consequently (5) and (7) are a matching Jacobian/transpose pair.

Although (7) is written in standard MLP coordinates, the implemented transpose must match whichever exact coordinate change and certified evaluator are used. For a coordinate map \(w=Ta\), it is \(T^\top J_w^\top\). For truncated/local evaluation, take the exact transpose of that implemented approximate linear map; do not combine it with an unrelated exact-dense transpose.

## 4. Unknown viscosity fields: preserve the declared stress law

A scalar coefficient multiplying \(\Delta u\) is a legitimate declared PDE operator, but it is not generally the Newtonian variable-viscosity stress divergence. For constant density and scalar viscosity, the latter is

\[
\mathcal V=\nabla\cdot[\nu(G+G^\top)]
=\nu H+(G+G^\top)n,
\quad H=\Delta u+\nabla D,\quad n=\nabla\nu.
\tag{8}
\]

The \(\nabla D\) term cannot be discarded when evaluating the off-solution residual merely because continuity is another equation. Different residual extensions away from \(D=0\) can have different conditioning. A solver must use the declared extension consistently.

If \(\nu\) is another network output with derivative profile \(V_m^{[1]}\), then

\[
n=\gamma\sum_ma_mV_m^{[1]},\qquad
H=\gamma^2\sum_m[\|a_m\|^2I+a_ma_m^\top]U_m^{[2]},
\]
\[
(G+G^\top)n
=\gamma\sum_m\left[U_m^{[1]}(a_m\cdot n)+a_m(U_m^{[1]}\cdot n)\right].
\tag{9}
\]

Each rank-one spatial contraction costs \(O(d)\), so this physically important variant also avoids assembling full Hessians. The momentum equation is now \(R=u_t+Gu+\nabla p-\mathcal V-f\).

Its viscosity variation is exactly

\[
\delta\mathcal V=\delta\nu H+\nu\delta H
+(\delta G+\delta G^\top)n+(G+G^\top)\delta n.
\tag{10}
\]

For completeness, the matching per-point velocity-readout transpose contribution is

\[
\begin{aligned}
\psi(s)G^\top z
&+\gamma\psi'(s)\left[
(\tau_m+a_m\cdot u-a_m\cdot n)z
-(a_m\cdot z)n+\zeta a_m\right]\\
&-\nu\gamma^2\psi''(s)
[\|a_m\|^2z+a_m(a_m\cdot z)].
\end{aligned}
\tag{11}
\]

The pressure contribution remains as in (7). If \(\chi_{jm}\) is a viscosity readout, its contribution is

\[
\psi(s)(-z\cdot H)
-\gamma\psi'(s)a_m\cdot[(G+G^\top)z].
\tag{12}
\]

All terms follow from (10) by contraction; physical viscosity constraints, data coverage, and identifiability are separate aspects of the stated inverse problem. No coefficient prior or positivity parameterization is silently introduced by these formulas.

## 5. Initial/boundary conditions, data, and pressure freedom

Initial velocity values and Dirichlet boundary values use the order-zero bank. Normal derivatives contract the first derivative bank with \(a_m\cdot\eta\), where \(\eta\) is the actual outward spatial normal. Traction uses

\[
\sigma\eta=-p\eta+\nu(G+G^\top)\eta,
\]
\[
(G+G^\top)\eta
=\gamma\sum_m\left[U_m^{[1]}(a_m\cdot\eta)+a_m(U_m^{[1]}\cdot\eta)\right].
\tag{13}
\]

This applies at arbitrary boundary sample locations and arbitrary normals, without changing the geometry of the neural network or assuming a special physical domain. Velocity/pressure observations, derivative observations, and local nonlinear boundary laws append their own residual rows. Nonlocal measurements require their actual observation operators and quadrature costs.

Interior unsteady incompressible equations determine pressure only up to \(p(x,t)+c(t)\). Boundary traction or pressure data may fix this freedom; otherwise the gauge must be resolved at each relevant time (for example by a time-dependent spatial mean constraint). One global scalar pressure-mean condition does not remove arbitrary time-dependent pressure freedom. Any representable time-only pressure modes, and near-null approximations to them, matter to the discrete solve. A gauge is imposed only where there is a real physical invariance; it must not contradict prescribed pressure/traction data.

If a mixed saddle-point solver or preconditioner is proposed, its discrete velocity-pressure inf-sup bound must be proved in the chosen norms. For strong-residual least squares, the corresponding issue is observability/coercivity of the full sampled operator after removing its genuine gauge: sharing tanh features does not prove either bound. This note does not claim that the textbook mixed finite-element inf-sup condition, unchanged, is necessary for every least-squares formulation. The pressure-nullspace and block-preconditioning issues also appear in the primary [Firedrake Navier–Stokes demo](https://www.firedrakeproject.org/firedrake/demos/navier_stokes.py.html); that implementation is a reference for those issues, not a proposed replacement for the neural model.

For inverse viscosity, a steady zero-velocity field with spatially constant pressure gives zero viscosity sensitivity: (7) vanishes. Thus arbitrary observation availability does not imply identifiability. Rich enough flow/data and the relevant sensitivity lower bound must be supplied for the actual inverse problem. Variable coefficient fields can have further unobserved directions.

## 6. Propagating neural-jet errors into actual residual errors

At one physical point, let the exact network jets be \(u,G,T=u_t,L=\Delta u,P=\nabla p\). Let approximate evaluation errors satisfy

\[
\|\widetilde u-u\|\le\epsilon_u,\quad
\|\widetilde G-G\|_2\le\epsilon_G,\quad
\|\widetilde T-T\|\le\epsilon_T,\quad
\|\widetilde L-L\|\le\epsilon_L,\quad
\|\widetilde P-P\|\le\epsilon_P,
\]

and \(|\widetilde\nu-\nu|\le\epsilon_\nu\). Set \(U=\|u\|\), \(G_*=\|G\|_2\), and \(L_*=\|L\|\). Expansion of the two products in (3) gives

\[
\boxed{
\|\widetilde R-R\|
\le\epsilon_T+G_*\epsilon_u+U\epsilon_G+\epsilon_u\epsilon_G
+|\nu|\epsilon_L+\epsilon_\nu(L_*+\epsilon_L)
+\epsilon_P+\epsilon_f.
}
\tag{14}
\]

Here \(\epsilon_f\) is forcing-evaluation error and can be zero. The continuity error satisfies
\(|\widetilde D-D|\le d\epsilon_G\); with a Frobenius gradient-error bound it improves to \(\sqrt d\epsilon_G\). Direct divergence evaluation can instead use its own tighter contracted error bound. Dirichlet and initial residual errors are at most \(\epsilon_u\). For traction with a unit normal,

\[
\|\widetilde{\sigma\eta}-\sigma\eta\|
\le\epsilon_p+2|\nu|\epsilon_G
+2\epsilon_\nu(G_*+\epsilon_G).
\tag{15}
\]

For variable-viscosity stress, let \(\epsilon_H,\epsilon_n\) bound errors in \(H,n\). The additional viscous error is bounded by

\[
|\nu|\epsilon_H+\epsilon_\nu(\|H\|+\epsilon_H)
+2G_*\epsilon_n+2\epsilon_G(\|n\|+\epsilon_n).
\tag{16}
\]

Equations (14)–(16) are errors relative to the **actual neural residual**, not a surrogate spectral field. They can be combined with the profile tail bounds in [local_structure.md](local_structure.md), coordinate-synthesis errors, Fourier/interpolation errors if those are later used, and arithmetic bounds. Large gradients or coefficient sensitivities demand a smaller evaluator error budget. No fixed cutoff certifies all flows.

Small evaluation error and small sampled residual do not automatically imply a small physical solution error. For illustration, for sufficiently regular divergence-free flows with compatible energy boundary conditions, comparison of a forced approximate flow to a reference solution gives an energy estimate containing factors like

\[
\exp\left(2\int_0^T\|\nabla u(t)\|_{L^\infty}\,dt\right)
\]

and, when residuals are measured in \(H^{-1}\), a factor \(1/\nu\) from absorbing forcing against viscous dissipation. This follows by bounding the linearized stretching term \(\int e\cdot(\nabla u)e\) by \(\|\nabla u\|_\infty\|e\|_2^2\), applying Young's inequality to the residual term, and then Gronwall's inequality. Nonzero continuity and boundary errors add terms that this illustration does not bound. It is not an assumption that the application's domain or conditions are of this special kind.

Such sensitivity is physical and may worsen at high Reynolds number or long time. The current algebra does not restrict to small Reynolds number or short time, but neither does it prove width-uniform global convergence, long-time trajectory accuracy, or regularity of arbitrary 3D flows. A machine-epsilon field target needs the relevant physical stability bound in addition to machine-epsilon operator application.

## 7. Actual operation counts and nonlinear sampling

Let \(K\) be the total number of physical sites used in one residual/Jacobian application and \(s\) the certified radial stencil size. The velocity and pressure profiles require only activation orders 0, 1, and 2. Unknown variable viscosity additionally requires its orders 0 and 1. All \(d\) velocity heads are counted.

| Work | Cost / storage |
|---|---|
| Form space–time projections at arbitrary sites | \(O(KM(d+1))\) work; \(O(KM)\) storage if cached |
| Evaluate necessary profiles and contract residuals | \(O(KMds)\) work for fixed derivative orders and number of heads relative to \(d\); \(O(Kd+MNd)\) streamed storage |
| Apply an exact linearization or its matching transpose | A fixed number of the same profile passes, \(O(KMds)\), without constructing \(d^2\) gradient arrays |
| Store the model | \(O(MN(d+1))\); one extra scalar head for viscosity fields |
| Construct all gradient entries, if explicitly requested by data or diagnostics | Additional \(O(KMd^2)\) contraction work and \(O(Kd^2)\) storage if retained |
| Independent constraint validation | The same costs at its additional sample sites; no free validation |

For the forward residual, evaluate \(u\) first, then use the projected coefficients \(a_m\cdot u\) in a second pass, or cache the needed profiles. For Jacobian actions first obtain \(\delta u\); for transpose actions first obtain \(G^\top z\) through (6). A constant number of streamed passes avoids an \(O(KMd)\) profile cache. The variable-viscosity contractions (9)–(12) also require only fixed additional passes.

These counts exploit contraction of the PDE, not a claim that an arbitrary declared differential tensor can be produced in \(O(d)\). With \(K=\Theta(MN)\), the displayed work is \(O(M^2Nds)\), so the full aggregation bottleneck remains. Nonlinear and linear iteration counts, preconditioner setup/applications, unknown-parameter identification, angular resolution, and the precision dependence of \(s\) are not included in one application and must enter a total complexity claim.

Computing \(Gu\) at scattered points from the evaluated neural jets introduces no spectral aliasing by itself. It computes the nonlinear residual of that same network at those points. Undersampling can nevertheless miss a large residual between them. If a subsequent fast scheme projects products back into a finite transform basis, truncation/aliasing becomes an additional error. A fixed \(3/2\) padding rule for bandlimited quadratic Fourier products is not a universal exactness rule for tanh fields, irregular samples, or general nonlinearities. Primary [Dedalus documentation](https://dedalus-project.readthedocs.io/en/gpu/notebooks/dedalus_tutorial_3.html) explicitly changes the dealiasing factor for a cubic nonlinearity; any transform used here needs its own justified approximation and sampling rule. No nonlinear-product projection has been adopted in this note.

## 8. Native-network preconditioning and admissible next theorem

The unknown remains the readout of (1), and every accepted candidate is evaluated with (3), or the explicitly declared stress variant (8), plus its true IC/BC/data/gauge constraints. An invertible right coordinate map \(\delta w=Pz\), a preconditioner for the original normal equations, or an exactly equivalent constrained formulation may use a reference operator without changing the network or its solution set. A reference approximation can use the fixed geometry and principal derivative banks; it does not supply an external field that is later encoded into the network.

Care is required with left preconditioning: an invertible left multiplier preserves the roots of an exactly consistent equation, but minimizing a new Euclidean norm of a left-scaled **overdetermined residual** generally changes the least-squares objective. To preserve the stated objective, carry its transformed metric or precondition the original linear system without redefining that objective. Similarly, solving only in a small coarse space changes the candidate family unless a full fine correction remains available.

The admissible remaining target is a stable, fast realization of the coupled native Jacobian on physically resolved flows, including pressure constraints, rather than a restriction to low-Reynolds-number or low-degree examples. The directional contractions above provide a concrete reduction from full derivative tensors to three shared activation banks. They do not yet establish fast all-direction aggregation or a preconditioner robust to angular redundancy, pressure coupling, thin layers, inverse sensitivity, and nonlinear stretching. Those are explicit theorem obligations, not numerical tuning opportunities.

All exact chain-rule contractions, the point–direction rank bound, and the JVP/VJP identities extend to any \(C^2\) activation for the stated second-order equations. The tanh partition, exponential locality, and radial conditioning bounds do not automatically extend: a replacement activation needs its own evaluation/localization and stability theorem. The compatibility argument therefore supports the original \(\Phi^{(r)}D\) framework without identifying arbitrary activations with tanh's additional structure.

Verification: exact rational-arithmetic audits checked the adjoint identity for the constant-viscosity and variable symmetric-stress velocity/viscosity blocks, including velocity biases and the continuity row. These were finite algebraic identity checks, not PDE solution experiments. Pressure blocks follow directly from the linear gradient contraction in (2).
