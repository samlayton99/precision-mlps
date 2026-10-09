# Claude continuation: evidence audit, October 3, 2026

This is a read-only source-and-record audit for Sam's explanation request. No solver, experiment, test, or benchmark was run. I read the new repository's requirements and handoff, followed the main methods into source, and checked the central numbers against saved JSON. This report organizes by mechanism rather than by the sequence of agent experiments. It does not propose new work.

Root of the audited repository: `/Users/sam/my-repos/research/collaborations/quill-radon`.

Path abbreviations below:

- `S/` = `agent/workspace/2026-10-02_structured_solver/`.
- `H/` = `agent/workspace/2026-10-02_route2_hard_cases/`.
- `T/` = `agent/workspace/2026-10-02_pde_theory_leads/`.
- `R/` after a workspace name = that workspace's `results/` directory.

Companion audit: `theory_history_evidence.md` contains direct user-authority quotes and the theory agent's construction/count records. The parent coordinator separately audits old route 2 and literature standards.

## 1. The central finding

The continuation contains several real numerical advances, but they are advances in different components. There is no single new solver whose evidence covers all of the reported problems.

1. **Actual tanh-readout solving:** the October 2 work applies differential operators to readout sequences through FFTs in a special translation-invariant setting; it also solves a variable-coefficient disk problem directly in tanh readouts with a dense pseudoinverse preconditioner. These are genuine attempts to compute the network state without first computing a different field. The first has special forcing/domain assumptions; the second retains the dense scaling wall. Attempts to make the second fast and general failed.
2. **Stable compilation of an already computed field:** the Helmholtz advance is an improved choice of ridge profiles for a polynomial already solved by classical collocation. It prevents enormous intermediate coefficients from ruining the final tanh sum. This is a useful compiler advance. It is not a new way to discover the PDE solution.
3. **Classical structured PDE solves, followed by network construction:** most October 3 results use Fourier/Chebyshev time slabs, ball polynomials, or boundary integrals/FFT to obtain a field. The field is then transformed into ridge profiles and encoded as one tanh layer. Some results stop at the classical field; others evaluate the actual tanh sum. Inverse Burgers and initial-condition assimilation stop at the classical field.

This distinction must be explained before any accuracy table. A number can be a field-solver error, a network error, an encoding discrepancy against the input field, or a boundary discrepancy. The handoff occasionally uses one label for another.

### Later permission matters

The new repository's `human/REQUIREMENTS.md` is an early condensation and is marked awaiting confirmation. It cannot alone settle whether later polynomial cores were authorized. Direct user text in the Claude transcript `/Users/sam/.claude/projects/-Users-sam-my-repos-research-collaborations-precisionMLPs/3bf10edb-b6ec-4265-ada4-d7d58e892472.jsonl:2008`, timestamp `2026-10-03T06:19:15.618Z`, says: **“If we are constructing it from a polynomial method, but it offers everything else, that is fine.”** The same message retains one hidden layer and asks for comparisons with spectral methods. Line 2024 confirms pushing route 2. Line 4129 allows dense solves provided they do not cap scaling. These are conditional relaxations: a polynomial core is not automatically a violation, but neither is every solver followed by network encoding automatically authorized.

Sam's latest clarification explicitly distinguishes this conditional polynomial allowance from permission to substitute arbitrary boundary-integral solvers or time-steppers and then encode their output. Those later substitutions must be disclosed as departures, not retroactively justified by the polynomial quote. The source audit verifies that both kinds of pipeline occurred. It also verifies test changes: the duct result has round rods, not a general elliptical-obstacle geometry, and the square-with-holes result uses compatible constant boundary data that permit exact reflections, not a general corner treatment. The companion report records the direct historical authority; Sam's current clarification controls its interpretation.

## 2. What the mathematical objects actually are

An ordinary one-hidden-layer network on input (x\in\mathbb R^d) is

\[
u_B(x)=b+\sum_{m=1}^{M}\sum_{j=1}^{H_m}a_{mj}\tanh\!\left(\gamma_m(v_m\cdot x-c_{mj})\right).
\]

Here (v_m\in\mathbb R^d) is a direction, (c_{mj}\in\mathbb R) a center, (\gamma_m>0) the slope, and (a_{mj}\in\mathbb R) the output weight. (B=\sum_mH_m) is the number of neurons. Direction weights can be absorbed in (a_{mj}); shifts of the input center can be absorbed in the first-layer biases. Thus a structured evaluator that loops over directions can still evaluate precisely the function of one flat MLP.

However, the unknowns being solved need not be these (B) weights. In the new methods the solve state is variously:

- the (B) readouts themselves;
- (P=\binom{p+d}{d}) independent polynomial coefficients;
- (N_xN_t) values on a space-time slab;
- a boundary density at (N_b) boundary nodes;
- a grid field with (G=n^d) values;
- a small vector of physical parameters.

This is why “millions of neurons” does not establish “a million-dimensional nonlinear readout solve.” The constructor can manufacture many linked weights from a much smaller previously solved state.

Let (E) denote the linear map from profile coefficients to tanh weights, where the encoder is used linearly. Let (A) denote a field-sampling map from profile coefficients to real function samples. These are different maps. A polynomial field may have many coefficient vectors (c) with the same (Ac), but very different encoded weights (Ec). The stable-profile advance exploits precisely that freedom.

## 3. Worked recipe A: native readout operators and nonlinear solving

### 3.1 Uniform translations make constant-coefficient operators simple

Source: `S/probe1_readout_operators_1d.py:24–48,91–140`; raw result `S/results/probe1_1d.json`.

The grid has 481 centers from (-12) to (12), spacing (h=0.05), (\lambda=0.25), and (\gamma=\lambda/h=5). Define (A_c=\pi/(2\gamma)), the complex shift; the script's variable is simply `A`. For an entire, decaying function (F), it computes the leading infinite-line density formula

\[
a_j={h\over2A_c}\operatorname{Im}F(c_j+iA_c),\qquad
b=\sum_j a_j.
\]

The bias enforces (u(-\infty)=0). This particular probe uses a long real-line grid, not the production short-halo finite-contour compiler. Gaussian decay makes the chosen truncation benign in this example.

For a constant-coefficient differential operator (L), differentiation commutes with translation. On a resolved band, applying (L) to the represented function corresponds approximately to applying its Fourier symbol to the readout sequence:

\[
L_a a=\operatorname{IFFT}\left[\ell(\omega)\operatorname{FFT}(a)\right].
\]

The arrays are padded by a factor of four before this action. For (L=-\partial_x^2+1), (\ell(\omega)=\omega^2+1), so the inverse divides by (\omega^2+1). This is not solving a global neuron matrix. It is exploiting translation invariance and the Fourier diagonalization of that operator. Boundary truncation, band truncation, and aliasing still matter; it is not an exact arbitrary-domain identity.

The checks observe encoding error (1.407\times10^{-14}), derivative error (5.089\times10^{-16}), second-derivative error (1.627\times10^{-14}), and inverse-operator reconstruction error (3.615\times10^{-14}), all versus the selected exact decaying functions.

### 3.2 How it handles a nonlinear term

A product of two ridge sums is generally not a ridge sum on the same discrete dictionary. The 1D probe handles this by evaluating the current tanh network on a grid four times finer, computing the product pointwise, then converting those real samples back to readouts:

\[
\mathcal P(g)_j={h\over2}\left[\operatorname{IFFT}\left(
{i\sinh(A_c\omega)\over A_c}\mathbf1_{|\omega|<\omega_c}\operatorname{FFT}(g)
\right)\right]_{4j}.
\]

The multiplier is the complex-shift density expressed on real-line Fourier samples. It avoids asking for an analytic continuation of the current numerical network. Its rapidly growing magnitude also explains why a band cutoff is essential. The cutoff in the final nonlinear probe is (0.4\pi/h). This is an approximation choice, and the saved product sweep demonstrates that under-resolution matters. Product projection plateaus around (3.8\times10^{-13}), whereas a separate dense least-squares control reaches (4.20\times10^{-14}).

### 3.3 The actual nonlinear solve

Problem: (-u''+u+u^3=f) on the line with decaying solution. The manufactured exact solution is

\[
u_*(x)=0.8e^{-x^2/2}(1+0.3\sin3x).
\]

The formula constructs the forcing and checks the output. The solve begins at zero readouts; it does not initialize from (u_*). The forcing is available at complex arguments for its initial density construction.

For current readouts (a), evaluate the current network (u_a), then form the **readout-coordinate residual**

\[
r(a)=L_a a+\mathcal P(u_a^3)-a_f.
\]

Given a trial correction (\delta a), the code applies the linearized residual without forming a Jacobian:

\[
J(a)\delta a=L_a\delta a+\mathcal P(3u_a^2u_{\delta a}).
\]

GMRES approximately solves (J(a)\delta a=-r(a)), using the FFT inverse of (L_a) as its preconditioner. Then (a\leftarrow a+\delta a). This is Newton's method on the nonlinear equation in readout coordinates. The preconditioner only helps solve the correction equation; it does not supply the desired nonlinear solution. No line search is present in this prototype.

Saved progression in relative field error: (0.359,0.0551,1.42\times10^{-3},9.49\times10^{-7},4.12\times10^{-13},3.52\times10^{-14},3.55\times10^{-14}). GMRES takes 6,14,12,13,13,13,12 iterations. The result is checked through an explicit NumPy tanh sum (`net`, line 36), not a separate Torch export. The recorded nonlinear residual is the readout-coordinate residual, not an independent raw physical PDE audit at held-out points. The field comparison is independent of the iterative state but uses the known manufactured formula.

This is a real native demonstration. Its reach is limited: 1D, a decaying analytic forcing with complex access, a long uniform grid, a chosen spectral cutoff, and an FFT-compatible principal operator. It is closely equivalent to a Fourier/sinc spectral method in different state coordinates; that mathematical equivalence does not invalidate the computation, but it limits novelty claims and generality.

### 3.4 Its 3D constant-coefficient extension

Source: `S/probe2_radon_3d_solve_inverse.py:32–165`. Command documented in the script/workspace: `~/venv/quill-radon/bin/python probe2_radon_3d_solve_inverse.py` from `S/`.

For (-\nabla\cdot(K\nabla u)+\kappa^2u=f), a ridge (q(v\cdot x)) obeys

\[
Lq(v\cdot x)=-(v^TKv)q''(v\cdot x)+\kappa^2q(v\cdot x).
\]

Each forcing profile is encoded first; the solver then divides its readout FFT by ((v^TKv)\omega^2+\kappa^2). Thus directions are independent in this setting. The forcing is an anisotropic Gaussian with known analytic Radon profiles. The angular rule is 32 positive (z) nodes times 64 azimuths, (M=2048). There are 176 retained centers per direction, (B=360448). Padding uses 704 entries per direction. Weights outside the retained interval are folded into the bias using tanh saturation in the evaluation region.

Raw result `S/results/probe2_3d.json`: 0.0200 seconds for the FFT division after forcing construction; 0.885 seconds to evaluate values and Hessians at 600 points; normalized raw PDE residual (1.873\times10^{-14}). The claimed field-reference discrepancy (6.804\times10^{-14}) uses a long, fine Fourier solve of the **same analytic Radon profile ODEs and same angular decomposition**. It checks encoding/profile discretization; it is not an independent check of the angular/Radon physics. The forcing normalization is calibrated from its known value at the origin (`:64–65`). No complicated boundary or nonlinear coupling is present.

The inverse experiment recovers six diffusion-tensor entries through a Cholesky parameterization and (\kappa) from 200 observations. Its Jacobian differentiates the same rational Fourier multiplier. Clean errors are (8.25\times10^{-14}) in (K), (1.30\times10^{-13}) in (\kappa). At 0.01%/0.1%/1% noise relative to maximum solution value, (K) errors are (1.27\times10^{-4},2.55\times10^{-3},4.07\times10^{-2}). Data use the related spectral-profile reference, so the clean floor shares a discretization/Radon assumption; the workspace itself acknowledges that inverse-crime concern. One realization at each noise level does not establish calibrated uncertainty generally.

### 3.5 Bounded ball correction

`S/probe3_ball_dirichlet.py` adds a homogeneous screened-Poisson correction to the particular solution. It obtains spherical-harmonic coefficients of the boundary defect by quadrature and converts them into stable real ridge profiles through Funk–Hecke. It does not solve a volume least-squares matrix. Raw `probe3_ball.json`: 360448 particular neurons +783120 homogeneous neurons =1143568; 400 random interior and 400 random sphere checks; relative sampled interior residual (1.11\times10^{-13}), sampled boundary error (3.33\times10^{-13}).

The reported `certificate_max_error_rel_to_u=3.67e-13` is **not a rigorous certificate**. Line 186 substitutes the maxima on those finite point sets into the maximum-principle bound that requires continuous supremum norms. A valid description is “sampled maximum-principle error indicator.” The raw per-degree encoding check is (2.76\times10^{-9}), even though the final weighted correction is much more accurate; small high-degree boundary coefficients suppress the imperfect high-degree encodings. Do not claim every profile was individually encoded to machine precision.

## 4. Worked recipe B: directly solve actual tanh residuals in 2D

Sources: `S/probe4_varcoef_disk.py:37–57,67–157`; `S/probe4b_gmres_trace.py:29–89`; raw `S/results/varcoef_gmres_trace_rot0.37.json`. Reproduction command recorded by source conventions: `ROT=0.37 ~/venv/quill-radon/bin/python probe4b_gmres_trace.py` in `S/`.

Problem: (-\nabla\cdot(a(x)\nabla u)+u=f) on a disk, Dirichlet trace supplied. The smooth exact field (\sin(1.3x+0.7y)e^{0.4xy}+0.5\cos2y) and coefficient (1+c[1+\sin(1.5x-y+0.3)]/2) are both rotated by 0.37 radians. Forcing and boundary values are manufactured from that field; its interior values are used for reporting, not as solve rows.

1. Fix (M=N) uniformly spaced directions and (N+2\lceil\sqrt N\rceil+1) centers per direction, including endpoints/halos. Append a free bias. At (N=48), there are 3024 neurons plus bias, hence 3025 unknowns.
2. Evaluate every tanh feature, its first derivatives, and Laplacian at all interior and boundary points. These are **actual tanh formulas**, not polynomial surrogates.
3. Assemble rectangular (A\in\mathbb R^{Q\times B}), whose interior columns apply the full variable-coefficient operator to each neuron and whose boundary columns are the neuron values. Form (b) from forcing and Dirichlet data. Boundary rows are weighted to balance the point counts.
4. Construct (A_0) using the constant mean coefficient in the same dictionary. Compute its dense SVD and truncate singular values below (10^{-14}\sigma_{\max}). The preconditioner is (Z=A_0^+), the truncated pseudoinverse.
5. Starting from zero weights, run GMRES on (ZAa=Zb). A Krylov step applies (A), then (Z); it seeks combinations of previously obtained correction directions. This is legitimate preconditioning, not an external solution being encoded.
6. For diagnostics, also compute the direct dense-SVD solution of (Aa=b), then compare iteration traces with it and with the analytic field. The stopping/count headline is measured **afterward using exact field error**: “first iteration within 10× the direct solve's field error.” That is not an implemented blind stopping rule.

At contrast 3, first counts meeting that diagnostic are 40,45,40,35 for 841,1441,3025,5185 unknowns. At contrast10 and3025 unknowns it is85. At3025/contrast3, the best recorded field error is (2.74\times10^{-16}); final iteration300 is (2.96\times10^{-15}); physical residual relative norm about (3.4\times10^{-14}). At1441 the best is (4.89\times10^{-16}), final300 is (8.95\times10^{-14}), and physical residual around (4.2\times10^{-13}). These are impressive approximation/conditioning observations but show why “iterations to floor” and an operational stopping rule differ.

The preconditioned residual itself stalls near (10^{-4}) because (Z) amplifies near-null-space rounding. The field can be excellent while that residual is not small. Continuing can make field error worse. This is one reason the stopping metric must be stated.

**Scaling:** this experiment materializes several (Q\times B) feature/operator matrices, an SVD, and the direct-SVD control. (Q\sim3B), so memory is quadratic in neuron count and setup cubic. At5185 unknowns, one (Q\times B) double matrix is already roughly650MB; several coexist. The result does not solve Sam's million-neuron memory requirement.

### Fast replacements failed

The next probes replace the dense (Z) with extension→Fourier slice→per-direction division→encoder→boundary correction. A (C^3) reflection/window extension (`probe5_structured_Z_disk.py`, `probe5b_right_precond.py`) gives only (10^{-4})-level inverse action on a smooth test and stalls the PDE iteration around (10^{-6})–(10^{-5}). An entire erfc-window attempt (`probe6_analytic_window_Z.py`) improves that isolated test to roughly (2\times10^{-5}) but the iteration fails near (10^{-1}) field error: the unconstrained off-domain extension contaminates the inverse. The mechanism did not become a robust general fast solver at this stage.

## 5. Worked recipe C: the Helmholtz gain is a stable compiler

Sources: `H/helmholtz.py:23–51`, `H/helm_tikhonov.py` (`solve_poly`), `H/r2core.py:66–145`, `H/helm_ladder3.py:16–55`. Raw results `helm_ladder3_nall.json`, `helm_ladder3_n46.json`; preceding controls `helmholtz.json`, `helm_ladder*.json`. Commands: `python helm_ladder3.py` or `python helm_ladder3.py 4 6` in `H/`, using the repo venv. The module executes on import; this audit did not import it.

Problem: (-\Delta u-k^2u=k^2\sin(kx)\sin(ky)) on ([0,1]^2), zero boundary values, (k=2\pi n). The target is known for verification; forcing/boundary data alone determine the spectral solve.

1. Represent a field by a tensor Chebyshev polynomial (u_p=\sum_{r,s=0}^p b_{rs}T_r(2x-1)T_s(2y-1)). There are ((p+1)^2) coefficients.
2. Evaluate its exact derivatives at oversampled Chebyshev points. Assemble the Helmholtz residual matrix and boundary rows. Column-normalize and solve by dense QR plus a small-factor least-squares step. This computes the polynomial field **before** constructing its network.
3. Choose (M=2p+1) ridge directions. Give each direction a Chebyshev profile of degree (D=2p+20) on its projected interval. (K=M(D+1)) is the number of redundant profile coefficients. Let (c\in\mathbb R^K) be those coefficients and (A\in\mathbb R^{Q_c\times K}) their summed real values on compiler sample points. Let (y\in\mathbb R^{Q_c}) be the solved polynomial at those points.
4. Many (c)'s give almost the same (Ac=y). The naive minimum-Euclidean-norm profile vector can grow enormously at complex points even if the summed real field stays small. QUILL samples those complex points; it then creates huge canceling readouts. This is the observed compiler instability.
5. Record the actual complex points at which the encoder queries a profile. If $Q_t$ maps one profile's Chebyshev coefficients to those complex values, stack its real and imaginary parts and QR-factor them: $Q_t=Q_RR$. The norm $\|Rc_m\|_2$ equals the sampled tube norm of profile $m$.
6. Solve the regularized compiler problem

\[
\min_c\|Ac-y\|_2^2+\mu\sum_m\|Rc_m\|_2^2.
\]

With block-diagonal (R_b=I_M\otimes R), set (z=R_bc). Then solve

\[
\min_z\|AR_b^{-1}z-y\|_2^2+\mu\|z\|_2^2.
\]

If (AR_b^{-1}=U\Sigma V^T), (z=V\operatorname{diag}(\sigma_i/(\sigma_i^2+\mu))U^Ty), then recover (c=R_b^{-1}z). The script uses QR first and SVD of the QR factor.
7. Encode each resulting full profile with the paper-1 finite-contour encoder, flatten its directions/centers/gammas/readouts, and evaluate the actual tanh MLP. The compiler tries five (μ)'s from (10^{-26}) to (10^{-18}), choosing by agreement with the already computed polynomial. It does not need the exact PDE field to rank those discrepancies (the reporting normalization is common across candidates).

**What changes mathematically?** The right change of variables (z=R_bc) alone does not change the represented family or objective. Adding the Tikhonov penalty **does** change the compiler optimization: it permits a small real-domain approximation error to reduce complex growth. Adding20 profile degrees enlarges the candidate ridge-profile space beyond the base polynomial degree. At finite (μ) this is not a pure exact-nullspace gauge transformation, nor a mere diagonal rescaling of the original PDE operator. It changes the chosen approximate representation of the already solved field. The PDE solve itself is not regularized by this penalty.

**Verified numbers.** For (n=4,p=36), the polynomial error is (7.6949\times10^{-12}); the tube-penalty MLP reaches (7.7275\times10^{-12}), with discrepancy from polynomial (9.046\times10^{-13}), readout (\ell_1=595.9), (M=73), (B=29857). Larger (μ) lowers (\ell_1) to about36 while slightly increasing the discrepancy. For (n=6,p=44), polynomial error (7.2697\times10^{-11}), best compiler-discrepancy candidate MLP error (2.2196\times10^{-10}), (\ell_1=1194), (M=89), (B=43877). Neither is a (10^{-14}) PDE field result. The useful advance is recovering essentially the available polynomial quality instead of losing many orders in the compiler.

At (n=2), the current raw tube file has (\ell_1=16.25) for its best discrepancy, whereas an older README table says29; cite the raw variant rather than conflating runs.

**Cost:** (n=4) builds (A) with20736 rows and6789 columns, about1.13GB just for that float64 array, before QR/SVD work. The factor step takes75–96s in two recorded invocations. (n=6) uses28224×9701, about2.19GB, factor time245s. This is a dense prototype; it is not the scalable million-neuron compiler. The later Fourier-slice construction replaces this fitting stage on other problems.

The tube ladder checks actual tanh values (`r2core.eval` computes `P @ W1.T + b1`, tanh, final dot product). It does **not** repeat the earlier raw PDE/boundary audit for every regularized candidate. The best tube network claims are therefore field-comparison claims, not a full exported-neural PDE residual certificate.

## 6. The newer space-time engine and its inverse experiments

### 6.1 The engine is an ordinary classical spectral PDE solver

Source: `H/st_spectral.py:35–172`; general systems `st_general.py`.

For Burgers (u_t+c uu_x=\nu u_{xx}), or Allen–Cahn (u_t=\epsilon u_{xx}+5(u-u^3)), put a periodic Fourier grid in (x) and Chebyshev time nodes in each short slab. Unknown (U\in\mathbb R^{(N_t+1)\times N_x}) stores values at those nodes. Its initial row is fixed by the initial condition or preceding slab. The residual is

\[
R(U)=D_tU-\epsilon D_{xx}U-N(U,D_xU),
\]

excluding the fixed initial row. Derivatives in (x) are FFT multipliers; (D_t) is a small dense time differentiation matrix. The initial guess repeats the initial row through the slab. A Newton step solves the linearized residual by GMRES, then adds the correction. There is no network evaluation or QUILL map inside this solve.

The preconditioner drops variable nonlinear terms and uses (P_\sigma=\partial_t+\sigma-\epsilon\partial_{xx}). Fourier transforming in space reduces its inverse to independent (N_t\times N_t) matrices, one per Fourier mode. That is why it is cheap: spatial coupling has been diagonalized, leaving small temporal blocks. Source `:55–59` explicitly forms those small inverse blocks. Numerical solves are still performed, but no global dense space-time Jacobian is formed.

At (N_x=2048,N_t=16), one slab has32768 unknowns. Burgers to (t=1) at slab0.05 takes13.56s in saved `st_spectral_checks.json` and has relative error (1.686\times10^{-15}) versus Cole–Hopf, max absolute error (1.22\times10^{-14}), max31 GMRES iterations. The same file includes failures/limits:1024 spatial points leave (5.63\times10^{-9}), and raising (N_t) to24 gives (1.02\times10^{-14}), not monotonic improvement in finite precision. Most slabs hit the maximum12 Newton steps despite excellent field error; do not describe every stop as residual tolerance convergence.

Allen–Cahn needs tiny graded initial slabs because (u_0=x^2\cos(\pi x)) has mismatched periodic derivatives. `make_fields.py` uses `(1e-5,4.0)` initial grading, Fourier2048, slab0.05, time16. Saved `fields_info.json` checks the resulting field only on (t\in[0.05,1]), reporting (1.414\times10^{-12}) versus an exact-initial-Fourier-coefficient ETDRK4 reference. The output field stored to (t=4) was checked over that shorter positive-time interval, not uniformly at the initial corner. Ungraded versions in `st_spectral_checks.json` remain around (2.3\times10^{-5}).

The per-slab memory includes several (N_xN_t) arrays, the Fourier inverse blocks (O(N_xN_t^2)), and GMRES's retained Krylov vectors (restart200). Storing all slabs adds a time-history factor. For (s) spatial dimensions on a tensor Fourier grid, replace (N_x) with (n^s). This method avoids dense (B^2) neuron matrices, but its grid still has the usual dimensional curse. It is not general arbitrary-domain high-dimensional solving.

### 6.2 How the spectral field becomes one MLP

The solved `Field` evaluates by Fourier interpolation in space and barycentric interpolation within each time slab; it is not itself an MLP. `T/burgers_construction.py` and associated runners then extend this field beyond the target space-time rectangle, window it, compute Fourier/Radon profiles, split directions into coarse/global and fine/front bands, and apply QUILL. The final sum is a global one-hidden-layer network; the construction did use a sequence of solved time slabs internally.

Verified final Burgers record:1,271,906 neurons, sampled relative field error (6.3\times10^{-10}) versus independently evaluated Cole–Hopf, shock-layer max error (6.7\times10^{-9}). The prior spectral field's (10^{-15}) result is not the final network's error. The front fan/window choices were designed around this problem; they are not the output of a frozen general adaptation policy. Exact formulas are not used as field input in this final variant, but the compiler uses known shock structure. The companion report provides exact source/JSON pointers and distinguishes the earlier analytic-field variants.

Allen–Cahn6.42M-neuron construction reports (1.97\times10^{-4}); doubling the fine fan to11.86M barely changes it. Critically, that metric uses4000 seeded random points throughout the unmasked domain against the **same spectral field fed to the constructor** (`T/run_ladder.py:57–62`). It is an encoding discrepancy, not a published benchmark-grid error against an independent PDE solution. The handoff's “full-grid” wording is inaccurate. Its positive-time spectral solver can be very accurate while its global MLP compiler is not. A further proposed construction change was stopped, so no result exists for it.

### 6.3 Inverse Burgers: real parameter inference, no MLP used

`H/inverse_burgers.py:14–59`: unknowns ((c,\nu)), exact IC (-\sin\pi x), period2, 2000 random observations on (x\in[-1,1],t\in[0,1]), true ((1,0.01/\pi)), start ((0.5,0.01)). Observations come from the independent Cole–Hopf expression plus iid Gaussian noise with standard deviation1% of the clean sample standard deviation. This is stronger information than a PINN receiving only interior measurements if that PINN does not also receive the exact IC/boundary information; reported published comparisons need that protocol qualification.

For each parameter candidate, run the spectral PDE solver, then propagate sensitivities (s_k=\partial u/\partial\theta_k) using the already formed linearized slab operators. At observations assemble (J_{ik}=s_k(x_i,t_i)). SciPy Levenberg–Marquardt minimizes the data residual. This is standard PDE-constrained parameter estimation; the numerical forward solver is integral to it. The QUILL construction is absent.

Clean raw result: relative (c) error (5.55\times10^{-16}), relative viscosity error (5.04\times10^{-15}),10 function evaluations,194.9s optimization time. Five extra forward/tangent calls at the known truth check derivatives before that timer; the headline time omits this diagnostic overhead. The known truth in that sensitivity diagnostic does not seed the optimization, which has its own start.

Seven noisy results exist. Six viscosity relative errors range0.0355%–0.3124%; the seventh is1.1593%, about3.15 estimated standard deviations. Convection errors range0.0155%–0.0714%. The reported covariance (\hat\sigma^2(J^TJ)^{-1}) is a local linearized/Gaussian uncertainty approximation, not an exact finite-sample guarantee. Seven realizations are suggestive checks, not a proof that the estimator universally attains the information limit. Including the outlier matters.

`H/assimilate_burgers.py` instead estimates Fourier coefficients of the initial condition. In the clean test8 unknown coefficients from500 measurements at (t\ge0.2) recover to (2.36\times10^{-16}), field relative error (1.77\times10^{-15}). Noisy tests use32 coefficients and1000 observations; three field errors are0.96%,0.51%,2.67%, with early unobserved-time errors1.86%,0.92%,5.11%. The exact IC lies in the chosen finite Fourier family. This is a legitimate constrained inverse example, not arbitrary unknown-field recovery. Again, the network is not involved.

### 6.4 Other PDEs: distinguish field results from network results

- **KdV spectral engine:** saved `bench_kdv_Nx512.json` relative error (1.1745\times10^{-11}),75.45s, reference self-gap (5.09\times10^{-11}). This is spectral field output, max339 GMRES iterations in its harder steps; it does not support uniformly tiny iteration counts.
- **KdV network:** a separate runner `bench_kdv_net.py` uses an ETDRK4 Fourier time-stepper (not the Newton slab engine), also solving backward in time for extension. It constructs653360 neurons and obtains (1.2828\times10^{-11}) relative discrepancy against a finer ETDRK4 reference on512×201 points;660.5s total. This is explicitly numerical-solve-then-encode. Its `field_rel_l2=0.01159` diagnostic is wrong because line56 rounds non-grid-aligned space/time queries to construction indices; the independent network-vs-reference evaluation at lines51–57 uses the actual points and is not that same bug.
- **Kuramoto–Sivashinsky:** default512-space run is (1.02\times10^{-6}). After changing slab/time resolution,256-space runs reach (1.90\times10^{-9}) against a reference whose own refinement gap is (1.21\times10^{-8}). Report selected settings and reference limitation. No compiled MLP result was found for KS.
- **Complex Ginzburg–Landau2D+t:**128² spatial grid, relative error (1.80\times10^{-6}),44.18s, reference gap (4.50\times10^{-8}). This is a spectral field, not a tanh network. The larger run was not completed.

## 7. High dimension: genuine gains with important boundaries

### 7.1 Better sphere/ridge representation for known functions

`H/hd_ball.py:22–34,100–143` chooses quasi-uniform directions by repulsive Riesz-energy descent rather than an expensive tensor sphere rule. For each polynomial degree (n), ridge Gegenbauer functions have an explicit zonal Gram kernel (K_n(v,w)=\lambda_nC_n^{d/2}(v\cdot w)). A blocked conjugate-gradient solve finds profile coefficients degree by degree. The RHS is computed from a closed-form **ball-restricted Radon transform of Gaussian terms**. This shortcut is available because the function's mixture parameters are known; it is not a generic black-box analysis routine.

At6D,p14,M17000, a four-Gaussian known target is represented at relative error (1.41\times10^{-12}) before compilation. The final tanh sum (`hd_network.py:18–24`) has1,887,000 neurons and (1.3045\times10^{-12}) relative error at1000 held-out ball points, max (3.859\times10^{-12}), readout (\ell_1=1.327). With (N=90) cells the encoder produces111 centers including halo/endpoints. The shared encoder's matrix multiplication takes0.0057s, but this excludes direction generation, projection, and checking. Direction preparation took about16min in the recorded run; the projection around75s. The broad Gaussian widths and dimension-dependent center scaling make this a selected smooth target family, not arbitrary6D complexity.

A crucial distinction: the blocked implementation avoids keeping the full17000² Gram in memory, but still computes its all-pairs interactions every iteration. Time is (O(pM^2)) per coupled CG sweep, plus direction-generation (O(I_RdM^2)). Memory includes several block×M panels (block2048 gives34.8M doubles,278MB per panel), profile arrays, and output readouts. It is memory-bounded in the blocking sense, not inference-linear construction time.

### 7.2 Poisson on the ball is not the same artifact

`H/run_poisson_ball.py:25–49` first expands the source in ridge polynomials, double-integrates each profile to obtain a particular solution, samples the boundary defect, solves a sphere-analysis interpolation system by CG, and represents the harmonic correction in ridges. This is mathematically structured PDE solving with no dense volume matrix. It depends specifically on the ball's harmonic structure and known Gaussian-source Radon transform.

At6D,p14,M17000,24000 boundary analysis points, `poisson_ball_d6_p14_L14_M17000.json` reports:

- harmonic-polynomial test interior max error (2.953\times10^{-14});
- nonzero Gaussian-source case boundary max discrepancy (6.528\times10^{-14});
- that case interior PDE residual max (3.891\times10^{-11});
-1847s for the harmonic solve and2103s for the source solve,97/106 sphere CG iterations.

**This runner never encodes or evaluates tanh.** It evaluates Chebyshev ridge profiles (`PB.eval_profiles`) and saves `V,Pc`. The nonzero-source solution has no independent known field truth in this runner. “6D Poisson at (6.5\times10^{-14})” is therefore a boundary discrepancy for a ridge-polynomial PDE solution, not a final network field-error claim.

Sphere analysis caches a full matrix only below50M entries, otherwise streams512×MS blocks (`poisson_ball.py:91–120`). AtMS24000 it is quadratic-time blocked work. The17k-direction representation result and24k-boundary Poisson result must not be combined into one claimed performance number.

### 7.3 Nonlinear6D: blind polynomial PDE solve, then compile

Problem (-0.2\Delta u+u^3-u=g) with manufactured broad Gaussian-mixture field and Dirichlet data on the unit ball. Forcing and boundary data are generated from the field. The unknown interior coefficients begin at zero in the least-squares variant; no target coefficients or target sample rows enter the solve. This is legitimate manufactured verification.

`H/nl_ball.py` uses an explicit orthonormal ball basis (hyperspherical harmonics times radial Jacobi polynomials), a block-diagonal exact Laplacian, a tau boundary-row replacement, and Newton–GMRES. Product quadrature evaluates nonlinear products. At3D,p16,P969,Q11664 it gets (2.669\times10^{-13});4D,p16,P4845,Q209952 gets (4.340\times10^{-12});5D,p10,P3003,Q497664 gets (1.153\times10^{-7}). Quadrature is the dimensional bottleneck.

The6D version switches to least-squares collocation (`nl_ball_ls.py:15–92`). Interior radii follow a boundary-enriched beta distribution, with importance weights; random directions are used. This is an asymptotic Christoffel-inspired density, not literally evaluation of the exact finite-space Christoffel function. (P=8008,Q=40040,Q_b=15015). Weighted residual:

\[
R(a)=\begin{bmatrix}W^{1/2}[-0.2\Delta u_a+u_a^3-u_a-g]\\W_b^{1/2}[u_a-g_b]\end{bmatrix}.
\]

Newton's Jacobian acts by (-0.2\Delta\delta u+(3u_a^2-1)\delta u) plus boundary values. LSQR solves each rectangular linearized problem through forward/transpose products, right-preconditioned by small radial tau solves for (-0.2\Delta+0.5). A backtracking rule accepts reductions in weighted residual. The actual evaluated field throughout is the polynomial, not the tanh MLP. The PDE is checked only through the sampled polynomial system during solving.

Raw `nl_ls_d6_p10.json`: field relative error (1.0344\times10^{-6}), maximum (1.832\times10^{-6}), final sampled residual max (4.878\times10^{-5}). Point setup252.7s, solve438.2s. The first meaningful LSQR counts7,12,45,41 are followed by tiny ineffective corrections; it runs to the10-Newton cap rather than proving convergence to a certified best approximation. The error is plausibly dominated by low degree, but no theorem from this run proves that raising degree alone will succeed.

`ball_to_net.py` converts saved ball coefficients to ridge RHSs, solves per-degree zonal systems, and applies the shared encoder. At4500 directions and89 centers, the400500-neuron tanh sum has relative error (1.0612\times10^{-6}),21s compile. This confirms the solver's field quality survives encoding. The MLP was evaluated by explicit NumPy tanh contractions, not separately exported to Torch or audited with ordinary autodifferentiated PDE residuals.

**Actual solve cost:** `synth_at/adjoint_at` repeatedly form3000×P basis panels (`:20–29`). Memory avoids a completeQ×P array, but each operator call performs (O((Q+Q_b)P)) dense-basis work, effectively (O(P^2)) whenQ grows withP. This implementation does not meet the theory document's hoped-for (O(dpP)) transform cost. AtP8008 one panel alone is192MB. Higher-resolution6D predictions in the handoff were not run.

### 7.4 Very high ambient dimension exploits exact structure

`dim_ridge_torus.py`: the manufactured forcing is exactly five periodic ridge functions; each unknown integer direction has only3–5 nonzero entries in ±1,±2. The algorithm receives exact gradients and Hessian-vector products, recovers the active subspace by SVD of200 gradients, separates directions by simultaneous diagonalization of two projected Hessians, rounds the directions to the integer lattice, fits405 Fourier profile coefficients from2430 samples, divides by the constant-coefficient Fourier symbol, then encodes. It finds the correct directions and3992-neuron functions at2–5e-15 normalized max error even atd3000. That is an interesting structure-recovery demonstration. It is not a generic3000D PDE result. Sparse integer directions keep the profile interval widths fixed asd increases; dense arbitrary ridge directions would not share that exact count.

`dim_subspace3.py`: a forcing is exactly a rotated3D Gaussian-mixture problem in ambient10/100/1000 dimensions. Gradients identify a rank3 subspace; a140³ FFT grid solves the reduced problem;3D Fourier slice builds the network. Outputs around75–78k neurons have normalized max errors (4.6\times10^{-15},1.2\times10^{-13},3.7\times10^{-14}). **Hidden oracle use:** lines54–61 compare backprojection to `w_ex`, the exact solution, to decide direction refinement. The advertised black-box solver is not wholly oracle-free in that configuration. This does not mean its gradient recovery is fake; it means the angular-resolution selection used extra knowledge.

`dim_ball_simple.py` uses exact algebraic identities for constant/affine sources on a ball. Its direction count grows with dimension; raw1000D errors are approximately (2.9\times10^{-12}) for constant and (9.8\times10^{-11}) for affine sources. These do not show dimension-independent rounding accuracy.

## 8. General 2D domains: classical boundary solving, then excellent network copying

### 8.1 The solve

`H/geom2d.py:148–225` solves Poisson as (u=u_p+u_h).

- (u_p) is a tensor-Chebyshev particular solution, obtained by dense minimum-norm least squares of (\Delta u_p=f) at interior points, with no imposed box boundary condition.
- (u_h) is harmonic, represented by a Laplace double-layer boundary density plus one logarithmic term per hole. A Nyström boundary discretization leads to a dense LU factorization in boundary unknowns.
- A barycentric Cauchy formula evaluates the harmonic field accurately close to smooth walls. It must be applied to holomorphic boundary values, not directly to the density; a failed version made precisely that error.

These are conventional PDE numerical mechanisms with a useful implementation. They are equation-specific to Laplace/Poisson at this stage. Boundary unknowns are fewer than volume unknowns, but memory is (O(N_b^2)) and factorization (O(N_b^3)), not a universal linear-memory inverse.

### 8.2 The scalable compiler

`H/geom_construct.py:223–337` starts from an evaluable solved field, extends it smoothly across boundaries, windows it, and samples it on a2D rectangular grid. It computes Fourier samples on radial lines by NUFFT, multiplies by the ramp filter, and obtains entire exponential-sum ridge profiles. These complex-evaluable profiles are exactly what QUILL needs. Directions are increased until a40-point backprojection check against the input field passes; encoder cells are allocated by the directional bandwidth rule.

The final evaluator is a literal sum of tanh units with only linear input projections and output accumulation. It has one-hidden-layer algebra. Saving grouped `(direction,centers,gamma,readouts,bias)` arrays is sufficient to flatten it. However, no independent flat Torch export/forward check was found. Its derivative accuracy is generally not checked after compilation in these runners; matching values to a high-accuracy field does not establish equally accurate PDE derivatives automatically.

For smooth2D boundaries the extension uses complex analytic continuation of a harmonic function. Its depth depends on singularities outside the domain, curve geometry, holes, and windows. Thus geometry can make the compiler dramatically larger even when solving the interior field is easy. This is an important gain/cost distinction, not merely a numerical implementation detail.

### 8.3 Verified examples

**Annuli:** `geom_construct_annulus_a{0.5,0.35,0.25}_eps1e-12_erfc.json`, exact manufactured harmonic/log fields. Maximum errors roughly (7.1\times10^{-14},8.7\times10^{-14},1.0\times10^{-13});605k,1.14M,2.14M neurons. Smaller hole size forces steeper continuation/windows; observed bandwidth proportional to inverse radius and neuron count proportional to bandwidth squared. These are deliberately controlled geometric tests, not a frozen broad benchmark.

**Duct with three round rods:** `run_geom_duct.py` explicitly solves **two-dimensional scalar** (\Delta w=-1,w=0) on all walls. This models fully developed pressure-driven axial velocity in a duct; the advective term vanishes under that specialization. It is not an arbitrary3D Navier–Stokes computation. Main solve uses3000 boundary nodes; reference doubles them. Network2627526 neurons,752 directions,1218² construction grid. Relative-to-max-field discrepancy against refined BIE: interior (2.696\times10^{-13}), wall layer atdistance1e-3 (2.229\times10^{-12}), exact-wall sample discrepancy (1.452\times10^{-12}). Construction881.8s plus evaluation134.5s. No independent physical field truth; this is self-refinement evidence.

**PINNacle Poisson2d-C:** harmonic field on square with four holes; one on square, zero on circles. `pinn_poisson2dc.py` solves6784 boundary unknowns with panels refined toward corners, then uses odd reflection across square sides and Laurent continuation into holes. Exact odd reflection applies because the specific Dirichlet data are constant/compatible on perpendicular sides; it is not a generic corner solution.

The final907-direction,2895255-neuron network has sampled interior relativeL2 (2.822\times10^{-14}), maximum (5.840\times10^{-14}), and near-wall-layer maximum (7.784\times10^{-12}), against a finer version of the same BIE. Source uses6000 interior random points and layers at1e-2,1e-3,1e-6. These are not the downloaded PINNacle reference data/point set. Published PINN errors printed in the JSON therefore provide context, not a matched experiment. The code explicitly depends on the equation and boundary compatibility.

### 8.4 Bratu: nonlinear classical engine, including an undisclosed oracle setup

`H/bratu.py` solves (\Delta u+\lambda e^u=0) by a fixed-point map (F=L(-\lambda e^F)) on a periodic extension grid. Each (L) applies FFT Poisson to a source, restores its mean with a quadratic particular solution, enforces the physical boundary by the BIE, and refreshes the smooth extension. Newton–GMRES differentiates this map. Once residual falls below1e-6 it changes to an interior-only Jacobian action with barycentric evaluation, because analytically continuing roundoff-shaped Krylov vectors amplified errors catastrophically. That2-phase rule is a tuned implementation response to a real failure, not evidence for arbitrary-domain/nonlinearity universality.

On a unit disk, exact Liouville solutions check a branch fromlambda0.5 to1.99. Atlambda1 the source solver has normalized maxfield error (3.007\times10^{-15}), while its213707-neuron network has (8.594\times10^{-14}). Newton startszero and uses the PDE source, but **`run_bratu.py:9–12` first calls a Poisson solve with source `-lam*np.exp(u_exact(P))` to choose collar depth and grid spacing**. The exact answer influences setup/resolution. This is not a blind from-declaration demonstration. The handoff does not surface that dependency.

On the rod duct the same oracle is absent: `run_bratu_duct.py:8–9` uses the linear proxy source (-\lambda) to choose depth. Atlambda2, self-refinement comparison is (9.36\times10^{-13}) max and (1.53\times10^{-13}) interior. The3.115M-neuron network has (5.965\times10^{-13}) overall discrepancy and (1.877\times10^{-13}) interior against that finer solve. It is encouraging nonlinear geometry evidence with a self-reference, not independent certification at that many digits.

## 9. What the evidence does and does not support

| Result family | Actual solve state | Final tanh evaluated? | Best defensible evidence | Important limit |
|---|---|---|---|---|
| 1D native nonlinear probe | 481 readouts | Yes, explicit tanh sum | $3.5\times10^{-14}$ sampled field error | Special whole-line forcing, band approximation; no independent raw PDE audit |
| 3D whole-space screened Poisson | Directional readout FFTs | Yes, tanh values/Hessians | $1.9\times10^{-14}$ sampled normalized PDE residual | Analytic Gaussian profiles; shared-formula field reference |
| 2D variable coefficients | Actual neuron readouts | Yes, actual features | $10^{-15}$-scale field with dense preconditioner | Dense quadratic memory/cubic setup; oracle-derived count metric |
| Helmholtz compiler | Polynomial solve, then dense profile fit | Yes, flat NumPy MLP | $n=4$: $7.7\times10^{-12}$; $n=6$: $2.2\times10^{-10}$ field | Compiler regularization, dense arrays, selected penalty |
| Sharp Burgers spectral solve | Fourier/Chebyshev slab values | No at this stage | $1.7\times10^{-15}$ versus Cole–Hopf | Classical periodic 1D engine |
| Sharp Burgers network | Same field, then tailored Radon construction | Yes | 1.27M neurons, $6.3\times10^{-10}$ sampled field | Fan/window tailored; not the spectral solver's floor |
| Allen–Cahn network | Same field, then tailored construction | Yes | 6.42M neurons, $1.97\times10^{-4}$ sampled encoding error | Same input field is reference; not benchmark grid |
| Inverse/data assimilation | Physical/IC parameters plus spectral solves | No | Clean near floor; noisy errors near statistical scale | Exact known IC or finite IC family; no network inverse |
| 6D known Gaussian mixture | Degree-wise frame coefficients | Yes | 1.887M neurons, $1.30\times10^{-12}$ field | Analytic Radon right-hand side; quadratic direction work |
| 6D Poisson on ball | Ridge-polynomial profiles | No | Boundary $6.53\times10^{-14}$; residual $3.89\times10^{-11}$ | No source-case field truth |
| 6D nonlinear ball | 8,008 polynomial coefficients | Yes, after solve | 400,500 neurons, $1.06\times10^{-6}$ field | Quadratic-in-$P$ operator work; smooth manufactured example |
| Rod-duct Poisson | Boundary density + particular polynomial | Yes | 2.63M neurons; interior $2.7\times10^{-13}$ self-comparison | Scalar 2D Poisson; wall error $2.2\times10^{-12}$ |
| Bratu disk | Extended FFT grid + BIE | Yes | 213,707 neurons, $8.6\times10^{-14}$ field | Exact solution used to select collar/grid |
| Bratu duct | Extended FFT grid + BIE | Yes | 3.11M neurons, $6\times10^{-13}$ self-comparison | Laplace-based 2D engine; tuned two-phase Newton |
| 3000D sparse ridges | Five recovered profiles | Yes | 3,992 neurons, a few times $10^{-15}$ | Exactly five ridges, sparse integer structure, derivative access |

The table uses compact notation only to aid scanning; every metric above has its definition and source earlier. No claim is a general-purpose solver guarantee.

## 10. Scaling ledger from implemented code

Let (B) be neurons, (P) independent field coefficients, (Q) solve samples, (M) directions, (N) profile cells, (N_b) boundarynodes, (G=n^s) spatial gridpoints, (b_s) a fixed streaming panel size, and (I) iteration count.

| Implementation | Dominant stored objects | Main computation |
|---|---|---|
| Global actual-neuron least squares/dense preconditioner | Multiple $Q\times B$ arrays, singular vectors | $O(QB^2)$ factorization when $Q\ge B$; cubic when $Q\sim B$ |
| Uniform directional FFT inverse | $M\times N$ readouts/padding | $O(MN\log N)$, but only for decoupled constant-coefficient profiles |
| Helmholtz tube-gauge compiler | $Q_c\times K$ array, $K\times K$ QR factor | Dense QR/SVD; $K=M(D+1)$, not scalable to millions of independent coefficients |
| Ball frame projection | $b_s\times M$ kernel panels, $M\times p$ coefficients | $O(IpM^2)$; Riesz preparation $O(I_RdM^2)$ |
| Ball boundary analysis | Blocked $b_s\times M_S$, or cached $M_S^2$ if small | $O(ILM_S^2)$ plus projection to directions |
| Nonlinear ball least squares | $b_s\times P$ evaluation panels, $Q$-vectors, small radial blocks | $O(I(Q+Q_b)P)$ plus basis generation; $Q\sim P$ means quadratic time |
| Periodic space-time engine | $GN_t$ state, $GN_t^2$ inverse blocks, Krylov vectors | Per-action FFTs $O(N_tG\log G)$ plus time products $O(N_t^2G)$ |
| 2D boundary-integral engine | $N_b^2$ matrix/LU plus grid | Dense boundary factor $O(N_b^3)$; field evaluation $O(GN_b)$ |
| Grid Fourier-slice compiler | $G$ grid points, $M\times n_\omega$ complex spectra, output $B$ weights | NUFFT plus per-direction exponential-sum evaluation and encoding; **$G=n^d$** for an unstructured tensor grid |
| Ordinary network evaluation | Streaming activation panel and weights | $O(Q_{eval}dB)$ flat; sharing directions reduces repeated dot products to $Md$, with $B$ nonlinearities per point |

The million-neuron milestone is real for final construction/evaluation. It has not removed all quadratic unknown-count work, all dense boundary algebra, or tensor-grid growth. The best memory improvement is avoiding global neuron-space least squares where structure permits. The stronger advertised target “construction limited asymptotically only by inference” is demonstrated in specific analytic/translation-invariant settings, not all the newer PDE engines.

## 11. Claim hygiene and omitted failures

The handoff is substantially more candid than several live INBOX items. `CLAIMS.md` has no accepted Paper2 claims. All new workspace evidence is preliminary under the repo's own rules.

Do not omit:

- the failed fast structured inverses after the dense native disk success;
- the original global/tensor compile failure behind the Helmholtz fix;
- dense greedy/direct-neuron attempts on fronts, which reach only modest accuracy and run into matrix/sampling memory aroundthousands, notmillions, ofsteepneurons;
- larger Allen–Cahn backgrounds that entered bad Newton basins;
- the difference between the1e-15Burgers spectralfield and6e-10network;
- the unchanged Allen–Cahn error after roughlydoublingitsfan;
- reference self-gaps onKS/KdV and exclusion of the Allen–Cahn initialcorner in the highaccuracyfieldcheck;
- exactfield-assisted Bratu setup and active-subspace angularcheck;
- degree/direction/step choices changed after failures, notonefrozenconfiguration;
- no new general3DNavier–Stokes experiment in the continuation audited here;
- no matched-cost comparison against a tuned classical solver, and no identical-data/protocol rerun of the published PINN baseline figures printed inJSON.

These qualifications do not erase the gains. They locate them: better operator coordinates can make solving much easier; smoother, better-scaled ridge decompositions can preserve digits during compilation; existing structured PDE machinery can produce accurate input fields; the Radon–QUILL map can turn some of those fields into surprisingly large and accurate shallow networks. What is absent is evidence that those ingredients have already been assembled into one routine that handles arbitrary equations, boundaries, noisy inverse data, and high dimensionality under the original scaling goals.

## 12. Audit coverage and limits

This audit traces the materially different October 2–3 mechanisms in the structured-solver, hard-cases, and theory-leads workspaces. It checks source and saved records for the headline examples above, including the nonlinear native probe, dense-preconditioned native disk solve, tube-gauge compiler, time-slab solver, inverse/assimilation, high-dimensional ball and structured examples, general-domain Poisson, and Bratu. It does not claim to inventory every parameter sweep or abandoned script. The companion report audits known-function scaling and construction allocation separately, including omitted misses and changing halo constants.

Nothing was rerun. Saved JSON and source establish what the recorded implementation reports, not independent numerical reproduction. Current source can differ from the version that originally produced an artifact; explicit mismatches found here are reported, but there is no immutable per-result source hash for every run. Most final networks are verified by literal NumPy tanh evaluation, which establishes their one-hidden-layer algebra; a separate ordinary-library export and autodifferentiated PDE audit is absent in the newer construction records inspected. Those limits apply to both positive results and failures.
