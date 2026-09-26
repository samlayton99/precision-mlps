# Frozen geometry: capacity and access — review and handoff

Status: both PDFs saved and read; independent PR implementation review complete. No new training launched. Sam requested storage, comparison, interpretation of Junmiao's message, and identification of remaining work/confounds. The delegated findings and exact code reuse points are saved in [pr_conditioning_review.md](/Users/sam/my-repos/research/collaborations/precisionMLPs/docs/frozen_geometry_capacity_access/pr_conditioning_review.md).

## Sources and revision relationship

- [First theorem](/Users/sam/my-repos/research/collaborations/precisionMLPs/papers/optimization_notes/bounded_slope_weak_access_theorem.pdf): *Bounded slopes and weak access to function corrections*, four pages.
- [Revised note](/Users/sam/my-repos/research/collaborations/precisionMLPs/papers/optimization_notes/frozen_geometry_capacity_access_note.pdf): *Capacity versus accessibility in frozen tanh networks*, five pages. The later conversation identifies this as the note with the intended experiments.
- [Merged PR #2](https://github.com/samlayton99/precision-mlps/pull/2), reviewed source head `b71a03396990900e94e3853b65dc36c506761bcb`, merge commit `d7d26e66d96141fca578cf22c77ce752c404a1f6`. It has already been merged and pulled locally. The relevant new experiment code is `experiments/expD06_fixed_center_scales/`.
- Sam's pasted conversation is the source for Junmiao's intended division of work and hardware comments. No Slack messages were fetched or sent.

## Same mechanism, stronger second formulation

Both documents concern the readout with hidden geometry fixed. If every hidden slope is bounded by Gamma, each tanh feature is close to a low-degree polynomial. A unit correction orthogonal to all sampled polynomials up to degree k therefore has small inner products with the features. That is weak readout access: a small coefficient step cannot make much progress in that direction.

This is not the earlier Fourier scale-tangent theorem. Its derivative is with respect to readout coefficients, its projection is onto sampled polynomials, and its domain is the actual finite sample grid. It does not establish that joint training keeps gamma bounded.

The first note bounds the entire polynomial-orthogonal operator by sqrt(W) E_k(Gamma). It gives frozen-readout flow/GD lower bounds when the initial residual is completely polynomial-orthogonal. The residual need not remain orthogonal later.

The revised note adds:

1. **Explicit capacity versus access.** An invertible readout map c=M theta preserves the exact feature span, but changes the optimization matrix from A to B=AM. For a unit probe q, squared access mu(q)=||B^T q||² factors as its squared overlap with the feature span times the curvature of its normalized attainable component. Low access can therefore mean missing capacity, weak positive curvature, or both.
2. **Necessary target tails.** The flow lower bound uses the amount of the initial residual beyond each polynomial cutoff that must actually be removed to attain tolerance epsilon, divided by its squared access. This handles general initial residuals; exact low-degree removal is no longer needed to obtain a nontrivial certificate.
3. **Sharper structural bounds.** The pole-based exponent is arsinh(pi/(2 Gamma)). The first note uses a freely chosen strip parameter below pi/2. The revised exponent and worst-case width dependence are sharp in the stated sense; uniform prefactors and saturation by our particular dictionaries/targets are not asserted optimal.
4. **Coefficient-budget capacity.** Accurate representation may exist while requiring a large coefficient norm. Equal native-coordinate budgets under different M are not equal physical coefficient budgets.
5. **Separate flow and discrete-GD statements.** Exact discrete spectral curves and bounds are supplied; gradient-flow time cannot simply be read as a number of GD updates. Adam remains an empirical comparison.
6. **One-step damped-GN measurement.** Sweep fixed damping from the same residual and examine which corrections become recoverable. The zero-damping limit diagnoses missing span; the approach to it diagnoses weak positive curvature.

A simple capacity example is the dictionary p and p+epsilon*q. Both columns are almost p, yet they span q exactly. Producing q requires subtracting the columns with coefficients of size 1/epsilon. Near-polynomial columns need not imply a polynomial exact span.

## Important content of Junmiao's conversation

- Intended theorem sequence: bounded gamma limits readout access; a further theorem must explain why gamma dynamics remain bounded; another part addresses remaining readout conditioning. These are separate claims.
- Desired empirical question: how much finite-budget optimization error is attributable to gamma after controlling other conditioning effects. Attributing the entire gap down to 1e-14 is explicitly not the objective.
- Primary comparison: actual independent-grid relative L2 error for ordinary GD/Adam across frozen gammas. Numerical least-squares fits measure available approximation and are supporting diagnostics.
- Reuse the merged PR's neighboring representation, characteristic readout scales, and parameter-coordinate maps. Do not equate a coordinate reparameterization with a changed physical initialization.
- Junmiao offered H200 access and described long-update throughput. That is his report, not a benchmark or verified access in this task. Runpod access remains to be confirmed before planning wall-clock throughput around it.
- Sam volunteered to handle experiments that evening and reported substantial availability before the Friday deadline. This is context from the pasted exchange, not a newly created calendar commitment. No exact calendar date for that deadline is inferred.
- The later five-page note is the proposed protocol to use when planning the new experiment. It does not establish that its experiment results already exist in PR #2.

## Confounds to separate

| Confound | Why it matters | Control in the revised protocol |
|---|---|---|
| Exact span versus slow access inside it | A small gradient can reflect an impossible correction, an accessible but weak direction, or both | Span-error decomposition, full-target and projected-target diagnostics, coefficient-budget fits, damping limit |
| Raw cumulative tanh coordinates | Strongly overlapping step-like columns can make readout optimization slow independently of the proposed bounded-slope rate | Invertible neighboring differences with the final anchor preserved; compare matched physical starts |
| Coordinate metric and global scale | c=M theta changes physical GD to a preconditioned update and changes damping/step norms | Measure B=AM, curvature-normalized access, eta proportional to 1/||B||², relative damping; keep M fixed across gamma |
| Initialization | A smaller physical random readout changes the initial residual, independently of optimization coordinates | Zero, sqrt(alpha)-Xavier, and alpha-Xavier as separate physical-start ablations; re-encode the same draw across maps |
| Bias and halo scaling | Unobserved or weakly observed features and a differently scaled constant column can dominate numerical conditioning | Preserve full span and anchor; selected unscaled-bias/ordinary-halo controls |
| Unnecessary target tails | A tiny response to a tail below the requested tolerance does not prevent attaining that tolerance | Measure D_k or initial-residual tail, use [delta_k-epsilon]_+, maximize over relevant degrees; quadratic control |
| Short horizon or optimizer tuning | A fixed short run or a poor common rate can exaggerate an apparent gamma effect | Fixed update budgets, late-window statistics, normalized stable GD rate, equal-budget Adam calibration |
| Numerical rank and sampled error | Tiny positive singular modes can be discarded or corrupted; a small training error can miss between-sample errors | Factor B rather than its rounded Gram matrix; independent evaluation grid; selective 80/120-digit reevaluation; cutoff variation |
| Geometry/readout coupling | Joint GD changes the object whose access is being compared | Freeze all hidden slopes and centers for this theorem audit |

Polynomial degree and Fourier frequency are distinct diagnostics. The new finite-grid polynomial projection avoids importing the whole-line Fourier assumptions into a finite-window training objective.

## Where the existing work leaves us

The session already has frozen-readout spectral predictions (D26/D30), corrected geometry/readout comparisons (D24), and current-readout gamma-gradient decomposition (D35). These supply useful methods and baseline observations. D35 concerns the geometry gradient under joint training; it does not measure the polynomial-tail readout access in these new notes.

The merged PR contributes conditioning maps, matched-physical-initialization machinery, numerical diagnostics, and longer optimizer runs. Its existing joint or handoff results are not the proposed matched frozen-gamma sweep. Reuse components, not their curves as if they answer the new question.

The independent implementation review confirms that the PR's individual-coordinate comparison preserved the collective physical start. The new alpha-Xavier physical start is a distinct ablation. It also identifies concrete reuse traps: the frozen-readout GD entry point uses Armijo and fixes gamma to .25/h; the joint loop updates slopes; one mapped-feature helper silently routes unsupported map labels into its neighboring branch. The proposed constant-rate sweep therefore requires a small new runner, even though the coordinate transforms and damped QR machinery already exist.

The next scientific deliverable is a controlled frozen-geometry gamma audit showing (a) actual trained error at a fixed budget, (b) access to necessary target corrections, and (c) which part survives readout-coordinate controls. A new general optimizer is not needed to ask that question.

## Proposed next work, not launched

1. Reuse the PR's model/map/initialization code. Verify evaluation identities and matched physical initial states, with geometry fully frozen. Begin with zero readout to remove initialization variation, then add the two physical Gaussian starts as separate controls.
2. Plot actual independent-grid relative L2 against gamma at equal GD/Adam update budgets, with coordinate maps separated. Also retain error versus updates so crossings and late drift remain visible. Use least-squares results as secondary capacity diagnostics.
3. On the same matrices and samples, build an orthonormal polynomial basis and measure necessary tail sizes, direct squared access, normalized access, and structural bounds. Compare exact discrete-GD predictions and the relevant time/step certificates with the runs.
4. Run the proposed **single-step** damped-readout-GN sweep on the same probes across gamma/maps. It is not the PR's joint nonlinear-GN training loop. Plot fraction remaining against relative damping, with span limits and step norms.
5. Audit difficult numerical cells and then test a second width. A diminishing gamma effect after conditioning controls is an informative result; the proposed explanation must account for the measured effect rather than be assumed dominant.

The revised note proposes N=512, gamma in {1,2,4,8,16,32,50,64}, 16N+1 endpoint-inclusive training points, 32,768 midpoint evaluation points, and budgets 10k/50k/200k. Its targets are sqrt(2) sin(2pi x), sqrt(5) x², [sin(2pi x)+0.1 sin(20pi x)]/sqrt(0.505), plus Runge. This mixed sine differs from our earlier three-carrier target. Its halo is ceil(sqrt(N))=23 at N512, whereas Sam's prior session default is 24 per side; that choice must be stated before claiming an exact PR-matched reproduction. Gamma64, not gamma16, corresponds to lambda=.25 at N512.

The full proposed maps-by-starts-by-seeds-by-optimizers grid is substantial. Stage it around one controlled change at a time and available compute; do not silently replace the specified targets, halo, or metric while claiming to reproduce the full protocol.
