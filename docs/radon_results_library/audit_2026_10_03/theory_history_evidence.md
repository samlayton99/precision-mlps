# Theory-lane and permission evidence audit, October 3, 2026

Read-only source/result audit for the coordinator. No numerical experiments, reruns, solver modifications, or external messages were performed. The only new file is this report. The sibling repository's agent reports are evidence, not accepted claims. Paths below prefixed `quill/` mean `/Users/sam/my-repos/research/collaborations/quill-radon/`; `theory/` means its `agent/workspace/2026-10-02_pde_theory_leads/`; `handoff/` means its `agent/workspace/2026-10-03_handoff_record/`.

## 1. What Sam actually authorized later

Direct user transcript evidence exists. These are not peer-agent paraphrases or compacted summaries. Source is `/Users/sam/.claude/projects/-Users-sam-my-repos-research-collaborations-precisionMLPs/3bf10edb-b6ec-4265-ada4-d7d58e892472.jsonl`.

- **Line 2008, timestamp `2026-10-03T06:19:15.618Z` (October 2, 11:19 p.m. PDT):** “THE THING THAT MATTERS IS DELIVERING TO THE PINNS COMMUNITY A PINN THAT ACTUALLY WORKS. If we are constructing it from a polynomial method, but it offers everything else, that is fine.” The same user message asks whether depending on a polynomial spectral solver prevents inverse problems and learning from measurements, retains a one-hidden-layer MLP as “non negotiable,” and requests comparison against spectral methods.
- **Line 2024, `2026-10-03T06:24:21.571Z`:** “so lets push with 2, but have the theory agent think about the structure”; “if a representation is possible, then it should be able to be found through our method.” This confirms the polynomial route as a current base while asking for structural advances.
- **Line 4090, `2026-10-03T15:18:52.410Z`:** Sam rejects being made to process inbox items and says millions of neurons are the point; a full dense solve that limits the method is unacceptable.
- **Line 4129, `2026-10-03T15:25:05.384Z`:** “Like dense solves are fine, it just can’t limit our scaling, whatever method we have should not be capped by this. Use common sense here”.
- **Line 5119, `2026-10-03T20:43:40.791Z`:** requests complicated geometry and says it must be robust, not hacky.
- **Line 5443, `2026-10-03T22:57:20.200Z`:** “As long as we have clear scaling laws, no need to do the longest ones”; asks to fix Allen–Cahn and do geometry before the benchmark; explicitly worries the polynomial setup limits dimension and requests alternatives for high-dimensional constant/affine problems.

Therefore `quill/human/REQUIREMENTS.md`'s earlier absolute “No dressed-up solver” wording cannot be treated as the last word about a polynomial core. It predates a direct relaxation. The handoff's statement that a polynomial core was allowed is verified. I found no comparably explicit user message authorizing boundary-integral and time-stepping replacements by name; the handoff admits extending the polynomial allowance to these without directly asking (`handoff/00_README.md:25`, `01_overview_and_ideas.md:159`). Do not automatically call these cheating: explain what changed and how far the demonstrated utility extends. The relevant scientific question is whether those alternatives deliver the requested capabilities and scaling, not the optimizer's name.

The theory session received the polynomial decision as a peer message at its JSONL line 977, not directly from Sam. The underlying direct message above resolves its provenance. Transcript times are UTC; handoff “Oct 3” labels often refer to UTC rather than the user's local date.

## 2. Three distinct objects that the theory documents blur

1. A **known-function compiler** takes a supplied field or its directional profiles and constructs tanh weights. Its tests can use an exact analytic solution without being a PDE solve.
2. A **PDE solver in polynomial/profile coordinates** discovers a solution from forcing and boundary/initial data. Its state may be polynomial or Fourier coefficients; a final compiler turns the answer into a tanh network.
3. A **native neural-residual solver** uses the actual tanh network in its residual and acceptance tests throughout the solve. The original Codex native route 2 had this distinction. Do not assume every later item called “route 2” preserves it.

The theory's main gain is that it identifies structured coordinates and constant-coefficient inverses. It does not turn the unknown PDE solution into a known profile for free. In much of the later work a classical solver supplies that missing profile.

## 3. The useful mathematical mechanism: solve an easy operator exactly, correct the remainder

For a unit direction vector `v`, a ridge `q(v·x)` satisfies `Delta[q(v·x)] = q''(v·x)`. If forcing is already decomposed into ridge profiles, a particular Poisson solution follows by integrating each profile twice. This only solves the interior equation. On a disk, evaluate the particular solution's boundary mismatch, expand it in angular Fourier modes, and add each mode's harmonic extension `r^|n| exp(i n theta)`. The extra term solves the homogeneous interior equation and corrects the boundary.

For a variable coefficient, the constant-coefficient inverse becomes a preconditioner: apply it to a residual, evaluate the true variable-coefficient operator on the proposed correction, and use Krylov combinations to reduce the remaining mismatch. This is a legitimate numerical method using classical elliptic structure. It is not a formula eliminating all numerical solution work for arbitrary PDEs.

**Literal implementation check:** `theory/probe_radon_exact_2d.py:3` explicitly says the state is analytic polynomial profiles, never the tanh network. Lines 65–75 build dense basis evaluations and Gram matrices and invoke `np.linalg.solve` for projections. Lines 77–88 build a dense harmonic-extension map. Lines 124–125 form a dense operator for eigenvalue diagnostics. Thus “zero linear solves” is a description of the ideal Poisson inversion formula after analytic transforms are available, not a literal description of this probe's implementation. This script has no neural export.

Saved data:

| File under `theory/results/` | Size | Poisson one-shot relative error | Variable-coefficient relative error | Recorded GMRES callbacks |
|---|---:|---:|---:|---:|
| `radonexact_p31_a0.0.json` | `p=31`, `P=528` | `5.654118047694452e-15` | `6.034240479503218e-15` | 3 |
| `radonexact_p31_a0.9.json` | `p=31`, `P=528`, contrast 19 | same | `5.232186997926986e-15` | 69 |

These errors use the polynomial quadrature points also used in the method, not a separate held-out neural audit (`probe_radon_exact_2d.py:99–122`). The direct forcing and boundary are derived from the manufactured analytic reference; this is ordinary manufactured-solution verification, but the resulting accuracy is a polynomial discretization result. The handoff sometimes reports iteration counts only until the history first crosses `1e-13`; the stored total callback count is larger. State which count is quoted.

## 4. Profile gauge: why good polynomial answers became bad networks

Different directional profiles can sum to the same function on the domain. Some choices cancel to a small field on the real domain while becoming very large just outside it or at the complex arguments used by QUILL's encoder. The encoder then produces huge weights; ordinary floating-point summation loses digits. This is a concrete explanation for a spectral solution being accurate while its compiled network is inaccurate.

The useful diagnostic in `theory/structure_theory.md:31–39` is the location where profiles must stay moderate: not only real sample points, and not an unnecessarily large complex ellipse, but the real halo plus the complex sample region actually used by the encoder. The proposed repair adds a penalty to profile selection:

`min_c ||A c - u_p||^2 + mu ||Q c||^2`.

Here `c` is the vector of profile expansion coefficients, `u_p` the already solved polynomial field sampled on the fitting domain, `A c` the sum of those profiles at those samples, and `Q c` their values on the encoder's complex sample region. This is a numerical regularized least-squares fit selecting a better representation of the already computed field. It is not just a change of coordinates: nonzero `mu` trades exact field matching for smaller profile growth.

The valid elementary bound is conditional. If an available coefficient vector `c*` obeys `||A c* - u_p|| <= eps` and `||Q c*|| <= K`, then taking `mu=eps^2/K^2` ensures the minimizer has objective at most `2 eps^2`; hence its fitting error is at most `sqrt(2) eps` and its `Q` norm at most `sqrt(2) K`. That is ordinary Tikhonov comparison with a feasible competitor. It does not prove the competitor exists in the chosen finite dictionary, that its `K` is known, that a finite sampled `Q` norm controls the entire complex region, or that conversion remains uniformly accurate as degree grows. The handoff's phrase “compile theorem ... uniformly in degree” skips these substantive hypotheses.

Related overreach: `structure_theory.md:35` says polynomial profiles are wrong “at any N,” but its own mechanism is growth at a halo/complex distance shrinking with N. The example demonstrates severe growth for the tested allocation, not an impossibility for every N. The cited equispaced-sampling impossibility theorem is not automatically a theorem about this encoder, which has analytic complex-value access. Pinkus-type profile nonuniqueness also must not be confused with an exact null space in a finite tanh dictionary: tiny singular values of a discretized neural matrix are numerical near-dependencies; the polynomial dimension count is a different exact statement.

## 5. Burgers and Allen–Cahn million-neuron constructions: exactly what went in

The implementation is `theory/burgers_construction.py`, selected by `run_ladder.py`.

1. Supply a field. Earlier Burgers counts call the known Cole–Hopf solution directly (`burgers_construction.py:33–54`, 180–182). Later counts load the other agent's already solved space-time spectral field (`run_ladder.py:45`; `burgers_construction.py:183–191`; `spectral_field.py:13–59`).
2. Continue below initial time with a finite Taylor expansion whose derivatives are derived recursively from the PDE and initial data (`burgers_construction.py:108–154`). Positive times still come from the analytic oracle or stored numerical field. This extension is finite-order and only validated locally; calling it a globally exact analytic continuation is too strong.
3. Multiply by windows that are almost one in the requested domain and decay outside. This creates an integrable whole-space field suitable for the Radon transform. Edges, widths, cutoffs, shock/interface locations, integration panels and spectral caps are selected per problem (`burgers_construction.py:57–106`). Burgers has a specified formation point; Allen–Cahn supplies fronts at `x=-1,-0.5,0.5,1`.
4. For each direction, numerically integrate along chords to obtain a Radon profile, transform along offsets, and multiply its spectrum by `|omega|/2` (the 2D filter). Split into a smooth background and higher-frequency part. Encode the background on a broad interval; encode fine content on shorter offset intervals where it is significant. Every piece is still a ridge, so summing them is algebraically one hidden layer.
5. Use a broad angular quadrature for smooth content and denser direction panels for sharp content. Cache per-direction weights. Evaluate actual tanh sums in blocks (`burgers_construction.py:404–416`). This avoids a least-squares solve over millions of neurons. It does not avoid the previous PDE solve or the cost of obtaining the Radon profiles by chord integration.

**What “no fit” means here:** no coefficient regression is needed after the field and Radon profiles are supplied. It does not mean equation/BC/IC are directly converted to the solution without a numerical PDE solver. The spectral field files are the answer being encoded.

**Manual adaptation:** `configure` in `burgers_construction.py:93–106` has explicit equation-specific front lists and windows. `precision` at lines 77–91 has separate hand-set cutoffs and resolutions for requested tolerances. `run_ladder.py` offers per-panel fan counts. The handoff accurately admits these were chosen by the agents, not by a frozen general automatic policy.

## 6. Audited million-neuron outcomes and the metric correction

| Saved result under `theory/results/` | Neurons | Reported relative L2 | Maximum sampled error, normalized by max reference | Source used to construct |
|---|---:|---:|---:|---|
| `count_burgers_spectral_p9_eps1e-09_M684.json` | 1,271,906 | `6.290342547453687e-10` | `6.664872497313701e-9` | stored spectral PDE solution |
| `count_allen_cahn_spectral_eps1e-06_M562.json` | 6,423,858 | `1.967777463092988e-4` | `7.646365288199699e-4` | stored spectral PDE solution |
| `count_allen_cahn_spectral_eps1e-06_M1122.json` | 11,859,648 | `1.963739579772563e-4` | `7.644459118613106e-4` | same stored solution |

**The handoff's “full-grid” wording is inaccurate.** `run_ladder.py:57–62` uses 4,000 seed-0 uniformly random interior points for these main metrics. A separate 39 by 25 plot grid covers `x in [-.95,.95]`, `t in [.02,.98]`, omitting exact boundaries and initial time. There is no masking in the final main metrics, which is useful, but “unmasked domain samples” is not the same as a published benchmark grid. Burgers truth is independent Cole–Hopf; Allen–Cahn truth is `FastField(path)`, the same stored solution that is encoded. Its number is primarily a representation/encoding error relative to that field, with the reference's independent accuracy needing separate evidence.

The stored `seconds` values are about 2,613 / 4,895 / 5,684 seconds. They begin after the fixed coarse sets are built (`run_ladder.py:77–81`), may reuse cached directions, and exclude the original PDE solve. They cannot be presented as end-to-end cold solve times. Likewise `M_fan` in the JSON is a rung multiplier (`0.5` or `1.0`); the actual count is `M_fan_total` (684/562/1122).

Doubling the Allen–Cahn fine-direction set leaves the error essentially unchanged. That demonstrates this refinement no longer helps. The proposed explanation is underresolved coarse angular content, based on an aliasing estimate. The proposed doubled-coarse-set repair was stopped and is untested; an observed plateau does not by itself prove the diagnosis.

## 7. The angular count idea: useful, but not a universal certificate

A Fourier component of frequency `omega` changes its phase across spatial extent `D` when the direction changes. Requiring its angular phase change not to be too large leads to the practical spacing rule `Delta phi <= 2 pi/(omega D)`. `D` must account for where features sit relative to the evaluation region, including artificial window features. This explains why adding more centers on each ridge could fail while adding directions helped, and why high-frequency and low-frequency parts can need different direction sets.

The handoff calls the displacement/alias formula exact for uniform angular grids. It is best presented as a local phase/Nyquist estimate plus measured support, not a complete finite-domain approximation theorem with its constants established. The angular dependence `omega v(phi)·x` is nonlinear in angle; the elementary linearized displacement argument alone does not certify all targets or nonuniform quadratures.

The theory revised its Burgers estimate from fewer than 2,000 neurons to hundreds of thousands after recognizing that the shock-formation region spreads directional content. `handoff/05b_failures_theory.md:11–35` records this and many implementation corrections. The later measured ladder is useful evidence of a particular scaling mechanism; it is not validation of every earlier “law.”

## 8. Known-function scaling results, including misses omitted by the short handoff

`theory/scaling_campaign.py:38–115` supplies closed-form target families and analytic directional-profile access: radial Gaussians, Poisson-kernel-type analytic fields, Bessel-harmonic fields, and finite anisotropic Gaussian mixtures. This is approximation, not a blind PDE test. Most families reuse one scalar encoded profile across all directions (`:155–160`), whereas the mixture has direction-specific profiles. Costs cannot be generalized from radial profile reuse to arbitrary fields.

At requested `eps_target=1e-14`, the saved 4D data include:

| Target | N | Directions M | Neurons B | Relative L2 | Normalized max error |
|---|---:|---:|---:|---:|---:|
| Gaussian sigma .6 | 85 | 12,167 | 1,289,702 | `4.72e-13` | `2.50e-13` |
| Gaussian sigma .35 | 88 | 29,791 | 3,247,219 | `1.10e-13` | `3.33e-14` |
| Bessel k=8, angular degree 0 | 85 | 4,913 | 520,778 | `1.59e-11` | `1.85e-12` |
| Bessel k=8, angular degree 4 | 85 | 6,859 | 727,054 | `2.99e-14` | `1.14e-14` |
| Poisson-kernel a=.5 | 85 | 21,952 | 2,326,912 | `4.45e-14` | `3.39e-14` |
| Poisson-kernel a=1 | 151 | 85,184 | 15,162,752 | `7.17e-15` | `6.51e-15` |

Source: the correspondingly named `theory/results/scaling_*_d4_eps1e-14.json` files. The tightest requested error is not reached uniformly. The library has an actual 15-million-neuron known-function construction, but that does not imply a blind 15-million-unknown PDE solve.

The handoff conflates two allocation versions. The current `scaling_campaign.py:119–140` uses the requested `eps` in the bandwidth expression and `HALO_RATE=.4`; its own `scaling_law.md:127` calibrates a revised `.33` constant after misses. The Burgers construction later uses the stricter fixed encoder tolerance. Therefore the final handoff's single supposedly frozen allocation rule should not be retroactively attributed to every campaign result.

The familiar `P=binomial(p+d,d)` counts polynomial coefficients of total degree at most `p`; it is a property of the space, not a special discovery of this method. The heuristic `B approximately 7dP` combines a near-minimal direction count with an empirical number of centers per profile; its constant depends on frame quality, halo and polynomial compile stability. It is not a universal optimality theorem. The dominant `log(1/eps)^d` analytic-class pattern is reasonable with controlled analytic norms, but a smooth/analytic label alone does not bound the constants needed for a real target.

## 9. Scaling caveats that matter to Sam's central concern

- `theory/scaling_law.md:25–46` is an aspirational cost ledger. It explicitly states the multidimensional fast polar-transform rows were design, not measurements. One must not report its `O(d p P)` fast surrogate step as the actual cost of every subsequent implementation.
- `theory/engine_theory.md:52` describes an `M by M` direction Gram matrix shared across degrees. At `M=17,000` that would be 289 million doubles, roughly 2.15 GiB before temporaries. **The implementation is better than that prose:** the parent auditor traced `hd_ball.py:100–134` to blocked all-pairs products, with a `2048 by M` block (~278 MB at M=17k), not a stored full Gram. The all-pairs arithmetic still scales quadratically in M; `poisson_ball.py:91–120` caches a full matrix only below 50 million elements and uses blocks for larger sets. A matrix whose size is directions rather than neurons can still be a serious time and memory bottleneck because directions grow with dimension and degree. These implementation details are the parent's independent report, not a rerun by this helper.
- The claim that no high-strength spherical designs “exist” is too strong. The operational issue is availability/construction of useful designs at the required size; that wording is not a nonexistence theorem.
- The few-ridge and low-active-dimension examples reduce intrinsic dimension by construction. Their constant neuron counts as ambient dimension grows are interesting structured results, not evidence about arbitrary genuinely high-dimensional data. Input-weight storage and dot products still depend on ambient dimension unless special factoring is retained.
- Millions of neurons with a small polynomial/profile state are different from millions of independently solved unknowns. This distinction is central to reporting the gains accurately.

## 10. What is solid theory versus conditional or unfinished theory

Solid classical ingredients: ridge differentiation/intertwining, harmonic boundary correction on a disk/ball, spectral equivalence for appropriate coercive variable-coefficient elliptic problems, local Newton convergence under invertibility and basin hypotheses, and the elementary Tikhonov comparison bound above. These legitimately explain parts of the gains.

`structure_theory.md:13–15` labels a “findability theorem” but assumes a stable polynomial discretization, an effective inverse, and that iteration starts inside Newton's convergence basin. `:112` explicitly says the full theorem and collocation quasi-optimality step remain to be written. It therefore does not prove Sam's desired implication “if a network representation exists, this solver will find it.” It assumes away significant parts of that question.

The Volterra argument in `engine_theory.md:28–30` gives factorial bounds for exact causal inversion followed by a bounded zeroth-order perturbation. It can explain why some parabolic corrections need few iterations independent of spatial resolution. It still depends on the time horizon, operator amplitudes, norm, requested tolerance, and faithful discretization. In that source, `epsilon` sometimes means Allen–Cahn diffusion, not accuracy; do not misread “independent of epsilon.” Burgers also has a first-order term whose kernel is singular. The original factorial argument for it was explicitly corrected; the replacement estimate based on `||u_x|| T` is a useful observed guide but not a proof that the full variable-transport Newton operator always has that bound.

Likewise “Gauss–Newton attains the Cramér–Rao bound” is an asymptotic/local statistical statement requiring identifiability, a correct noise model and sufficiently accurate optimization/forward model. It is not an unconditional finite-data guarantee. Six noisy trials are a small empirical check. Parent is auditing the saved inverse experiment directly.

For the final explanation, the defensible story is: there are useful analytical transforms and conditioning repairs; some supply classical PDE solvers with good coordinates and preconditioners, and some compile an already solved field into a one-layer network. These stages have distinct error and scaling limits. The current evidence does not collapse them into one automatic, generally predictive, million-unknown PDE solver.
