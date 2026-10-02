# Single-gamma geometry theory — Codex, 2026-09-29

Sam's question: predict the solved readout as one center's gamma changes, with an infinite fixed-spacing lattice as the idealization; inspect saved Adam and floor-reaching VarPro geometries for smooth readout branches sharing gamma.

Coordinator: root. Active work: exact one-column least-squares derivation, normalization and endpoint audit, and a small numerical check. Worker `trained_gamma_branches` owns only `trained_runs/` here and analyzes saved checkpoints without training. Worker `single_gamma_math_review` independently reviews the mathematics without edits. Existing apps and unrelated research remain untouched.

Context: local shared-context revision `1a626ad59f8931c5592c625afdfd0da80cb0bb5a`; installation check passed; synchronization failed at Git/transport, so freshness is unverified.

Status: complete. No new training and no app changes.

Completed authorized consolidation: [self-contained theory checkpoint](../../../../docs/geometry_readout_theory_codex/checkpoint.md), indexed in `docs/INDEX.md`. It contains full intermediate derivations, exact and approximate claims, controlled multi-neuron predictions, and activation-selection theory with traced evidence. Coordinator owned `docs/geometry_readout_theory_codex/` and `checkpoint_validation/` here. Independent read-only reviews by `multi_defect_review` and `activation_evidence` checked the formulas and evidence. Their corrections to weighted dual rows, baseline differentiation, normalization limits, tent remainder details, periodic zero modes, and cutoff descriptions were incorporated.

New bounded validation: the explicit analytic center and width response sequences predict four simultaneous mixed tent changes without a fit. A separate continuous-L2 solve is used only as reference. At scales 1, 1/2, 1/4, 1/8, coefficient-change relative errors are 7.49%, 3.92%, 2.01%, 1.02%; absolute error decreases approximately quadratically. Quadrature order3 versus5 changes coefficients by under 5e-15. [Script](checkpoint_validation/multiple_tent_prediction.py), [measurements](checkpoint_validation/multiple_tent_prediction.json), [figure](checkpoint_validation/multiple_tent_prediction.png). Figure inspected; document local links, contents anchors, unique equation labels, and math delimiter balance checked. No app edits, broad new sweeps, or training.

Presentation follow-up completed: [results report with inline figures](checkpoint_validation/results_report.md), [four-panel tent figure](checkpoint_validation/four_neuron_full_report.png), and [activation evidence figure](checkpoint_validation/activation_evidence_report.png). The new figures use saved arrays and existing repository measurements; no new fits or training were run for this presentation. The report distinguishes the new tent validation, earlier ReLU experiment, and historical activation evidence, and states that the broader proposed program has not been launched.

Completed follow-up: extend the coefficient-analysis identity to second derivatives (GELU/ReLU). A pure second-derivative identity requires each nonlinear coefficient functional to annihilate both constants and linear functions; an explicit affine baseline in identifiable ordinary least squares supplies this condition. GELU's curvature kernel is `(2-z^2) phi(z)` with mass one, giving the local/QI normalization `gamma*v approximately h*f''`. For positive-gamma ReLU, gamma only rescales the feature; `gamma*v` is the exact slope jump.

Worker `relu_readout_quick` produced the [four-target ReLU experiment](relu_test/report.md) and [figure](relu_test/relu_coefficients.png), using 63 interior knots, an explicit affine baseline, and continuous function-value L2 fitting. The normalized coefficients follow target curvature; the quadratic gives `w/h=2` within approximately 1e-12 relative error. Independent hat-basis, quadrature, and gamma-rescaling checks passed. No new training or app changes.

Follow-up: [exact tent-kernel deletion and width tests](theory/tent_derivation.md), with [figure](theory/tent_defects.png). Continuous-L2 quadrature checks agree with the infinite-lattice coefficient formulas within 3.7e-12. This case separates alternating compensation from severe ill-conditioning: its Gram condition number is only 3.

Further follow-up: [eightfold tent width](theory/tent_overlap_notes.md), [main figure](theory/tent_overlap_w8.png), and [nearby-width control](theory/tent_overlap_w8.25.png). The periodic half-width-8 grid has seven exact null directions: deleting one neuron gives a repeating zero/one-seventh readout and an exact constant. Half-width 8.25 removes that special redundancy and produces small residuals and modulated coefficients. Domain and minimum-norm conventions are stated explicitly in the notes and figures.

- [Derivation, assumptions, endpoint qualifications, and numerical checks](derivation.md).
- [Saved-run branch/gamma report](trained_runs/report.md), with source hashes in `trained_runs/summary.json`.
- [Single-gamma profiles](theory/single_gamma_profiles.png), computed with analytic Fourier features for periodized normalized bumps.
- [Floor-run readouts colored by gamma](trained_runs/varpro_floor_zoom.png).

Independent mathematical review agreed with the one-column formula and identified the importance of distinguishing function-LS from derivative-LS, and fixed-rank solves from global TSVD. The periodic numerical response matches fresh full SVD solves within 5.8e-10 relative coefficient norm in tested cases. The broadest diagnostic case is not boundary-converged and is excluded from the illustrated range.

The floor run reproduces approximately 6.6e-15 relative L2. Its visible near-zero and negative readout bands have markedly different gamma regimes; this is descriptive evidence, not a universal branch classification. The saved ordinary Adam controls are short runs; the nearly uniform QI-initialized run fits well, while the two Xavier variants have limited convergence. No general conclusion about converged ordinary Adam is claimed.
