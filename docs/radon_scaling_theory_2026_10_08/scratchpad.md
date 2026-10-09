# QI–Radon scaling theorem: working record

Date: 2026-10-08. Coordinator: Codex /root. Status: mathematical writeup and independent audit complete; ready for presentation. No Lean work authorized or started.

## Scope and acceptance criteria

Known-function, computable one-hidden-layer QI–Radon construction. Prove uniform error, count actual halo neurons, expose dimension and accuracy dependence, and derive allocation of angular directions versus scalar centers. Distinguish activation classes; no universal exponential claim for ReLU/polynomial activations. No PDE experiments, hidden least-squares solve, output replacement by a polynomial, or unsupported float64 claims. Sam receives the mathematical argument in chat before any Lean formalization.

## Sources being audited

- `../radon_known_function_construction.tex`: finite analytic tanh construction, contour moments, prescribed rational halo correction, and existing error bounds.
- `../appendix_notes/paper_v5_replacements/organized_v7_package/sections/7_appendix/01_Construction/09_26_sl.tex`: original construction appendix and GELU/SiLU extensions.
- `../../experiments/expF19_radon_direct_pde/quill_boundary.py`: implemented explicit tanh coefficients; default double precision is not an asymptotic accuracy certificate.
- Earlier chat: Fourier-slice derivation, normalized spherical weights, recursive Gauss–Jacobi quadrature, and the exact single-mode center-alias calculation.

## Findings so far

1. A fixed overlap lambda has an actual O(N^-1 exp(-pi^2/lambda)) sampled-density alias contribution. It cannot support a literal epsilon→0 geometric claim.
2. Existing tanh bounds *can* yield O(log(1/epsilon)) scalar width: fix physical halo thickness and gamma, refine spacing, hence lambda=gamma h decreases and halo count is linear in interior count. Alternatively explicit accuracy recipe lambda=pi^2/T, R=ceil(2T/pi), n proportional to T satisfies all three bounds. This is a disclosed departure from a fixed sqrt(N) halo schedule, not a new representation.
3. Analyticity of f in input space does not imply analytic canonical Radon angular profiles. Counterexample f(x)=product_j (1+x_j^2)^-1 has Fourier transform pi^d exp(-||xi||_1), producing directional cusps. Need a separate quantitative angular regularity assumption.
4. Positive exact angular quadrature passes uniform per-spoke errors through without a factor M. Full degree-(2q-1) tensor rule has 2q^(d-1) directions; antipodal folding applies to the even Radon integrand and leaves q^(d-1).
5. Activation-independent exponential rate is impossible. ReLU restrictions to a line are piecewise linear, giving an Omega(W^-2) error lower bound for x^2 even though the target is analytic. A general theorem must expose activation-specific constructive scalar rates.
6. Corrected tanh coefficients can have exponentially growing conditioning as lambda→0. Existing bound predicts O(log(1/epsilon)) working bits for the logarithmic-width schedule, not uniform float64 stability. Moment quadrature must be tolerance-controlled.

## Workstreams

- `scaling_scalar_audit`: finite tanh certificate, parameter optimization, halo conditioning and computability.
- `scaling_angular`: angular regularity, polynomial exactness, dimension constants and allocation.
- `activation_scope_audit`: activation obstructions and explicit Gaussian/GELU scalar construction.

## Consolidation and checks

- Completed `theory.md` plus independent scalar, angular, activation, and stable finite-smooth tanh proof modules.
- Independent review checked the Gaussian-family alias sum and finite-tail constants, the sampled polynomial filter, and all angular allocation formulas.
- Corrected a pointwise-representative gap: harmonic L2 assumptions must be accompanied by continuity/equality to the continuous harmonic representative at quadrature nodes.
- Corrected precision scope: logarithmic bits apply to the fixed analytic and stable Fourier classes, not the local finite-smooth polynomial fallback.
- Corrected adaptive error-budget normalization to include the global reanchoring factor two.
- Checked 48 choices of analytic margin/grid parameters and 246 finite-band alias parameter cases numerically as arithmetic sanity checks. These do not replace the proofs and are not network benchmarks. Results are in `verification.json`.
- Found and independently checked a stable finite-smooth tanh route: Fourier cutoff, finite grid, ordinary halo, algebraic width, bounded readout norm. This complements the local C^s construction with a stronger global data assumption and better conditioning.
- Completed `radial_extension_certificate.md`: a prescribed smooth radial cutoff represented as a positive mixture of balls derives both required analytic regularities directly from a target holomorphic on a complex neighborhood of a larger real ball. The proof covers both odd and even dimensions; independent review found no material gap. Its finite continuation formulas replace the unjustified use of a complex argument in the compact extension's full Fourier integral. Added the target-side corollary to the main theorem.
- Fixed all missing mathematical display delimiters found in final review. No existing user work was reverted, no code was committed or published, and no PDE/training experiments were run.

## Presentation and authorization boundary

- Present the assumptions, finite construction, error propagation, explicit width law, allocation, activation boundaries, and practical costs in chat. The target-side boundary constants are conservative upper bounds, not empirical optimal settings.
- Do not start Lean until Sam explicitly authorizes it after reviewing the completed argument.
