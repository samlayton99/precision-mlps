# Current status

The current mathematical presentation is the [PDF note](dist/qi_radon_theorems.pdf),
organized around four main theorems. It includes the corrected finite-halo tanh
proof, explicit construction algorithm, and empirical checks. Exact Lean coverage
and limitations are in [docs/theorems.md](docs/theorems.md).

Coordinator: Codex. Cleanup on 2026-10-09, with no subagents:

- Source grouped by mathematical responsibility; declaration namespaces retained.
- Public entry reduced to 13 imports with the same complete dependency closure.
- Three unused source files and historical reports isolated in `archive/`.
- Structural validation checks archive exclusion and source reachability.
- Current build and axiom evidence live in `verification/`.

The stronger tanh boundary correction is part of the human proof and numerical
construction. Its contour and rational-correction estimates are not yet in Lean;
the existing Lean tanh endpoint retains its generic d+1 exponent. The corrected
count also improves the human local algebraic tanh result by using the existing
entire sample preparation. Full numerical coefficient acquisition and lambda
optimization remain unformalized. Historical status
and worker notes were consolidated into the archive rather than kept as parallel
current descriptions.

The complete human-readable theorem note is in docs/qi_radon_theorems.tex,
rendered to dist/qi_radon_theorems.pdf. The release ZIP is made with
scripts/make_release.py. The final warning cleanup retains all mathematical
hypotheses, apart from removing unused typeclass dependencies; the package
build and 71-declaration axiom audit pass.

The empirical note now separates a single Gaussian family's d=1..4 scaling
from five Gaussian-based families' robustness. Additional checks cover the
scalar nonanalytic Fourier-moment construction, ReLU's algebraic rate, and
the sufficient direction/center allocation. Dashed predictions include the
halo correction and do not include a fitted empirical floor. Readout precision
is isolated from float64 geometry/profile/evaluation errors.
