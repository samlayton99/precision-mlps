# QI–Radon: verified construction and approximation bounds

This Lean library constructs finite shallow networks for known functions and
proves their approximation error, halo-inclusive width, and resolution allocation.
It does not solve an unknown PDE or establish the cost of computing every coefficient.

The complete [PDF note](dist/qi_radon_theorems.pdf) is organized around four
results: analytic scaling, algebraic scaling, resolution allocation, and
numerical accuracy. It includes the construction algorithm and empirical
comparisons. Its editable source is
[docs/qi_radon_theorems.tex](docs/qi_radon_theorems.tex).
For a shorter overview, start with [the argument](docs/argument.md), then use the
[theorem map](docs/theorems.md) to locate the precise statements and assumptions.
[Verification](verification/README.md) records the build and axiom checks.

The construction section includes a practical lambda sweep recipe (Section 4.6)
and the complex-shift tanh readout corollary (Section 4.8), including the halo
correction and output bias. The lambda seed uses the kernel Fourier ratio
(approximately 0.25 for float64 tanh); `2*pi/R` is an exact-arithmetic
halo/alias balance, not a replacement practical default. The sweep range is
a starting recommendation, not a proven optimum. Run
`python empirical/lambda_consistency.py` for the algebraic bridge and the
recorded scalar-profile checks; these do not benchmark a universal selector.

The note includes the finite-halo correction in its main tanh proof and uses
the resulting `log^d(1/epsilon)` width law. The Lean library currently verifies
the generic tanh `log^(d+1)` bound, not that corrected contour argument. The
note also shows how the correction improves the local algebraic tanh count;
that substitution is a human proof until the scalar correction is formalized.
No Lean theorem has been weakened or relabeled to claim this coverage.

The [empirical results](empirical/results/summary.md) separate one-family
dimension scaling from a multi-family robustness figure. Run
`python empirical/run.py --suite all` and `python empirical/plot.py` to reproduce
them. All halo neurons are counted; no readouts are fitted.

## Build

The toolchain is pinned to Lean 4.33.0 and the Mathlib revision in `lake-manifest.json`.
Install the Lean toolchain manager elan, then enter this directory.
The pinned toolchain is selected automatically. On a first setup, run
<code>lake exe cache get</code> to obtain cached Mathlib dependencies. Then:

```sh
python3 scripts/check_structure.py
lake build
lake env lean Audit.lean
```

`QIRadon.lean` imports the public endpoints. Every active module is reachable
from those imports. No archived source is part of the build.

## Source layout

| Directory | Responsibility |
|---|---|
| `QIRadon/Radon/` | Boundary extension, normalized reconstruction, and target-to-profile regularity |
| `QIRadon/Kernel/` | Deconvolution, Poisson aliases, finite halo certificate, kernel envelopes |
| `QIRadon/Scalar/` | Sample preparation, anchoring, one-spoke error and width |
| `QIRadon/Quadrature/` | Positive spherical rules, harmonic tails, angular regularity, error transport |
| `QIRadon/Architecture/` | Conversion of ridge sums and anchors into ordinary shallow MLPs |
| `QIRadon/Construction/` | Uniform multidimensional endpoints and full neuron counts |
| `QIRadon/Allocation/` | Certified angular/row allocation and per-spoke multiplier solver |
| `QIRadon/Activations/` | Concrete admissible activations and approximation obstructions |
| `QIRadon/Validation/` | Data-based certificates, selection, and finite-sample limitations |
| `QIRadon/Numerics/` | Proved perturbation and preparation-stability bounds |

In the working repository, archive/ holds historical reports, unused source,
and unfinished exploration. It is excluded from the release ZIP and from the
proof dependency graph. Public theorem namespaces were preserved
when the source files were grouped by purpose.

To rebuild the PDF, use a LaTeX installation with the packages listed in its
preamble, for example <code>tectonic --outdir dist docs/qi_radon_theorems.tex</code>.
After verifying the source and rendering the PDF,
<code>python3 scripts/make_release.py</code> creates the source ZIP and its
checksum manifest. No compiler binaries or third-party dependency caches are bundled.
