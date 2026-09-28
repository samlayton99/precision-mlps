# Section 3: standalone appendix source notes

Prepared 25 September 2026, using the supplied ICLR draft as the scope. These are readable, self-contained source notes for later appendix assembly. The current paper and its existing appendix have not been edited. Arithmetic circuits are excluded.

## Downloads and what each note adds

| Note | PDF | LaTeX | Main use in the current draft |
|---|---|---|---|
| 1. True-p-bit implementation | [PDF](01_true_pbit_implementation.pdf) | [Source](01_true_pbit_implementation.tex) | §3.2: specifies the actual rounded construction, solve, and inference used in C11. |
| 2. Finite-kernel eigenvalue ratios | [PDF](02_finite_kernel_ratios.pdf) | [Source](02_finite_kernel_ratios.tex) | §3.4: supplies the reviewed two-sided ratio proof, finite corrections, and error/time transfer. |
| 3. Geometry and readout mathematics | [PDF](03_geometry_readout.pdf) | [Source](03_geometry_readout.tex) | §3.5: exact decompositions, derivative distinctions, local curvature, and conditional slope-travel budgets. |
| 4. Bandwidth selection | [PDF](04_bandwidth_selection.pdf) | [Source](04_bandwidth_selection.tex) | §3.3: the full supplied bandwidth proof, with a reading guide and explicit separation of predictor and certificate. |
| 5. Neighboring and readout conditioning | [PDF](05_neighboring_conditioning.pdf) | [Source](05_neighboring_conditioning.tex) | §3.4 controls: proves which conditioning penalty a coordinate change removes and which small-bandwidth dependence remains. |

The accompanying ZIP contains all five PDFs and all LaTeX sources, the three existing figures used by Note 2, and the files needed to rebuild the documents. Each note can be read independently. The earlier unchanged source collection remains at `docs/appendix_materials/2026-09-25/`.

## Is this material orthogonal to the supplied notes?

**Not entirely.** The collection adds missing proof and implementation detail while deliberately consolidating one supplied proof.

- **Note 1 adds an implementation contract.** The supplied FP64 note proves guarantees for reference-scaled recovery; C11 implements an unscaled, precision-controlled DGELSS solve. They are different algorithms. This note documents the latter and its completed measurements without claiming it certifies the former.
- **Note 2 supplies the finite-kernel route behind §3.4.** It preserves the complete reviewed revision-4 proof. It is distinct from the supplied polynomial-tail accessibility argument, and it supports the theorem already stated in the paper rather than adding a separate competing theorem to the main text.
- **Note 3 consolidates prior geometry work and later corrections.** The familiar loss split is retained because the derivations depend on it. The useful additional material is the distinction between the current-readout projection and the exact profile gradient, the residual derivative, the metric-dependent drift budgets, and the limits of truncated-solve diagnostics.
- **Note 4 substantially overlaps the supplied bandwidth source.** It retains that proof rather than inventing a different derivation. Use this edition or the original as the bandwidth source; do not append both. The full-spectrum bounds, centroid limitations, endpoint conventions, and recovery-transfer hypotheses remain intact.
- **Note 5 is the additional note worth retaining.** It proves the neighboring-difference conditioning result and records the actual coordinate maps. General coordinate-control ideas also appear in other notes, but the explicit tanh bounds and finite-window halo obstruction add useful mathematical content.

## What is established, and what should not be claimed

### 1. True-p-bit construction

The note gives the arithmetic format, rounded primitives, centered feature evaluation, activation algorithm, complete least-squares path and cutoff, coefficient storage, and sequential inference order. It separates offline inputs and the external error observer from model arithmetic. Completed chirp and Runge results and historical validation are identified as such. It also corrects overly broad descriptions of IEEE equivalence, tanh accuracy, and replay coverage.

This is implementation documentation plus existing numerical evidence. It is not a new uniform floating-point error theorem, an end-to-end certificate for the supplied scaled solver, or a native low-bit hardware speed claim.

### 2. Finite-kernel ratio proof

The four-part argument is preserved: the gradient-descent recurrence; the center-integral decomposition; analytic grid/halo corrections and two-sided normalized eigenvalue bounds; and target-weighted residual and step-count intervals. Upper ratio endpoints give necessary times. Lower ratio endpoints give sufficient times. No earlier gamma is needed as an anchor.

The exact theorem and checked floating-point figures are distinguished. The displayed example uses 153 hidden neurons, 263 samples, and the 2π/6π/10π mixture. It must not be relabeled as the current paper's separate width-512, 2π/6π/14π experiment. The theorem does not assert universal monotonicity of every finite normalized eigenvalue; the source counterexample is retained. The actual target projections still enter the time bounds.

### 3. Geometry/readout mathematics

The note proves the exact `L = F + G` identity, differentiates the exact profile under local constant rank, relates its gradient to the current-readout span split, and derives the exact-fit profiled Hessian. It gives finite-horizon slope budgets with explicit rate, metric, residual, or sufficient-descent assumptions.

These ingredients do not yet prove universal joint-training failure or the necessity of the QI geometry for every accurate network. The missing approximation and trajectory premises are stated. Hard-truncated least-squares diagnostics are not silently treated as the exact profile.

### 4. Full bandwidth selection

The complete finite-contour proof, exact first-pair score, compact approximation, frequency-estimation conventions, scalar inversion, sensitivities, and recovery transfer are retained. The added opening explains how these pieces fit together. The scalar tables are inherited from the supplied note; their mentioned companion script was not supplied, and no new reproduction is claimed.

The mean-frequency prescription is a practical approximation. Its accurate approximation of a scalar score is different from a certificate for all target frequencies or the complete recovered-network error.

### 5. Neighboring conditioning

The note proves width-independent whole-line bounds for neighboring differences at fixed relative bandwidth, compares them with the same function space in physical zero-sum tanh coordinates, and derives the remaining exponential small-bandwidth dependence. It then separates finite-window halo effects and shows exactly how an invertible readout map changes the gradient metric while preserving the span.

These whole-line bounds do not automatically apply to the complete finite sampled matrix with bias, anchor, and halo columns. This is a useful conditioning control, not a universal target-specific training lower bound.

## What was deliberately left out

The supplied fixed-lambda representation and FP64 recovery proofs are already substantial and do not need duplicate standalone notes. The older polynomial argument, superseded gamma comparisons anchored to a reference slope, and unfinished C13 comparison results were not promoted into new notes. Activation extensions and auxiliary empirical diagnostics remain in the source inventory; they are not needed to complete this Section 3 packet.

## Build and verification

All five PDFs were compiled directly from the included LaTeX with Tectonic 0.17.0. The documents use standard AMS/LaTeX packages; `./build.sh` also supports `latexmk` or `pdflatex`. Run it from this folder (the script changes to its own directory). Note 2's figure paths are relative to `figures/`.

The final build checks equation references, missing fonts/glyphs, and oversized lines. Representative pages, theorem pages, and figures were visually inspected. Independent read-only checks confirmed that Note 2 preserves the reviewed revision-4 mathematics and that Note 3's key derivative and drift inequalities retain their assumptions and directions. This is not a claim of a new exhaustive formal proof audit. No training or precision sweep was rerun. `MANIFEST.json` records checksums of the delivered PDFs, sources, figures, and build files.
