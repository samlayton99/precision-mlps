# Construction appendix replacement

The complete replacement is `construction_appendix.tex`, based on the unchanged copy in `originals/construction_appendix.tex`. The edit follows the conservative ordering recommendation in `../appendix_structure_review_2026-09-26.md`.

## Changed blocks

- **Setup, lines 73–108:** state the real-sample/deterministic-envelope algorithm once, retain the successful-run qualification, update the sampling-condition reference, and remove the duplicate proof outline. The theorem's equations and all numerical allowances remain unchanged.
- **Tanh reference construction, lines 107–562:** consolidate the five former subsections into one subsection with paragraph headings for local representation, signed midpoint discretization, boundary correction, coefficient control, and the reference theorem. Former section labels `sec:local`, `sec:quadrature`, `sec:boundary`, `sec:coefficients`, and `sec:witness` are aliases of the consolidated subsection; placing them immediately after its heading avoids labels inheriting a theorem counter. The removed roadmap contained the only former subsection range in this file.
- **Common recovery, lines 564–649:** replace already-established activation-specific assertions with explicit dictionary hypotheses. These require the normalization and real-axis bound, a uniform reference error and positive coordinate envelopes with bounded sum, and holomorphy and bounded normalized features on the sampling ellipses. Numerical comparison/export, cutoff, input-rounding, and evaluation assumptions remain separate run conditions. Move the unchanged sampling conditions and tanh complex bound here, and verify all analytic hypotheses for tanh before recovery.
- **Recovery and continuous transfer, lines 652–1040:** retain the SVD derivation, cutoff, coefficient/sample bounds, continuous transfer, input rounding, evaluation, and tanh theorem assembly in one subsection. Continuous transfer now invokes the explicit complex-bound hypothesis. Compress only the sentence about the pairwise projection contribution, preserving its restriction to that contribution. Keep the maximum-residual certificate and the full fixed-design/certified-width-range qualification unchanged.
- **GELU/SiLU, lines 1041–1516:** move the full activation construction after tanh theorem assembly. Keep polynomial approximation, deconvolution, halo estimates, activation realization, native and normalized coefficient envelopes, and the complex-growth lemma in their original internal order. At the end, explicitly discharge the common dictionary hypotheses (including the SiLU pole-free strip), then give the unchanged `eq:psi-computed` conclusion under the numerical run conditions. Point to the width corollary with its existing qualifications.

## Preservation checks

Comparing the replacement against `originals/construction_appendix.tex`, with moved blocks compared as multisets:

- All **155 labels** retained exactly once; no label added or removed.
- All **106 numbered display environments** unchanged byte-for-byte.
- All **138 display-math blocks**, numbered and unnumbered, unchanged byte-for-byte.
- All **18 proof environments** unchanged byte-for-byte.
- All **13 citation occurrences** retained, with the same keys and multiplicities.
- No new unresolved local reference. The only external targets remain the main theorem and main corollary; the reference helper's `#1` is a macro parameter.

No compile was run by this worker; the coordinator is compiling the complete replacement bundle. No other source files, original attachments, or editor files were changed.
