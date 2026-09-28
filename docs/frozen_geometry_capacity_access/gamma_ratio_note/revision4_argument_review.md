# Revision 4 argument review

Reviewed `gamma_ratio_note_v4.md` after both placeholders were filled, together with `note_interval_figures/MANIFEST.md`, the saved numerical diagnostics and counts in `note_interval_figures/data.json`, and the separately evaluated attainable readout in `spectrum_mechanism_results.md`. This review covers inference, explanatory scope, and numerical claims; the separate mathematical review covers the Poisson and tail constants. No training was run and the note was not edited.

## Finding: tighten the local certification disclaimer

At line 284, “not certified integer bounds on all floating-point implementations of GD” states a weaker limitation than the numerical calculation actually has. The empirical numerical allowance and refinement checks do not certify the tabulated numbers as bounds for the exact, nominal-real tanh kernel either. The introductory and Appendix F disclaimers do state this more fully, so this is a local clarity issue rather than a defect in the theorem.

Suggested replacement: “These are checked floating-point evaluations of (11)–(12), not certified bounds for the exact tanh kernel or guarantees for a floating-point GD implementation.” Keep the statement that the gamma-4 displayed endpoints are rounded outward: that rounding is correct relative to the computed endpoints, but does not itself certify them.

## Logical completeness and limits

No blocking logical gap, inequality reversal, concealed target projection, or universal monotonicity claim was found in the completed version.

- Equations (1)–(3) put the normalization and target dependence in the right places. The result is specifically about zero-initialized Euclidean readout GD with the stated normalized learning rate. The scalar-schedule extension does not silently extend to momentum or arbitrary large steps.
- Equation (7) proves increasing quadratic forms and ordered eigenvalues for the projected whole-line integral. The note does not transfer this order directly to the actual finite ratios. The corrected matrix, normalization, interlacing indices, and explicit counterexample identify the missing steps in any such universal inference.
- The theorem bounds the actual ordered finite eigenvalues through a separate corrected-matrix spectrum. The remaining eigensolve is stated immediately before the theorem; the finite-feature products used for normalization and the actual feature-SVD projections used for the training calculation are stated in Sections 4 and Appendix D. There is no claim of a closed scalar gamma-to-rank formula or a target-independent prediction of training time.
- The two training implications use opposite endpoints correctly. An upper ratio gives a lower residual and a necessary time; a lower ratio gives an upper residual and a sufficient time. Unresolved energy is dropped only where dropping nonnegative terms preserves the lower residual, and it is retained in the upper residual. A zero lower rate is not mistaken for a genuinely zero kernel eigenvalue.

## Numerical claims checked against the saved sources

The count table agrees with the saved data. The gamma-4 necessary endpoint `920663607184` is rounded down to `9.20e11`; the sufficient endpoint `3303588843394` is rounded up to `3.31e12`. The middle count `1007616497993` rounds to `1.008e12`. The four other rows agree as written. The statements about approximately 91% and 3.3 times the spectral count are supported.

The gamma-4 comparison remains evidence of optimization delay at an attainable tolerance: the separately constructed readout has relative error `0.0005850007883`, with the source reporting a 70-digit reevaluation of saved coefficients at the same nominal geometry. Its error is far below the 1% threshold. The note does not need to identify every tiny positive eigenvalue or the true asymptotic nullspace floor to make that tolerance-level comparison.

The two independent SVD drivers and quadrature refinements support the reported scale separation. The gamma-4 necessary count changes by roughly `7.9e-9` relatively under refinement; the other displayed count rows are unchanged as integers. These checks support the numerical account but are not directed-rounding certificates. Larger tested gammas have substantially smaller necessary and sufficient counts; the text restricts the conclusion to those tested values rather than extrapolating to all gamma or all targets.
