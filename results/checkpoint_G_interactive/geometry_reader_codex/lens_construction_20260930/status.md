# Constructing the geometry lens — Codex

Coordinator: root. September 30, 2026. Completed construction and validation; no training.

User asks for an implemented, tested, predictive encoder, including finite width/center changes and activation transfer. The kernel-difference proposal is developed as exact reference-coordinate transport plus orthogonal leakage and residual forcing.

- Root owns `spectral/`, integrated report, and status. Explicit Fourier reference for periodic normalized tanh/GELU primitives; finite defect encoder; target transfer; controlled breakdown. Coefficients are explicitly constrained to sum zero; the output mean is separate.
- `geometry_object_audit` owns `theory.md`: derivation and mathematical audit of the small correction system, constrained reference spaces, and failure conditions.
- `sparse_centers_explanation` owns `tents/`: finite continuous fitting, analytic hat inverse and target measurements, finite interacting defects across targets, independent full-solve validation.
- `learned_solution_anatomy` owns `learned_transfer/`: low-order moment encoder built from saved broad-neuron geometries, new target transfer, tail-error prediction and limitations.

Prediction must not use the changed geometry's dense LS/SVD. Such solves are isolated reference checks. Report target and geometry selection, exact conventions, approximation errors, and numerical limitations. Existing application/checkpoint files remain unchanged. Context sync transport failed; local context is used, and context check passed.

Delivered: integrated report.md, theory.md, branch reports and figures, reusable finite-edit encoders, geometry-only saved matrices, learned-geometry sampled-input encoder and original-readout decoder. Local report links checked. Independent mathematical/implementation review found no remaining substantive corrections. Validation records accompany each branch.

Main results: twelve interacting edits predicted to relative coefficient discrepancies of 2.05e-12 (finite tents), 1.72e-11 (periodic tanh), and 8.36e-12 (periodic GELU). Ten-sample target transfer on the restricted saved Adam geometry ranges from 1.63e-10 for slow sine to 0.192 for sin(2πx); analytic omitted moments predict tested errors. Decoding original readouts is explicitly limited to the broad-neuron cohort. The general arbitrary-network encoder remains an open research problem, rather than a completed claim.

The existing theory checkpoint received a link to the new report; no earlier results were rewritten. No application, saved model, or unrelated repository changes were modified.
