# Fixed-solution geometry investigation — Codex

September 29, 2026. Coordinator: root.

Scope: characterize the function-space geometry represented by fixed one-layer tanh solutions; derive and validate an explicit multiscale encoding law; inspect saved learned solutions. No new training or training-dynamics claims.

- Root: operator/quotient theory, analytic multiscale prediction and validation, integrated report.
- `geometry_object_audit`: independent mathematical derivation and scope audit; no shared file edits.
- `learned_solution_anatomy`: saved-model analysis and figures under `learned_solution/`.
- `sparse_centers_explanation`: workbench deletion behavior and clustered-kernel explanation under `sparse_centers/`.

Status: completed. Integrated results and derivations are in [report.md](report.md); independent derivations are in [theory_audit.md](theory_audit.md). Saved-model anatomy and direct Taylor-moment decoding are in `learned_solution/`; the illustrative sparse-center coordinate comparison is in `sparse_centers/`; the explicit three-target no-LS interpolation encoder is in `analytic_encoder/`. Figures were inspected. Source hashes, grid checks, exact settings, and high-precision encoder identity checks are saved with the analyses. A separate mathematical review found no material error; its scope/wording refinements were incorporated. Local report links and all three encoder identities were checked.

The equal-width rational structure, stable coordinate comparison, and polynomial compensation are supported. An arbitrary unequal-width learned geometry has not been reduced to a universal closed-form coefficient predictor. Existing checkpoints and application code remain unchanged. Shared context check passed; sync transport was unavailable, so local context was used.
