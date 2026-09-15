# Gamma-barrier synthesis

Status: complete, 2026-09-14. Coordinator: main conversation.

Sam requests a coherent assessment of the complete session against a proposed paper claim: critical lambda and width-dependent pre-activation scales, with GD impracticality explained by limited scale drift and weakening useful geometry signal while error remains above the desired precision.

This task is analysis and internal writing only. No training, held experiments, optimizer changes, publication, or messages to collaborators are authorized by this synthesis request.

Work split: coordinator reviews construction/lambda scope and the note's Fourier/drift bounds, then writes the synthesis. One reviewer audits sections 3–4 against expD28/29. A second audits the empirical conclusions and counterexamples in expD24–27/30. Reviewer reports live beside this status. Existing research files are preserved.

A third reviewer independently verified the elementary energy/time drift bound and its limitations. All three reports are complete. The coordinator's synthesis is `synthesis.md`.

Latest user emphasis: identify what is certain about a necessity of unstable training, or an actual way around the obstruction. Conclusion: no necessity-of-instability theorem is established; restricted successful escapes already exist. The most defensible proof direction is finite-budget raw-GD drift, with an additional accuracy-versus-geometry necessity statement still required. Existing complete D26 trajectories provide a restrictive retrospective residual-budget certificate without new training.

The user also requested the separate Codex session **Junmi Optimization**. The coordinator located and read that thread directly (`01a08d13-ef01-7d51-b995-8ef9ed1f5632`), incorporated its exact F+G dynamics and whitening discussion, and checked the completed D28/D29 results against local reports. Its new separate-Adam trials were still running and have not been treated as evidence. This review did not modify or interrupt that session's work.

Follow-up: Sam requests a principled unified theory linking C07/C08 aliasing, readout conditioning, useful geometry gradients, and movement budgets. `theory_framework.md` combines the existing fiber projection law with exact readout-GD dynamics, specifies a target-weighted schedule lower bound, and separates this fixed-grid result from the missing joint-trajectory confinement theorem. Independent reviewers checked the algebra and the gradient/schedule qualifications. No new experiments were run.
