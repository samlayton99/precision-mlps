# Local ridge PDE feasibility — October 9, 2026

Coordinator: current Codex chat. User requested a repository audit and preliminary tests reaching millions of neurons. Existing unrelated edits are preserved.

- Audit: delegated read-only inspection of existing ridge solvers and scaling evidence.
- Implementation: isolated local-window tanh kernel with prefix-sum value tails and matching transpose; no production solver changes.
- Verification: dense values/derivatives, adjoint identity, nonlinear Jacobian finite differences, cutoff sensitivity, independent PDE validation.
- Experiments: operator scaling, independent-readout linear solves, and separately labelled reduced-coordinate linear/nonlinear solves.
- Status: complete. Operator benchmarked through 4,194,432 neurons; free linear/nonlinear solves tested at 1,048,704 neurons; reduced controls at 1,081,600 neurons. Free-readout runs hit iteration budgets and have materially larger held-out PDE residuals than reduced controls. Seventeen targeted tests pass. Report: `results/checkpoint_F_applications/expF20_local_ridge_pde/expF20_results.md`.
