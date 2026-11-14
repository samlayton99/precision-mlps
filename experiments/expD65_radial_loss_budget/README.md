# Radial loss-budget bound

Read-only theory audit on saved tanh GD trajectories. No training or optimizer changes.

The proposed bound controls RMS slope using a finite residual-reduction budget. It assumes sufficient GD loss decrease and a uniform bound on the signed outward slope-gradient ratio. It avoids the original self-fed parameter-norm recurrence and the failed absolute-moment remainder bound.

- `audit.py`: calibrates a single ratio threshold from default-init 20k–30k checkpoints, then checks all 24 functions at learning rates 0.001/0.0001 through 200k.
- `validate_extra_inits.py`: tests that unchanged threshold on the 2x and 4x first-layer initializations. It reports the Airy 2x / 0.001 failure explicitly.
- `check_proof.py`: independent autograd checks plus randomized checks of the discrete telescoping coefficients.
- `report.py`: writes the summary and primary figure.

Proof: `docs/review_process/iclr_2027/radial_loss_budget_bound.md`.
Results: `results/checkpoint_D_optimizers/expD65_radial_loss_budget/`.

The temporal split is exploratory, not preregistered independent confirmation. Saved checkpoints cannot establish a premise at every intervening update. No indefinite persistence, maximum-neuron bound, or target-error floor is claimed. The QUILL scale is a construction reference rather than a necessary accuracy threshold.
