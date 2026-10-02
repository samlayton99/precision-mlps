# Branch prediction — Codex

Coordinator: this Codex thread. September 30, 2026.

Objective: predict individual readout patterns from fixed geometry and target, beginning with the user-shared 17-neuron equal-width Runge state. No training or app mutation.

Completed this bounded investigation. The report contains the complete factorization and alternating-carrier proof, a three-pattern target encoder and its derivative-measurement form, controlled geometry/target transfer results, and the finite-edit extension. Four figures have PNG and PDF copies; all numerical inputs and results are saved here.

Main result: 0.43% relative coefficient error on the shared Runge state with three prescribed geometry patterns; 0.256% with seven patterns after four simultaneous center/gamma edits. The latter uses exact edit responses and approximate background encoding; its absolute compression error is independent of the new edit values for a fixed edited set.

Limitations: the three-pattern weights alone produce 42.24% sampled function error, versus 1.16% for the full fit. Accuracy varies substantially with target. This is an equal-gamma background theory plus finitely many edits, not a completed theory of arbitrary learned Adam geometries. Broadening to gamma 1.3 crosses the app cutoff, so that transfer row describes the untruncated solution. Higher cutoff changes also expose failures.

Independent verification: the complete recurrence agrees with an 80-digit direct QR reference to relative 1.95e-16; 70 versus 100 digits yields identical exported coefficients. The report distinguishes standard projection algebra from the tested structural compression claim. No training, app modification, commits, or publication.
