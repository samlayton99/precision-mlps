# Deterministic sample-based selection

`QIRadon/DeterministicSelection.lean` closes the final deterministic C_data connection. Its actual computed certificate is

`U_j = max_i |y_i - Q_j(x_i)| + noise + (Lf + Lj) radius`.

The finite sample set is nonempty and covers the target domain at the supplied radius. The target and candidates satisfy the supplied quantitative Lipschitz bounds; measurement error is at most `noise`. `sampleCertificate_sound` proves uniform error at most `U_j`.

`exists_sample_selected_near_best` chooses a minimum-width candidate among those with `U_j ≤ epsilon`. It proves true uniform error at most epsilon and width no greater than any tested candidate whose true uniform error `e` satisfies

`e + 2 noise + (Lf + Lj) radius ≤ epsilon`.

The second noise allowance is necessary: it pays for the difference between true error and the observed residual. `passing_exists_of_uniform_margin` derives existence of a passing certificate from exactly this comparison margin. Neither result claims global optimality over untested networks. Unlike statistical validation, deterministic coverage permits the candidates to depend on the same samples.

The maximum is an actual finite maximum, not a supplied error oracle. The regularity and coverage bounds remain explicit assumptions, as required by the impossibility of uniform certification from unqualified finite samples.

Verification: `lake build QIRadon.DeterministicSelection` succeeds without warnings. The four headline results audit to standard Lean foundations only (`propext`, `Classical.choice`, `Quot.sound`).
