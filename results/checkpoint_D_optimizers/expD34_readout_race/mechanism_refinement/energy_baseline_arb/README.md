# Verified initial conditions for the generic energy baseline

All 18 checked empirical states satisfy $M_0=\|(a,b,c)\|_2\le3$,
$L_0=\frac12\|f-y\|_m^2\le1$, and $\eta\le1/490$. These are seed-30
checkpoints at update 20,000, for six targets and three independently
initialized widths. They are the starts of the next 20,000-update windows,
not initialization at update zero.

| $N_{\rm ref}$ | Physical width | States | Rounded-up maximum $M_0$ | Rounded-up maximum $L_0$ |
|---:|---:|---:|---:|---:|
| 128 | 177 | 6 | 2.681334 | 0.416601 |
| 512 | 705 | 6 | 2.607185 | 0.416404 |
| 1024 | 1409 | 6 | 2.631417 | 0.416352 |

The [generic energy theorem](../../../../../docs/d34_energy_baseline.md)
therefore excludes every neuron from $\lambda=0.25$ at every prefix of the
next 20,000 updates in all 18 cases. This statement concerns exact
real-arithmetic GD on the archived binary64 dataset and state, with the
exact binary64 step size represented by `0.002`. It does not certify
preceding training or rounding in the GPU implementation.

The unchanged original Arb helper supplies an outward-rounded upper bound
$s_0$ for residual RMS. The loss check uses the upper endpoint of
$s_0^2/2$, and the norm check uses the first $3W$ parameters, excluding the
output bias. The input grid also satisfies $|x_i|\le1$. No effective-force,
tracking-force, or finite-width force-constant assumptions were evaluated.

[conditions.json](conditions.json) contains every enclosing interval and
condition flag; [summary.json](summary.json) contains per-width maxima.
The exact input packs, [verification command](verification.sbatch),
[stdout](verification.out), and [summary command](summary.sbatch) are
preserved. Job 1295 used the already-deployed original `point()` function,
100-bit Arb arithmetic, one CPU, and 103 seconds. Its source hash is
recorded in the data. No certificate helper was changed or uploaded.

This baseline makes a distant-threshold exclusion inexpensive. It does
not explain the signed direction of slope motion or the accurate
finite-window predictions of the evolving effective-force model.
