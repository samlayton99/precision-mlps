# Compact gamma note figure

`gamma_ratios_and_steps.png` is the only figure produced. It replots saved arrays; no eigensolve or training is run.

- Left: `../direct_ratio_interval/data.json`, ranks 26 and 33 over the saved gamma sweep; original two-sided ratio intervals.
- Right: `../note_interval_figures/data.json`, gamma 4, 8, 16, 32, 64; necessary, actual spectral, and sufficient counts to 1% relative training residual with eta=0.5/lambda_1.

The left panel uses the original selected-rank evaluations; the right panel uses the full residual calculation with its checked numerical guard and conservative unresolved-mass treatment documented in `../note_interval_figures/MANIFEST.md`. Neither panel is an interval-certified numerical calculation.
