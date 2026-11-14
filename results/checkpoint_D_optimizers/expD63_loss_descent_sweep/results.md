# Loss-descent bound: 24 functions, 200k steps

All 216 runs were evaluated from initialization through 200,000 updates. No new training.

| Learning rate | Default initialization | 2x initialization | 4x initialization |
|---|---:|---:|---:|
| 0.01 | 0/24 | 0/24 | 0/24 |
| 0.001 | 24/24 | 24/24 | 23/24 |
| 0.0001 | 24/24 | 24/24 | 24/24 |

**143/216 pass the sufficient step-size condition.** The sole failure at LR 0.001 is mixed_sine with 4x initialization (eta L_hat = 1.02338; passing requires <= 1).

Every measured RMS-slope trajectory stays below the candidate formula at all saved checkpoints. This observation does not establish the bound for failed-check cases. Solid curves show measured RMS slopes, dashed curves show supported ceilings, and dotted curves show formula-only evaluations.

Direct checks at consecutive saved updates find violations of the proof’s sufficient-decrease inequality in 10 LR-0.01 runs. The exponential target with 4x initialization and LR 0.01 also exceeds the full-parameter movement formula at step one (0.105346 versus 0.099955), while its RMS slope remains below the slope formula.

[Overview](figures/gamma_bound_comparison.png) · [Step-size checks](figures/step_size_condition.png) · [All functions separately](figures/all_24_functions.pdf) · [Every run (CSV)](run_results.csv)
