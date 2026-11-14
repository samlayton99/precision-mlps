# Readout spectrum: 24 functions

All 88,344 saved checkpoints were analyzed (216 runs × 409 checkpoints). No new training.

Rates are $\rho_i=\sigma_i^2/\sigma_{\max}^2$ across the 197 readout modes, including output bias.
The figures show the 10th–90th percentiles. CSV files also include the minimum and maximum.

Values at or below 5.170e-26 are numerically unresolved; plots place them on the bottom line.

Final checkpoint: ranges across all 24 functions and three learning rates within each initialization.

| Initialization | Unresolved modes | 90th-percentile rate |
|---|---:|---:|
| 1x | 85.3%–90.9% | Unresolved–1.66e-16 |
| 2x | 80.2%–85.8% | 3.60e-18–8.58e-12 |
| 4x | 73.1%–77.7% | 8.21e-12–2.76e-09 |

These are spectral percentiles, not uncertainty intervals or percentiles of error energy.
They show whether slow readout directions persist. They do not alone measure how much residual error occupies those directions.

[Overview](figures/spectrum_overview.png) · [All 24 function figures](figures/all_24_functions.pdf) · [All checkpoint percentiles](all_checkpoint_percentiles.csv)
