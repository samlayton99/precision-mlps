# Results for the displayed experiments

The scaling plot uses only f_d(x)=exp(-3*sum(x_k^2)), d=1..4. A separate robustness figure displays all five Gaussian-based families using the same construction schedule, with no precision panel.

| d | Best relative L2 | Width at best | Last relative L2 |
|---|---:|---:|---:|
| 1 | 6.43e-16 | 193 | 1.64e-15 |
| 2 | 1.3e-14 | 4,820 | 2.07e-14 |
| 3 | 2.32e-15 | 49,408 | 6.09e-15 |
| 4 | 3.87e-15 | 790,528 | 9.31e-15 |

## Allocation

| Tolerance | Predicted width | Best tested width | Ratio | Max sampled error |
|---|---:|---:|---:|---:|
| 1e-03 | 780 | 294 | 2.65 | 9.26e-08 |
| 1e-06 | 1,729 | 610 | 2.83 | 7.78e-11 |
| 1e-09 | 3,068 | 1,164 | 2.64 | 9.85e-14 |

The allocation meets every tolerance but is conservative. It is not an empirical optimality result.

The nonanalytic Gaussian-kernel experiment uses p_hat(xi)=(1+xi^2)^(-2), s=2, and A_s=pi+2. Max sampled error decreases from 0.00968 (43 neurons) to 3.97e-8 (1611 neurons). The sufficient bound is algebraic and conservative. Doubled spectral quadrature changes the largest run by about 2.0e-10, so the last digits are not claimed accurate.

The precision experiment rounds final readouts and bias only; geometry and evaluation remain float64. Dashed curves include the exact construction bound and the readout-rounding bound, with no measured error floor added. Unaccounted float64 errors limit measured target accuracy.

The full multidimensional nonanalytic theorem is not tested end to end. Width bounds do not assert efficient profile acquisition or a PDE solver.
