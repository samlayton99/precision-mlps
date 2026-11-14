# Direct GD loss-decrease check

Checked 58,752 adjacent-update probes from 144 saved runs: 24 functions, three initializations, two learning rates, width 196. Each probe uses the saved state at n and computes exactly one native GD update to n+1. No run was continued and no checkpoint was overwritten.

| Initialization | LR | Checks | Minimum c | Maximum c | Minimum after 30k | Violations of c >= 0.5 |
|---|---:|---:|---:|---:|---:|---:|
| 1x | 0.001 | 9,792 | 0.97862136 | 1.00000896 | 0.99993714 | 0 |
| 1x | 0.0001 | 9,792 | 0.99786213 | 1.00000090 | 0.99998886 | 0 |
| 2x | 0.001 | 9,792 | 0.95772641 | 1.00000541 | 0.99995952 | 0 |
| 2x | 0.0001 | 9,792 | 0.99577264 | 1.00000045 | 0.99995534 | 0 |
| 4x | 0.001 | 9,792 | 0.94196140 | 1.00000244 | 0.99996157 | 0 |
| 4x | 0.0001 | 9,792 | 0.99419614 | 1.00000005 | 0.99995747 | 0 |

The numerator is evaluated as -mean(e*delta_q)-mean(delta_q²)/2. The tanh addition identity evaluates delta_q without subtracting nearly equal predictions. Naive subtraction of the two rounded losses is retained only as a numerical diagnostic. Gradients include slopes, hidden biases, readout weights, and output bias; loss is half MSE.

The first two updates of every run match the original training kernel and saved next states (288 checks; largest saved-state parameter discrepancy 0).

Regular checkpoints are 500 steps apart; this checks only the single next step at each saved checkpoint, not every intervening update. The last probe is 199500 -> 199501, within the original 200k horizon. Losses separated by 500 updates are never substituted for adjacent losses. All ratios are floating-point evaluations, not interval certificates.
