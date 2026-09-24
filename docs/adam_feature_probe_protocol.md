# Width-512 Adam assay of acquired features

The question is whether joint training acquires features that make its target easier to learn. We train fresh width-512 networks, freeze their features at recorded checkpoints, restart their readouts, and compare with uniform-center dictionaries under the same readout budget. Cosine decay spans each entire run. Every halo feature counts toward the total hidden width.

## Protocol fixed before execution

- Set the **total hidden width to 512**. Invert the existing D06 halo rule $r=\lceil\sqrt{N}\rceil$ subject to $N+1+2r=512$: $N=467$ intervals, 468 centers on $[-1,1]$, and 22 halo centers on each side. The spacing is $h=2/467$; centers are $-1+jh$ for $j=-22,\ldots,489$. This preserves the frozen-kernel study's halo convention; it is not a claim that the separate uncorrected QI construction's default halo policy is identical.
- Use the existing D34 target $[\sin(2\pi x)+0.5\sin(6\pi x)+0.25\sin(14\pi x)]/0.8100925873009825$, with 2048 midpoint training samples on $[-1,1]$.
- Train five paired affine-Xavier seeds, with identical initial parameters across schedules and learning rates. Use the existing D24 initialization convention with the budget-derived grid resolution. Optimize all 1537 scalar parameters for 200,000 updates with Adam, at initial learning rates $0.0002,0.002,0.02$, each under constant and full-horizon cosine schedules. Preserve every raw error and parameters every 10,000 updates.
- Select one joint-training recipe per seed by final error on a 4096-midpoint validation grid. Freeze that trajectory's dictionaries at 20,000, 100,000, and 200,000 updates, together with the shared initialization. Selection uses the endpoint, so intermediate dictionaries describe retrospectively selected trajectories. Retain every joint recipe's traces and explicitly show the shared $0.002$ recipe separately.
- Compare seven uniform-center dictionaries with $\lambda=\gamma h\in\{0.03125,0.0625,0.09375,0.125,0.25,0.5,1\}$. Centers remain fixed across slopes. Always record physical $\gamma$, dimensionless $\lambda$, spacing, and total width.
- In every frozen assay reset readout coefficients, output bias, and Adam moments to zero. Preserve raw feature coordinates. Use full-batch half-MSE, FP64, $\beta_1=0.9$, $\beta_2=0.999$, and $\epsilon=10^{-8}$ in both joint and frozen runs.
- Run each frozen readout for 200,000 updates for each initial learning rate $10^{-5},10^{-4},10^{-3},0.002,0.01,0.1$, under both constant and cosine schedules. At update index $n=0,\ldots,199999$, cosine uses $\eta_n=\eta_0[1+\cos(\pi n/200000)]/2$. There is no early floor or pilot continuation.
- Preserve every frozen update's raw relative output error, every 1000th readout, and the final optimizer state. Show the shared $\eta_0=0.002$ comparison separately from equally tuned comparisons.
- Select one readout learning rate per geometry and schedule by final validation-grid error. Plot that fixed recipe's whole curve, never a pointwise best envelope. Evaluate selected endpoints on the 8192-midpoint grid. This grid is independent of training and selection, but has been inspected in previous work; it is a resolution check rather than a new untouched statistical test set.
- Measure residual energy along all three target harmonics and the orthogonal remainder; the components sum to squared relative output error. Tolerance crossings are secondary summaries.

The frozen comparison contains 27 dictionaries and 12 recipes each; joint training has 30 cases. Short timing runs preserve the full schedule clock and are discarded. Production restarts from initialization. Benchmark before production, use at most two allocated GPUs, and record actual compute costs. Historical width-177 and width-559 data remain historical; no padding or relabeling makes them width 512.

## Interpretation

Initial-versus-later frozen dictionaries test feature acquisition. Learned-versus-uniform dictionaries test feature quality for this target, width, and budget. Neither establishes a universal bandwidth threshold, nor does the frozen-feature GD theorem give an Adam rate law. Improved reset-readout training need not explain the original joint trajectory: readout history and feature motion also matter.

All seeds, learning rates, instability, and unfavorable slope orderings remain in the evidence. The three target components contain $16/21$, $4/21$, and $1/21$ of its energy; dropping the highest harmonic alone leaves about 21.8% relative output error. The target is inherited from the existing joint experiment, not selected for this comparison.
