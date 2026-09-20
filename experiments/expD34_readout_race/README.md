# Ordinary-GD readout competition

This experiment tests whether readout adaptation depletes residual moments before a substantial population of hidden slopes can grow. The evidence sought is an initialization-only prediction of signed scale trajectories and their changes across readout rates. Successful execution does not require the hypothesis to hold.

**Notation.** These symbols refer to physical raw coordinates throughout.

| Symbol | Meaning |
|---|---|
| $f=d+\sum_j c_j\tanh(a_jx+b_j)$ | Network with independently trained slopes, hidden biases, readouts, and output bias. |
| $\gamma_j=\lvert a_j\rvert$ | Reported slope magnitude; neuron indices and live signs are preserved. |
| $\lambda_j=(2/N)\gamma_j$ | Reporting scale tied to the reference budget, not to sample or evolving-center spacing. |
| $\eta$, $\kappa$ | Geometry rate and readout/geometry rate ratio. Output bias uses the readout rate. |
| Polynomial reference | Independently evolved degree-1/3/5/7 activation, using only its own state and fixed target moments. |

## Locked protocol

All four parameter blocks update simultaneously from the old state, using the loss $L=\tfrac12\operatorname{mean}(f-y)^2$. There is no momentum, normalization of training residuals, refit, clipping, freeze, parameter remapping, or schedule. Reported MSE is $2L$.

The main matrix uses $(N,H,W)=(64,12,89),(128,24,177),(256,48,353)$, seeds 0–4, and $\kappa\in\{10^{-4},10^{-3},10^{-2},0.1,1,10,100\}$. The geometry rate is $0.002$. All finite scientific trajectories receive at least 20,000 updates. Training has 2,048 midpoints on $[-1,1]$ at every width; evaluation uses 8,192 different midpoints. Two additional seed-0, width-177 baseline comparisons retain D28's 1,024 training samples for sine and Runge. Independent evaluation diagnoses interpolation error and is not used to choose a winning optimizer, rate, or checkpoint.

Initialization calls the unchanged D28 helper. Slopes and hidden biases are independent uniform Xavier draws with bound $\sqrt{6/(W+1)}$ from NumPy's stream seeded by `[seed, N]`. Readouts use a separate `[seed, N, 24]` stream and the same bound. Output bias is zero. There is no tanh gain, readout norm rescaling, reference-envelope multiplier, or sign canonicalization. Copies of these physical arrays initialize every paired rate and reference.

Experiment A uses the upstream sine and Runge target amplitudes. Experiment B uses empirical orthonormal Legendre polynomials fitted once on the training grid:

$$
y_k=0.3\phi_0+0.4\phi_1+\sqrt{0.75}\phi_k,\qquad k=3,5,9.
$$

Every target has training RMS one and the same first two target coefficients. The saved polynomial coefficient map evaluates the same function on independent grids. The affine references must agree across all three targets; the cubic references must agree for $k=5,9$. Actual tanh trajectories can differ immediately. The main matrix contains 525 tanh and 2,100 reference trajectories.

## Mathematical verification

`references.py` computes polynomial coefficients and the exact moment-gradient pullback. Its separate affine Gram recurrence retains the quadratic finite-step term and reconstructs individual states with a common $3\times3$ map. Higher-order references retain all parameters. `core.py` implements simultaneous GD and analytic tanh gradients; its stable expression for $\operatorname{sech}^2$ avoids saturation cancellation. FP64 is enabled by the experiment entry point or test environment, not as an import side effect of the kernels.

Tests compare the analytic gradients with independent sample-space differentiation, the affine recurrence with explicit polynomial GD at extreme rates, the initialization and early trajectory with the unchanged PyTorch D28 trainer, and the state-conditioned moment remainder bound. A reference loss computed from moments can suffer cancellation; it must be checked against direct sample evaluation at saved states and never silently clipped as a training objective.

## Execution and evidence

GPU computation must run inside Slurm on the allocated devices, with at most two H200s concurrently. The initial allocation budget is two GPU-hours. Core comparisons precede half-step, equal-geometry-time refinements. Remaining time funds balanced width-177 continuations toward 100k, seeds 5–9, then further common-horizon continuations. A finite trajectory interrupted by the allocation remains incomplete, not converged. Polynomial breakdown does not terminate its independently trained tanh counterpart.

The diagnostic record includes signed mean and median scale changes, quantiles, population thresholds, residual moments, coarse and remainder signed slope forces, parameter gradients and actual changes, readout norms, full states, reference prediction errors, and independent-grid errors. Hypothesis assessment must use paired rate contrasts and reference-validity checks, not just declining loss or a few large slopes. Implementation and analysis commands will accompany the runner; numerical findings belong in the results report after execution.
