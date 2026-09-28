# D34 effective-force extension: implemented experiment plan

The question is whether the GD observation survives changes in target and optimizer: after the affine transient, does the effective fine-residual force account for slope motion? For Adam, separate what is present in the raw gradient from what survives momentum and coordinate scaling. Large path length does not establish outward movement, population acquisition, or useful target access.

The training matrix and online measurements are fixed in commit `78c6e1b`. This document records that design and the subsequent analysis, rather than presenting a new preregistration. The [source guide](../experiments/expD34_readout_race/README.md) gives executable entry points. The [completed evidence package](../results/checkpoint_D_optimizers/expD34_readout_race/adam_force_extension/README.md) reports the outcomes.

## Comparisons and resource allocation

All cases use width 177, D34's random nonzero readout initialization, physical coordinates $(a,b,c,d)$, half empirical MSE, 2,048 fixed training midpoints, and 600,000 simultaneous full-batch updates. Evaluation uses 8,192 separate midpoints. No clipping, weight decay, schedule, checkpoint selection, or early stopping is applied.

| Comparison | Cases | Purpose |
|---|---:|---|
| Thirteen targets, seeds 0–4, GD and Adam at 0.002 | 130 | Paired primary comparison |
| All targets, seed 0, Adam at 0.0002 and 0.001 | 26 | Sensitivity to nominal learning rate |
| Degree 3, degree 9, mixed sine, chirp; seeds 0–2; EMA-only and adaptive-only | 24 | Separate first-moment memory from adaptive scaling |
| The same four targets, seed 0, Adam epsilon $10^{-12}$ | 4 | Sensitivity to the denominator floor |

Adam uses $(\beta_1,\beta_2)=(0.9,0.999)$, epsilon $10^{-8}$, and zero initial moments. EMA-only uses the bias-corrected first moment with denominator one. Adaptive-only sets $\beta_1=0$. Equal nominal rates do not imply equal coordinatewise steps or equal gradient-flow time.

The extension has a six allocated GPU-hour ceiling inside the earlier eight-hour allowance. Run two one-GPU Slurm tasks concurrently while two tasks remain; the per-user QoS enforces the global two-GPU limit. Include pilots, compilation, and unsuccessful allocations in accounting. All training ran on Runpod. Remaining CPU analysis, rendering, and tests also run through Runpod Slurm; local use is limited to editing, file transfer, and inspection.

## Targets

Preserve the original sine $\sin(2\pi x)$, Runge $1/(1+25x^2)$, and coarse-plus-degree-3, 5, and 9 targets exactly. Here $q_k$ is the degree-$k$ empirical orthonormal polynomial on the original 2,048-point grid.

Add the following:

- Mixed sine: $\sin(2\pi x)+0.5\sin(6\pi x)+0.25\sin(14\pi x)$.
- Localized sine: that mixture multiplied by $\exp[-(x/0.4)^2/2]$.
- Chirp: $\sin[2\pi(u+4u^2)]$, $u=(x+1)/2$.
- Even-degree control: $0.3q_0+0.4q_1+\sqrt{0.75}q_4$.
- Four controlled mixtures: $0.3q_0+0.4q_1+\sqrt{0.75}[s q_3+\sqrt{1-s^2}q_9]$, with $s=-0.1,0.01,0.1,0.3$. The original degree-9 and degree-3 targets supply the endpoints $s=0,1$.

Normalize the three new nonpolynomial targets to unit RMS on the original training grid, without subtracting their mean. Reuse that normalization and polynomial map on every evaluation or refined grid. The mixture sweep preserves total target energy and its two affine coefficients; inspect the degree-9 residual separately so fitting the added cubic cannot masquerade as fitting the difficult component.

## Measurements that distinguish the mechanisms

Let $J_C$ be the two-row constant/linear Jacobian with respect to **all** parameters, and $e_C$ the corresponding residual coefficients. Use the entire orthogonal residual complement online:

$$
g_H=g-J_C^Te_C,\qquad C=J_CJ_C^T,\qquad b=C^{-1}J_Cg_H,
$$
$$
g^{\rm eff}=g_H-J_C^Tb,\qquad
g^{\rm track}=J_C^T(e_C+b),\qquad
g=g^{\rm eff}+g^{\rm track}.
$$

Their slope blocks are the effective fine force and coarse-tracking correction. No Taylor approximation of tanh or finite residual truncation enters training or these online measurements. An unresolved coarse solve goes into an explicit unknown channel; it does not change the optimizer.

Maintain a first-moment history for each channel and apply the actual full-gradient Adam denominator to every channel. This gives an additive accounting of actual steps. Running separate second-moment streams would describe different optimizers and would fail this purpose. At saved states, also inspect the instantaneous balance defined by $C_P=J_CPJ_C^T$ with the actual diagonal preconditioner $P$. Retain momentum lag and finite-step defect; this balance is not an assumed Adam equilibrium.

Record four different links:

1. Raw force, first moment, scaled current gradient, and actual step contributions.
2. Every-update component norm budgets, signed outward motion, positive/negative neuronwise travel, and the exact remainder when a signed slope crosses zero.
3. Slope distributions and acquired populations, retaining actual endpoints and all failures.
4. Actual fit and a common frozen-geometry assay: reset the diagnostic readout to zero and run readout GD at 0.002 through 600,000 updates. Cross old/new slopes and biases to inspect their interaction. This assay is separate from the actual trained readout and is not a universal approximation threshold.

Useful possible outcomes include persistence of effective-force dominance, amplification of the tracking channel by the optimizer, and substantial motion that mostly cancels or fails to improve target access. These outcomes can coexist in different windows or targets. Do not force them into a single success/failure label.

## Verification and interpretation gates

Compare analytic gradients and coarse Jacobians with independent autodiff. Check actual Adam updates against PyTorch, including cancellation and tiny gradients; check exact continuation of all optimizer and diagnostic buffers. Replay all 25 original GD target/seed cases against archived states.

Compare retained residual degrees 65 and 129 with the full-residual force. At fixed seed-0 endpoints, double the training and evaluation quadratures while preserving target definitions; refine again when the effective-force difference exceeds one percent. These fixed-state checks do not substitute for a retrained quadrature study.

Inspect every target and seed. Use exact online sums for motion-window claims, and label sampled force traces as sampled observations. A terminal-state diagnostic of the *next* Adam update is virtual; it is not part of the 600,000 trained updates. Report finite-horizon mechanism evidence, target/rate dependence, and numerical limitations without turning an observed small force into an initialization-only trapping theorem.
