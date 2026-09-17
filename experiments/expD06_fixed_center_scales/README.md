# Fixed-center bandwidth acquisition

This experiment tests whether readout and bandwidth update scales change acquisition of useful tanh geometry. It uses JAX and Optax in FP64, with fixed grid and halo centers, independent signed slopes, and paired physical initialization across Raw and Both coordinates. The reference bandwidth is $\lambda=0.25$ and the halo radius is $R=\lceil\sqrt N\rceil$.

Install the optional `jax-experiments` dependency group for CPU development. On the remote H200 host, use an isolated environment with the CUDA 12 JAX extra and record its exact installed versions. This experiment does not change the repository's existing PyTorch runners.

The two parameter blocks minimize one mean squared error objective. Their separate learning rates control their relative contributions to learning. Every finite scientific run must receive at least 20,000 updates; rates remain fixed within a run. Shorter executions are implementation checks or throughput benchmarks, never scientific comparisons or rate-selection evidence.

`core.py` defines the reference envelopes, paired initialization, coordinate maps, and compiled training chunks. The tanh derivative uses the equivalent stable $4e^{-2|u|}/(1+e^{-2|u|})^2$ expression to avoid cancellation in $1-\tanh^2 u$.

Focused verification: `python -m pytest -q tests/test_expD06_fixed_center_scales.py`.

## Diagnostic conventions

The training residual and feature matrix include the $1/\sqrt M$ normalization. Diagnostic refits use `scipy.linalg.svd` with the `gesdd` driver on $AD_{\rm ref}$ and retain singular values above $10^{-12}\sigma_{\max}$. The fitted coefficients are exported back to physical readout units. The retained projector describes numerically accessible directions, not the exact full feature span.

Fourier bands use an orthonormal DFT on the actual training sample vector. Its implied period is $M\Delta x$, including the endpoint convention. DC and paired positive/negative dyadic bands partition the vector exactly. The band gradients are $J_\lambda^T Q_b r$, $(P_\tau J_\lambda)^TQ_b r$, and $((I-P_\tau)J_\lambda)^TQ_b r$. This order makes both the frequency sum and the within-band parallel/perpendicular sum reconstruct the original gradient without assuming the two projectors commute.

Detached analyses preserve the live residual, its two projected components, the actual refitted residual, complex spectra, per-neuron band gradients, unit-RMS sine/cosine sensitivity probes, and signed forces in the collective magnitude-increase direction. The readout correction from a full truncated refit is recorded separately from $P_\tau r$: discarding a live coefficient component can make them differ. A Hann-window residual spectrum is a boundary-sensitivity diagnostic and does not change the objective.
