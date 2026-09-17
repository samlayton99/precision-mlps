# Fixed-center bandwidth acquisition

This experiment tests whether readout and bandwidth update scales change acquisition of useful tanh geometry. It uses JAX and Optax in FP64, with fixed grid and halo centers, independent signed slopes, and paired physical initialization across Raw and Both coordinates. The reference bandwidth is $\lambda=0.25$ and the halo radius is $R=\lceil\sqrt N\rceil$.

Install the optional `jax-experiments` dependency group for CPU development. On the remote H200 host, use an isolated environment with the CUDA 12 JAX extra and record its exact installed versions. This experiment does not change the repository's existing PyTorch runners.

The two parameter blocks minimize one mean squared error objective. Their separate learning rates control their relative contributions to learning. Every finite scientific run must receive at least 20,000 updates; rates remain fixed within a run. Shorter executions are implementation checks or throughput benchmarks, never scientific comparisons or rate-selection evidence.

`core.py` defines the reference envelopes, paired initialization, coordinate maps, and compiled training chunks. The tanh derivative uses the equivalent stable $4e^{-2|u|}/(1+e^{-2|u|})^2$ expression to avoid cancellation in $1-\tanh^2 u$.

Focused verification: `python -m pytest -q tests/test_expD06_fixed_center_scales.py`.
