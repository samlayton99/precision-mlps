# D57 — Two-function GELU scale-acquisition comparison

Status: both 200k-step sweeps running on the Mac mini, November 11, 2026. All 36 100-step pilot runs passed. The first matched 5,000-step round also completed for all 36 runs without nonfinite states or numerical errors. No final scientific conclusion yet.

## Question and design

Does the scale-acquisition pattern persist for exact GELU under GD and Adam? Two targets are fixed in advance: sin(4πx) and 1/(1+100x²). Each optimizer has 18 runs at width 196: two targets, three learning rates (0.01, 0.001, 0.0001), and first-layer initialization multipliers 1, 2, 4. All runs use the same seed, normalized targets, sample grids, and 200k budget as D56.

The [experiment README](../../../experiments/expD57_gelu_scale_acquisition/README.md) records the full protocol, default Adam settings, reference-line provenance, and execution commands.

## Results and figures

Live data: `sweep/gd/runs/` and `sweep/adam/runs/`. The runner refreshes both figures after each 5,000-step round:

- `sweep/gd/figures/scale_acquisition.png`: latest GD progress, then complete GD results.
- `sweep/adam/figures/scale_acquisition.png`: latest Adam progress, then complete Adam results.

During training these images explicitly say partial data and give the displayed versus planned step count; `_PARTIAL` copies are also retained. The stable filenames receive complete 200k data once each optimizer finishes.

The primary curves are per-function medians of absolute neuron slopes, with 1st–99th percentile bands over the 196 neurons. Colors identify learning rates and shades identify functions. The comparison QUILL/GELU slope reference is 51.9645, following the repo's lambda=0.707 convention. It is not an accuracy threshold.

## Verification

Three numerical tests passed before the full sweep: native exact-GELU gradients, PyTorch-equivalent GD/Adam updates plus exact stateful resume, and matrix/initialization matching. Full parameters and Adam moments are saved in checkpoints. The unprivileged parallel pilot encountered an OpenMP shared-memory restriction before its workers started; those infrastructure errors are separate from numerical training failures.

## Interpretation

Pending measured trajectories. This extension tests empirical relevance across activations and optimizers; it does not extend the tanh GD theorem to GELU or Adam. Runs that diverge are retained rather than retuned away.
