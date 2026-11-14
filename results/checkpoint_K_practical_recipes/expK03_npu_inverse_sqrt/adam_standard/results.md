# Conventional Adam baseline: replacement for the rejected multiscale initialization

**The random-multiscale result is excluded from this baseline and from the current plot.** It is preserved only as an archived exploratory result.

Selected: **pytorch_lr3e-02_batch1024**, 128 neurons / 385 parameters, checkpoint **94,000**, relative L2 **0.753494153**.

## Allowed techniques

- Unmodified PyTorch Linear initialization, or library Xavier uniform (tanh gain 5/3 in the hidden layer, gain 1 in the linear readout, zero biases). Initial hidden weights are at most 1 in magnitude and independent of epsilon and the target function.
- Ordinary 128-neuron single-hidden-layer tanh MLP. The same fixed affine input centering as before; no log inputs, engineered feature scales, center placement or activation reparameterization.
- Raw unweighted MSE on exactly the same C17 halo-center labels. No output gain or target rescaling.
- Adam rates 0.001, 0.01, 0.03; batches 100 or 1024; rate held for 50k updates then cosine-decayed to 1e-5 by 100k. One fixed seed 0, float64, no clipping or weight decay.

This makes 12 new 100k-step runs. Four earlier eligible standard 128-neuron runs (three constant rates and one batch-512 cosine run) remain candidates; the selection does not throw away a better allowed result. All selected checkpoints must have at least 50k updates. Only order-32 validation scores select a model; independent order-256/512 quadrature determines the reported test error.

| Trial | Init | Batch | Peak LR | Selected step | Test relative L2 |
|---|---|---:|---:|---:|---:|
| pytorch_lr1e-03_batch100 | pytorch | 100 | 0.001 | 84,500 | 1.72593593 |
| pytorch_lr1e-02_batch100 | pytorch | 100 | 0.01 | 89,500 | 1.07317224 |
| pytorch_lr3e-02_batch100 | pytorch | 100 | 0.03 | 89,000 | 0.787063487 |
| xavier_uniform_tanh_lr1e-03_batch100 | xavier_uniform_tanh | 100 | 0.001 | 89,500 | 1.84246618 |
| pytorch_lr1e-02_batch1024 | pytorch | 1024 | 0.01 | 87,500 | 0.90894628 |
| pytorch_lr1e-03_batch1024 | pytorch | 1024 | 0.001 | 93,000 | 1.4248772 |
| xavier_uniform_tanh_lr1e-02_batch100 | xavier_uniform_tanh | 100 | 0.01 | 89,500 | 1.07403598 |
| pytorch_lr3e-02_batch1024 | pytorch | 1024 | 0.03 | 94,000 | 0.753494153 |
| xavier_uniform_tanh_lr3e-02_batch100 | xavier_uniform_tanh | 100 | 0.03 | 82,000 | 0.864460362 |
| xavier_uniform_tanh_lr1e-03_batch1024 | xavier_uniform_tanh | 1024 | 0.001 | 93,000 | 1.33337884 |
| xavier_uniform_tanh_lr1e-02_batch1024 | xavier_uniform_tanh | 1024 | 0.01 | 77,500 | 0.924282896 |
| xavier_uniform_tanh_lr3e-02_batch1024 | xavier_uniform_tanh | 1024 | 0.03 | 94,000 | 0.80263996 |
| prior_pytorch_lr0.01 | pytorch | 100 | 0.01 | 96,500 | 1.03421631 |
| prior_pytorch_lr0.001 | pytorch | 100 | 0.001 | 93,000 | 1.63391613 |
| prior_pytorch_lr0.0001 | pytorch | 100 | 0.0001 | 100,000 | 3.0757644 |
| standard_batch512_decay | pytorch | 512 | 0.01 | 92,000 | 0.983073801 |

## Verification and scope

All 16 eligible candidates were reloaded. Initial tensors match the corresponding standard initializer exactly and do not change when epsilon is changed. Every new run completed 100k updates. Checkpoint selection, independent test quadrature, and bitwise equality of the halo labels were checked. Fourteen implementation tests passed, including standard-library Xavier equivalence and domain independence.

The target, halo-center dataset (N=256, R=32), and physical dx relative-L2 metric on [0.0001,1] are unchanged. Training still uses unweighted halo-sample MSE, whose domain/weights differ from the evaluation metric. This is a conventional Adam baseline under the stated protocol, not a lower bound on all possible training. NPU/RealNPU curves and their earlier budgets are unchanged.

See specification.json, summary.csv, selected_reference.json, validation.json, and all per-trial configs, initial models, optimizer checkpoints, histories and predictions.
