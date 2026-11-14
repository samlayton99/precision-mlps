# QUILL versus Neural Power Units on the C17 extension

The requested overlay is `inverse_sqrt_npu_comparison.png` / `.pdf`. It shows
only matched-halo baselines. All reported errors below are physical-domain
relative L2 on [1e-4,1], in float64.

| Method | Model size | Best validation checkpoint: test error | Step | Final error at 20k |
|---|---:|---:|---:|---:|
| QUILL | 4,815 tanh neurons, 14,446 parameters | 4.95315e-14 | no training | — |
| NPU | 6 power units, 19 parameters | 0.585038 | 16,750 | 0.687816 |
| RealNPU | 6 power units, 13 parameters | 0.255534 | 750 | 1.15755 |
| Adam MLP | 4,815 tanh neurons, 14,446 parameters | 2.55313 | 19,500 | 2.73513 |

The NPU models use the paper's six-unit power layer followed by its unregularized,
bias-free linear NAU, adapted from two inputs/four outputs to one input/output.
Glorot uniform real/readout weights, zero imaginary weights, gates 0.5; Adam
learning rate 0.001, batch 100, 20,000 updates, seed 0, no warmup or decay. This
follows the paper's stated rate; the released script uses 0.005. Float64 is an
adaptation to match QUILL, not the original Julia Float32 experiment.

The exact same `Extension(eps=1e-4, order=16)` callable supplies all labels.
Training samples are drawn from the same 4,815 physical QI block sampling nodes
as the selected construction, including both halos and repeated nodes. The node
range is [-0.4096,3.6864]. QUILL's relative error on these extension-node targets
is 4.16594e-14, independently checked. No changed cutoff, truncated target, or
alternate extension is used for NPU.

The width sweep is QUILL's existing unsmoothed data. NPU horizontal lines are
fixed six-unit references with the halo matched to the 4,815-neuron construction;
they are not a trained NPU width sweep. QUILL accesses analytic derivatives and
an anchor, while NPU trains on function values. Thus the function, halo geometry,
precision, evaluation interval and metric match, but the architecture, parameter
budget, and information supplied to the methods do not. The result establishes
the measured gap for this protocol, not the best possible NPU result.

Validation checkpoints were logged every 250 steps, using independent order-32
quadrature. Best-checkpoint selection for the plotted references was chosen
after running to show the strongest measured baseline checkpoint. The original
final-iterate results are retained above. Orders 256 and 512 reproduce baseline
errors to roundoff; QUILL varies from 4.95315e-14 to 5.01167e-14 at its numerical
floor. Saved checkpoints were reloaded and their reported scores reproduced.

Seven implementation tests pass: independent published-formula evaluation,
gradient checks, one-unit exact-power representation, clipping behavior,
NPU/RealNPU agreement on positive inputs, halo-node equality and shared physical
L2 scoring, and optimizer checkpoint continuation (some tests cover two checks).

Files: `summary.json`, `validation.json`, `provenance.json`, `figure_provenance.json`,
`matched_halo_data.npz`, per-model histories/predictions, initial and 5k-interval
optimizer checkpoints, and best-validation model checkpoints. Downloaded Julia
reference sources are preserved under `reference/` with source URLs and hashes.
Previously completed interval-only controls are retained in raw files but are
excluded from this requested figure and table.

## Added ordinary Adam reference

The Adam MLP matches the selected QUILL network's architecture, width, parameter
count, affine input centering, and float64 precision. All weights and biases are
trained, starting from PyTorch's default Linear initialization with seed 0. Its
target values and halo nodes are bitwise equal to those in the NPU runs. It uses
the same minibatch RNG seed, batch 100, Adam lr=0.001, 20k updates, no warmup,
regularization or decay, and the same independent validation and test rules.

The initial relative L2 is 1.05718, better than every logged trained checkpoint.
The overlay explicitly selects the best **positive-step** validation checkpoint
for all trained baselines. For Adam this is step 19,500, error 2.55313; NPU and
RealNPU selections are unchanged. The final error is 2.73513. This single run
does not improve the original-interval metric over initialization; its finite
halo-node training objective is different from that evaluation metric. Do not
describe the initialization's 1.05718 error as an attained trained solution.

To recover the best trained checkpoint, resume the saved step-15k optimizer/RNG
state, retain the step-19.5k model, and continue to step 20k. The replay's final
parameters are bitwise equal to the original final checkpoint. Order-256/512
test quadrature checks the recovered model. Details are in
`adam_mlp/recovery_validation.json`; checkpoints/history/data are in `adam_mlp/`.
Nine implementation tests passed, including the MLP architecture/input-map
equivalence and bounded-memory evaluation checks.
