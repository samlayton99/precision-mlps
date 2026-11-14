# NPU on the QUILL inverse-square-root target

**Current Adam baseline policy:** standard PyTorch or standard Xavier initialization
only. The user rejected the random-multiscale initializer as inappropriate for
this baseline. Its results remain archived under `adam128_refine/` but are excluded
from the current figure. The renderer reads `adam_standard/selected_reference.json`
and rejects multiscale initializers and output-gain reparameterizations.

Local float64 PyTorch adaptation of Heim, Pevný, and Šmídl, *Neural Power Units*
(NeurIPS 2020), section 4.2. This is a new experiment on our function/domain, not
a reproduction of their published table values.

**Current requested figure:** `inverse_sqrt_npu_comparison.png` / `.pdf` uses only
the matched-halo runs. After the user's scope correction, interval-only controls
were excluded from presentation; their already-completed files remain saved for
provenance. The source config now reproduces only the matched-halo runs. The
frozen results config records the full run that actually happened.

The overlay uses each baseline's **best saved validation checkpoint**, selected
on order-32 quadrature and scored separately with order-256/512 quadrature. This
presentation choice was made after running and disclosed to the user; final
20,000-step measurements remain in the result table. It is deliberately more
favorable to the baselines than showing just their worse final iterates.

## Protocol fixed before training

- Target: the **same `Extension(eps=1e-4, order=16)`** from expK02, a bounded C17
  extension of `1/sqrt(x)`. No new smoothing, clipping of target values, or
  different extension. Inputs to NPU are physical x: its architecture internally
  uses log/exp/cos; QUILL's affine centering is an implementation coordinate.
- Architecture: `NPU(1,6) -> NAU(6,1)`, or its RealNPU variant, following the
  released simple-task script. The reference NAU is a plain unconstrained linear
  map, without a bias. NPU has 19 parameters; RealNPU has 13.
- Glorot uniform real/readout weights, imaginary weights zero, gates 0.5.
  Forward clips gates into [0,1]; parameters themselves are not projected.
- One fixed seed (0), as requested. Adam, 20,000 steps, batch 100, learning rate
  0.001, no warmup/decay/regularization. The paper states 0.001; the released
  script instead says 0.005. We follow the paper, and do not select rates/seeds.
- Float64 matches QUILL; the original code uses Float32. NPU's additive input
  epsilon is `finfo(dtype).eps`, exactly as in the reference method.
- **Matched-halo arm:** all 4,815 physical sample locations used by the fixed
  QUILL reference with N=256 and halo R=32 at each of its 15 scales. These are
  `x[j,n]=4*s[j]*n/256`, n=-32,...,288, including repeated locations. Their union
  spans [-0.4096,3.6864]. Targets are the exact same C17 extension evaluated at
  those locations. Draw 100 indices with replacement per training step.
- **Interval-only control:** fresh uniform physical inputs on [1e-4,1], with
  the same extension callable as target. This checks whether the halo itself
  disadvantages this architecture. NPU/RealNPU share initialization and batches.
- Evaluate both arms on [1e-4,1] using **physical dx relative L2**, exactly as
  expK02: dyadic log-coordinate integration is a quadrature change of variables,
  not a log-input transform. Final evaluations use orders 256 and 512 per panel.
- Primary results are final checkpoints at 20,000 updates. Best validation
  checkpoints (independent order-32 quadrature) are saved as secondary results.

## What this controls, and what it does not

Target, extension, scored interval, precision, and error metric are shared.
The halo locations match the **4,815-neuron reference**, not every width in the
QUILL scaling sweep. QI gets analytic derivative samples and a function anchor;
NPU trains on function values. This is a comparison of the requested methods,
not identical information access or identical architectures/parameter budgets.
Horizontal NPU references on the width plot are fixed six-unit models, not a
claim that NPU was trained at every plotted QUILL width. One seed cannot estimate
success rates, or establish a method's best achievable accuracy.

NPU's positive-input branch with imaginary weights initialized at zero has zero
imaginary gradient, so matched NPU/RealNPU interval runs should agree. Signed
halo inputs can activate the imaginary branch. We check this rather than
adding noise to deliberately break the published initialization.

## Implementation checks

`tests/test_neural_power_unit.py` compares against the independent NumPy form of
the published complex equations; finite-difference checks parameter gradients;
checks one-unit inverse-square-root representation, gate clipping, parameter
counts, positive-input NPU/RealNPU equality, exact QI halo coordinates, the
physical L2 metric, and bitwise Adam checkpoint continuation.

This experiment uses stock Adam, not a new optimizer mechanism. One forward and
backward per step; two parameter-sized moment buffers; no Krylov solve, curvature
probe, adaptive control, or dataset-sized optimizer state. Saved training nodes
are the finite target dataset. The broader optimizer-program litmus tests do not
apply to reproducing an architecture-specific published baseline.

## Reproduce

```sh
~/venv/quill-radon/bin/python -m pytest -q tests/test_neural_power_unit.py
~/venv/quill-radon/bin/python experiments/expK03_npu_inverse_sqrt/run.py
~/venv/quill-radon/bin/python experiments/expK03_npu_inverse_sqrt/plot.py
```

Results and checkpoints: `results/checkpoint_K_practical_recipes/expK03_npu_inverse_sqrt/`.
Downloaded reference Julia sources and SHA-256 hashes live in its `reference/`
directory. Existing results are preserved: use a new `--output` to repeat runs.

Measured best-checkpoint relative L2: NPU 0.585038 at step 16,750; RealNPU
0.255534 at step 750. The 4,815-neuron QUILL reference is 4.953e-14 (order 256)
and 5.012e-14 (order 512). These are the results of the stated single-seed budget,
not a lower bound on NPU accuracy or a matched-parameter comparison.

## Ordinary Adam baseline

`config_adam.json` adds an ordinary `Linear(1,4815) -> Tanh -> Linear(4815,1)`
network, with all 14,446 parameters trainable. This **matches the selected QUILL
architecture and parameter count**, with the same affine input centering. It
starts from PyTorch's default Linear initialization (seed 0). The target, halo
nodes, minibatch RNG seed, batch size, Adam settings, 20k-step budget, precision,
validation grid, test metric and best-checkpoint selection match the NPU runs.
No geometry, readout coefficients, or initialization are taken from QUILL.

```sh
~/venv/quill-radon/bin/python experiments/expK03_npu_inverse_sqrt/run.py --config experiments/expK03_npu_inverse_sqrt/config_adam.json --output results/checkpoint_K_practical_recipes/expK03_npu_inverse_sqrt/adam_mlp
~/venv/quill-radon/bin/python experiments/expK03_npu_inverse_sqrt/recover_best_trained.py
~/venv/quill-radon/bin/python experiments/expK03_npu_inverse_sqrt/plot.py
```

The plotted Adam reference is a fixed-width trained model, not a width sweep.
Its checkpoints and history are saved separately in `adam_mlp/`. Nine tests now
also check the ordinary MLP's equivalence to QUILL's network evaluation and the
batched evaluation used to bound memory at this width.

The Adam run's best validation score including initialization occurs at step 0.
The displayed reference uses the best **trained** checkpoint instead: step 19,500,
relative L2 2.55313. This positive-step restriction also applies to NPU/RealNPU
and leaves their selected checkpoints unchanged. `recover_best_trained.py`
recovers that saved-history optimum from the step-15k checkpoint and checks
bitwise agreement with the original final parameters after replay to 20k.

## NPU capacity sweep

`width_sweep.py` expands NPU to widths 6, 32, 128, 512, 2,048, and 4,815, and
RealNPU to the same widths plus 7,222. The parameter counts are 3h+1 and 2h+1,
respectively: the largest models have 14,446 and 14,445 parameters, essentially
matching the selected QUILL model's 14,446. The existing six-unit runs are reused.

The same C17 target, fixed 4,815-point QUILL halo dataset, minibatch sequence,
one seed, Adam rate, 20k-step budget, and validation/test metric apply to every
width. Neither halo extent nor target weighting changes with NPU capacity.
The best **positive-step** validation checkpoint is saved automatically, along
with the final model and optimizer states at each 5k checkpoint. No learning-rate
or seed search is performed; this is capacity scaling at a fixed training recipe,
not a claim that every width has exhausted its optimization potential.

```sh
~/venv/quill-radon/bin/python experiments/expK03_npu_inverse_sqrt/width_sweep.py --workers 2
~/venv/quill-radon/bin/python experiments/expK03_npu_inverse_sqrt/validate_width_sweep.py
~/venv/quill-radon/bin/python experiments/expK03_npu_inverse_sqrt/plot_width_sweep.py
```

Outputs are under `width_sweep/` in this experiment's results directory. The
capacity figure uses **total parameter count**, so NPU and RealNPU appear as
measured curves rather than horizontal references. Adam MLP remains a single
measured point. The QUILL data are the original unsmoothed construction sweep.
Its halo matches the baseline exactly at the 4,815-neuron reference; other QUILL
widths retain their original 32-cell halo recipe. The shared extension itself
and evaluation domain are unchanged. QUILL still uses derivative information,
while the trained baselines use function values.

`validate_width_sweep.py` reloads every selected checkpoint, checks the parameter
counts and validation-selection steps, recomputes errors at quadrature orders
256 and 512, and verifies bitwise equality of the shared target dataset. Ten
implementation tests cover the runner's positive-step checkpoint selection as
well as the prior model/metric/halo checks.

## Identical construction widths (current requested comparison)

The user subsequently requested **hidden widths**, not parameter-count matching,
at every original construction point. `matched_width_sweep.py` runs both NPU
variants at all 22 widths in `scaling_1k_5k.json` (990 through 5,055). At each
width, N and R match that QUILL model, so the halo-center grid changes exactly
with the construction. The two existing h=4,815 runs remain reusable; 42 new
runs supply the remaining points. Four one-thread CPU jobs run concurrently.

Here “matching the halo grid” means the mathematical **neuron-center grid**,
`x[j,n]=4*s[j]*n/N` for n=-R,...,N+R. The cardinal construction additionally
evaluates derivatives on Kc=160 stencil-padding locations on each side. All
those extra locations are outside the prototype derivative's compact support
and have exactly zero derivative. They are not added as redundant constant-tail
function-value observations in the training dataset. This does not equate the
methods' information access: QI uses derivative values and an anchor; trained
models use extension values at the halo-center grid. Bitwise dataset checks
refer to the baseline arrays constructed from this shared mathematical grid.

```sh
~/venv/quill-radon/bin/python experiments/expK03_npu_inverse_sqrt/matched_width_sweep.py --workers 4
~/venv/quill-radon/bin/python experiments/expK03_npu_inverse_sqrt/validate_matched_widths.py
~/venv/quill-radon/bin/python experiments/expK03_npu_inverse_sqrt/plot_width_sweep.py --matched-widths
```

Results are under `matched_construction_widths/`; the figure is
`matched_width_scaling.png` / `.pdf`. The original coarse parameter-count sweep
is preserved separately in `width_sweep/`. The new figure uses hidden width on
the x-axis, unsmoothed measured values, and white-centered markers at the actual
construction widths. The Adam MLP remains a single reference at h=4,815.

### Current Adam reference: 1,000 units

The user subsequently replaced the Adam marker with a dotted horizontal
reference trained at **exactly 1,000 units** (3,001 parameters). Only the Adam
model width changes: the existing N=256/R=32 halo-center dataset, C17 target,
input centering, seed, minibatches, optimizer and 20k-step budget are retained.
Its best positive-step checkpoint is step 19,500, with physical relative L2
2.25522585; the final error is 2.46476417. The below-legend description is removed
from the current matched-width figure. Original 4,815-unit results are preserved.

```sh
~/venv/quill-radon/bin/python experiments/expK03_npu_inverse_sqrt/run.py --config experiments/expK03_npu_inverse_sqrt/config_adam_1k.json --output results/checkpoint_K_practical_recipes/expK03_npu_inverse_sqrt/adam_mlp_1k
~/venv/quill-radon/bin/python experiments/expK03_npu_inverse_sqrt/plot_width_sweep.py --matched-widths
```

### Current Adam baseline: pilot-selected width and nine 100k runs

The user then requested a stronger baseline: 10k-step width pilots followed by
three widths at three rates for 100k steps each. `adam_tuning.py` runs 24 pilots
(widths 8, 16, 32, 64, 128, 256, 512, 1000; rates 1e-2, 1e-3, 1e-4). It ranks
widths by their best order-32 validation error in steps 5k–10k across the rates.
The three selected widths were **8, 16, 128**. All nine width/rate combinations
then completed **100,000 steps**, restarting from the same initialization and
minibatch sequence. The first 10k updates match each corresponding pilot bitwise.

For the displayed dotted line, selection is restricted to steps **50k–100k**, so
the chosen model has actually trained for at least 50k steps. The winner is width
16, lr=0.01, step 85,250, with independent test relative L2 **0.971605074**.
All target/halo/sampling/precision settings remain unchanged from the earlier
Adam baseline. The line is labeled simply **Adam baseline**, with no description
under the plot. NPU/RealNPU curves retain their original 20k budgets.

```sh
~/venv/quill-radon/bin/python experiments/expK03_npu_inverse_sqrt/adam_tuning.py --workers 4
~/venv/quill-radon/bin/python experiments/expK03_npu_inverse_sqrt/validate_adam_tuning.py
~/venv/quill-radon/bin/python experiments/expK03_npu_inverse_sqrt/plot_width_sweep.py --matched-widths
```

The current renderer requires `adam_tuning/selected_reference.json`. Historical
20k Adam references remain saved but no longer define the matched-width figure's
line. The search specification, width ranking, all pilot/long-run tables,
checkpoints, validation audit and report are saved under `adam_tuning/`.

### Current baseline: improved 128-neuron Adam

The user then requested fixing width at 128 and improving optimization. Six
100k-step trials in `adam128_refine.py` screened larger batches, cosine LR decay,
random multiscale initialization, and optional output normalization. The winner
is **128 neurons / 385 parameters**, relative L2 **3.49909e-4**, at step 100k.
It uses 112 randomly jittered log-scale local features plus 16 broad features,
all trained by Adam; batch 512; LR .003 held to 50k then cosine-decayed to 1e-5.
Its output scale is 1. Target, halo data and relative loss weights are unchanged.

This is **Adam with random multiscale initialization**, a domain-scale-informed
prior, not default PyTorch initialization. The feature geometry is randomized;
no QI coefficients, derivative labels or coefficient solve initialize the model.
All 385 parameters are trainable. The initializer addresses the missing sharp
features near epsilon: initial hidden slopes reach about 1e5, compared with only
about 25 in the earlier trained network. Output normalization was tested but
does not appear in the winning recipe.

The user briefly allowed capacity expansion with minimum width 64, then accepted
the 128-neuron result. No larger/smaller capacity trials were launched. The
matched-width renderer now reads `adam128_refine/selected_reference.json`.
All previous runs are retained. See `adam128_refine/results.md`, `summary.csv`,
`validation.json`, and the exported ordinary MLP in `multiscale_gain1/`.

### Superseding correction: conventional initialization only

The user subsequently disallowed the multiscale feature initialization. The
3.49909e-4 result above is an archived exploratory result, **not an eligible Adam
baseline**. `adam_standard.py` performs 12 new 100k-step runs at width 128 using
only unmodified PyTorch initialization or Xavier uniform (standard tanh gain,
zero biases). Rates are .001, .01 and .03, batches 100/1024, with cosine decay
after 50k. No feature placement, scale priors, log inputs, output gains or changed
loss weighting are allowed. Existing eligible 128-neuron standard runs are also
considered; width-8/16 and all multiscale/output-scaled runs are excluded.

```sh
~/venv/quill-radon/bin/python experiments/expK03_npu_inverse_sqrt/adam_standard.py
~/venv/quill-radon/bin/python experiments/expK03_npu_inverse_sqrt/validate_adam_standard.py
~/venv/quill-radon/bin/python experiments/expK03_npu_inverse_sqrt/plot_width_sweep.py --matched-widths
```

Initial tensors must exactly match the standard library implementation and must
not change with the target-domain epsilon. The allowed initial hidden weights
have magnitude at most 1. The initializer policy is enforced both in training
and in the figure renderer; the saved validation audit checks every candidate.
The same C17 target, fixed halo-center dataset and physical-domain error metric
remain in use. All previously saved results are preserved.

All 12 new trials completed. The selected conventional baseline is
`pytorch_lr3e-02_batch1024`: width 128, unmodified PyTorch initialization,
batch 1024, LR .03 held to 50k then cosine-decayed to 1e-5 by 100k. Its selected
checkpoint is step 94,000, relative L2 **0.753494153**. The audit checks all 16
eligible candidates (12 new plus four prior), including exact standard-library
initial tensors, domain independence, unchanged labels, and independent
quadrature. The figure rejects multiscale/output-scaled candidates explicitly.

Primary sources:
- https://proceedings.neurips.cc/paper_files/paper/2020/file/48e59000d7dfcf6c1d96ce4a603ed738-Paper.pdf
- https://github.com/nmheim/NeuralPowerUnits/blob/submission/src/npu.jl
- https://github.com/nmheim/NeuralPowerUnits/blob/submission/src/nau.jl
- https://github.com/nmheim/NeuralPowerUnits/blob/submission/scripts/simple/run.jl
