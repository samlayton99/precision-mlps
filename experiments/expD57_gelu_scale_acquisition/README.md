# D57 — GELU scale acquisition with GD and Adam

This is a matched extension of D56 to exact GELU, using two targets with distinct structure: the oscillatory `sin(4*pi*x)` and the localized rational function `1/(1+100*x*x)`. Selection precedes the runs. The target names do not imply equal optimization difficulty.

## Protocol

- 18 full-batch GD runs and 18 full-batch Adam runs: two targets × three initialization scales × three learning rates per optimizer.
- Exactly 196 hidden GELU neurons, scalar input/output, FP64, 200,000 updates, one paired seed (30).
- Exact GELU, `z*Phi(z)`, matching `torch.nn.functional.gelu(..., approximate="none")`. No tanh approximation.
- The same default float64 PyTorch Linear weights as D56. First-layer weights and biases are multiplied by 1, 2, or 4; readout initialization is unchanged. All parameters train freely.
- Rates 0.01, 0.001, 0.0001; constant throughout training. Loss is half mean squared error. No clipping, weight decay, warmup, schedule, geometry constraints, or readout solves.
- Adam uses beta1=0.9, beta2=0.999, epsilon=1e-8, global step bias correction, and saved first/second moments across every checkpoint. GD has no momentum.
- Same 1,024 midpoint training samples and 4,096 disjoint midpoint evaluation samples on [-1,1]. Targets are divided by their training RMS without mean subtraction.

## Reference and plots

Two separate figures reuse the approved D56 format: three initialization panels, linear axes, blue/orange/purple learning-rate families, function-specific shades, thin opaque median lines, transparent central 98% neuron intervals, and visible dotted interval boundaries. Absolute input weights are the plotted scales. Each final median annotation is the median of the six final run medians in that panel.

The GELU reference uses lambda=0.707, the convention documented in `results/checkpoint_D_optimizers/expD19_gelu_init/expD19_results.md` and attributed there to C07. With the same comparison geometry (147 intervals, 148 interior centers, 48 halos, 196 total neurons), h=2/147 and gamma=51.9645. This is a construction-scale reference, not a necessary threshold or a new precision validation of a GELU construction. Standard random initialization has no designated halo neurons.

No tanh-specific growth or error theorem is applied to these GELU runs, and no GD theorem is asserted for Adam. The experiment measures trajectories.

## Validation and execution

Tests check exact-GELU predictions and gradients against PyTorch autograd, both optimizer trajectories against PyTorch SGD/Adam, bitwise equality of split versus continuous updates with saved Adam moments, and identical initial parameters to the corresponding D56 runs. The 36-case short pilot tests execution and rendering before the full sweep.

```bash
OMP_NUM_THREADS=1 .venv/bin/python -m pytest -q tests/test_expD57_gelu_scale_acquisition.py
.venv/bin/python experiments/expD57_gelu_scale_acquisition/run.py pilot
.venv/bin/python experiments/expD57_gelu_scale_acquisition/run.py launch
.venv/bin/python experiments/expD57_gelu_scale_acquisition/run.py status
.venv/bin/python experiments/expD57_gelu_scale_acquisition/plot.py
```

Run on the Mac mini using four one-thread workers. The detached process advances every case in 5,000-step rounds, so both optimizers and all initialization panels acquire data early. It updates both plots after every round. Checkpoints contain every logged parameter vector and Adam's exact state; source hashes guard training resumes. The original tanh data is unchanged.

Outputs: `results/checkpoint_D_optimizers/expD57_gelu_scale_acquisition/sweep/{gd,adam}/`. Each optimizer has `runs/`, a manifest, and `figures/`. Early plots have `_PARTIAL` names and end at observed checkpoints. Final plots omit that suffix. Nonfinite and error statuses are retained. The failed sandbox pilot, if present, is separate from `pilot_verified/` and has no experimental interpretation.
