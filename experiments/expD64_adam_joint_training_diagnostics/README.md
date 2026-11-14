# Adam joint-training diagnostics

Matched counterpart of D62: tanh, widths 64/196/384/768, targets x²/sin(4πx)/1/(1+100x²), learning rates 0.001/0.0001, and 50,000 updates. All parameters train jointly using full-batch float64 Adam (betas 0.9/0.999, epsilon 1e-8, no weight decay). Initialization, seed 30, 1,024 training midpoints, 4,096 evaluation midpoints, unit training-RMS target normalization, and half-MSE loss match D62.

The figure retains the original arrangement and colors. It tracks the measured hidden parameter norm, concentration chi6, raw-gradient affine-tracking norm R, and fixed suffix maximum nonaffine residual E0. All four concentration moments are recorded. Snapshots are saved at steps 0/1/2/5/10/20/50/100/200 and every 500 thereafter; resumable checkpoints include both Adam moments.

The pink curve evaluates the paper's original GD recurrence with the diagnostics observed along Adam's trajectory. It is **not an Adam upper bound**: Adam transforms the gradient through its moving averages and coordinatewise denominator. R and F are raw gradient components, not Adam displacement components. Between-checkpoint allowance interpolation is retrospective rather than certified. Curves are produced for both the 1k and 30k analysis starts.

The update implementation was compared against PyTorch autograd plus `torch.optim.Adam` for 23 updates (maximum parameter difference 3.33e-16). Splitting at step 7 and resuming gives exactly the same parameters and both moments as the uninterrupted implementation.

```sh
.venv/bin/python experiments/expD64_adam_joint_training_diagnostics/run.py --verify
.venv/bin/python experiments/expD64_adam_joint_training_diagnostics/run.py
.venv/bin/python experiments/expD64_adam_joint_training_diagnostics/plot.py --restart 1000
.venv/bin/python experiments/expD64_adam_joint_training_diagnostics/plot.py --restart 30000
```

Training data and results are saved under `results/checkpoint_D_optimizers/expD64_adam_joint_training_diagnostics/`. D62 and the earlier 24-function GD distribution plots are unchanged.
