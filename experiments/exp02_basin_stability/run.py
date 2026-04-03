"""Experiment 2: QI Basin Stability and Path Experiments.

Core:
- Low LR from construction: train from QI with small LR, monitor drift.
- Directional perturbation profiles: perturb gamma/centers/readout, measure loss.
- Path interpolation: interpolate QI <-> trained solutions, check for barriers.

Additional:
- Hessian-eigenvector perturbations.
- Frozen-subset drift (freeze gamma only, centers only, readout only).
- Isotropic vs Hessian-aligned noise recovery.
"""

# TODO: implement
# Low-LR drift experiment:
#   1. Construct QI, initialize model.
#   2. Train with lr=1e-5 Adam, no stochastic noise (full batch).
#   3. Log lambda, gamma, weight norms, loss every step.
#   4. Compare SGD vs Adam.
#
# Perturbation profiles:
#   1. From QI solution, sweep perturbation magnitude in [1e-8, 1].
#   2. For each direction: gamma, centers, readout, random, Hessian eigenvecs.
#   3. Plot loss increase vs magnitude.
#
# Path interpolation:
#   1. Construct QI solution.
#   2. Train a separate model to convergence (standard training).
#   3. Match neurons (solve permutation).
#   4. Interpolate, track loss along path.
