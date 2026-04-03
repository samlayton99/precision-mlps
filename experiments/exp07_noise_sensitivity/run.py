"""Experiment 7: Noise Sensitivity.

Core:
- Y-noise: Gaussian noise on function values before construction.
  Plot construction error vs noise level at different widths.
- X-noise: Gaussian noise on grid positions. Same analysis.

Additional:
- Train on noisy data and compare noise-sensitivity curves
  (trained vs constructed).
- Gradient noise during training: simulated SGD noise at controlled magnitude.
"""

# TODO: implement
# NOISE_LEVELS = [0, 1e-14, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4]
#
# For Y-noise:
#   for sigma in NOISE_LEVELS:
#       config = modify_noise(base_config, y_noise_std=sigma)
#       # Construct QI on noisy data, evaluate on clean data
#
# For X-noise:
#   for sigma in NOISE_LEVELS:
#       config = modify_noise(base_config, x_noise_std=sigma)
#       # Construct QI on perturbed grid, evaluate on clean grid
