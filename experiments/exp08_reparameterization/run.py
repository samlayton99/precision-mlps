"""Experiment 8: Reparameterization.

Core (head-to-head comparison):
- Raw (standard): W, b
- Gamma-center (gamma_linear): gamma, centers
- Log-gamma (gamma_exp): log_gamma, centers, with h = 2/N
- Global bandwidth: single learnable lambda, gamma = lambda/h, centers on grid

For each: train at widths {16, 32, 64, 128} with Adam -> LBFGS.
Log final error, learned lambda, outer weight norms, convergence speed.

Additional:
- Alpha readout: learn alpha = a * gamma instead of a directly.
- Combined reparameterization: log-gamma + dimensionless centers + alpha readout.
- Test whether reparameterization alone is sufficient or must combine
  with exact readout / constrained geometry.
"""

# TODO: implement
# from src.config.loader import load_config
# from src.models.mlp import QIMlp
# from src.data.dataset import build_dataset
# from src.training.train_loop import run_training
#
# LAYER_TYPES = ["standard", "gamma_linear", "gamma_exp"]
# For each layer_type:
#   config = modify_layer_type(base_config, layer_type)
#   # build dataset, create model, train, collect results
