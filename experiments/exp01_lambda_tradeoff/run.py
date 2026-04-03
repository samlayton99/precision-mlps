"""Experiment 1: Lambda Tradeoff Verification.

Core:
- Construct QI MLPs across widths and sweep lambda_star.
- Plot L_inf error (y) vs lambda (x), one curve per width.
- Expect U-shaped curves with shared optimal lambda* ~ 1.5.

Additional:
- Fixed QI geometry + least-squares readout (is U-shape geometric or coefficient-dependent?).
- Overlay trained-network lambda values on the same plot.
"""

# TODO: implement
# from src.config.loader import load_config, expand_sweep
# from src.data.dataset import build_dataset
# from src.models.mlp import QIMlp
# from src.construction.qi_jax import construct_qi
#
# def run():
#     base_config = load_config("experiments/exp01_lambda_tradeoff/config.yaml")
#     # Sweep over widths and lambda values
#     # Construct QI MLP for each, evaluate, collect results
#     # Post-processing: group by width, plot error vs lambda
