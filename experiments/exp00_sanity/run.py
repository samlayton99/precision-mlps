"""Experiment 0: Numerics Sanity Checks.

Tests:
1. Compare exact-readout solves: lstsq vs QR vs SVD vs ridge.
2. Verify QI + exact readout reproducibility across solver backends.
3. Track residual norms of linear solves.
4. Test whether denser eval grid changes L_inf.
5. (Additional) Extended precision for linear solve or evaluation only.
6. (Additional) tanh stability at large arguments, cond(Phi) vs cond(Phi^T Phi).
"""

# TODO: implement
# from src.config.loader import load_config
# from src.construction.readout import solve_readout
# from src.construction.qi_jax import construct_qi
# from src.models.mlp import QIMlp
#
# SOLVE_METHODS = ["lstsq", "qr", "svd", "ridge"]
# EVAL_DENSITIES = [256, 512, 1024, 2048, 4096]
#
# def run():
#     base_config = load_config("experiments/exp00_sanity/config.yaml")
#     for method in SOLVE_METHODS:
#         config = modify_readout_solve(base_config, method)
#         # build dataset, construct QI, evaluate, save results
