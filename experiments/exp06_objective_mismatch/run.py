"""Experiment 6: Objective Mismatch.

Core:
- Compare: uniform-grid MSE, denser-grid MSE, Lp (p=6) approximating L_inf,
  hybrid boundary-weighted MSE.
- Check whether optimizer still drives lambda -> 0 under all objectives.

Additional:
- Chebyshev-weighted MSE vs uniform sampling.
- MSE + derivative matching (if analytic derivatives available).
- Where is remaining error concentrated (boundaries vs interior vs high-freq)?
"""

# TODO: implement
# from src.training.losses import get_loss_fn
#
# LOSS_VARIANTS = [
#     get_loss_fn("mse"),
#     get_loss_fn("lp_approx", p=6.0),
#     get_loss_fn("hybrid_boundary", boundary_weight=5.0, boundary_fraction=0.1),
#     get_loss_fn("weighted_mse"),  # Chebyshev weights
# ]
