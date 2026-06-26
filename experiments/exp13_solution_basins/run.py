"""Experiment exp13: Solution basins -- the landscape near the optimal solution.

Merges the former basin-stability and Hessian-landscape studies into one
experiment that asks: what is actually going on at the high-precision solution,
and why does an optimizer drift away from it?

Reference point: the QI construction (mpmath, a fixed exact solution), and the
matched least-squares / trained solutions on the same geometry.

Core:
- Hessian spectrum at the solution. Compare the full-parameter Hessian to the
  readout-only Hessian (Phi^T Phi, convex) and the reduced/Gauss-Newton
  (geometry-only, readout eliminated) Hessian. If the readout-only block is well
  conditioned and the full block is not, the difficulty is in the inner layer.
- Basin geometry. From the QI solution, perturb along isolated directions
  (global gamma, per-neuron gamma, center shifts, readout weights, and the
  Hessian eigenvectors), sweep magnitude, measure the loss increase, then
  re-optimize and record recovery probability and final error.
- Path interpolation. Interpolate (neuron-matched) between the QI solution and a
  trained solution; track loss along the path. A barrier => different basins; a
  flat path => the difference is parameterization, not landscape.

Additional:
- Gauss-Newton J^T J vs the full Hessian (is the quadratic model accurate?).
- Chebyshev decomposition of the residual for QI vs trained (is the gap spectral?).
- Hessian-spectrum evolution along the training trajectory.

Use mpmath for the QI reference (fixed exact solution); fp64 for everything else.
"""

# TODO: implement
# from src.construction.qi_mpmath import construct_qi, default_halo
# from src.construction.readout import build_phi, solve_readout_with_bias
# from src.models.mlp import QIMlp
# from src.data.targets import get_target
#
# 1. Build the QI reference solution (and the matched lstsq / trained solutions).
# 2. Hessian blocks: full vs readout-only (Phi^T Phi) vs reduced (geometry-only).
# 3. Directional + Hessian-eigenvector perturbation -> recovery profiles.
# 4. Neuron-matched path interpolation QI <-> trained; loss along the path.
