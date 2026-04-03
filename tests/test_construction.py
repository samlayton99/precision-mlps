"""Tests for QI construction and readout solve.

Test cases:
- construct_qi returns QIResult with correct shapes and types
- QIResult.gamma = lambda_star / h
- QIResult.centers are on uniform grid
- build_phi produces correct shape [n_points, width]
- solve_readout with known Phi and y recovers exact weights
- solve_readout_with_bias: bias is unpenalized in ridge
- lstsq, qr, svd solvers agree on well-conditioned problems
- Kahan summation is more accurate than naive sum
- QI construction achieves < 1e-13 L_inf on sine target at width 64
"""

# TODO: implement
