"""Experiment 4: Hessian Landscape.

Core:
- Hessian eigenspectrum at QI solution (readout-only vs full-parameter).
- Compare Hessian at QI, trained-MLP, and geometry-ladder solutions.
- If readout-only Hessian is well-conditioned (it's Phi^T Phi), the
  optimization difficulty is entirely in the inner layer.

Additional:
- Gauss-Newton J^T J vs full Hessian comparison.
- Chebyshev residual expansion (is the gap spectral?).
- Hessian evolution along training trajectory.
"""

# TODO: implement
# 1. Construct QI, compute Hessian at QI solution (wrt="all", "readout", "inner")
# 2. Train model to convergence, compute Hessian at trained solution
# 3. Run geometry ladder (from exp03), compute Hessian at each level
# 4. Compare eigenspectra, condition numbers, negative eigenvalue counts
# 5. Compute Gauss-Newton at QI and trained, compare to full Hessian
# 6. Chebyshev expansion of residuals at QI and trained solutions
