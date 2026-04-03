"""Tests for analysis modules.

Test cases:
- compute_full_hessian is symmetric
- hessian_eigenspectrum eigenvalues are real
- Gauss-Newton J^T J is positive semi-definite
- compute_phi_conditioning returns correct keys
- feature_rank_diagnostics: effective_rank <= width
- stable_rank <= effective_rank
- directional_perturbation_profile: loss increases with perturbation magnitude
- path_interpolation: endpoints match model losses
"""

# TODO: implement
