"""Rational target filters from structured solves, without eigendecomposition.

The residual-squared and kernel-perturbation formulas are rigorous given
exact residuals and an operator error bound. The separate floating-point
sensitivity allowance is heuristic and is not an interval certificate.
"""
from __future__ import annotations

import numpy as np
from scipy.special import expit


def signed_lowrank_solver(diagonal, factors, signed_values,
                          to_basis=None, from_basis=None):
    """Cache a Woodbury solver for D+U diag(signed_values) U*.

    Optional unitary transforms map physical coefficients to/from the known
    diagonal basis. Both vector and multiple-column right sides are accepted.
    Negative real shifts are allowed for capacity witnesses; filter shifts
    are nonreal. No eigenvalues are computed here.
    """
    diagonal = np.asarray(diagonal, dtype=float)
    factors = np.asarray(factors)
    signed_values = np.asarray(signed_values, dtype=float)
    keep = signed_values != 0
    weighted = factors[:, keep]*np.sqrt(np.abs(signed_values[keep]))
    signs = np.sign(signed_values[keep])

    def solve(rhs, shift):
        rhs = np.asarray(rhs)
        vector = rhs.ndim == 1
        rhs = rhs[:, None] if vector else rhs
        mapped = to_basis(rhs) if to_basis is not None else rhs
        inverse = 1/(diagonal-shift)
        result = np.asarray(inverse[:, None]*mapped,
                            dtype=np.result_type(mapped, weighted, shift))
        if len(signs):
            scaled = inverse[:, None]*weighted
            small = np.diag(signs)+weighted.conj().T@scaled
            result -= scaled@np.linalg.solve(small, weighted.conj().T@result)
        result = from_basis(result) if from_basis is not None else result
        return result[:, 0] if vector else result

    return solve


def rational_filter_mass(forcing, target_norm_squared, scale, order, solve,
                         action, kernel_error=0., residual_guard=0.,
                         arithmetic_factor=64.):
    """Target expectation of f(a)=1/(1+(a/scale)**order).

    forcing=sqrt(eta)*Jhat.T@y is REAL in physical coefficient coordinates;
    action(v)=eta*Jhat.T@(Jhat@v) uses the UNCOMPRESSED constructed model.
    solve(forcing,z) may use a compressed diagonal+signed-low-rank model.
    target_norm_squared contains one physical sample-space norm per target.

    kernel_error bounds ||eta*K-eta*Khat|| in SAMPLE space, so it accounts
    for changing both the coefficient Gram and forcing between models.
    It is not merely an error bound on the coefficient Gram.

    residual_guard is an additional norm uncertainty per target, or a
    callback(v,z,residual) returning that uncertainty. arithmetic_factor
    adds a dot-product sensitivity allowance. Set it to zero for exact-
    arithmetic diagnostics; ordinary floating-point output is not certified.
    """
    if scale <= 0 or order < 2 or order % 2 or kernel_error < 0:
        raise ValueError('require positive scale, even order >=2, nonnegative error')
    forcing = np.asarray(forcing)
    if np.iscomplexobj(forcing) and np.any(forcing.imag != 0):
        raise ValueError('forcing must be real in physical coordinates')
    forcing = forcing.real
    if forcing.ndim == 1:
        forcing = forcing[:, None]
    norm2 = np.broadcast_to(np.asarray(target_norm_squared, dtype=float), (forcing.shape[1],))
    if np.any(norm2 <= 0):
        raise ValueError('targets must have positive norm')
    value = np.ones(forcing.shape[1])
    solve_radius = np.zeros_like(value)
    arithmetic_radius = np.zeros_like(value)
    max_residual = np.zeros_like(value)
    perturbation_factor = 0.
    # Real physical H and g give conjugate quadratics. Use only upper poles.
    for j in range(order//2):
        shift = scale*np.exp(1j*(2*j+1)*np.pi/order)
        v = np.asarray(solve(forcing, shift))
        residual = forcing-(action(v)-shift*v)
        # Ordinary transpose is essential: H-zI is complex symmetric.
        leading = np.sum(forcing*v, axis=0)
        correction = np.sum(v*residual, axis=0)
        value -= 2*np.real(leading+correction)/(order*norm2)
        rn = np.linalg.norm(residual, axis=0)
        vn = np.linalg.norm(v, axis=0)
        max_residual = np.maximum(max_residual, rn)
        solve_radius += 2*rn**2/(order*norm2*abs(shift.imag))
        guard = (residual_guard(v, shift, residual) if callable(residual_guard)
                 else np.asarray(residual_guard))
        if np.any(np.asarray(guard) < 0):
            raise ValueError('residual guard must be nonnegative')
        # If ||r_exact-r_computed||<=guard, the additional analytic error
        # is ||v||guard+[(||r||+guard)^2-||r||^2]/|Im z|.
        additional = vn*guard+(2*rn*guard+guard**2)/abs(shift.imag)
        dot_scale = np.sum(np.abs(forcing*v)+np.abs(v*residual), axis=0)
        additional += arithmetic_factor*np.finfo(float).eps*dot_scale
        arithmetic_radius += 2*additional/(order*norm2)
        perturbation_factor += 2*abs(shift)/(order*shift.imag**2)
    construction_radius = np.full_like(value, kernel_error*perturbation_factor)
    radius = solve_radius+construction_radius+arithmetic_radius
    return dict(mass=value, lower=np.maximum(0., value-radius),
                upper=np.minimum(1., value+radius), solve_radius=solve_radius,
                construction_radius=construction_radius, arithmetic_radius=arithmetic_radius,
                radius=radius, maximum_solve_residual=max_residual,
                construction_lipschitz=float(perturbation_factor),
                scale=float(scale), order=int(order),
                numerical_status='Residual and kernel formulas with a separately disclosed floating-point sensitivity allowance; not interval certified.')


def filter_cdf_bounds(lower_filter, upper_filter, cutoff):
    """Monotone filter expectations bound mass on [0,cutoff].

    Use a filter centered below cutoff for the lower bound and one above for
    the upper bound. Any positive scales are valid, though poor scales may
    make the bounds vacuous. The nullspace is INCLUDED.
    """
    if cutoff <= 0:
        raise ValueError('cutoff must be positive')
    q_lower = expit(-lower_filter['order']*np.log(cutoff/lower_filter['scale']))
    q_upper = expit(-upper_filter['order']*np.log(cutoff/upper_filter['scale']))
    if q_lower == 1 or q_upper == 0:
        raise ValueError('scales make the CDF formula numerically singular')
    return dict(lower=np.maximum(0., (lower_filter['lower']-q_lower)/(1-q_lower)),
                upper=np.minimum(1., upper_filter['upper']/q_upper),
                lower_threshold=float(q_lower), upper_threshold=float(q_upper))
