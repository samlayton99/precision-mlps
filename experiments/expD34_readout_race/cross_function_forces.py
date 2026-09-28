"""Sampled force identities for the existing cross-function GD checkpoints.

No trajectory integral or future-error certificate is computed here. The input
basis fixes the retained modes (normally degrees 2--65). Configure FP64 in the
remote launcher, as for the existing effective-feedback kernel.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from . import effective_feedback_kernel as ef

BLOCK_NAMES = ('a', 'b', 'c', 'd')


def gain_parts(p, context):
    """Full parameter-by-fine-mode gains D, C, with effective gain T=D-C."""
    matrices = ef.matrices(p, context)
    return matrices['J_H'].T, matrices['J_C'].T @ matrices['B']


def make_analyzer(include_derivatives=True):
    """Return a pure, JIT-compatible ``analyze(p, context, D0, C0, e_hat)``.

    D0/C0 are full gains from ``gain_parts`` at the forecast fork; e_hat is
    the corresponding fixed-map residual forecast at this sampled time.
    ``balanced`` always includes its minus sign: -C e. Signed projections
    are contributions to mean absolute-slope velocity, -sign(a).F / width.
    Derivatives are instantaneous gradient-flow derivatives at the GD state,
    not finite-step changes. Block arrays have order a,b,c,d.
    """
    def analyze(p, context, D0, C0, e_hat):
        width = (p.shape[0]-1)//3
        g, channels = ef.field(p, context)
        matrices = ef.matrices(p, context)
        JH, JC = matrices['J_H'], matrices['J_C']
        D, C = JH.T, JC.T @ matrices['B']
        T, T0 = D-C, D0-C0
        e = channels['eH']
        direct, balanced = D @ e, -C @ e
        effective = direct+balanced
        tracking = JC.T @ channels['zC']
        c, h, sech = ef._features(p, context['x'])
        coefficients = jnp.concatenate((channels['eC'], e))
        omitted = ef._pullback(channels['residual']-context['q'] @ coefficients,
                               context['x'], c, h, sech)
        remainder = tracking+omitted
        signed = -jnp.sign(p[:width])/width
        projection = lambda v: signed @ v[:width]
        map_direct = (D[:width]-D0[:width]) @ e
        map_balanced = -(C[:width]-C0[:width]) @ e
        map_defect = map_direct+map_balanced
        error_defect = T0[:width] @ (e-e_hat)
        forecast_defect = g[:width]-T0[:width] @ e_hat
        modal_direct = D[:width]*e[None, :]
        modal_balanced = -C[:width]*e[None, :]
        fine_effective = JH @ effective
        fine_tracking = JH @ tracking
        fine_omitted = JH @ omitted
        residual_velocity = -JH @ g
        residual_force_derivative = T[:width] @ residual_velocity
        result = dict(
            gradient=g, direct=direct, balanced=balanced, effective=effective,
            tracking=tracking, omitted=omitted, remainder=remainder,
            eH=e, eC=channels['eC'], zC=channels['zC'],
            coarse_resolved=channels['coarse_resolved'],
            coarse_min_eigenvalue=channels['coarse_min_eigenvalue'],
            reconstruction=g-effective-tracking-omitted,
            reconstruction_norm=jnp.linalg.norm(g-effective-tracking-omitted),
            map_direct_defect_a=map_direct, map_balanced_defect_a=map_balanced,
            map_defect_a=map_defect, residual_defect_a=error_defect,
            remainder_defect_a=remainder[:width], forecast_defect_a=forecast_defect,
            defect_identity=forecast_defect-map_defect-error_defect-remainder[:width],
            defect_identity_norm=jnp.linalg.norm(
                forecast_defect-map_defect-error_defect-remainder[:width]),
            coarse_projector_identity_norm=jnp.linalg.norm(JC @ T),
            Schur_identity_norm=jnp.linalg.norm(JH @ T-T.T @ T),
            direct_gain_a=D[:width], balanced_gain_a=-C[:width],
            effective_gain_a=T[:width],
            modal_direct_a=modal_direct, modal_balanced_a=modal_balanced,
            modal_effective_a=modal_direct+modal_balanced,
            modal_direct_A=signed @ D[:width],
            modal_balanced_A=-signed @ C[:width],
            modal_effective_A=signed @ T[:width],
            modal_direct_velocity=signed @ modal_direct,
            modal_balanced_velocity=signed @ modal_balanced,
            modal_effective_velocity=signed @ (modal_direct+modal_balanced),
            fine_effective_forcing=fine_effective,
            fine_tracking_forcing=fine_tracking,
            fine_omitted_forcing=fine_omitted,
            fine_residual_velocity=residual_velocity,
            fine_effective_forcing_norm=jnp.linalg.norm(fine_effective),
            fine_tracking_forcing_norm=jnp.linalg.norm(fine_tracking),
            fine_omitted_forcing_norm=jnp.linalg.norm(fine_omitted),
            fine_residual_velocity_norm=jnp.linalg.norm(residual_velocity),
            # No denominator floor: zero channels give inf or nan, not a
            # silently regularized finite ratio. Interpret alongside raw terms.
            fine_tracking_to_effective_ratio=(jnp.linalg.norm(fine_tracking)
                                              /jnp.linalg.norm(fine_effective)),
            fine_tracking_to_effective_modal_ratio=(jnp.abs(fine_tracking)
                                                    /jnp.abs(fine_effective)),
            effective_norm=jnp.linalg.norm(effective[:width]),
            remainder_norm=jnp.linalg.norm(remainder[:width]),
            map_defect_norm=jnp.linalg.norm(map_defect),
            residual_defect_norm=jnp.linalg.norm(error_defect),
            effective_residual_velocity_norm=jnp.linalg.norm(JH @ effective),
            residual_tracking_norm=jnp.linalg.norm(JH @ tracking),
            residual_omitted_norm=jnp.linalg.norm(JH @ omitted),
            # Frobenius norms of slope gain matrices, not induced/operator norms.
            gain_direct_norm=jnp.linalg.norm(D[:width]),
            gain_balanced_norm=jnp.linalg.norm(C[:width]),
            residual_force_derivative_a=residual_force_derivative,
            residual_force_derivative_signed=signed @ residual_force_derivative,
        )
        for name, force in [('gradient', g), ('direct', direct), ('balanced', balanced),
                            ('effective', effective), ('tracking', tracking),
                            ('omitted', omitted), ('remainder', remainder)]:
            result[name+'_slope_norm'] = jnp.linalg.norm(force[:width])
            result[name+'_signed_velocity'] = projection(force)
        for name in ('map_direct_defect_a', 'map_balanced_defect_a', 'map_defect_a',
                     'residual_defect_a', 'remainder_defect_a', 'forecast_defect_a'):
            result[name+'_norm'] = jnp.linalg.norm(result[name])
            result[name+'_signed'] = signed @ result[name]
        if include_derivatives:
            parts = lambda v: gain_parts(v, context)
            edges = (0, width, 2*width, 3*width, 3*width+1)
            direct_terms, balanced_terms = [], []
            for lower, upper in zip(edges[:-1], edges[1:]):
                velocity = jnp.zeros_like(p).at[lower:upper].set(-g[lower:upper])
                _, (Ddot, Cdot) = jax.jvp(parts, (p,), (velocity,))
                direct_terms.append(Ddot[:width] @ e)
                balanced_terms.append(-Cdot[:width] @ e)
            direct_dot = jnp.stack(direct_terms)
            balanced_dot = jnp.stack(balanced_terms)
            gain_dot = direct_dot+balanced_dot
            # Differentiate the whole effective force, including its residual,
            # independently of the blockwise product-rule construction above.
            def whole_force(v):
                return ef.field(v, context)[1]['effective_a']
            _, whole_dot = jax.jvp(whole_force, (p,), (-g,))
            result.update(
                block_direct_derivative_a=direct_dot,
                block_balanced_derivative_a=balanced_dot,
                block_gain_derivative_a=gain_dot,
                block_direct_derivative_norm=jnp.linalg.norm(direct_dot, axis=1),
                block_balanced_derivative_norm=jnp.linalg.norm(balanced_dot, axis=1),
                block_gain_derivative_norm=jnp.linalg.norm(gain_dot, axis=1),
                block_direct_derivative_signed=direct_dot @ signed,
                block_balanced_derivative_signed=balanced_dot @ signed,
                block_gain_derivative_signed=gain_dot @ signed,
                full_force_derivative_a=whole_dot,
                derivative_identity=whole_dot-jnp.sum(gain_dot, axis=0)
                                    -residual_force_derivative,
                derivative_identity_norm=jnp.linalg.norm(
                    whole_dot-jnp.sum(gain_dot, axis=0)-residual_force_derivative),
            )
        return result
    return analyze
