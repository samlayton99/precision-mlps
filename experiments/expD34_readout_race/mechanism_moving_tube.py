"""Prospective moving-ball enclosure around the pure frozen-effective model.

Every reference state is generated from the fork, with no ordinary-GD future
inputs. Analytic real-arithmetic bounds are evaluated in FP64: this is NOT a
directed-rounding certificate. The defect uses the full ordinary-GD gradient.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from . import effective_feedback as ef, effective_feedback_kernel as kernel, transport
from .run import write_json


def reference_state(p, x, y):
    """Gradient and signed neuron residual-curvature blocks, without full Hessian."""
    width = (p.shape[0]-1)//3
    a, b, c = p[:-1].reshape(3, width)
    u = x[:, None]*a+b
    h = jnp.tanh(u)
    ex = jnp.exp(-2*jnp.abs(u)); s = 4*ex/(1+ex)**2
    residual = h@c+p[-1]-y
    m = x.shape[0]
    gradient = jnp.concatenate((c*((residual*x)@s)/m, c*(residual@s)/m,
                                residual@h/m, jnp.mean(residual)[None]))
    t = -2*h*s
    blocks = jnp.zeros((width, 3, 3), dtype=p.dtype)
    blocks = blocks.at[:, 0, 0].set(c*((residual*x*x)@t)/m)
    blocks = blocks.at[:, 0, 1].set(c*((residual*x)@t)/m)
    blocks = blocks.at[:, 1, 0].set(blocks[:, 0, 1])
    blocks = blocks.at[:, 1, 1].set(c*(residual@t)/m)
    blocks = blocks.at[:, 0, 2].set((residual*x)@s/m)
    blocks = blocks.at[:, 2, 0].set(blocks[:, 0, 2])
    blocks = blocks.at[:, 1, 2].set(residual@s/m)
    blocks = blocks.at[:, 2, 1].set(blocks[:, 1, 2])
    values = jnp.linalg.eigvalsh(blocks)
    jnorm = jnp.sqrt(jnp.sum(c*c*jnp.mean((1+x[:, None]**2)*s*s, axis=0))
                      +jnp.sum(jnp.mean(h*h, axis=0))+1)
    return dict(g=gradient, jacobian_bound=jnorm,
                residual_rms=jnp.sqrt(jnp.mean(residual**2)),
                curvature_min=values.min(), curvature_norm=jnp.max(abs(values)))


def ball_map_bound(p, state, radius, eta):
    """Same analytic tanh constants as persistence_theory, centered at this reference."""
    width = (p.shape[0]-1)//3
    cmax = jnp.max(abs(p[2*width:3*width]))+radius
    m2 = 8*cmax/(3*jnp.sqrt(3.))+jnp.sqrt(2.)
    m3 = 4*jnp.sqrt(2.)*cmax+8/jnp.sqrt(3.)
    residual_drift = state['jacobian_bound']*radius+.5*m2*radius**2
    curvature_drift = m2*residual_drift+m3*state['residual_rms']*radius
    negative = jnp.maximum(0., -state['curvature_min'])+curvature_drift
    upper = (state['jacobian_bound']+m2*radius)**2+state['curvature_norm']+curvature_drift
    beta = jnp.maximum(1+eta*negative, abs(1-eta*upper))
    return beta, negative, upper


SAVE_UPDATES = (100, 1000, 10000, 20000, 50000, 100000, 200000, 1000000)


@partial(jax.jit, static_argnames=('steps', 'save_updates'))
def enclose(p0, x, y, T0, e0, eta, h, threshold, *, steps, save_updates=SAVE_UPDATES):
    """Inductively propagate a full-parameter Euclidean error radius.

    For reference b, defect ||G(b)-b_next|| and uniform map Lipschitz beta
    on B(b,r), r_next=beta*r+defect encloses the true next state. The full
    ball is convex; the bound includes both ends of the Hessian spectrum.
    """
    width = (p0.shape[0]-1)//3
    coupling = T0.T@T0
    initial_max = h*abs(p0[:width])
    saved = jnp.asarray(tuple(n for n in save_updates if n <= steps), dtype=jnp.int64)
    checkpoints = jnp.full((len(saved), len(p0)), jnp.nan)
    checkpoint_bounds = jnp.full((len(saved), width), jnp.inf)

    def step(carry, index):
        p, e, radius, maximum, active, saved_p, saved_bounds = carry

        def compute(_):
            state = reference_state(p, x, y)
            reference_gradient = T0@e
            defect = eta*jnp.linalg.norm(state['g']-reference_gradient)
            beta, negative, upper = ball_map_bound(p, state, radius, eta)
            radius_next = beta*radius+defect
            p_next = p-eta*reference_gradient
            e_next = e-eta*coupling@e
            maximum_next = jnp.maximum(maximum, h*(abs(p_next[:width])+radius_next))
            valid = jnp.isfinite(radius_next) & jnp.all(jnp.isfinite(p_next))
            # Once every neuron is allowed, further growth cannot restore
            # prefix exclusion. Stop expensive evaluations and mark censoring.
            continue_run = valid & jnp.any(maximum_next < threshold)
            save_now = (saved == index+1)[:, None]
            next_saved_p = jnp.where(save_now, p_next[None, :], saved_p)
            next_saved_bounds = jnp.where(save_now, maximum_next[None, :], saved_bounds)
            output = dict(radius=radius_next, defect=defect, beta=beta,
                          negative=negative, upper=upper,
                          maximum_lambda_upper=maximum_next.max(),
                          excluded=jnp.sum(maximum_next < threshold),
                          evaluated=jnp.asarray(True), finite=valid)
            return (p_next, e_next, radius_next, maximum_next, continue_run,
                    next_saved_p, next_saved_bounds), output

        def stopped(_):
            output = dict(radius=radius, defect=jnp.asarray(jnp.nan), beta=jnp.asarray(jnp.nan),
                          negative=jnp.asarray(jnp.nan), upper=jnp.asarray(jnp.nan),
                          maximum_lambda_upper=jnp.asarray(jnp.inf), excluded=jnp.asarray(0),
                          evaluated=jnp.asarray(False), finite=jnp.asarray(False))
            return carry, output

        return jax.lax.cond(active, compute, stopped, operand=None)

    final, trace = jax.lax.scan(step,
        (p0, e0, jnp.asarray(0.), initial_max, jnp.asarray(True), checkpoints, checkpoint_bounds),
        xs=jnp.arange(steps))
    return trace | dict(checkpoint_updates=saved, checkpoint_p=final[5],
                        checkpoint_lambda_upper=final[6], last_evaluated_p=final[0],
                        final_lambda_prefix_upper=final[3])


def finite_number(value):
    value = float(value)
    return value if np.isfinite(value) else None


def evaluate_case(p, x, y, q, case, args, output):
    context = kernel.fork_context(p, x, y, q=q)
    matrices = kernel.matrices(context['p0'], context)
    eig = np.linalg.eigvalsh(np.asarray(matrices['C']))
    if eig[0] <= 64*np.finfo(float).eps*max(1., eig[-1]):
        return dict(case=case, status='unresolved_initial_coarse')
    T0 = np.asarray(matrices['T']); e0 = np.asarray(context['eH0'])
    if args.eta*np.linalg.norm(T0, 2)**2 > 1:
        return dict(case=case, status='unsupported_reference_step')
    output.mkdir()
    arrays = jax.tree.map(np.asarray, enclose(*map(jnp.asarray, (p, x, y, T0, e0)),
        args.eta, args.h, args.threshold, steps=args.steps))
    np.savez_compressed(output/'enclosure.npz', p0=p, T0=T0, e0=e0, **arrays)
    summaries = []
    _, singular, right = np.linalg.svd(T0, full_matrices=False)
    for checkpoint_index, n in enumerate(arrays['checkpoint_updates']):
        n = int(n); k = n-1; evaluated = bool(arrays['evaluated'][k])
        row = dict(updates=n, evaluated=evaluated, finite=bool(arrays['finite'][k]),
            radius=finite_number(arrays['radius'][k]) if evaluated else None,
            excluded=int(arrays['excluded'][k]),
            maximum_lambda_upper=finite_number(arrays['maximum_lambda_upper'][k]))
        if evaluated and row['finite']:
            point = arrays['checkpoint_p'][checkpoint_index]
            _, ch = kernel.field(jnp.asarray(point), context)
            mat = kernel.matrices(jnp.asarray(point), context)
            actual_e = np.asarray(ch['eH'])
            model_e = right.T@((1-args.eta*singular**2)**n*(right@e0))
            model_e += e0-right.T@(right@e0)
            map_force = (np.asarray(mat['T'])-T0)@actual_e
            residual_force = T0@(actual_e-model_e)
            tracking = np.asarray(mat['J_C']).T@np.asarray(ch['zC'])
            omitted = np.asarray(ch['full_gradient'])-np.asarray(mat['T'])@actual_e-tracking
            width = (len(p)-1)//3
            row.update(reference_map_defect_norm=float(np.linalg.norm(map_force)),
                reference_residual_defect_norm=float(np.linalg.norm(residual_force)),
                reference_signed_pair_norm=float(np.linalg.norm(map_force+residual_force)),
                reference_tracking_norm=float(np.linalg.norm(tracking)),
                reference_slope_tracking_norm=float(np.linalg.norm(tracking[:width])),
                reference_residual_tracking_norm=float(np.linalg.norm(np.asarray(mat['J_H'])@tracking)),
                reference_omitted_norm=float(np.linalg.norm(omitted)))
        summaries.append(row)
    summary = dict(case=case, eta=args.eta, h=args.h, threshold=args.threshold,
        steps=args.steps, evaluated_steps=int(arrays['evaluated'].sum()),
        status='complete' if arrays['evaluated'].all() else 'censored_after_loss_of_exclusion_or_nonfinite_bound',
        issued_utc=datetime.now(timezone.utc).isoformat(), input_sha256=ef.digest(args.inputs),
        source_sha256=ef.digest(__file__), numerical_certificate=False,
        scope='Ordinary-GD moving-ball analytic enclosure evaluated in FP64; no directed rounding. Reference-channel norms are point diagnostics, not uniform hypotheses.',
        horizons=summaries)
    write_json(output/'summary.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--target', default='moment9')
    parser.add_argument('--steps', type=int, default=20000)
    parser.add_argument('--eta', type=float, default=.002)
    parser.add_argument('--h', type=float, default=1/64)
    parser.add_argument('--threshold', type=float, default=.25)
    args = parser.parse_args()
    if not jax.config.x64_enabled:
        raise ValueError('FP64 required')
    pp, x, yy, cases = ef.load_inputs(args.inputs)
    if np.max(abs(x)) > 1 or args.eta <= 0:
        raise ValueError('Bounds require |x|<=1 and positive eta')
    selected = [i for i, case in enumerate(cases) if args.target == 'all' or case['target'] == args.target]
    if not selected or (args.target != 'all' and len(selected) != 1):
        raise ValueError('Select all, or exactly one target instance in this input cohort')
    q = transport.basis(x, 65)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    summaries = []
    for i in selected:
        summary = evaluate_case(pp[i], x, yy[i], q, cases[i], args,
                                args.output/f'{i:03d}_{cases[i]["target"]}')
        summaries.append(summary)
        print(cases[i]['target'], summary.get('status'), summary.get('evaluated_steps'), flush=True)
    write_json(args.output/'summary.json', summaries)


if __name__ == '__main__':
    main()
