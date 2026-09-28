"""Own-state projected polynomial dynamics; no tracking and no fitted constants."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from math import comb
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from . import effective_feedback as ef, transport
from .mechanism_splitting_diagnostics import write_csv


def modal_setup(x, y, degree):
    q = transport.basis(x, degree)
    monomials = np.asarray(x)[:, None]**np.arange(degree+1)
    return q.T@monomials/len(x), q.T@np.asarray(y)/len(x)


def coefficients_jacobian(p, degree):
    """Exact binomial coefficients and their derivatives in parameter order a,b,c,d."""
    w = (p.shape[0]-1)//3
    a, b, c = p[:-1].reshape(3, w)
    rows, ja, jb, jc = [], [], [], []
    terms = ((1, 1.), (3, -1/3), (5, 2/15))
    for r in range(degree+1):
        value = jnp.zeros_like(a); da = jnp.zeros_like(a); db = jnp.zeros_like(a)
        for power, factor in terms:
            if power > degree or r > power:
                continue
            scale = factor*comb(power, r)
            value = value+scale*a**r*b**(power-r)
            if r:
                da = da+scale*r*a**(r-1)*b**(power-r)
            if power > r:
                db = db+scale*(power-r)*a**r*b**(power-r-1)
        rows.append(jnp.sum(c*value)+(p[-1] if r == 0 else 0.))
        ja.append(c*da); jb.append(c*db); jc.append(value)
    bias = jnp.zeros((degree+1, 1)).at[0, 0].set(1.)
    return jnp.stack(rows), jnp.concatenate((jnp.stack(ja), jnp.stack(jb), jnp.stack(jc), bias), axis=1)


def field(p, transform, target, degree):
    coefficients, jacobian = coefficients_jacobian(p, degree)
    modal = transform@jacobian
    JC, JH = modal[:2], modal[2:]
    residual = (transform@coefficients-target)[2:]
    raw = JH.T@residual
    coarse = JC@JC.T
    force = raw-JC.T@jnp.linalg.solve(coarse, JC@raw)
    return force, coarse


def predictor(degree, eta, steps):
    def one(p, transform, target):
        def update(_, old):
            return old-eta*field(old, transform, target, degree)[0]
        return jax.lax.fori_loop(0, steps, update, p)
    return jax.jit(jax.vmap(one, in_axes=(0, None, 0)))


def run(args):
    if jax.default_backend() != 'cpu':
        raise RuntimeError('This surrogate experiment is authorized for CPU only')
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []; sources = {}
    for n in (128, 512, 1024):
        source = args.root/'width_inputs'/f'N{n}_fork20000.npz'
        pp, x, yy, cases = ef.load_inputs(source)
        sources[str(source)] = ef.digest(source)
        actual_path = args.root/'widths_post20k'/f'N{n}'/'snapshots/000020000.npz'
        with np.load(actual_path) as data:
            actual = data['p'].copy()
        sources[str(actual_path)] = ef.digest(actual_path)
        benchmark = args.root/'width_predictions'/f'N{n}'/'predictions.npz'
        with np.load(benchmark) as data:
            hi = list(data['horizons']).index(args.steps)
            constant = data['constant_effective_p'][0, :, hi].copy()
            frozen = data['effective_pure_p'][0, :, hi].copy()
        sources[str(benchmark)] = ef.digest(benchmark)
        w = (pp.shape[1]-1)//3
        results = dict(p0=pp)
        for degree in (3, 5):
            transform, _ = modal_setup(x, yy[0], degree)
            targets = np.stack([modal_setup(x, y, degree)[1] for y in yy])
            states = np.asarray(predictor(degree, args.eta, args.steps)(jnp.asarray(pp), jnp.asarray(transform), jnp.asarray(targets)))
            results[f'poly{degree}'] = states
            for i, case in enumerate(cases):
                motion = np.linalg.norm(actual[i, :w]-pp[i, :w])
                force, coarse = field(jnp.asarray(pp[i]), jnp.asarray(transform), jnp.asarray(targets[i]), degree)
                row = dict(case, degree=degree, steps=args.steps, finite=bool(np.isfinite(states[i]).all()),
                    initial_coarse_min_eigenvalue=float(np.linalg.eigvalsh(np.asarray(coarse))[0]),
                    initial_slope_force_norm=float(np.linalg.norm(np.asarray(force)[:w])),
                    actual_slope_motion=float(motion))
                for name, state in [('polynomial', states), ('constant_effective', constant), ('frozen_effective', frozen)]:
                    error = float(np.linalg.norm(state[i, :w]-actual[i, :w]))
                    row[name+'_error'] = error
                    row[name+'_relative_error'] = error/motion if motion else np.nan
                    row[name+'_lambda_change'] = float(2/n*np.mean(abs(state[i, :w])-abs(pp[i, :w])))
                rows.append(row)
        np.savez_compressed(args.output/f'N{n}.npz', **results)
    write_csv(args.output/'scores.csv', rows)
    (args.output/'manifest.json').write_text(json.dumps(dict(source_sha256=ef.digest(__file__), sources=sources,
        issued_utc=datetime.now(timezone.utc).isoformat(), eta=args.eta, steps=args.steps,
        role='retrospective model evaluation', degrees=[3, 5],
        statement='Own-state polynomial Jacobian and coarse projector every update; all parameter blocks; tracking omitted; no input quadrature during stepping'), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=20000)
    parser.add_argument('--eta', type=float, default=.002)
    run(parser.parse_args())
