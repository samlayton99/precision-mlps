"""Compare physical frozen-geometry readouts with the construction's h scale.

Sine only, zero-start physical-coordinate GD and Adam. GD uses the existing
finite-time SVD recurrence; Adam iterates a QR factorization of the same loss.
This exploratory check writes numerical evidence, never a prose report.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from . import adam_analyze, mechanism, targets

HORIZONS = (0, 2000, 20000, 100000, 600000)
LAMBDAS = (.25, .5, 1.)
ADAM_RATES = (.002, .0002)


def dictionary(n, lam, m=2048):
    """Preserve D34's physical halo extent as the center spacing is refined."""
    h = 2/n
    halo = 3*n//16
    centers = -1+h*np.arange(-halo, n+halo+1)
    gamma = lam/h
    x = targets.grid(m)
    y = np.sin(2*np.pi*x)
    A = mechanism.design(np.full(len(centers), gamma), -gamma*centers, x)
    q, factor = np.linalg.qr(A/np.sqrt(m), mode='reduced')
    return centers, x, y, A, factor, q.T @ (y/np.sqrt(m))


def initial(batch, size):
    z = jnp.zeros((batch, size), dtype=jnp.float64)
    return z, z, z, jnp.array(0, dtype=jnp.int64)


@jax.jit
def advance(state, factor, rhs, rates, steps):
    """Ordinary Adam, beta=(.9,.999), epsilon=1e-8, no decay or weight penalty."""
    def step(_, carry):
        c, first, second, count = carry
        residual = jnp.einsum('bij,bj->bi', factor, c)-rhs
        g = jnp.einsum('bij,bi->bj', factor, residual)
        count = count+1
        first = .9*first+.1*g
        second = .999*second+.001*g*g
        delta = rates[:, None]*(first/(1-.9**count))/(jnp.sqrt(second/(1-.999**count))+1e-8)
        return c-delta, first, second, count
    return jax.lax.fori_loop(0, steps, step, state)


def metrics(c, centers, n, gamma, x, y, A):
    h = 2/n
    xe = targets.grid(8192)
    ye = np.sin(2*np.pi*xe)
    Ae = mechanism.design(np.full(len(centers), gamma), -gamma*centers, xe)
    weights = c[:-1]
    row = dict(relative_train_mse=float(np.mean((A @ c-y)**2)/np.mean(y*y)),
               relative_eval_mse=float(np.mean((Ae @ c-ye)**2)/np.mean(ye*ye)),
               hidden_l1=float(np.sum(abs(weights))), hidden_l2=float(np.linalg.norm(weights)),
               bias=float(c[-1]), width=len(centers), h=h, gamma=gamma)
    for label, mask in [('interior', abs(centers)<=.75), ('core', abs(centers)<=1),
                        ('halo', abs(centers)>1), ('all', np.ones(len(centers), dtype=bool))]:
        values = weights[mask]
        row.update({label+'_rms_over_h': float(np.sqrt(np.mean(values**2))/h),
                    label+'_max_over_h': float(np.max(abs(values))/h),
                    label+'_median_abs_over_h': float(np.median(abs(values))/h)})
    # Exact ordinary interior density from the construction, without halo corrections.
    z = np.pi*np.pi/gamma
    reference = np.pi*np.sinh(z)/z*np.cos(2*np.pi*centers)
    mask = abs(centers)<=.75
    row['density_reference_rms_over_h'] = float(np.sqrt(np.mean(reference[mask]**2)))
    row['density_reference_relative_error'] = float(np.linalg.norm(weights[mask]/h-reference[mask])
                                                  /np.linalg.norm(reference[mask]))
    # Frozen geometry: this is an assay of the signal a joint update would see.
    u = gamma*(x[:, None]-centers)
    sech2 = 1-np.tanh(u)**2
    residual = A @ c-y
    raw_slope = weights*(x @ (residual[:, None]*sech2))/len(x)
    centered_slope = weights*np.mean(residual[:, None]*(x[:, None]-centers)*sech2, axis=0)
    row['virtual_raw_slope_gradient_norm'] = float(np.linalg.norm(raw_slope))
    row['virtual_fixed_center_slope_gradient_norm'] = float(np.linalg.norm(centered_slope))
    return row


def run(output, widths=(64, 128, 256), horizons=HORIZONS, m=2048):
    if not jax.config.x64_enabled:
        raise ValueError('Run with JAX_ENABLE_X64=true')
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    rows, checks, controls, windows, packs = [], [], [], [], {}
    for n in widths:
        cases, factors, right, designs = [], [], [], []
        for lam in LAMBDAS:
            centers, x, y, A, factor, rhs = dictionary(n, lam, m)
            gamma = lam*n/2
            curves, _, coefficients = mechanism.frozen_curves(
                np.full(len(centers), gamma), -gamma*centers, x, y,
                targets.grid(8192), np.sin(2*np.pi*targets.grid(8192)), horizons=horizons)
            for k, (count, c) in enumerate(zip(horizons, coefficients)):
                row = dict(optimizer='gd', n=n, lam=lam, eta=.002, step=count,
                           **metrics(c, centers, n, gamma, x, y, A))
                assert abs(row['relative_eval_mse']-curves[k]['relative_heldout_mse'])<1e-12
                rows.append(row)
                packs[f'gd_n{n}_l{lam}_s{count}'] = c
            for rate in ADAM_RATES:
                cases.append(dict(optimizer='adam', n=n, lam=lam, eta=rate))
                factors.append(factor); right.append(rhs); designs.append((centers, x, y, A))
        factors, right = np.stack(factors), np.stack(right)
        state = initial(len(cases), factors.shape[-1])
        rates = jnp.asarray([case['eta'] for case in cases])
        previous = 0
        for count in horizons:
            state = advance(state, jnp.asarray(factors), jnp.asarray(right), rates, count-previous)
            cs = np.asarray(state[0])
            for i, case in enumerate(cases):
                centers, x, y, A = designs[i]
                c = cs[i]
                direct = A.T @ (A @ c-y)/len(x)
                factored = factors[i].T @ (factors[i] @ c-right[i])
                checks.append(dict(**case, step=count, gradient_difference=float(np.max(abs(direct-factored)))))
                rows.append(dict(**case, step=count, **metrics(c, centers, n, case['lam']*n/2, x, y, A)))
                packs[f'adam_n{n}_l{case["lam"]}_e{case["eta"]}_s{count}'] = c
            previous = count
            adam_analyze.write_csv(output/'measurements.csv', rows)
            adam_analyze.write_csv(output/'verification.csv', checks)
            np.savez_compressed(output/'coefficients.npz', **packs)
            print(json.dumps(dict(n=n, step=count, elapsed=time.perf_counter()-started)), flush=True)
        # Consecutive states distinguish an endpoint from an Adam oscillation.
        @jax.jit
        def window(state):
            def step(carry, _):
                nxt = advance(carry, jnp.asarray(factors), jnp.asarray(right), rates, 1)
                return nxt, nxt[0]
            return jax.lax.scan(step, state, None, length=256)
        _, window_cs = window(state)
        window_cs = np.asarray(window_cs)
        for i, case in enumerate(cases):
            centers, x, y, A = designs[i]
            coeff = window_cs[:, i]
            interior = abs(centers)<=.75
            scale = np.sqrt(np.mean(coeff[:, :-1][:, interior]**2, axis=1))/(2/n)
            errors = np.mean((A @ coeff.T-y[:, None])**2, axis=0)/np.mean(y*y)
            windows.append(dict(**case, start=horizons[-1]+1, end=horizons[-1]+256,
                interior_rms_over_h_min=float(scale.min()), interior_rms_over_h_max=float(scale.max()),
                relative_train_mse_min=float(errors.min()), relative_train_mse_max=float(errors.max())))
            packs[f'window_n{n}_l{case["lam"]}_e{case["eta"]}'] = coeff
        if n==128:
            # Independent representation of exactly the same objective; no QR.
            direct_factors = jnp.array(np.stack([d[3]/np.sqrt(len(d[1])) for d in designs]))
            direct_rhs = jnp.array(np.stack([d[2]/np.sqrt(len(d[1])) for d in designs]))
            direct = advance(initial(len(cases), factors.shape[-1]), direct_factors, direct_rhs, rates, 20000)
            for i, case in enumerate(cases):
                centers, x, y, A = designs[i]
                coeff = np.asarray(direct[0][i])
                controls.append(dict(**case, step=20000, **metrics(coeff, centers, n, case['lam']*n/2, x, y, A)))
                packs[f'direct_n{n}_l{case["lam"]}_e{case["eta"]}'] = coeff
        adam_analyze.write_csv(output/'terminal_windows.csv', windows)
        adam_analyze.write_csv(output/'direct_controls.csv', controls)
        np.savez_compressed(output/'coefficients.npz', **packs)
    manifest = dict(widths=list(widths), lambdas=LAMBDAS, horizons=list(horizons), samples=m,
        eval_samples=8192, target='sin(2*pi*x), unnormalized',
        initialization='zero physical readouts and bias', gd_eta=.002,
        adam_rates=ADAM_RATES, adam_betas=[.9,.999], adam_epsilon=1e-8,
        terminal_window='256 consecutive additional updates', direct_controls='N128, first 20k updates, all gammas/rates',
        geometry='fixed centers -1+j*(2/N), j=-3N/16,...,N+3N/16; gamma=lambda/h',
        gd_method='finite-time SVD recurrence', adam_method='QR-factor residual, ordinary Adam updates',
        no_readout_norm_constraint=True, no_model_selection=True, backend=jax.default_backend(),
        jax_version=jax.__version__, numpy_version=np.__version__, elapsed_seconds=time.perf_counter()-started,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--widths', type=int, nargs='+', default=[64,128,256])
    parser.add_argument('--horizons', type=int, nargs='+', default=list(HORIZONS))
    parser.add_argument('--samples', type=int, default=2048)
    args = parser.parse_args()
    run(args.output, tuple(args.widths), tuple(args.horizons), args.samples)
