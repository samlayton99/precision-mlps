"""Interval-check an archived readout as a capacity witness, never as GD."""
from __future__ import annotations

import argparse
from pathlib import Path
import time

import numpy as np

from . import core


def certify_readout(x, centers, gamma, theta, target, precision=128):
    from flint import arb, ctx
    old = ctx.prec; ctx.prec = precision
    begun = time.monotonic()
    try:
        x, centers, theta, target = map(np.asarray, [x, centers, theta, target])
        if theta.shape != (len(centers)+1,) or target.shape != x.shape:
            raise ValueError('Capacity witness dimensions do not match the dictionary')
        coefficients = [arb(float(t)) for t in theta]
        nodes = [arb(float(c)) for c in centers]
        slope = arb(float(gamma)); scale = arb(len(x)).sqrt()
        error, energy = arb(0), arb(0)
        for point, truth in zip(x, target):
            point = arb(float(point)); truth = arb(float(truth))
            value = coefficients[0]+sum((weight*(slope*(point-center)).tanh()
                       for center, weight in zip(nodes, coefficients[1:])), arb(0))
            error += (value/scale-truth)**2
            energy += truth**2
        relative = (error/energy).sqrt()
        return dict(status='interval_certified_capacity',
            relative_error_upper=float(np.nextafter(float(relative.upper()), np.inf)),
            relative_error_lower=max(0., float(np.nextafter(float(relative.lower()), -np.inf))),
            coefficient_norm_upper=float(np.nextafter(float(sum((c*c for c in coefficients), arb(0)).sqrt().upper()), np.inf)),
            precision_bits=precision, seconds=time.monotonic()-begun,
            grid_hash=core.array_hash(x), centers_hash=core.array_hash(centers),
            target_hash=core.array_hash(target), theta_hash=core.array_hash(theta),
            input_semantics='saved binary inputs and coefficients; real tanh and sample normalization')
    finally:
        ctx.prec = old


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--n', type=int, default=512)
    parser.add_argument('--caps', nargs='+', type=float, default=[4, 8, 16])
    args = parser.parse_args()
    common = np.load(args.archive/f'common/N{args.n}/arrays.npz')
    rows = []
    for cap in args.caps:
        path = args.archive/'dictionaries'/f'N{args.n}_raw_g{cap:g}'/'spectrum.npz'
        theta = np.load(path)['refit_theta'][:, 0]
        proof = certify_readout(common['x_train'], common['centers'], cap, theta, common['y_train'][:, 0])
        proof.update(n=args.n, cap=cap, target='sine_mix_2_6_10', source=str(path.relative_to(args.archive)))
        rows.append(proof)
        core.write_json(args.output, rows)
        print('CAPACITY', cap, proof['relative_error_upper'], proof['seconds'], flush=True)


if __name__ == '__main__':
    main()
