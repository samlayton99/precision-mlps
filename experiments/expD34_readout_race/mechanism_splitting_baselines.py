"""Issue checkpoint-only constant-driver and Schur forecasts before continuations.

Also accepts a genuine-width checkpoint. All matrices are at the supplied fork;
the ordinary training runners and their source hashes remain unchanged.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from . import effective_feedback as ef, mechanism_splitting as ms, transport
from .effective_feedback_predict import frozen_effective_forecast
from .run import write_json


def matrices(p, x, y, d, q):
    w = (len(p)-1)//3
    a, b, c = p[:-1].reshape(3, w)
    u = x[:, None]*a+b
    features = np.tanh(u)
    exponential = np.exp(-2*abs(u)); derivative = 4*exponential/(1+exponential)**2
    J = np.concatenate((x[:, None]*derivative*c, derivative*c, features, np.ones((len(x), 1))), axis=1)
    residual = features@c+p[-1]-y
    modal = q.T@J/len(x); JC, JH = modal[:2], modal[2:]
    C = (JC*d)@JC.T
    eigen = np.linalg.eigvalsh(C)
    if eigen[0] <= 64*np.finfo(float).eps*max(1., eigen[-1]):
        raise ValueError('Unresolved metric coarse inverse; forecast not issued')
    B = np.linalg.solve(C, (JC*d)@JH.T)
    T = d[:, None]*(JH.T-JC.T@B)
    eH = q[:, 2:].T@residual/len(x)
    gradient = d*(J.T@residual/len(x))
    return gradient, T, JH, eH


def issue(args):
    pp, x, yy, cases = ef.load_inputs(args.inputs)
    sources = {str(args.inputs): ef.digest(args.inputs)}
    if args.snapshot:
        with np.load(args.snapshot) as data:
            if np.any(data['failed']):
                raise ValueError('Cannot issue forecasts from failed checkpoint cases')
            pp = data['p'].copy(); offset = int(data['offset'])
        cases = [dict(c, start=c['start']+offset) for c in cases]
        sources[str(args.snapshot)] = ef.digest(args.snapshot)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    horizons = np.array([int(v) for v in args.horizons.split(',')])
    arms = ms.ARMS if args.all_policies else ('original',)
    q = transport.basis(x, 65)
    qcheck = transport.basis(x, 129) if args.snapshot else None
    arrays = dict(p0=pp, horizons=horizons)
    names = ('constant_full_p', 'constant_effective_p', 'effective_pure_p',
             'effective_remainder_p', 'gradient0', 'effective0', 'S0', 'S_blocks',
             'fine_projection_difference', 'effective_norm')
    collected = {name: [] for name in names}
    for arm in arms:
        arm_arrays = {name: [] for name in names}
        for p, y in zip(pp, yy):
            w = (len(p)-1)//3
            d = ms.mobility(w, arm)
            g, T, JH, eH = matrices(p, x, y, d, q)
            effective = T@eH
            forecasts = frozen_effective_forecast(p, g, T, JH, eH, args.eta, horizons)
            discrepancy = np.nan
            if qcheck is not None:
                _, fineT, _, finee = matrices(p, x, y, d, qcheck)
                discrepancy = np.linalg.norm((fineT@finee-effective)[:w])
            # These are signed residual-forcing contributions, not PSD energy
            # allocations. PSD block forms are T_block.T D_block^-1 T_block.
            blocks = [JH[:, lo:hi]@T[lo:hi] for lo, hi in ((0, w), (w, 2*w), (2*w, 3*w), (3*w, 3*w+1))]
            values = dict(constant_full_p=p-args.eta*horizons[:, None]*g,
                constant_effective_p=p-args.eta*horizons[:, None]*effective,
                effective_pure_p=forecasts['pure']['p'], effective_remainder_p=forecasts['with_remainder']['p'],
                gradient0=g, effective0=effective, S0=forecasts['S0'], S_blocks=np.asarray(blocks),
                fine_projection_difference=discrepancy, effective_norm=np.linalg.norm(effective[:w]))
            for name in names:
                arm_arrays[name].append(values[name])
        for name in names:
            collected[name].append(arm_arrays[name])
    arrays.update({name: np.asarray(value) for name, value in collected.items()})
    np.savez_compressed(args.output/'predictions.npz', **arrays)
    write_json(args.output/'manifest.json', dict(cases=cases, arms=arms, eta=args.eta,
        sources=sources, source_sha256=ef.digest(__file__),
        issued_utc=datetime.now(timezone.utc).isoformat(), horizons=horizons.tolist(),
        evidence_role=args.role, degree=65, audit_degree=129 if qcheck is not None else None,
        blocks=['a', 'b', 'c', 'd'], block_meaning='S_blocks are signed J_H,block T_velocity,block contributions, not PSD energy shares',
        h_by_case=[2/c.get('nref', 128) for c in cases],
        prediction_sha256=ef.digest(args.output/'predictions.npz'),
        statement='All models use only the supplied checkpoint. Constant-driver models expose immediate mobility scaling; Schur models additionally evolve fine residuals.'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--snapshot', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--all-policies', action='store_true')
    parser.add_argument('--horizons', default='1,10,1000,10000,20000,80000')
    parser.add_argument('--eta', type=float, default=.002)
    parser.add_argument('--role', choices=('prospective', 'retrospective_reference'), default='prospective')
    issue(parser.parse_args())


if __name__ == '__main__':
    main()
