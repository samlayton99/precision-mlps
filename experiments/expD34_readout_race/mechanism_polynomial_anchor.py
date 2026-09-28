"""Checkpoint-anchored quintic field; the fixed correction need not remain tangent."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from datetime import datetime, timezone
import jax
import jax.numpy as jnp
import numpy as np
from . import effective_feedback as ef, transport, mechanism_polynomial as poly
from .mechanism_splitting_baselines import matrices
from .mechanism_splitting_diagnostics import write_csv


def anchored_predictor(eta, steps):
    def one(p, transform, target, correction):
        def update(_, old):
            return old-eta*(poly.field(old, transform, target, 5)[0]+correction)
        return jax.lax.fori_loop(0, steps, update, p)
    return jax.jit(jax.vmap(one, in_axes=(0, None, 0, 0)))


def predict(pp, x, yy, steps=20000):
    transform, _ = poly.modal_setup(x, yy[0], 5)
    targets = np.stack([poly.modal_setup(x, y, 5)[1] for y in yy])
    correction = []
    q = transport.basis(x, 65)
    for p, y, target in zip(pp, yy, targets):
        _, T, _, e = matrices(p, x, y, np.ones(len(p)), q)
        initial = np.asarray(poly.field(jnp.asarray(p), jnp.asarray(transform), jnp.asarray(target), 5)[0])
        correction.append(T@e-initial)
    correction = np.asarray(correction)
    states = np.asarray(anchored_predictor(.002, steps)(jnp.asarray(pp), jnp.asarray(transform), jnp.asarray(targets), jnp.asarray(correction)))
    return states, correction


def coarse(p, x):
    a, b, c = p[:-1].reshape(3, -1)
    return transport.basis(x, 1).T@(np.tanh(x[:, None]*a+b)@c+p[-1])/len(x)


def retrospective(root):
    out = root/'polynomial_anchor'; out.mkdir(exist_ok=False)
    rows = []; sources = {}
    for n in (128, 512, 1024):
        source = root/'width_inputs'/f'N{n}_fork20000.npz'
        pp, x, yy, cases = ef.load_inputs(source)
        states, correction = predict(pp, x, yy)
        np.savez_compressed(out/f'N{n}.npz', p0=pp, anchored5=states, correction=correction)
        future = root/'widths_post20k'/f'N{n}'/'snapshots/000020000.npz'
        with np.load(future) as data:
            actual = data['p'].copy()
        with np.load(root/'polynomial'/f'N{n}.npz') as data:
            pure = data['poly5'].copy()
        w = (pp.shape[1]-1)//3
        for i, case in enumerate(cases):
            motion = np.linalg.norm(actual[i, :w]-pp[i, :w])
            row = dict(case, degree=5, steps=20000, finite=bool(np.isfinite(states[i]).all()),
                actual_slope_motion=float(motion), anchored_relative_error=float(np.linalg.norm(states[i, :w]-actual[i, :w])/motion),
                pure_relative_error=float(np.linalg.norm(pure[i, :w]-actual[i, :w])/motion))
            c0 = coarse(pp[i], x)
            for label, state in [('anchored', states[i]), ('pure', pure[i]), ('actual', actual[i])]:
                row[label+'_exact_coarse_drift'] = float(np.linalg.norm(coarse(state, x)-c0))
            rows.append(row)
        sources[str(source)] = ef.digest(source); sources[str(future)] = ef.digest(future)
    write_csv(out/'scores.csv', rows)
    (out/'manifest.json').write_text(json.dumps(dict(source_sha256=ef.digest(__file__),
        kernel_sha256=ef.digest(poly.__file__), issued_utc=datetime.now(timezone.utc).isoformat(),
        sources=sources, role='retrospective anchored-quintic reference',
        statement='Constant full-parameter initial exact-minus-quintic force correction; not constrained to instantaneous coarse tangent space away from fork'), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    retrospective(parser.parse_args().root)
