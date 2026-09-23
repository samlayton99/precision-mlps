"""Locked seeds 32/33: prepare ordinary-GD inputs, then forecast from 20k forks."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from types import SimpleNamespace
import jax
import jax.numpy as jnp
import numpy as np
from . import adam_run as ar, effective_feedback as ef, targets, mechanism_widths as mw
from . import mechanism_polynomial as poly, mechanism_splitting_baselines as baseline
from . import mechanism_polynomial_anchor as anchor


def prepare(root):
    out = root/'inputs'; out.mkdir(parents=True, exist_ok=False)
    for nref, halo in mw.WIDTHS:
        pp, yy, yeval, cases = [], [], [], []
        for seed in (32, 33):
            z, d = targets.initial(nref, halo, seed)
            for name in mw.TARGETS:
                x, y, _, scale = mw.data(name)
                xe, ye, _, _ = mw.data(name, 8192)
                pp.append(np.r_[z.ravel(), d]); yy.append(y); yeval.append(ye)
                cases.append(dict(target=name, seed=seed, start=0, eta=.002,
                    width=nref+2*halo+1, nref=nref, halo=halo, target_scale=scale))
        ar.atomic_npz(out/f'N{nref}.npz', p=np.asarray(pp), x=x, y=np.asarray(yy),
            x_eval=xe, y_eval=np.asarray(yeval), cases=np.array(json.dumps(cases)),
            sources=np.array(json.dumps(dict(helper_sha256=ef.digest(__file__),
                initialization_sha256=ef.digest(targets.__file__), seeds=[32, 33],
                statement='Locked same six targets and three widths; independent unchanged Xavier initialization'))))


def predict(root):
    if jax.default_backend() != 'cpu':
        raise RuntimeError('CPU-only forecast issuance')
    out = root/'forecasts'; out.mkdir(parents=True, exist_ok=False)
    forks = root/'fork_inputs'; forks.mkdir()
    for n, _ in mw.WIDTHS:
        source = root/'inputs'/f'N{n}.npz'
        _, x, yy, cases = ef.load_inputs(source)
        snapshot = root/'burnin'/f'N{n}'/'snapshots/000020000.npz'
        with np.load(snapshot) as data:
            if np.any(data['failed']) or int(data['offset']) != 20000:
                raise ValueError('Incomplete or failed burn-in')
            pp = data['p'].copy()
        with np.load(source) as data:
            xe, ye = data['x_eval'].copy(), data['y_eval'].copy()
        fork = forks/f'N{n}.npz'
        cases = [dict(c, start=20000) for c in cases]
        ar.atomic_npz(fork, p=pp, x=x, y=yy, x_eval=xe, y_eval=ye,
            cases=np.array(json.dumps(cases)), sources=np.array(json.dumps({str(source): ef.digest(source), str(snapshot): ef.digest(snapshot)})))
        arrays = dict(p0=pp)
        for degree in (3, 5):
            transform, _ = poly.modal_setup(x, yy[0], degree)
            target = np.stack([poly.modal_setup(x, y, degree)[1] for y in yy])
            arrays[f'poly{degree}'] = np.asarray(poly.predictor(degree, .002, 20000)(jnp.asarray(pp), jnp.asarray(transform), jnp.asarray(target)))
        arrays['anchored5'], arrays['anchor_correction'] = anchor.predict(pp, x, yy)
        path = out/f'N{n}.npz'
        np.savez_compressed(path, **arrays)
        (out/f'N{n}.json').write_text(json.dumps(dict(cases=cases, input_sha256=ef.digest(fork),
            prediction_sha256=ef.digest(path), source_sha256=ef.digest(__file__),
            kernel_sha256=ef.digest(poly.__file__), issued_utc=datetime.now(timezone.utc).isoformat(),
            anchor_sha256=ef.digest(anchor.__file__), models=['poly3', 'poly5', 'anchored5'],
            anchor_statement='Fixed initial exact-minus-quintic force correction; not continuously coarse tangent',
            role='prospective fixed-degree confirmation', degrees=[3, 5], eta=.002, steps=20000), indent=2))
        baseline.issue(SimpleNamespace(inputs=fork, snapshot=None, output=root/'baselines'/f'N{n}',
            all_policies=False, horizons='20000', eta=.002, role='prospective'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'predict'))
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    (prepare if args.command == 'prepare' else predict)(args.root)
