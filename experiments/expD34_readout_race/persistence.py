"""Paired degree-nine GD forecasts with progressively evolving sensitivities."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np

from . import adam_forces as af, adam_run as ar, targets, transport

MODELS = ('full', 'five_mode', 'ten_mode', 'linear_features')


def features(p, p0, x, linear=False):
    (a, b, c), d = af.unpack(p)
    if linear:
        (a0, b0, _), _ = af.unpack(p0)
        u0 = x[:, None]*a0+b0
        h0 = jnp.tanh(u0)
        ex = jnp.exp(-2*jnp.abs(u0)); s = 4*ex/(1+ex)**2
        h = h0+s*(x[:, None]*(a-a0)+(b-b0))
    else:
        u = x[:, None]*a+b
        h = jnp.tanh(u)
        ex = jnp.exp(-2*jnp.abs(u)); s = 4*ex/(1+ex)**2
    return h, s, c, d


def field(p, p0, x, y, q, model, mass=None):
    mass = jnp.ones_like(x)/len(x) if mass is None else mass
    h, s, c, d = features(p, p0, x, model == 'linear_features')
    residual = h @ c+d-y
    active = q @ (q.T @ (mass*residual)) if q.shape[1] else residual
    weighted = mass*active
    g = jnp.r_[c*((mass*x) @ (active[:, None]*s)),
               c*(weighted @ s), h.T @ weighted, weighted.sum()]
    qc = jnp.stack((jnp.ones_like(x), x/jnp.sqrt(mass @ (x*x))))
    jq = qc*mass
    jc = jnp.concatenate(((jq @ (x[:, None]*s))*c,
                         (jq @ s)*c, jq @ h,
                         jq.sum(axis=1)[:, None]), axis=1)
    return g, residual, jc


def initial(p):
    w = (p.shape[-1]-1)//3
    return dict(p=p, positive=jnp.zeros_like(p[..., :w]),
                negative=jnp.zeros_like(p[..., :w]),
                tracking=jnp.zeros_like(p[..., :w]),
                path=jnp.zeros(p.shape[:-1]), tracking_path=jnp.zeros(p.shape[:-1]),
                crossing=jnp.zeros(p.shape[:-1]), min_coarse=jnp.full(p.shape[:-1], jnp.inf))


def linear_context(p0, x, y):
    """Exact fixed-feature Gram calculation; no rank or quadrature truncation."""
    h, s, c, d = features(p0, p0, x, True)
    psi = jnp.concatenate((x[:, None]*s, s, h, jnp.ones((len(x), 1))), axis=1)
    qc = jnp.stack((jnp.ones_like(x), x/jnp.sqrt(jnp.mean(x*x))))
    return psi.T @ psi/len(x), psi.T @ (h @ c+d-y)/len(x), qc @ psi/len(x)


def linear_field(p, p0, context):
    (a, b, c), d = af.unpack(p); (a0, b0, c0), d0 = af.unpack(p0)
    gram, v0, qc = context; w = len(a)
    delta = jnp.r_[c*(a-a0), c*(b-b0), c-c0, d-d0]
    v = v0+gram @ delta
    g = jnp.r_[c*v[:w], c*v[w:2*w], v[2*w:3*w]+(a-a0)*v[:w]+(b-b0)*v[w:2*w], v[-1]]
    jc = jnp.concatenate((qc[:, :w]*c, qc[:, w:2*w]*c,
        qc[:, 2*w:3*w]+qc[:, :w]*(a-a0)+qc[:, w:2*w]*(b-b0), qc[:, -1:]), axis=1)
    return g, jc


def advance_factory(x, q, model, eta, mass=None):
    def one(state, p0, y, length):
        context = linear_context(p0, x, y) if model == 'linear_features' else None
        def step(_, old):
            p = old['p']; w = (len(p)-1)//3
            if model == 'linear_features':
                g, jc = linear_field(p, p0, context)
            else:
                g, _, jc = field(p, p0, x, y, q, model, mass)
            C = jc @ jc.T; rhs = jc @ g
            det = C[0, 0]*C[1, 1]-C[0, 1]**2
            z = jnp.array([C[1, 1]*rhs[0]-C[0, 1]*rhs[1],
                           C[0, 0]*rhs[1]-C[0, 1]*rhs[0]])/det
            tracking = (jc.T @ z)[:w]
            mineig = .5*(jnp.trace(C)-jnp.sqrt((C[0, 0]-C[1, 1])**2+4*C[0, 1]**2))
            pn = p-eta*g
            delta = jnp.abs(pn[:w])-jnp.abs(p[:w])
            return dict(p=pn, positive=old['positive']+jnp.maximum(delta, 0),
                negative=old['negative']+jnp.maximum(-delta, 0),
                tracking=old['tracking']-eta*jnp.sign(p[:w])*tracking,
                path=old['path']+eta*jnp.linalg.norm(g[:w]),
                tracking_path=old['tracking_path']+eta*jnp.linalg.norm(tracking),
                crossing=old['crossing']+jnp.mean(delta+eta*jnp.sign(p[:w])*g[:w]),
                min_coarse=jnp.minimum(old['min_coarse'], mineig))
        return jax.lax.fori_loop(0, length, step, state)
    return jax.jit(jax.vmap(one, in_axes=(0, 0, 0, None)))


def load_inputs(source, start, seeds):
    if source.is_file():
        with np.load(source) as f:
            if str(f['target']) != 'moment9' or float(f['eta']) != .002:
                raise ValueError('Expected the curated degree-nine GD input pack')
            positions = np.flatnonzero(f['steps'] == start)
            if len(positions) != 1: raise ValueError(f'Missing checkpoint {start} in {source}')
            indices = [np.flatnonzero(f['seeds'] == seed)[0] for seed in seeds]
            pp, x, y = f['p'][indices, positions[0]], f['x'], f['y']
        return pp, x, y, {str(source): hashlib.sha256(source.read_bytes()).hexdigest()}
    pp = []; hashes = {}
    for seed in seeds:
        folder = source/f'primary_{seed}'
        cases = json.loads((folder/'manifest.json').read_text())['cases']
        matches = [i for i, c in enumerate(cases) if c['target'] == 'moment9' and c['optimizer'] == 'gd']
        if len(matches) != 1 or cases[matches[0]]['eta'] != .002:
            raise ValueError('Expected a unique unchanged degree-nine GD trajectory')
        path = folder/'snapshots.npz'
        with np.load(path) as f:
            positions = np.flatnonzero(f['steps'] == start)
            if len(positions) != 1: raise ValueError(f'Missing checkpoint {start} in {path}')
            pp.append(f['p'][matches[0], positions[0]])
        hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    x, y, _, _ = af.data('moment9')
    return np.stack(pp), x, y, hashes


def run(args):
    if not jax.config.x64_enabled: raise ValueError('FP64 is required')
    out = args.output; out.mkdir(parents=True, exist_ok=True)
    if args.backend == 'gpu':
        from .run import verify_gpu
        verify_gpu(out)
    elif any(d.platform != 'cpu' for d in jax.devices()):
        raise ValueError('CPU job exposed an accelerator backend')
    seeds = [int(s) for s in args.seeds.split(',')]
    p0, x, y, hashes = load_inputs(args.source, args.start, seeds)
    empirical_x = x.copy(); mass = np.ones_like(x)/len(x)
    if args.quadrature:
        if args.model == 'linear_features': raise ValueError('Linear features already use an exact Gram evaluation')
        from .persistence_quadrature import empirical_rule
        x, mass = empirical_rule(empirical_x, args.quadrature)
        mapping = targets.polynomial_map(empirical_x)
        y = targets.values('moment9', x, mapping)
        q = np.polynomial.legendre.legvander(x, 9) @ mapping
    else:
        q = transport.basis(x, 9)
    columns = (0, 1, 2, 3, 9) if args.model == 'five_mode' else tuple(range(10))
    q = q[:, columns] if args.model in ('five_mode', 'ten_mode') else np.empty((len(x), 0))
    factor = round(.002/args.eta)
    if args.eta not in (.002, .001): raise ValueError('Use reference or matched half step')
    manifest = dict(model=args.model, seeds=seeds, start=args.start, end=args.end, eta=args.eta,
        reference_eta=.002, width=(p0.shape[1]-1)//3, samples=len(empirical_x), backend=args.backend,
        quadrature_points=args.quadrature,
        source_commit=os.environ.get('RACE_SOURCE_COMMIT'), input_hashes=hashes,
        model_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        evidence_role='Deterministic training-grid optimization and autonomous forecasts')
    manifest_path = out/'manifest.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError('Changed continuation protocol')
    manifest_path.write_text(json.dumps(manifest, indent=2)+'\n')
    state = initial(jnp.asarray(p0)); snapshots = {args.start: jax.device_get(state)}; step = args.start
    prior_seconds = 0.
    if (out/'snapshots.npz').exists():
        with np.load(out/'snapshots.npz') as f:
            snapshots = {int(t): {k: f[k][:, i] for k in state} for i, t in enumerate(f['steps'])}
        step = max(snapshots); state = jax.tree.map(jnp.asarray, snapshots[step])
        prior_seconds = json.loads((out/'status.json').read_text())['seconds']
    advance = advance_factory(jnp.asarray(x), jnp.asarray(q), args.model, args.eta, jnp.asarray(mass))
    begun = time.monotonic(); yy = jnp.asarray(np.broadcast_to(y, (len(seeds), len(y))))
    def save():
        tt = sorted(snapshots); host = snapshots[step]; w = manifest['width']
        ar.atomic_npz(out/'snapshots.npz', steps=np.array(tt), p0=p0, x=x, y=y, mass=mass, empirical_x=empirical_x,
                      **{k: np.stack([snapshots[t][k] for t in tt], axis=1) for k in state})
        status = dict(step=step, complete=step == args.end,
            seconds=prior_seconds+time.monotonic()-begun,
            motion_identity=float(np.max(abs(host['positive']-host['negative']-(abs(host['p'][:, :w])-abs(p0[:, :w]))))),
            minimum_coarse_eigenvalue=float(host['min_coarse'].min()))
        (out/'status.json').write_text(json.dumps(status, indent=2)+'\n')
        print(json.dumps(status), flush=True)
    while step < args.end:
        end = min(step+args.stride, args.end)
        state = advance(state, jnp.asarray(p0), yy, (end-step)*factor)
        host = jax.device_get(state)
        if not all(np.all(np.isfinite(v)) for v in host.values()): raise FloatingPointError('Nonfinite forecast')
        if np.min(host['min_coarse']) <= 64*np.finfo(float).eps: raise ValueError('Unresolved coarse decomposition')
        step = end; snapshots[step] = host
        save()
        if time.monotonic()-begun >= args.max_seconds: break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--start', type=int, default=100000)
    parser.add_argument('--end', type=int, default=600000)
    parser.add_argument('--model', choices=MODELS, required=True)
    parser.add_argument('--eta', type=float, default=.002)
    parser.add_argument('--seeds', default='0,1,2,3,4')
    parser.add_argument('--stride', type=int, default=10000)
    parser.add_argument('--max-seconds', type=float, default=3300)
    parser.add_argument('--backend', choices=('cpu', 'gpu'), default='gpu')
    parser.add_argument('--quadrature', type=int, default=0)
    run(parser.parse_args())
