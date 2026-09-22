"""Exact effective-force derivatives and frozen-tangent forecasts at D34 states."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from . import adam_analyze as aa, adam_forces as af, targets


def prediction(p, x):
    (a, b, c), d = af.unpack(p)
    return jnp.tanh(x[:, None]*a+b) @ c+d


def tangent(p, x):
    (a, b, c), _ = af.unpack(p)
    u = x[:, None]*a+b
    ex = jnp.exp(-2*jnp.abs(u)); s = 4*ex/(1+ex)**2
    return jnp.concatenate((x[:, None]*s*c, s*c, jnp.tanh(u), jnp.ones((len(x), 1))), axis=1)


def coarse(x):
    return jnp.stack((jnp.ones_like(x), x/jnp.sqrt(jnp.mean(x*x))))


def effective(p, residual, x):
    """All-parameter effective gradient: project off the coarse Jacobian rows."""
    J = tangent(p, x); jc = coarse(x) @ J/len(x)
    g = J.T @ residual/len(x)
    return g-jc.T @ jnp.linalg.solve(jc @ jc.T, jc @ g)


@jax.jit
def diagnostics(p, m, v, cm, count, y, x, settings):
    eta, b1, b2, eps, adaptive = settings
    w = (len(p)-1)//3
    g, r, jc, ec = af.field(p, x, y)
    channels, info = af.split(g, jc, ec)
    _, _, _, mh, ch, inverse = af.moments(g, channels, m, v, cm, count+1, b1, b2, eps, adaptive)
    velocity = -inverse*mh
    ve, vz = -inverse*ch[0], -inverse*ch[1]
    F = effective(p, r, x)[:w]
    fn = lambda pp: effective(pp, r, x)[:w]
    shape = jax.jvp(fn, (p,), (velocity.at[2*w:].set(0.),))[1]
    readout = jax.jvp(fn, (p,), (velocity.at[:2*w].set(0.),))[1]
    J = tangent(p, x)
    residual_eff = effective(p, J @ ve, x)[:w]
    residual_track = effective(p, J @ vz, x)[:w]
    total = jax.jvp(lambda pp: effective(pp, prediction(pp, x)-y, x)[:w], (p,), (velocity,))[1]
    parts = jnp.stack((shape, readout, residual_eff, residual_track))
    f2 = F @ F; fine = r-coarse(x).T @ ec
    target = effective(p, -y, x)[:w]
    generated = effective(p, prediction(p, x), x)[:w]
    actual_next = effective(p+eta*velocity, prediction(p+eta*velocity, x)-y, x)[:w]
    return dict(force=F, parts=parts, total=total, velocity=velocity[:w],
        force_norm=jnp.sqrt(f2), fine_norm=jnp.linalg.norm(fine)/jnp.sqrt(len(x)),
        coupling=f2/jnp.mean(fine*fine), alignment=-jnp.sign(p[:w]) @ F/jnp.sqrt(w*f2),
        participation=f2*f2/(w*jnp.sum(F**4)),
        log_rates=parts @ F/f2, driver_norms=jnp.linalg.norm(parts, axis=1),
        total_log_rate=F @ total/f2, target_force_norm=jnp.linalg.norm(target),
        relative_vector_rate=jnp.linalg.norm(total)/jnp.sqrt(f2),
        derivative_cancellation=jnp.linalg.norm(total)/jnp.maximum(jnp.sum(jnp.linalg.norm(parts,axis=1)),1e-300),
        generated_force_norm=jnp.linalg.norm(generated),
        target_outward=-jnp.mean(jnp.sign(p[:w])*target),
        generated_outward=-jnp.mean(jnp.sign(p[:w])*generated),
        force_identity=jnp.linalg.norm(F-channels[0, :w]),
        derivative_identity=jnp.linalg.norm(total-parts.sum(axis=0)),
        projection_identity=jnp.linalg.norm(jc @ effective(p, r, x)),
        target_identity=jnp.linalg.norm(F-target-generated), resolved=info['resolved'],
        next_step_relative_remainder=jnp.linalg.norm(actual_next-F-eta*total)/jnp.maximum(jnp.linalg.norm(actual_next-F), 1e-300),
        mean_gamma=jnp.mean(abs(p[:w])), relative_mse=jnp.mean(r*r)/jnp.mean(y*y),
        readout_l2=jnp.linalg.norm(p[2*w:3*w]))


def frozen_tangent(p, x, y, horizons, eta=.002):
    """Exact discrete GD on the local linearized model; no fitted coefficients."""
    J = np.asarray(tangent(jnp.asarray(p), jnp.asarray(x)))
    r = np.asarray(prediction(jnp.asarray(p), jnp.asarray(x)))-y
    g = J.T @ r/len(x)
    values, vectors = np.linalg.eigh(J.T @ J/len(x))
    if values[0] < -1e-12*max(1., values[-1]): raise ValueError('Non-PSD tangent Gram matrix')
    values = np.maximum(values, 0.)
    if eta*values[-1]>=1: raise ValueError('Frozen forecast requires nonoscillating GD')
    jc = np.asarray(coarse(jnp.asarray(x))) @ J/len(x); loading = vectors.T @ g
    result = []
    for n in horizons:
        exponent = n*np.log1p(-eta*values); decay = np.exp(exponent)
        integral = np.divide(-np.expm1(exponent), values, out=np.full_like(values, eta*n), where=values>0)
        gn = vectors @ (decay*loading)
        force = gn-jc.T @ np.linalg.solve(jc @ jc.T, jc @ gn)
        result.append((force[:(len(p)-1)//3], p-vectors @ (integral*loading)))
    return result


DRIVERS = ('shape', 'readout', 'residual_effective', 'residual_tracking')


def analyze(root, output, seed_limit=5, snapshots_root=None):
    output.mkdir(parents=True, exist_ok=True)
    source = snapshots_root if snapshots_root is not None else root/'raw'
    rows = []; windows = []; forecasts = []; vectors = []; identities = []; hashes = {}
    for seed in range(seed_limit):
        folder = source/f'primary_{seed}'
        cases = json.loads((folder/'manifest.json').read_text())['cases']; f = np.load(folder/'snapshots.npz')
        required=(20000,100000,200000,400000,600000)
        if not set(required).issubset(set(f['steps'])):
            raise ValueError(f'{folder} omits required states; use the full training archive, not curated snapshots')
        hashes[str(folder/'snapshots.npz')]=hashlib.sha256((folder/'snapshots.npz').read_bytes()).hexdigest()
        for i, case in enumerate(cases):
            x, y, _, _ = af.data(case['target']); xx = jnp.asarray(x); yy = jnp.asarray(y)
            settings = jnp.array([case[k] for k in ('eta', 'beta1', 'beta2', 'epsilon', 'adaptive')]); records = {}
            for step in required:
                si = int(np.flatnonzero(f['steps']==step)[0])
                args = [jnp.asarray(f[k][i, si]) for k in ('p', 'm', 'v', 'channel_m', 'count')]
                d = jax.device_get(diagnostics(*args, yy, xx, settings))
                row = dict(target=case['target'], optimizer=case['optimizer'], seed=seed, step=step)
                row.update({k:float(v) for k,v in d.items() if np.ndim(v)==0})
                for k, name in enumerate(DRIVERS):
                    row['rate_'+name] = float(d['log_rates'][k]); row['norm_'+name] = float(d['driver_norms'][k])
                rows.append(row); vectors.append(d['force']); records[step] = d
                identities.append(max(row[k] for k in ('force_identity', 'derivative_identity', 'projection_identity', 'target_identity')))
            for start, end in ((20000, 100000), (100000, 600000), (400000, 600000)):
                a,b = records[start],records[end]; fa,fb = a['force'],b['force']
                windows.append(dict(target=case['target'], optimizer=case['optimizer'], seed=seed, start=start, end=end,
                    force_ratio=b['force_norm']/a['force_norm'], residual_ratio=b['fine_norm']/a['fine_norm'],
                    coupling_ratio=b['coupling']/a['coupling'], force_cosine=fa @ fb/(np.linalg.norm(fa)*np.linalg.norm(fb)),
                    start_alignment=a['alignment'], end_alignment=b['alignment'], mean_gamma_change=b['mean_gamma']-a['mean_gamma']))
            if case['optimizer']=='gd':
                si = int(np.flatnonzero(f['steps']==100000)[0]); p = f['p'][i, si]
                predicted = frozen_tangent(p, x, y, [100000, 300000, 500000])
                for end, (force, pp) in zip((200000, 400000, 600000), predicted):
                    actual = records[end]['force']; base = records[100000]['force']
                    forecasts.append(dict(target=case['target'], seed=seed, start=100000, end=end,
                        actual_norm=np.linalg.norm(actual), frozen_norm=np.linalg.norm(force), constant_norm=np.linalg.norm(base),
                        frozen_relative_vector_error=np.linalg.norm(force-actual)/np.linalg.norm(actual),
                        constant_relative_vector_error=np.linalg.norm(base-actual)/np.linalg.norm(actual),
                        actual_mean_gamma=records[end]['mean_gamma'], predicted_mean_gamma=np.mean(abs(pp[:177]))))
            print(json.dumps(dict(seed=seed, target=case['target'], optimizer=case['optimizer'])), flush=True)
    aa.write_csv(output/'states.csv', rows); aa.write_csv(output/'windows.csv', windows); aa.write_csv(output/'forecasts.csv', forecasts)
    np.savez_compressed(output/'forces.npz', force=np.stack(vectors))
    (output/'audit.json').write_text(json.dumps(dict(states=len(rows), maximum_identity_error=float(max(identities)),
        input_hashes=hashes,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        unresolved=sum(not bool(r['resolved']) for r in rows),
        derivative_units='per t=eta*n; Adam is a next-step directional derivative, not a flow theorem'), indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True); parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--snapshots-root',type=Path)
    parser.add_argument('--seed-limit', type=int, default=5); args=parser.parse_args()
    analyze(args.root, args.output, args.seed_limit,args.snapshots_root)
