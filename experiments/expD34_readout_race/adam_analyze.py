"""Numerical evidence for D34 Adam; prose is authored separately."""
from __future__ import annotations

import argparse
import csv
from functools import lru_cache
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from . import adam_forces as af, adam_run as ar, mechanism, targets, transport


def write_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'wt', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader(); writer.writerows(rows)


def field(p, x, y):
    a, b, c = p[:-1].reshape(3, -1)
    u = x[:, None]*a+b; h = np.tanh(u)
    ex = np.exp(-2*abs(u)); s = 4*ex/(1+ex)**2
    r = h @ c+p[-1]-y; m = len(x)
    weighted = r[:, None]*s
    g = np.r_[c*(x @ weighted)/m, c*weighted.sum(axis=0)/m, h.T @ r/m, r.mean()]
    q = np.stack((np.ones_like(x), x/np.sqrt(np.mean(x*x))))
    j = np.c_[(q @ (x[:, None]*s)/m)*c, (q @ s/m)*c, q @ h/m, q.mean(axis=1)]
    return g, r, j, q @ r/m, h, s


def split(g, j, ec, mobility=None):
    mobility = np.ones_like(g) if mobility is None else mobility
    C = (j*mobility) @ j.T
    ev = np.linalg.eigvalsh(C)
    if ev[0] <= 64*np.finfo(float).eps*max(1., ev[-1]):
        return None
    fine = g-j.T @ ec; b = np.linalg.solve(C, j @ (mobility*fine))
    return fine-j.T @ b, j.T @ (ec+b), ec+b, C


def ratio(a, b):
    return float(a/b) if b>0 else np.nan


@lru_cache(maxsize=8)
def modes(m, degree):
    return transport.basis(targets.grid(m), degree)


def state_metrics(p, m, v, cm, count, case, x, y):
    g, r, j, ec, h, s = field(p, x, y)
    width = (len(p)-1)//3
    parts = split(g, j, ec)
    b1, b2, eps = case['beta1'], case['beta2'], case['epsilon']
    mh = (b1*m+(1-b1)*g)/(1-b1**(count+1))
    vh = (b2*v+(1-b2)*g*g)/(1-b2**(count+1))
    P = 1/(np.sqrt(vh)+eps) if case['adaptive'] else np.ones_like(g)
    weighted = split(g, j, ec, P)
    delta = -case['eta']*P*mh
    _, rn, _, en, _, _ = field(p+delta, x, y)
    row = dict(step_role='virtual next update; endpoint update is not trained',
        relative_train_mse=np.mean(r*r)/np.mean(y*y),
        mean_gamma=np.mean(abs(p[:width])), median_gamma=np.median(abs(p[:width])),
        q90_gamma=np.quantile(abs(p[:width]), .9), max_gamma=max(abs(p[:width])),
        readout_l2=np.linalg.norm(p[2*width:3*width]),
        raw_norm=np.linalg.norm(g[:width]), step_norm=np.linalg.norm(delta[:width]),
        coarse_norm=np.linalg.norm(ec), gd_balance_resolved=parts is not None,
        adam_balance_resolved=weighted is not None,
        coarse_step_defect=np.linalg.norm(en-ec-j @ delta),
        coarse_step_actual=np.linalg.norm(en-ec),
        coarse_momentum_lag=np.linalg.norm(case['eta']*j @ (P*(mh-g))))
    q1 = x/np.sqrt(np.mean(x*x)); fine2 = np.mean((r-ec[0]-ec[1]*q1)**2)
    row['fine_residual_norm'] = np.sqrt(fine2)
    if parts is not None:
        effective, tracking, z, C = parts
        cc = np.stack((effective, tracking, np.zeros_like(g)))
        ch = (b1*cm+(1-b1)*cc)/(1-b1**(count+1))
        dc = -case['eta']*P*ch
        row.update(effective_norm=np.linalg.norm(effective[:width]),
            tracking_norm=np.linalg.norm(tracking[:width]),
            effective_coupling=ratio(effective[:width] @ effective[:width], fine2),
            raw_tracking_ratio=ratio(np.linalg.norm(tracking[:width]), np.linalg.norm(g[:width])),
            step_tracking_ratio=ratio(np.linalg.norm(dc[1, :width]), np.linalg.norm(delta[:width])),
            tracking_metric=np.sqrt(max(0., z @ C @ z)))
    if weighted is not None:
        eff, track, z, C = weighted
        expected = -case['eta']*(C @ z+j @ (P*(mh-g)))
        row.update(preconditioned_raw_tracking_ratio=ratio(np.linalg.norm(track[:width]), np.linalg.norm(g[:width])),
            preconditioned_scaled_tracking_ratio=ratio(np.linalg.norm(P[:width]*track[:width]), np.linalg.norm(P[:width]*g[:width])),
            preconditioned_coarse_identity_error=np.linalg.norm(expected-j @ delta),
            preconditioned_tracking_norm=np.linalg.norm(track[:width]),
            preconditioned_effective_norm=np.linalg.norm(eff[:width]))
    return row, (g, r, j, ec, h, s)


def modal_metrics(p, x, arrays, degree):
    g, r, jc, ec, h, s = arrays
    width = (len(p)-1)//3; c = p[2*width:3*width]
    q = modes(len(x), degree); e = q.T @ r/len(x)
    J = np.c_[(q.T @ (x[:, None]*s)/len(x))*c,
              (q.T @ s/len(x))*c, q.T @ h/len(x), q.mean(axis=0)]
    C = J[:2] @ J[:2].T
    ev = np.linalg.eigvalsh(C)
    if ev[0] <= 64*np.finfo(float).eps*max(1., ev[-1]):
        return dict(degree=degree, resolved=False), e
    B = np.linalg.solve(C, J[:2] @ J[2:].T)
    T = J[2:, :width].T-J[:2, :width].T @ B
    effective = T @ e[2:]; tracking = J[:2, :width].T @ (e[:2]+B @ e[2:])
    omitted = g[:width]-J[:, :width].T @ e
    full = split(g, jc, ec)
    row = dict(degree=degree, resolved=True, effective_norm=np.linalg.norm(effective),
        tracking_norm=np.linalg.norm(tracking), omitted_norm=np.linalg.norm(omitted),
        effective_vector_difference=np.linalg.norm(effective-full[0][:width]) if full else np.nan,
        reconstruction_error=np.linalg.norm(g[:width]-effective-tracking-omitted))
    for mode in (2, 3, 4, 5, 9):
        vector = T[:, mode-2]*e[mode]
        row[f'mode_{mode}_force_norm'] = np.linalg.norm(vector)
        row[f'mode_{mode}_outward'] = -np.mean(np.sign(p[:width])*vector)
    return row, e


def geometry(p, x, y, xe, ye, identity, kind):
    z = p[:-1].reshape(3, -1)
    curves, capacity, _ = mechanism.frozen_curves(z[0], z[1], x, y, xe, ye)
    return [dict(**identity, kind=kind, **row) for row in curves]


def analyze_bundle(source, destination):
    destination.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((source/'manifest.json').read_text())
    status = json.loads((source/'status.json').read_text())
    if not status['complete']:
        raise ValueError(f'Incomplete scientific bundle: {source}')
    f = dict(np.load(source/'snapshots.npz')); steps = f.pop('steps')
    t = dict(np.load(source/'trace.npz'))
    state_rows = []; modal_rows = []; residuals = []; curves = []; endpoint_rows = []; windows = []
    trace_rows = []
    primary = source.name.startswith('primary_')
    for i, case in enumerate(manifest['cases']):
        identity = dict(bundle=source.name, case_index=i, **case)
        x, y, mapping, scale = af.data(case['target'], manifest['m'])
        xe = targets.grid(8192); ye = af.target_values(case['target'], xe, mapping)/scale
        previous = None
        for k, step in enumerate(steps):
            p = f['p'][i, k]
            row, arrays = state_metrics(p, f['m'][i, k], f['v'][i, k], f['channel_m'][i, k],
                                        int(f['count'][i, k]), case, x, y)
            state_rows.append(dict(**identity, step=int(step), **row))
            if (primary and step in (0, 2000, 20000, 100000, 600000)) or (not primary and step==600000):
                for degree in (65, 129):
                    modal, e = modal_metrics(p, x, arrays, degree)
                    modal_rows.append(dict(**identity, step=int(step), **modal))
                    if degree==129:
                        residuals.append(dict(**identity, step=int(step), **{f'e_{j}': v for j, v in enumerate(e)}))
            selected_geometry = step in (0, 2000, 20000, 100000, 600000) if primary else step==600000
            if selected_geometry:
                mark = dict(**identity, step=int(step))
                curves.extend(geometry(p, x, y, xe, ye, mark, 'learned'))
                if previous is not None:
                    old = previous.copy(); new = p.copy(); width = (len(p)-1)//3
                    new[width:2*width] = old[width:2*width]
                    old[width:2*width] = p[width:2*width]
                    curves.extend(geometry(new, x, y, xe, ye, mark, 'new_slopes_old_biases'))
                    curves.extend(geometry(old, x, y, xe, ye, mark, 'old_slopes_new_biases'))
                previous = p
            if step==600000:
                z = p[:-1].reshape(3, -1)
                re = np.tanh(xe[:, None]*z[0]+z[1]) @ z[2]+p[-1]-ye
                endpoint_rows.append(dict(**identity, **row, relative_eval_mse=np.mean(re*re)/np.mean(ye*ye),
                    eval_mse=np.mean(re*re), path=f['path'][i, k], positive_mean=f['positive'][i, k].mean(),
                    negative_mean=f['negative'][i, k].mean(), failed=int(f['failed'][i, k]),
                    unresolved_steps=int(f['unresolved_steps'][i, k]), loss_increases=int(f['loss_increases'][i, k]),
                    frozen_relative_mse=curves[-1]['relative_heldout_mse'] if not primary else
                        next(r['relative_heldout_mse'] for r in reversed(curves) if r['kind']=='learned' and r['updates']==600000)))
        for start, end in ((0, 200), (200, 2000), (2000, 20000), (20000, 100000),
                           (100000, 600000), (20000, 600000), (0, 600000)):
            a = int(np.flatnonzero(steps==start)[0]); b = int(np.flatnonzero(steps==end)[0])
            row = dict(**identity, start=start, end=end, path=f['path'][i, b]-f['path'][i, a],
                mean_gamma_change=np.mean(abs(f['p'][i, b, :177])-abs(f['p'][i, a, :177])),
                crossing=f['crossing'][i, b]-f['crossing'][i, a],
                positive_mean=np.mean(f['positive'][i, b]-f['positive'][i, a]),
                negative_mean=np.mean(f['negative'][i, b]-f['negative'][i, a]))
            for j, channel in enumerate(af.CHANNELS):
                for name in ('channel_path', 'raw_channel_path', 'signed_channels', 'projected_energy'):
                    row[f'{name}_{channel}'] = f[name][i, b, j]-f[name][i, a, j]
            row['step_energy'] = f['step_energy'][i, b]-f['step_energy'][i, a]
            row['tracking_path_ratio'] = ratio(row['channel_path_tracking'], row['path'])
            row['raw_tracking_fraction'] = ratio(row['raw_channel_path_tracking'],
                row['raw_channel_path_tracking']+row['raw_channel_path_effective'])
            row['step_tracking_fraction'] = ratio(row['channel_path_tracking'],
                row['channel_path_tracking']+row['channel_path_effective'])
            windows.append(row)
        for k, end in enumerate(t['ends']):
            trace_rows.append(dict(**identity, start=int(t['starts'][k]), end=int(end),
                gradient_step=int(end)-1, **dict(zip(manifest['metrics'], t['values'][i, k]))))
        print(json.dumps(dict(bundle=source.name, case=i, target=case['target'], optimizer=case['optimizer'])), flush=True)
    for name, rows in [('states.csv.gz', state_rows), ('modal.csv', modal_rows), ('residuals.csv.gz', residuals),
                       ('geometry.csv.gz', curves), ('endpoints.csv', endpoint_rows), ('windows.csv', windows),
                       ('trace.csv.gz', trace_rows)]:
        write_csv(destination/name, rows)
    (destination/'audit.json').write_text(json.dumps(dict(source=str(source), cases=len(manifest['cases']),
        source_hashes={name: hashlib.sha256((source/name).read_bytes()).hexdigest()
                       for name in ('manifest.json', 'snapshots.npz', 'trace.npz', 'status.json')},
        identity_max=status['identity_max'], motion_identity_max=status['motion_identity_max']), indent=2)+'\n')


def construction_references(destination):
    destination.mkdir(parents=True, exist_ok=True)
    rows = []
    centers = -1+2*np.arange(-24, 153)/128
    for target in af.TARGETS:
        x, y, mapping, scale = af.data(target); xe = targets.grid(8192)
        ye = af.target_values(target, xe, mapping)/scale
        for gamma in mechanism.GAMMAS:
            a = np.full(177, gamma); b = -gamma*centers
            p = np.r_[a, b, np.zeros(178)]
            rows.extend(geometry(p, x, y, xe, ye, dict(target=target, gamma=gamma), 'construction'))
    write_csv(destination/'construction.csv', rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--construction', action='store_true')
    args = parser.parse_args()
    if args.construction:
        construction_references(args.output)
    else:
        analyze_bundle(args.source, args.output)


if __name__ == '__main__':
    main()
