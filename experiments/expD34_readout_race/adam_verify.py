"""Fixed-state numerical checks and compact evidence export; no report prose."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from . import adam_analyze as aa, adam_forces as af, mechanism, targets


def verify(root):
    destination = root/'verification'; destination.mkdir(exist_ok=True)
    rows = []; replay = []
    source = root/'raw'/'primary_0'
    cases = json.loads((source/'manifest.json').read_text())['cases']
    f = np.load(source/'snapshots.npz'); end = int(np.flatnonzero(f['steps']==600000)[0])
    for i, case in enumerate(cases):
        p = f['p'][i, end]; gradients = []; effective = []; errors = []; frozen = []
        for m in (2048, 4096):
            x, y, mapping, scale = af.data(case['target'], m)
            g, r, j, ec, _, _ = aa.field(p, x, y)
            gradients.append(g[:177]); effective.append(aa.split(g, j, ec)[0][:177])
            xe = targets.grid(m*4); ye = af.target_values(case['target'], xe, mapping)/scale
            re = aa.field(p, xe, ye)[1]; errors.append(np.mean(re*re)/np.mean(ye*ye))
            curves, _, _ = mechanism.frozen_curves(p[:177], p[177:354], x, y, xe, ye, horizons=(600000,))
            frozen.append(curves[0]['relative_heldout_mse'])
        rows.append(dict(**case, gradient_absolute_change=np.linalg.norm(gradients[1]-gradients[0]),
            gradient_relative_change=aa.ratio(np.linalg.norm(gradients[1]-gradients[0]), np.linalg.norm(gradients[1])),
            effective_absolute_change=np.linalg.norm(effective[1]-effective[0]),
            effective_relative_change=aa.ratio(np.linalg.norm(effective[1]-effective[0]), np.linalg.norm(effective[1])),
            error_8192=errors[0], error_16384=errors[1], frozen_original=frozen[0], frozen_refined=frozen[1]))
        if rows[-1]['effective_relative_change']>.01:
            x, y, _, _ = af.data(case['target'], 8192)
            g, _, j, ec, _, _ = aa.field(p, x, y)
            refined = aa.split(g, j, ec)[0][:177]
            difference = np.linalg.norm(refined-effective[1])
            rows[-1].update(effective_4096_to_8192_absolute_change=difference,
                effective_4096_to_8192_relative_change=aa.ratio(difference, np.linalg.norm(refined)),
                effective_observed_order=np.log2(rows[-1]['effective_absolute_change']/difference))
    aa.write_csv(destination/'grid.csv', rows)
    oldroot = root.parent/'useful_slopes'/'curated'/'states'
    for name in ('signal_recovery', 'missing'):
        old = np.load(oldroot/name/'compact_states.npz')
        for oi, case in enumerate(json.loads(str(old['cases']))):
            if isinstance(case, list): case = dict(seed=case[0], target=case[1])
            source = root/'raw'/f"primary_{case['seed']}"
            cases = json.loads((source/'manifest.json').read_text())['cases']
            ni = next(i for i, c in enumerate(cases) if c['target']==case['target'] and c['optimizer']=='gd')
            new = np.load(source/'snapshots.npz')
            for step in np.intersect1d(old['steps'], new['steps']):
                a = int(np.flatnonzero(old['steps']==step)[0]); b = int(np.flatnonzero(new['steps']==step)[0])
                original = np.r_[old['z'][oi, a].ravel(), old['d'][oi, a]]
                replay.append(dict(**case, step=int(step), max_parameter_difference=float(np.max(abs(original-new['p'][ni, b])))))
    aa.write_csv(destination/'replay.csv', replay)
    statuses = [json.loads(p.read_text()) for p in (root/'raw').glob('*/status.json')]
    result = dict(cases=sum(s['cases'] for s in statuses), all_complete=all(s['complete'] for s in statuses),
        failed=sum(sum(s['failed']) for s in statuses), unresolved_steps=sum(sum(s['unresolved_steps']) for s in statuses),
        identity_max=np.max([s['identity_max'] for s in statuses], axis=0).tolist(),
        motion_identity_max=max(s['motion_identity_max'] for s in statuses),
        replay_max=max(r['max_parameter_difference'] for r in replay),
        replay_cases=len({(r['seed'], r['target']) for r in replay}))
    (destination/'numerical.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


def curate(root):
    """Keep selected states and coarsened traces with exact interval envelopes."""
    destination = root/'curated'; destination.mkdir(exist_ok=True)
    hashes = {}
    keep = (0, 1, 10, 100, 200, 1000, 2000, 20000, 100000, 600000)
    for source in sorted((root/'raw').iterdir()):
        output = destination/source.name; output.mkdir(exist_ok=True)
        status = json.loads((source/'status.json').read_text())
        if not status['complete']: raise ValueError(source)
        for name in ('manifest.json', 'status.json'):
            shutil.copy2(source/name, output/name)
        for path in source.glob('environment_*.json'): shutil.copy2(path, output/path.name)
        f = dict(np.load(source/'snapshots.npz')); steps = f.pop('steps'); select = np.isin(steps, keep)
        np.savez_compressed(output/'snapshots.npz', steps=steps[select], **{key: value[:, select] for key, value in f.items()})
        t = dict(np.load(source/'trace.npz')); ends = t['ends']
        stride = np.where(ends<=200, 10, np.where(ends<=2000, 100, np.where(ends<=20000, 1000, 5000)))
        indices = np.flatnonzero((ends % stride==0) | (ends==ends[-1]))
        starts = np.r_[0, indices[:-1]+1]
        compact = dict(starts=t['starts'][starts], ends=ends[indices], values=t['values'][:, indices])
        for key, reduction in [('minimum', np.min), ('maximum', np.max)]:
            compact[key] = np.stack([reduction(t[key][:, lo:hi+1], axis=1) for lo, hi in zip(starts, indices)], axis=1)
        np.savez_compressed(output/'trace.npz', **compact)
        hashes[source.name] = dict(original={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir() if p.is_file()},
            curated={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir() if p.is_file()})
    (destination/'hashes.json').write_text(json.dumps(hashes, indent=2)+'\n')


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--curate', action='store_true')
    args = parser.parse_args()
    if args.curate: curate(args.root)
    else: verify(args.root)
