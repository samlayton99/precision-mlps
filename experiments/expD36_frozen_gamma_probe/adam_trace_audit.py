"""Extract and independently audit the archived N512 raw Adam scalar traces.

Run ``extract`` inside a CPU Slurm step on the archive host; stdout is an NPZ.
``verify`` runs locally and keeps only a compact numeric audit and plot envelopes.
No training, remote writes, SSH configuration, or report generation occurs here.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import sys

import numpy as np

GAMMAS = [8, 12, 16, 64]
TOLERANCES = [1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12]
PREFIX = 'N512_raw'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def extract(root):
    train = root / 'training'
    pilot, cont = [train / f'{PREFIX}_adam_{s}' for s in ['pilot', 'continue']]
    case = read_json(cont / 'case.json')
    selection = read_json(train / f'{PREFIX}_selection.json')
    gis = [case['gammas'].index(g) for g in GAMMAS]
    indices = np.array(selection['indices'])[gis]
    errors = np.full((200001, len(gis), len(case['columns'])), np.nan)
    metadata = dict(gammas=GAMMAS, columns=case['columns'],
                    rates=np.array(case['rates'])[gis].tolist(), source_files=[],
                    endpoint_differences=[], extraction_source_sha256=digest(__file__))
    for folder in [pilot, cont]:
        for path in sorted(folder.glob('trace_*.npz')):
            start = int(path.stem.split('_')[1])
            raw = path.read_bytes()
            with np.load(io.BytesIO(raw)) as data:
                values = data['trace'][..., 0][:, gis, :]
            if folder == pilot:
                values = np.take_along_axis(values, indices[None, :, :], axis=2)
            if start == 50000 and np.isfinite(errors[start]).all():
                metadata['endpoint_differences'].append(dict(
                    step=start, max_difference=float(np.max(np.abs(errors[start]-values[0])))))
            errors[start:start+len(values)] = values
            metadata['source_files'].append(dict(
                path=str(path.relative_to(root)), sha256=hashlib.sha256(raw).hexdigest(),
                bytes=len(raw), start=start, length=len(values)))
        final = read_json(folder / 'evaluations.json')[-1]
        with np.load(folder / 'state.npz') as state:
            count = int(state['count'])
        values = np.array(final['train'])[gis]
        if folder == pilot:
            values = np.take_along_axis(values, indices, axis=1)
        errors[count] = values
        for name in ['case.json', 'evaluations.json', 'hitting_audit.npz']:
            path = folder / name
            metadata['source_files'].append(dict(path=str(path.relative_to(root)),
                sha256=digest(path), bytes=path.stat().st_size))
    path = train / f'{PREFIX}_selection.json'
    metadata['source_files'].append(dict(path=str(path.relative_to(root)),
        sha256=digest(path), bytes=path.stat().st_size))
    with np.load(cont / 'hitting_audit.npz') as data:
        archived = {k: data[k][gis] for k in ['first', 'last_above', 'sustained']}
    with np.load(cont / 'state.npz') as state:
        failed = state['failed'][gis]
    if not np.isfinite(errors).all():
        raise ValueError('Incomplete or nonfinite scalar archive')
    metadata['pilot_continuation_boundary_included'] = True
    np.savez_compressed(sys.stdout.buffer, errors=errors, failed=failed,
                        metadata=json.dumps(metadata), **archived)


def verify(root, archive, output):
    manifest = read_json(root / 'provenance/compact_manifest.json')
    checks = []
    for item in manifest['artifacts']:
        if item['path'].startswith(('training/N512_raw_adam_', 'training/N512_raw_selection')):
            actual = digest(root / item['path'])
            checks.append(dict(path=item['path'], expected=item['sha256'],
                               actual=actual, matches=actual == item['sha256']))
    with np.load(archive) as data:
        errors, failed = data['errors'], data['failed']
        meta = json.loads(str(data['metadata']))
        archived = {k: data[k] for k in ['first', 'last_above', 'sustained']}
    if errors.shape != (200001, 4, 10) or not np.isfinite(errors).all():
        raise ValueError('Expected all 200001 scalar states for four gammas and ten views')
    if meta['gammas'] != GAMMAS:
        raise ValueError('Unexpected gamma order')
    first = np.full((4, 10, len(TOLERANCES)), -1, dtype=np.int64)
    last = first.copy()
    sustained = first.copy()
    for gi in range(4):
        for ci in range(10):
            for ei, epsilon in enumerate(TOLERANCES):
                hit = np.flatnonzero(errors[:, gi, ci] <= epsilon)
                above = np.flatnonzero(errors[:, gi, ci] > epsilon)
                if len(hit):
                    first[gi, ci, ei] = hit[0]
                if len(above):
                    last[gi, ci, ei] = above[-1]
                if last[gi, ci, ei] < 200000 and failed[gi, ci] == 0:
                    sustained[gi, ci, ei] = last[gi, ci, ei]+1
    comparisons = {k: bool(np.array_equal(value, archived[k])) for k, value in
                   [('first', first), ('last_above', last), ('sustained', sustained)]}
    cont = root / 'training' / f'{PREFIX}_adam_continue'
    case = read_json(cont / 'case.json')
    gis = [case['gammas'].index(g) for g in GAMMAS]
    with np.load(cont / 'hitting_audit.npz') as data:
        local_comparisons = {k: bool(np.array_equal(archived[k], data[k][gis]))
                             for k in archived}
    hash_checks = [dict(path=a['path'], matches=digest(root/a['path']) == a['sha256'])
                   for a in meta['source_files'] if (root/a['path']).exists()]
    selection = read_json(root / 'training' / f'{PREFIX}_selection.json')
    indices = np.array(selection['indices'])[gis]
    pilot = np.array(read_json(root / 'training' / f'{PREFIX}_adam_pilot/evaluations.json')[-1]['train'])[gis]
    pilot = np.take_along_axis(pilot, indices, axis=1)
    endpoint_difference = float(np.max(np.abs(pilot-errors[50000])))
    if not (all(comparisons.values()) and all(local_comparisons.values())
            and all(c['matches'] for c in checks+hash_checks)):
        raise AssertionError('Archived hitting times or source hashes disagree')
    if endpoint_difference > 1e-12:
        raise AssertionError('Pilot endpoint and continuation start disagree')
    # Half-open bins partition every integer state, including the final endpoint.
    edges = np.unique(np.r_[0, np.ceil(np.geomspace(1, 200001, 401)).astype(np.int64)])
    low = np.stack([errors[a:b].min(axis=0) for a, b in zip(edges[:-1], edges[1:])])
    high = np.stack([errors[a:b].max(axis=0) for a, b in zip(edges[:-1], edges[1:])])
    steps = np.unique(np.r_[edges[:-1], 50000, 200000]).astype(np.int64)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output/'trace_envelopes.npz', bin_edges=edges,
                        bin_min=low, bin_max=high, steps=steps, errors=errors[steps],
                        gammas=GAMMAS, columns=json.dumps(meta['columns']))
    rows = [dict(gamma=GAMMAS[gi], **col, first=first[gi, ci].tolist(),
                 last_above=last[gi, ci].tolist(), sustained=sustained[gi, ci].tolist(),
                 final_error=float(errors[-1, gi, ci]), base_rate=meta['rates'][gi][ci])
            for gi in range(4) for ci, col in enumerate(meta['columns'])]
    result = dict(source_sha256=digest(__file__), scalar_archive_sha256=digest(archive),
        manifest_sha256=digest(root/'provenance/compact_manifest.json'),
        final_audit_sha256=digest(root/'validation/final_audit.json'),
        local_manifest_checks=checks, remote_metadata=meta, thresholds=TOLERANCES,
        shape=list(errors.shape), failed=int(np.count_nonzero(failed)), comparisons=comparisons,
        remote_vs_local_hitting=local_comparisons, remote_metadata_vs_local_hashes=hash_checks,
        pilot_endpoint_vs_continuation_start_max_abs=endpoint_difference, rows=rows,
        envelope=dict(file='trace_envelopes.npz', sha256=digest(output/'trace_envelopes.npz'),
            bins=len(edges)-1, sampled_states=len(steps),
            convention='Half-open integer bins [edge[i],edge[i+1]); minima/maxima of all actual scalar errors. '
                       'Raw sampled points at bin starts plus updates 50000 and 200000. No EMA or smoothing.'))
    (output/'trace_audit.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(comparisons=comparisons, remote_vs_local=local_comparisons,
                         bins=len(edges)-1, rows=len(rows))))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    extraction = sub.add_parser('extract')
    extraction.add_argument('--root', type=Path, required=True)
    verification = sub.add_parser('verify')
    verification.add_argument('--root', type=Path, required=True)
    verification.add_argument('--archive', type=Path, required=True)
    verification.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'extract':
        extract(args.root)
    else:
        verify(args.root, args.archive, args.output)


if __name__ == '__main__':
    main()
