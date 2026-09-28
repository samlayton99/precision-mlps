"""Locked held-out functions and CPU-only input/fork preparation.

GPU runners consume the resulting arrays without importing this module. Existing
target definitions and immutable GPU source capsules remain unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from . import targets

FAMILIES = dict(exp_right='exponential', exp_left='exponential',
                gauss_left='gaussian', gauss_right='gaussian',
                bump_left='compact_bump', bump_right='compact_bump',
                step_left='tanh_step', step_right='tanh_step',
                kink_abs='kink', kink_relu='kink')
TARGETS = tuple(FAMILIES)


def raw_values(name, x):
    x = np.asarray(x, dtype=np.float64)
    if name == 'exp_right':
        return np.exp(2*x)
    if name == 'exp_left':
        return np.exp(-4*x)
    if name == 'gauss_left':
        return np.exp(-((x+.35)/.22)**2)
    if name == 'gauss_right':
        return np.exp(-((x-.42)/.09)**2)
    if name in ('bump_left', 'bump_right'):
        u = (x+.30)/.40 if name == 'bump_left' else (x-.35)/.22
        y = np.zeros_like(x)
        inside = abs(u) < 1
        y[inside] = np.exp(1-1/(1-u[inside]**2))
        return y
    if name == 'step_left':
        return np.tanh(6*(x+.27))
    if name == 'step_right':
        return np.tanh(14*(x-.31))
    if name == 'kink_abs':
        return abs(x+.23)
    if name == 'kink_relu':
        return np.maximum(x-.37, 0)
    raise ValueError(name)


def data(name, m=2048):
    """Use the original 2048-grid RMS for every training/evaluation grid."""
    original = targets.grid(2048)
    scale = float(np.sqrt(np.mean(raw_values(name, original)**2)))
    x = targets.grid(m)
    return x, raw_values(name, x)/scale, None, scale


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_pack(path, **arrays):
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp.npz')
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)
    print(json.dumps(dict(path=str(path), sha256=digest(path), cases=len(arrays['p']))))


def prepare(args):
    cases, parameters, labels = [], [], []
    for seed in (22, 23):
        z, d = targets.initial(128, 24, seed)
        p = np.r_[z.ravel(), d]
        for name in TARGETS:
            x, y, _, scale = data(name)
            cases.append(dict(target=name, family=FAMILIES[name], seed=seed, start=0,
                              eta=.002, width=(len(p)-1)//3, target_scale=scale))
            parameters.append(p)
            labels.append(y)
    # Capture loaded local initialization dependencies, including the upstream
    # initializer imported lazily by targets.initial, without copying code.
    root = Path(__file__).resolve().parents[2]
    sources = {}
    for module in tuple(sys.modules.values()):
        filename = getattr(module, '__file__', None)
        if filename:
            path = Path(filename).resolve()
            if path.is_relative_to(root/'experiments') and path.suffix == '.py':
                sources[str(path.relative_to(root))] = digest(path)
    write_pack(args.output, p=np.asarray(parameters), x=x, y=np.asarray(labels),
               cases=np.array(json.dumps(cases)), sources=np.array(json.dumps(sources)))


def export(args):
    manifest_file = args.source/'manifest.json'
    manifest = json.loads(manifest_file.read_text())
    if manifest['arm'] != 'joint':
        raise ValueError('Only ordinary-GD backbone states can define fresh forks')
    if manifest['input_sha256'] != digest(args.inputs):
        raise ValueError('Initial input does not match the backbone manifest')
    with np.load(args.inputs) as pack:
        initial_cases = json.loads(str(pack['cases']))
        selection = manifest['selected_indices']
        indices = np.arange(len(initial_cases)) if selection is None else np.asarray(selection)
        if [initial_cases[i] for i in indices] != manifest['cases']:
            raise ValueError('Backbone case order differs from initial inputs')
        x, y = pack['x'].copy(), pack['y'][indices].copy()
        sources = json.loads(str(pack['sources']))
    snapshot = args.source/'snapshots'/f'{args.offset:09d}.npz'
    with np.load(snapshot) as state:
        if np.any(state['failed']) or np.any(state['count'] != args.offset):
            raise ValueError('Failed or incompletely advanced backbone state')
        p = state['p'].copy()
    if len(p) != len(indices):
        raise ValueError('Snapshot case count differs from its manifest')
    cases = [dict(c, start=c['start']+args.offset) for c in manifest['cases']]
    sources.update({str(path): digest(path) for path in
                    (args.inputs, manifest_file, snapshot, Path(__file__))})
    write_pack(args.output, p=p, x=x, y=y, cases=np.array(json.dumps(cases)),
               sources=np.array(json.dumps(sources)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prep = commands.add_parser('prepare')
    prep.add_argument('--output', type=Path, required=True)
    fork = commands.add_parser('export')
    fork.add_argument('--inputs', type=Path, required=True)
    fork.add_argument('--source', type=Path, required=True)
    fork.add_argument('--offset', type=int, required=True)
    fork.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    (prepare if args.command == 'prepare' else export)(args)


if __name__ == '__main__':
    main()
