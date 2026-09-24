"""Prepare the missing width-705 target panel and export its ordinary-GD forks.

Run preparation in a CPU allocation, existing mechanism_widths.run on GPU,
and extraction in a CPU allocation. No change to initialization or GD rates.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from . import adam_forces as af, adam_run as ar, effective_feedback as ef
from . import effective_feedback_holdout as holdout, mechanism_widths as widths, targets


def prepare(output):
    names = [name for name in (*af.TARGETS, *holdout.TARGETS) if name not in widths.TARGETS]
    assert len(names) == 17
    pp, yy, ye, cases = [], [], [], []
    for seed in (30, 31):
        z, d = targets.initial(512, 96, seed)
        for name in names:
            x, y, _, scale = widths.data(name)
            xe, evaluation, _, _ = widths.data(name, 8192)
            pp.append(np.r_[z.ravel(), d]); yy.append(y); ye.append(evaluation)
            cases.append(dict(target=name, seed=seed, start=0, eta=.002,
                              width=705, nref=512, halo=96, target_scale=scale))
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    ar.atomic_npz(output, p=np.asarray(pp), x=x, y=np.asarray(yy),
                  x_eval=xe, y_eval=np.asarray(ye), cases=np.array(json.dumps(cases)),
                  source_sha256=np.array(ef.digest(__file__)))


def extract(source, snapshot, output):
    with np.load(source) as data:
        payload = {k: data[k].copy() for k in data.files}
    with np.load(snapshot) as state:
        if np.any(state['failed']) or np.any(state['count'] != 20000):
            raise ValueError('Expected complete finite 20k ordinary-GD states')
        payload['p'] = state['p'].copy()
    cases = json.loads(str(payload['cases']))
    payload['cases'] = np.array(json.dumps([dict(c, start=20000) for c in cases]))
    payload['sources'] = np.array(json.dumps({str(source): ef.digest(source),
                                             str(snapshot): ef.digest(snapshot)}))
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    ar.atomic_npz(output, **payload)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare'); p.add_argument('--output', required=True, type=Path)
    p = sub.add_parser('extract')
    p.add_argument('--source', required=True, type=Path)
    p.add_argument('--snapshot', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args.output)
    else:
        extract(args.source, args.snapshot, args.output)


if __name__ == '__main__':
    main()
