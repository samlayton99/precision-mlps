"""CPU-only archived-fork and exact first-update checks for the Adam probe."""
import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from . import adam_forces as af, mechanism_adam as ma


def verify(archive, inputs, predictions):
    pack = dict(np.load(inputs))
    cases = json.loads(str(pack['cases']))
    manifest = json.loads((archive/'manifest.json').read_text())
    snapshots = np.load(archive/'snapshots.npz')
    at = list(snapshots['steps']).index(600000)
    first = np.load(predictions/'first_step.npz')
    maximum = dict(m=0., v=0., p=0., channel_m=0., history_identity=0.)
    for i, case in enumerate(cases):
        source = next(j for j, c in enumerate(manifest['cases'])
                      if c['optimizer'] == 'adam' and c['target'] == case['target'])
        for key in ('p', 'm', 'v', 'count', 'channel_m'):
            np.testing.assert_array_equal(pack[key][i], snapshots[key][source, at])
        p = pack['p'][i]
        g, _, jc, ec = af.field(jnp.asarray(p), jnp.asarray(pack['x']), jnp.asarray(pack['y'][i]))
        channels, info = af.split(g, jc, ec)
        assert info['resolved']
        g, channels = np.asarray(g), np.array(channels, copy=True)
        width = (len(p)-1)//3
        gm, gv = g.copy(), g.copy()
        gm[:width] += (case['alpha_m']-1)*channels[1, :width]
        gv[:width] += (case['alpha_v']-1)*channels[1, :width]
        count = pack['count'][i]+1
        m = .9*pack['m'][i]+.1*gm
        v = .999*pack['v'][i]+.001*gv**2
        channels[1, :width] *= case['alpha_m']
        cm = .9*pack['channel_m'][i]+.1*channels
        pp = p-.002*(m/(1-.9**count))/(np.sqrt(v/(1-.999**count))+1e-8)
        for key, expected in (('m', m), ('v', v), ('p', pp), ('channel_m', cm)):
            error = float(np.max(abs(first[key][i]-expected)))
            maximum[key] = max(maximum[key], error)
            np.testing.assert_allclose(first[key][i], expected, atol=2e-14, rtol=2e-13)
        maximum['history_identity'] = max(maximum['history_identity'],
            float(np.max(abs(pack['channel_m'][i].sum(axis=0)-pack['m'][i]))))
    return dict(cases=len(cases), fork=600000, all_fork_buffers_bitwise_equal=True,
                maximum_absolute_errors=maximum, input_sha256=ma.digest(inputs),
                source_snapshot_sha256=ma.digest(archive/'snapshots.npz'),
                prediction_manifest_sha256=ma.digest(predictions/'manifest.json'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive', type=Path, required=True)
    p.add_argument('--inputs', type=Path, required=True)
    p.add_argument('--predictions', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if not jax.config.x64_enabled or any(d.platform != 'cpu' for d in jax.devices()):
        raise RuntimeError('Run verification on CPU with FP64')
    result = verify(args.archive, args.inputs, args.predictions)
    ma.write_json(args.output, result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
