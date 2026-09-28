"""Archived-state replay and actual process interruption on allocated Modal GPUs."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def replay(root, source, optimizer, mode):
    import jax
    import jax.numpy as jnp
    import numpy as np
    from . import adam_forces as af, adam_run as ar, plateau
    from .run import write_json

    if not jax.config.x64_enabled or len(jax.devices()) != 1 or jax.devices()[0].platform != 'gpu':
        raise RuntimeError('Pilot subprocess lost the parent GPU/FP64 context')
    folder = root / 'pilot' / optimizer
    folder.mkdir(parents=True, exist_ok=True)
    if mode == 'produce':
        manifest = json.loads((source / 'curated/primary_0/manifest.json').read_text())
        indices = [i for i, c in enumerate(manifest['cases']) if c['optimizer'] == optimizer
                   and c['target'] in ('moment9', 'mixed_sine')]
        cases = [manifest['cases'][i] for i in indices]
        with np.load(source / 'curated/primary_0/snapshots.npz') as archive:
            position = int(np.flatnonzero(archive['steps'] == 600000)[0])
            state = {k: archive[k][indices, position] for k in archive.files if k != 'steps'}
        settings = np.array([[c[k] for k in ('eta', 'beta1', 'beta2', 'epsilon', 'adaptive')] for c in cases])
        yy = np.stack([af.data(c['target'])[1] for c in cases])
        x = jnp.asarray(af.data(cases[0]['target'])[0])
        max_gradient_error = 0.
        max_split_error = 0.
        for p, y in zip(state['p'], yy):
            g, _, jc, ec = af.field(jnp.asarray(p), x, jnp.asarray(y))
            reference = jax.grad(lambda v: .5*jnp.mean((plateau.prediction(v, x)-y)**2))(jnp.asarray(p))
            np.testing.assert_allclose(g, reference, atol=2e-13, rtol=2e-11)
            channels, info = af.split(g, jc, ec)
            assert bool(info['resolved'])
            np.testing.assert_allclose(channels.sum(axis=0), g, atol=2e-13, rtol=2e-11)
            max_gradient_error = max(max_gradient_error, float(jnp.max(abs(g-reference))))
            max_split_error = max(max_split_error, float(jnp.max(abs(channels.sum(axis=0)-g))))
        ar.atomic_npz(folder / 'initial.npz', **state)
        ar.atomic_npz(folder / 'inputs.npz', settings=settings, yy=yy)
        advance = ar.advance_factory(2048)
        begun = time.monotonic()
        whole = jax.device_get(advance(jax.tree.map(jnp.asarray, state), jnp.asarray(yy), jnp.asarray(settings), 32)[0])
        compile_and_replay = time.monotonic() - begun
        begun = time.monotonic()
        part = jax.device_get(advance(jax.tree.map(jnp.asarray, state), jnp.asarray(yy), jnp.asarray(settings), 13)[0])
        replay_seconds = time.monotonic() - begun
        ar.atomic_npz(folder / 'whole.npz', **whole)
        ar.atomic_npz(folder / 'interrupted.npz', **part)
        write_json(folder / 'ready.json', dict(optimizer=optimizer, targets=[c['target'] for c in cases],
            device=jax.devices()[0].device_kind, container=os.environ.get('MODAL_TASK_ID'),
            compile_and_32_updates_seconds=compile_and_replay, thirteen_updates_seconds=replay_seconds,
            gradient_error=max_gradient_error, force_split_error=max_split_error))
        # The parent actually terminates this process at its durable checkpoint.
        while True:
            time.sleep(1)
    else:
        with np.load(folder / 'interrupted.npz') as f:
            part = {k: f[k] for k in f.files}
        with np.load(folder / 'inputs.npz') as f:
            yy, settings = f['yy'], f['settings']
        resumed = jax.device_get(ar.advance_factory(2048)(jax.tree.map(jnp.asarray, part),
                                 jnp.asarray(yy), jnp.asarray(settings), 19)[0])
        exact = True
        with np.load(folder / 'whole.npz') as whole:
            for k, value in resumed.items():
                exact = exact and np.array_equal(value, whole[k])
                np.testing.assert_allclose(value, whole[k], atol=2e-13, rtol=2e-12)
        with np.load(folder / 'initial.npz') as original:
            delta = np.mean(abs(resumed['p'][:, :177])-abs(original['p'][:, :177]), axis=1)
            accounted = (resumed['signed_channels']-original['signed_channels']).sum(axis=1)
            accounted += resumed['crossing']-original['crossing']
            np.testing.assert_allclose(delta, accounted, atol=1e-12, rtol=0)
        assert not np.any(resumed['failed']) and not np.any(resumed['unresolved_steps'])
        assert np.max(resumed['identity_max']) < 1e-9
        producer = json.loads((folder / 'ready.json').read_text())
        assert producer['container'] != os.environ.get('MODAL_TASK_ID')
        result = dict(passed=True, bitwise_equal=bool(exact), producer=producer,
                      consumer_container=os.environ.get('MODAL_TASK_ID'),
                      consumer_device=jax.devices()[0].device_kind,
                      signed_motion_error=float(np.max(abs(delta-accounted))),
                      identity_max=float(np.max(resumed['identity_max'])))
        write_json(root / 'pilot' / f'resumed_{optimizer}.json', result)


def run(root, source, optimizer, volume, deadline):
    """Each of two workers produces its state, then resumes the other's state."""
    from .run import write_json
    folder = root / 'pilot' / optimizer
    folder.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, '-u', '-m', __name__, '--root', str(root), '--source', str(source)]
    with (folder / 'producer.log').open('w') as log:
        child = subprocess.Popen(command + ['--optimizer', optimizer, '--mode', 'produce'],
                                 stdout=log, stderr=subprocess.STDOUT)
        try:
            while not (folder / 'ready.json').exists():
                if child.poll() is not None:
                    raise RuntimeError((folder / 'producer.log').read_text())
                if time.time() > deadline - 120:
                    raise RuntimeError('Pilot producer exhausted its original deadline')
                time.sleep(.5)
        finally:
            child.terminate()
            child.wait(timeout=10)
    if child.returncode != -15:
        raise RuntimeError(f'Unexpected interruption status: {child.returncode}')
    write_json(folder / 'interruption.json', dict(signal='SIGTERM', returncode=child.returncode))
    volume.commit()
    peer = 'adam' if optimizer == 'gd' else 'gd'
    while True:
        volume.reload()
        if (root / 'pilot' / peer / 'interruption.json').exists():
            break
        if time.time() > deadline - 60:
            raise RuntimeError('Other GPU did not publish a complete pilot checkpoint')
        time.sleep(1)
    with (folder / 'consumer.log').open('w') as log:
        subprocess.run(command + ['--optimizer', peer, '--mode', 'resume'], stdout=log,
                       stderr=subprocess.STDOUT, check=True, timeout=max(1, deadline-time.time()-5))
    volume.commit()
    return json.loads((root / 'pilot' / f'resumed_{peer}.json').read_text())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--optimizer', choices=('gd', 'adam'), required=True)
    parser.add_argument('--mode', choices=('produce', 'resume'), required=True)
    args = parser.parse_args()
    replay(args.root, args.source, args.optimizer, args.mode)
