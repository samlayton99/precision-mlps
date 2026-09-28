"""Immutable Figure 4 exports, acknowledged only after verified laptop copies."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import time

import numpy as np


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def export_checkpoint(output, destination, first, count, errors, rms,
                      snapshots, snapshot_steps, metadata, ack_timeout):
    destination.mkdir(parents=True, exist_ok=True)
    temporary = destination / f'.pending-step{count:08d}'
    complete = destination / f'step{count:08d}'
    temporary.mkdir()
    keep = (snapshot_steps >= first) & (snapshot_steps <= count)
    np.savez(temporary / 'trace.npz', error=errors[first:count+1],
             slope_rms=rms[first:count+1], checkpoint_steps=snapshot_steps[keep],
             parameters=snapshots[keep])
    shutil.copyfile(output / 'state.npz', temporary / 'state.npz')
    manifest = dict(first_step=first, completed_updates=count, metadata=metadata,
                    files={name: digest(temporary / name)
                           for name in ['trace.npz', 'state.npz']})
    (temporary / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    temporary.rename(complete)
    print(json.dumps(dict(backup_waiting=str(complete), update=count)), flush=True)
    deadline = time.monotonic() + ack_timeout
    while not (complete / 'ACK').exists():
        if time.monotonic() >= deadline:
            raise TimeoutError(f'No verified off-pod backup after {ack_timeout}s: {complete}')
        time.sleep(.5)
    print(json.dumps(dict(backup_acknowledged=str(complete), update=count)), flush=True)


def verify_packet(directory):
    manifest = json.loads((directory / 'manifest.json').read_text())
    for name, expected in manifest['files'].items():
        path = directory / name
        if digest(path) != expected:
            raise ValueError(f'Backup checksum mismatch: {path}')
        with path.open('rb') as stream:
            os.fsync(stream.fileno())
    count = manifest['completed_updates']
    with np.load(directory / 'state.npz') as state, np.load(directory / 'trace.npz') as trace:
        assert int(state['count']) == count
        assert len(trace['error']) == count - manifest['first_step'] + 1
        assert trace['checkpoint_steps'][-1] == count
        np.testing.assert_array_equal(trace['parameters'][-1], state['p'])
        assert state['p'].shape == state['m'].shape == state['v'].shape
    record = dict(verified_unix=time.time(), completed_updates=count,
                  manifest_sha256=digest(directory / 'manifest.json'))
    with (directory / 'verified.json').open('w') as stream:
        json.dump(record, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    return record


def collect(args):
    args.local_root.mkdir(parents=True, exist_ok=True)
    ssh = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
           '-o', 'ControlMaster=auto', '-o', 'ControlPersist=600',
           '-o', 'ControlPath=/tmp/figure4-ssh-%r-%h-%p',
           '-i', str(args.key.expanduser()), '-p', str(args.port)]
    deadline = time.monotonic() + args.seconds
    while time.monotonic() < deadline:
        try:
            subprocess.run(['rsync', '-az', '--partial', '--timeout=120',
                            '--exclude=.pending-*', '--exclude=ACK',
                            '-e', shlex.join(ssh),
                            f'{args.host}:{args.remote_root}/', str(args.local_root) + '/'],
                           check=True, timeout=300)
            acknowledgments = []
            for path in sorted(args.local_root.glob('**/step*/manifest.json')):
                directory = path.parent
                if not (directory / 'verified.json').exists():
                    record = verify_packet(directory)
                    print(json.dumps(dict(packet=str(directory), **record)), flush=True)
                if not (directory / 'ack_sent').exists():
                    acknowledgments.append(directory)
            if acknowledgments:
                remote_paths = [args.remote_root + '/' +
                                str(p.relative_to(args.local_root)) + '/ACK'
                                for p in acknowledgments]
                subprocess.run(ssh + [args.host, 'touch -- ' +
                                      ' '.join(shlex.quote(p) for p in remote_paths)],
                               check=True, timeout=30)
                for path in acknowledgments:
                    (path / 'ack_sent').touch()
            finished = list(args.local_root.glob('**/step05000000/ack_sent'))
            if args.expected_groups and len(finished) == args.expected_groups:
                (args.local_root / 'collector_complete.json').write_text(json.dumps(
                    dict(completed_groups=len(finished), verified_unix=time.time()), indent=2) + '\n')
                return
        except (subprocess.SubprocessError, OSError, ValueError, AssertionError) as error:
            print(json.dumps(dict(backup_error=str(error), unix=time.time())), flush=True)
        if args.once:
            return
        time.sleep(3)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', required=True)
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--key', type=Path, default=Path('~/.ssh/id_ed25519'))
    parser.add_argument('--remote-root', required=True)
    parser.add_argument('--local-root', type=Path, required=True)
    parser.add_argument('--seconds', type=int, default=14400)
    parser.add_argument('--expected-groups', type=int, default=0)
    parser.add_argument('--once', action='store_true')
    collect(parser.parse_args())
