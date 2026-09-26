"""Reconstruct ordinary Figure 4 arrays from checksum-verified export segments."""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np

from figure4_durable import digest


def restore_arrays(exports, output, through=None, allocate_through=None):
    packets = [(p.parent, json.loads(p.read_text()))
               for p in sorted(exports.glob('step*/manifest.json'))]
    if through is not None:
        packets = [(p, m) for p, m in packets if m['completed_updates'] <= through]
    if not packets:
        raise ValueError('No exported segments')
    count = packets[-1][1]['completed_updates']
    if through is not None and count != through:
        raise ValueError('Requested endpoint is not an exported checkpoint')
    allocated = count if allocate_through is None else allocate_through
    if allocated < count:
        raise ValueError('Allocated trajectory cannot truncate verified data')
    metadata = packets[0][1]['metadata']
    for directory, manifest in packets:
        if not ((directory/'verified.json').exists() or (directory/'ACK').exists()):
            raise ValueError(f'Export was not acknowledged: {directory}')
        for name, expected in manifest['files'].items():
            if digest(directory/name) != expected:
                raise ValueError(f'Checksum mismatch: {directory/name}')
        for key in ['config', 'input_sha256', 'width', 'recipes']:
            if manifest['metadata'][key] != metadata[key]:
                raise ValueError(f'Inconsistent {key}: {directory}')
    output.mkdir(parents=True, exist_ok=True)
    if (output/'relative_error.npy').exists():
        raise FileExistsError('Reconstruction requires a fresh output directory')
    with np.load(packets[-1][0]/'state.npz') as state:
        shape = state['p'].shape
    errors = np.lib.format.open_memmap(output/'relative_error.npy', mode='w+',
                                      dtype=np.float64, shape=(allocated+1, *shape[:2]))
    rms = np.lib.format.open_memmap(output/'slope_rms.npy', mode='w+',
                                   dtype=np.float64, shape=errors.shape)
    steps = np.unique(np.r_[np.arange(0, allocated+1, 10000), allocated])
    snapshots = np.lib.format.open_memmap(output/'parameter_checkpoints.npy', mode='w+',
                                          dtype=np.float64, shape=(len(steps), *shape))
    errors[:] = np.nan; rms[:] = np.nan; snapshots[:] = np.nan
    next_step = 0
    saved_steps = []
    for directory, manifest in packets:
        first, last = manifest['first_step'], manifest['completed_updates']
        if first != next_step:
            raise ValueError(f'Trace gap or overlap at {directory}: expected {next_step}')
        with np.load(directory/'trace.npz') as trace, np.load(directory/'state.npz') as state:
            assert trace['error'].shape == (last-first+1, *shape[:2])
            assert trace['slope_rms'].shape == trace['error'].shape
            assert int(state['count']) == last
            np.testing.assert_array_equal(trace['parameters'][-1], state['p'])
            actual_steps = trace['checkpoint_steps']
            expected_steps = steps[(steps >= first) & (steps <= last)]
            np.testing.assert_array_equal(actual_steps, expected_steps)
            errors[first:last+1] = trace['error']; rms[first:last+1] = trace['slope_rms']
            snapshots[np.searchsorted(steps, actual_steps)] = trace['parameters']
            saved_steps.extend(actual_steps.tolist())
        next_step = last+1
    np.testing.assert_array_equal(saved_steps, steps[steps <= count])
    errors.flush(); rms.flush(); snapshots.flush()
    np.save(output/'checkpoint_steps.npy', steps)
    shutil.copyfile(packets[-1][0]/'state.npz', output/'state.npz')
    (output/'metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    record = dict(completed_updates=count, allocated_through=allocated,
                  exports=str(exports), verified_segments=len(packets),
                  manifest_sha256=[digest(p/'manifest.json') for p, _ in packets])
    (output/'reconstruction.json').write_text(json.dumps(record, indent=2)+'\n')
    return count, metadata


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exports', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--through', type=int)
    args = parser.parse_args()
    count, _ = restore_arrays(args.exports, args.output, args.through)
    print(json.dumps(dict(completed_updates=count, output=str(args.output))))
