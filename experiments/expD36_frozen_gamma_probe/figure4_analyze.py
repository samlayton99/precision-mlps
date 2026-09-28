"""Reconstruct and select the completed four-width Figure 4 sweep."""
import argparse
import json
from pathlib import Path

from threadpoolctl import threadpool_limits

from figure4_restore import restore_arrays
from section34_analyze import analyze


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    summaries = {}
    for width in [128, 512, 1024, 256]:
        base = args.root/f'w{width}'
        runs = []
        for group in ['joint_adam_cosine', 'joint_adam_constant', 'joint_gd']:
            exports = args.root/'durable'/f'w{width}'/group
            if not (exports/'step05000000/verified.json').exists():
                raise ValueError(f'Incomplete experiment: {exports}')
            run = args.root/'reconstructed'/f'w{width}'/group
            if not (run/'reconstruction.json').exists():
                restore_arrays(exports, run, through=5000000)
            record = json.loads((run/'reconstruction.json').read_text())
            assert record['completed_updates'] == record['allocated_through'] == 5000000
            runs.append(('gd' if group == 'joint_gd' else 'adam', run))
        output = args.output/f'w{width}'
        analyze(base/'base', runs, output)
        summaries[str(width)] = json.loads((output/'summary.json').read_text())
        print(f'Completed analysis for W={width}', flush=True)
    (args.output/'width_summary.json').write_text(json.dumps(summaries, indent=2)+'\n')


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        main()
