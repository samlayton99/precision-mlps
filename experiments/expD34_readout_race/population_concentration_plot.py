"""Compare sampled force concentration on broad short and selected long GD paths."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .population_output_summary import load


def run(args):
    rows = load(args.states)
    endpoints = load(args.endpoints)
    groups = [defaultdict(list), defaultdict(list)]
    for row in rows:
        if row.get('reinforcement_status') != 'finite' or row.get('failed') is True:
            continue
        if row['width'] != 705 or row['start'] != 20000 or row['eta'] != .002:
            continue
        if row['arm'] == 'original' and row['panel'] in ('wide_N512_run', 'missing_wide_N512_run'):
            column = 0
        elif row['arm'] == 'repaired' and row['panel'] == 'wide_N512_long':
            column = 1
        else:
            continue
        groups[column][(row['target'], row['seed'])].append(row)
    if [len(group) for group in groups] != [46, 6]:
        raise ValueError('Expected 46 broad original and 6 long repaired trajectories')
    args.output.mkdir(parents=True, exist_ok=False)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex='col', sharey='row', constrained_layout=True)
    records = []
    for column, paths in enumerate(groups):
        for (target, seed), path in paths.items():
            path.sort(key=lambda r: r['horizon'])
            steps = [r['horizon'] for r in path]
            if steps[0] != 0:
                raise ValueError('Missing restart observation')
            concentrations = [r['reinforcement_hidden_force_energy_concentration'] for r in path]
            errors = [r['relative_l2'] for r in path]
            highlighted = column == 1 and target in ('bump_right', 'gauss_left')
            color = {'bump_right': 'C0', 'gauss_left': 'C1'}.get(target, '.6') if column else '.6'
            label = target.replace('_', ' ') if highlighted else None
            for ax, values in zip(axes[:, column], (concentrations, errors)):
                ax.plot(steps, values, 'o-', color=color, alpha=.95 if highlighted else .35,
                        markersize=4 if highlighted else 2, linewidth=1.7 if highlighted else .8, label=label)
            record = dict(panel=column, target=target, seed=seed, steps=steps,
                          I_F=concentrations, raw_relative_l2=errors)
            if column == 1:
                matches = [r for r in endpoints if r['target'] == target and r['seed'] == seed
                           and r['width'] == 705 and r['start'] == 20000 and r['arm'] == 'repaired'
                           and r['eta'] == .002 and r['requested_steps'] == 100000]
                if len(matches) != 1 or matches[0]['failed'] or matches[0]['completed_steps'] != 100000:
                    raise ValueError('Missing or incomplete matching long-run endpoint')
                endpoint = matches[0]
                record.update(first_hit_raw_error_001=endpoint['first_hit_relative_l2_0.01'],
                              independent_grid_endpoint_error=endpoint['relative_eval_l2'],
                              sampled_I32_crossing=bool(min(concentrations) <= 32 < max(concentrations)))
            records.append(record)
        axes[0, column].axhline(32, color='k', ls='--', lw=1, label='Chosen allowance I*=32')
        axes[1, column].axhline(.01, color='k', ls='--', lw=1)
        axes[0, column].set_yscale('log')
        axes[1, column].set_yscale('log')
        axes[1, column].set_xlabel('Further GD updates after 20k restart')
        for ax in axes[:, column]:
            ax.set_xlim(left=0)
            ax.grid(alpha=.15)
            ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
    axes[0, 0].set_title('23 targets × 2 seeds\nOriginal continuation: 20k further updates')
    axes[0, 1].set_title('Six targets × seed 30\nRepaired continuation: 100k further updates')
    axes[0, 0].set_ylabel('Effective-force concentration I_F')
    axes[1, 0].set_ylabel('Raw relative L2 error (training grid)')
    axes[0, 1].legend(fontsize=8, loc='upper left')
    axes[1, 1].text(.02, .07, 'Dashed: 1% error requirement', transform=axes[1, 1].transAxes, fontsize=9)
    fig.suptitle('A distributed-force regime has a finite empirical range\n'
                 'I*=32 is a chosen theorem allowance, not a universal transition; lines connect saved checkpoints', fontsize=11)
    fig.savefig(args.output/'concentration_and_output.png', dpi=180)
    plt.close(fig)
    facts = dict(trajectories=records,
                 sources={str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in (args.states, args.endpoints)},
                 helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                 scope='Sparse effective-force samples along full GD; first-hit counters separately cover every training update')
    (args.output/'facts.json').write_text(json.dumps(facts, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--states', type=Path, required=True)
    parser.add_argument('--endpoints', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
