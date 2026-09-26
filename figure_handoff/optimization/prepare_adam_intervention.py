"""Extract Figure 4(c) from completed pulse summaries and cumulative scalar histories."""
import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path


TARGETS = ('moment5', 'mixed_sine', 'gauss_left', 'bump_right', 'step_right', 'kink_abs')
POLICIES = {
    'gain_only': 'Scalar amplification',
    'tracking_attenuated_variance': 'Tracking-attenuated denominator',
}


def read(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def integer(value):
    number = float(value)
    if not math.isfinite(number) or not number.is_integer():
        raise ValueError(f'Expected integer identifier: {value!r}')
    return int(number)


def start(row):
    return row['target'], integer(row['width']), integer(row['seed'])


def index(rows, fields):
    result = {}
    for row in rows:
        key = (*start(row), *(integer(row[f]) if f == 'offset' else row[f] for f in fields))
        if key in result:
            raise ValueError(f'Duplicate match: {key}')
        result[key] = row
    return result


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(a, b, label):
    require(math.isclose(float(a), float(b), rel_tol=2e-13, abs_tol=2e-12),
            f'Inconsistent {label}: {a} vs {b}')


def extract(evidence):
    final = evidence/'adam_variance_final_20260926'
    runs = evidence/'adam_variance_runs_20260926'
    paths = [final/'denominator_contrasts.csv', final/'native_denominator_gains.csv',
             final/'facts.json', runs/'states.csv', runs/'runs.csv']
    summaries = index([r for r in read(paths[0]) if integer(r['offset']) == 10000], ('arm',))
    passive = index(read(paths[1]), ())
    facts = json.loads(paths[2].read_text())
    states = index(read(paths[3]), ('arm', 'offset'))
    records = index(read(paths[4]), ('arm',))
    starts = set(itertools.product(TARGETS, (705, 1409), (30, 31)))
    expected = {(*key, policy) for key in starts for policy in POLICIES}
    require(set(summaries) == expected,
            f'Pulse matches: missing={expected-set(summaries)}, unexpected={set(summaries)-expected}')
    require(set(passive) == starts, 'Passive-gain cohort differs from the 24 matched starts')
    all_branches = {(*key, arm) for key in starts for arm in ('native', *POLICIES)}
    require(set(records) == all_branches, 'Missing or unexpected branch completion records')
    expected_states = {(*key, offset) for key in all_branches for offset in range(0, 20001, 1000)}
    require(set(states) == expected_states, 'Missing or unexpected scalar history records')
    require(facts['complete_runs'] == 72 and facts['native_cases'] == 24,
            'Campaign facts do not confirm all 72 branches')

    for key in sorted(all_branches):
        require(records[key]['complete'] == 'True', f'Incomplete branch: {key}')
        initial = states[(*key, 0)]
        native_initial = states[(*key[:3], 'native', 0)]
        require(float(initial['fine_slope_path']) == 0, f'Path counter not reset at fork: {key}')
        for field in ('h', 'A', 'M', 'bias2', 'readout2', 'relative_error'):
            close(initial[field], native_initial[field], f'shared fork {key}/{field}')
        previous_path = 0.
        for offset in range(0, 10001, 1000):
            row = states[(*key, offset)]
            require(integer(row['age']) == 130000 and integer(row['step']) == 130000+offset,
                    f'Wrong pulse age: {key}/{offset}')
            path, rms, h = (float(row[f]) for f in ('fine_slope_path', 'slope_rms', 'h'))
            require(all(math.isfinite(v) for v in (path, rms, h)) and rms > 0 and h > 0,
                    f'Nonfinite or nonpositive scalar: {key}/{offset}')
            require(path >= previous_path and (offset == 0 or path > 0),
                    f'Invalid cumulative path: {key}/{offset}')
            close(h, native_initial['h'], f'matched spacing {key}/{offset}')
            close(rms, math.sqrt(float(row['A'])/key[1]), f'RMS from slope energy {key}/{offset}')
            close(row['lambda_rms'], h*rms, f'bandwidth {key}/{offset}')
            previous_path = path

    points = []
    for key in sorted(expected):
        target, width, seed, policy = key
        branch = states[(*key, 10000)]
        native = states[(target, width, seed, 'native', 10000)]
        p, p0 = float(branch['fine_slope_path']), float(native['fine_slope_path'])
        s, s0 = float(branch['slope_rms']), float(native['slope_rms'])
        x, y = p/p0, 100*(s/s0-1)
        summary = summaries[key]
        close(x, summary['fine_path_ratio'], f'path ratio {key}')
        close(y, summary['slope_effect_percent'], f'RMS response {key}')
        close(branch['lambda_rms'], summary['lambda_rms'], f'endpoint bandwidth {key}')
        close(float(branch['lambda_rms'])-float(native['lambda_rms']),
              summary['lambda_difference'], f'bandwidth difference {key}')
        points.append(dict(target=target, width=width, seed=seed, policy=policy,
            policy_label=POLICIES[policy], fork_age=130000, endpoint_age=140000,
            pulse_updates=10000, spacing=float(branch['h']),
            native_fine_path=p0, intervention_fine_path=p,
            native_endpoint_rms=s0, intervention_endpoint_rms=s,
            native_endpoint_bandwidth=float(native['lambda_rms']),
            intervention_endpoint_bandwidth=float(branch['lambda_rms']),
            fine_path_ratio=x, slope_effect_percent=y))

    checks = {}
    for policy in POLICIES:
        subset = [r for r in points if r['policy'] == policy]
        expected_stats = facts['arms'][f'{policy}_10000']
        for output_field, fact_field in [('fine_path_ratio', 'fine_path_ratio'),
                                         ('slope_effect_percent', 'slope_effect_percent'),
                                         ('intervention_endpoint_bandwidth', 'lambda_rms')]:
            for kind, fn in [('min', min), ('max', max)]:
                close(fn(r[output_field] for r in subset), expected_stats[fact_field][kind],
                      f'facts {policy}/{output_field}/{kind}')
        checks[policy] = dict(points=len(subset), positive=sum(r['slope_effect_percent'] > 0 for r in subset),
                             negative=sum(r['slope_effect_percent'] < 0 for r in subset))
    checks.update(points=len(points), matched_starts=len(starts), completed_branches=len(records),
        scalar_pulse_rows=len(records)*11, duplicate_matches=0, missing_matches=0,
        fine_path_range=[min(r['fine_path_ratio'] for r in points), max(r['fine_path_ratio'] for r in points)],
        response_percent_range=[min(r['slope_effect_percent'] for r in points), max(r['slope_effect_percent'] for r in points)],
        maximum_endpoint_bandwidth=max(r['intervention_endpoint_bandwidth'] for r in points),
        compact_bump_w705_seed30=[r for r in points if (r['target'], r['width'], r['seed']) == ('bump_right', 705, 30)])
    provenance = dict(inputs=[dict(path=str(p.resolve()), sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths],
        mapping=dict(summary_x='fine_path_ratio', summary_y='slope_effect_percent',
            history_path='fine_slope_path', history_rms='slope_rms', history_spacing='h',
            history_bandwidth='lambda_rms', selection='offset=10000; age=130000; step=140000',
            x='intervention_fine_path / native_fine_path',
            y='100 * (intervention_endpoint_rms / native_endpoint_rms - 1)'),
        checks=checks)
    return points, provenance


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent/'data')
    args = parser.parse_args()
    points, provenance = extract(args.evidence)
    args.output.mkdir(parents=True, exist_ok=True)
    destination = args.output/'adam_intervention.csv'
    with destination.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(points[0]), lineterminator='\n')
        writer.writeheader(); writer.writerows(points)
    provenance['plotted_csv_sha256'] = hashlib.sha256(destination.read_bytes()).hexdigest()
    (args.output/'adam_intervention_provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    print(json.dumps(provenance['checks'], indent=2))
