"""Baseline-relative matching and induced tracking in physical GD pulses.

This separate analysis helper does not change any issued pulse or forecast.
Run its numerical evaluation on CPU through Slurm. It writes data only.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from . import effective_feedback as ef, mechanism_persistence_kernel as kernel
from .run import write_json


def scalar(value):
    value = float(value)
    return value if np.isfinite(value) else None


def ratio(numerator, denominator):
    return scalar(numerator/denominator) if denominator > 0 else None


def write_csv(path, rows):
    if rows:
        with path.open('w', newline='') as stream:
            fields = list(dict.fromkeys(key for row in rows for key in row))
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader(); writer.writerows(rows)


@jax.jit
def inspect(p, x, y):
    state = kernel.decomposition(p, x, y)
    w = (p.size-1)//3
    return dict(F=state['F'], R=state['R'], ga=state['g'][:w], z=state['z'],
                coarse=state['z']-state['balance'])


def matching_rows(points, cases, values, diagnostics):
    """Pure array bookkeeping; retain zero-denominator ratios as undefined."""
    width = (points.shape[1]-1)//3
    source_info = {int(item['source_index']): item for item in diagnostics}
    rows, paired, families = [], [], {}
    for i, case in enumerate(cases):
        b = int(case['reference_baseline_index'])
        if float(cases[b]['amplitude']) != 0:
            raise ValueError('Pulse pack reference does not identify a baseline')
        families.setdefault(b, {})[float(case['amplitude'])] = i
        here, base = values[i], values[b]
        q, q0 = np.linalg.norm(here['F']), np.linalg.norm(base['F'])
        fa, fa0 = np.linalg.norm(here['F'][:width]), np.linalg.norm(base['F'][:width])
        ga0, z0 = np.linalg.norm(base['ga']), np.linalg.norm(base['z'])
        record = source_info[int(case['source_index'])]
        pulse = next((p for p in record.get('pulses', [])
                      if p['amplitude'] == case['amplitude']), None)
        row = dict(case, baseline_ga_norm=scalar(ga0), baseline_force_norm=scalar(q0),
            baseline_slope_force_norm=scalar(fa0), baseline_z_norm=scalar(z0),
            current_force_norm=scalar(q), current_slope_force_norm=scalar(fa),
            coarse_change=scalar(np.linalg.norm(here['coarse']-base['coarse'])),
            z_change=scalar(np.linalg.norm(here['z']-base['z'])),
            z_change_over_baseline_z=ratio(np.linalg.norm(here['z']-base['z']), z0),
            slope_gradient_change=scalar(np.linalg.norm(here['ga']-base['ga'])),
            slope_gradient_change_over_baseline_ga=ratio(np.linalg.norm(here['ga']-base['ga']), ga0),
            force_norm_change_over_baseline_force=ratio(abs(q-q0), q0),
            force_squared_change_over_baseline_squared=ratio(abs(q*q-q0*q0), q0*q0),
            force_vector_change_over_baseline_force=ratio(np.linalg.norm(here['F']-base['F']), q0),
            tracking_norm=scalar(np.linalg.norm(here['R'])),
            tracking_over_current_force=ratio(np.linalg.norm(here['R']), q),
            tracking_over_baseline_force=ratio(np.linalg.norm(here['R']), q0),
            induced_tracking_over_baseline_force=ratio(np.linalg.norm(here['R']-base['R']), q0),
            slope_tracking_over_current_slope_force=ratio(np.linalg.norm(here['R'][:width]), fa),
            induced_slope_tracking_over_baseline_slope_force=ratio(
                np.linalg.norm((here['R']-base['R'])[:width]), fa0),
            slopes_unchanged=bool(np.array_equal(points[i, :width], points[b, :width])))
        if pulse is not None:
            row.update(pulse['diagnostics'])
            row.update(coarse_resolved=pulse['coarse_resolved'], finite=pulse['finite'],
                intrinsic_k_change=pulse['intrinsic_k_change'])
            derivative = record['direction_diagnostics']['intrinsic_k_derivative']
            row['intrinsic_k_linear_prediction'] = scalar(case['amplitude']*derivative)
            row['loaded_log_rate_correction'] = (scalar(row['k']-row['k_pure'])
                if row['k'] is not None and row['k_pure'] is not None else None)
        rows.append(row)
    for b, family in families.items():
        base = values[b]
        q0 = np.linalg.norm(base['F']); ga0 = np.linalg.norm(base['ga'])
        for amplitude in (.01, .005, .0025):
            if amplitude not in family or -amplitude not in family:
                continue
            plus, minus = family[amplitude], family[-amplitude]
            row = dict(cases[plus], baseline_force_norm=scalar(q0), baseline_ga_norm=scalar(ga0))
            for key, denominator in (('R', q0), ('ga', ga0), ('F', q0)):
                odd = (values[plus][key]-values[minus][key])/2
                even = (values[plus][key]+values[minus][key])/2-base[key]
                label = dict(R='tracking', ga='slope_gradient', F='force_vector')[key]
                row[label+'_odd_norm'] = scalar(np.linalg.norm(odd))
                row[label+'_even_norm'] = scalar(np.linalg.norm(even))
                row[label+'_odd_over_baseline'] = ratio(np.linalg.norm(odd), denominator)
                row[label+'_even_over_baseline'] = ratio(np.linalg.norm(even), denominator)
                row[label+'_central_derivative_norm'] = scalar(np.linalg.norm(odd)/amplitude)
            for key in ('k_pure', 'k', 'intrinsic_k_change'):
                a, c = rows[plus].get(key), rows[minus].get(key)
                row[key+'_central_derivative'] = scalar((a-c)/(2*amplitude)) if a is not None and c is not None else None
            paired.append(row)
    return rows, paired


def outcome_summary(analysis):
    """Aggregate all issued amplitude/horizon pairs without dropping misses."""
    with (analysis/'paired.csv').open() as stream:
        pairs = list(csv.DictReader(stream))
    groups = {}
    for row in pairs:
        groups.setdefault((int(row['horizon']), float(row['amplitude'])), []).append(row)
    result = []
    for (horizon, amplitude), rows in sorted(groups.items()):
        item = dict(horizon=horizon, amplitude=amplitude, pairs=len(rows),
            failed=sum(row['failed'] == 'True' for row in rows),
            unsupported=sum(row['forecast_supported'] != 'True' for row in rows))
        for key in ('slope_relative_error', 'readout_relative_error',
                    'slope_halving_response_difference', 'full_halving_response_difference'):
            samples = [float(row[key]) for row in rows if row.get(key) not in ('', None)]
            finite = [value for value in samples if np.isfinite(value)]
            item[key] = dict(defined=len(samples), finite=len(finite),
                median=scalar(np.median(finite)) if finite else None,
                maximum=scalar(max(finite)) if finite else None)
        for observable in ('mean_lambda', 'q_change'):
            actual_key = 'actual_'+observable+'_response'
            predicted_key = 'predicted_'+observable+'_response'
            valid = [row for row in rows if row[actual_key] and row[predicted_key]]
            misses = [row for row in valid if np.sign(float(row[actual_key])) != np.sign(float(row[predicted_key]))]
            item[observable+'_signs'] = dict(defined=len(valid), matches=len(valid)-len(misses),
                misses=[{key: row[key] for key in ('target', 'seed', 'start', 'source_index',
                                                 actual_key, predicted_key)} for row in misses])
        result.append(item)
    return result


def summarize(pulses, out, analysis=None):
    pulses, out = Path(pulses), Path(out)
    if out.exists():
        raise FileExistsError(out)
    pp, x, yy, cases = ef.load_inputs(pulses/'inputs.npz')
    diagnostics = json.loads((pulses/'diagnostics.json').read_text())
    values = [jax.tree.map(np.asarray, inspect(jnp.asarray(p), jnp.asarray(x), jnp.asarray(y)))
              for p, y in zip(pp, yy)]
    rows, paired = matching_rows(pp, cases, values, diagnostics)
    out.mkdir(parents=True)
    write_csv(out/'matching.csv', rows); write_csv(out/'paired_matching.csv', paired)
    signed = [row for row in rows if float(row['amplitude']) != 0]
    over_one = [row for row in signed if row['tracking_over_current_force'] is not None
                and row['tracking_over_current_force'] > 1]
    summary = dict(source_sha256=ef.digest(__file__), kernel_sha256=ef.digest(kernel.__file__),
        input_sha256=ef.digest(pulses/'inputs.npz'), diagnostics_sha256=ef.digest(pulses/'diagnostics.json'),
        baseline_cases=sum(float(row['amplitude']) == 0 for row in rows),
        unresolved_cases=[item['case'] for item in diagnostics if item['status'] != 'resolved'],
        signed_pulses=len(signed), pairs=len(paired),
        tracking_larger_than_effective=[{key: row[key] for key in
            ('target', 'seed', 'start', 'source_index', 'amplitude', 'tracking_over_current_force')}
            for row in over_one],
        interpretation='First-order matching does not bound finite-amplitude tracking relative to a weak force.',
        paired_units='Odd=(plus-minus)/2; even=(plus+minus)/2-baseline; central derivative=odd/amplitude.',
        undefined_ratios='Zero denominators remain null; no numerical floor is substituted.',
        numerical_certificate=False)
    if analysis is not None:
        analysis = Path(analysis)
        summary['outcomes'] = outcome_summary(analysis)
        summary['paired_outcomes_sha256'] = ef.digest(analysis/'paired.csv')
        summary['outcome_manifest_sha256'] = ef.digest(analysis/'manifest.json')
    write_json(out/'summary.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pulses', type=Path, required=True)
    parser.add_argument('--analysis', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not jax.config.x64_enabled or any(device.platform != 'cpu' for device in jax.devices()):
        raise RuntimeError('FP64 CPU analysis required')
    summarize(args.pulses, args.output, args.analysis)


if __name__ == '__main__':
    main()
