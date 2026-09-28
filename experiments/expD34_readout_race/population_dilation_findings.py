"""Additional output/acquisition facts from already validated dilation summaries."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

from .population_output_summary import load, finite
from .mechanism_dilation_analysis import clean, digest, stats


TOLERANCES = (.1, .01, .001, .0001, .000001)
FIELDS = ('initial_relative_l2', 'relative_l2', 'relative_l2_own_start_ratio',
          'initial_relative_eval_l2', 'relative_eval_l2', 'relative_eval_l2_own_start_ratio',
          'relative_l2_minus_repaired', 'relative_l2_initial_minus_repaired',
          'initial_gamma_mean', 'gamma_mean', 'gamma_mean_own_start_ratio',
          'initial_lambda_mean', 'lambda_mean', 'initial_lambda_rms', 'lambda_rms',
          'Q_own_start_ratio', 'Fa_own_start_ratio', 'tracking_integrated_norm_to_effective',
          'tracking_signed_travel', 'effective_signed_travel', 'closure_error')


def group_facts(rows):
    complete = [r for r in rows if r.get('failed') is False
                and r['completed_steps'] == r['requested_steps']]
    thresholds = {}
    for threshold in TOLERANCES:
        key = f'first_hit_relative_l2_{threshold:g}'
        thresholds[str(threshold)] = dict(
            exact_training_event_records=sum(finite(r, key) for r in complete),
            already_met_after_repair=sum(r.get(key) == 0 for r in complete),
            newly_met_during_training=sum(r.get(key, -1) > 0 for r in complete if finite(r, key)),
            training_endpoint_meets=sum(r['relative_l2'] <= threshold for r in complete),
            evaluation_endpoint_meets=sum(r['relative_eval_l2'] <= threshold for r in complete
                                          if finite(r, 'relative_eval_l2')),
            evaluation_already_met_after_repair=sum(r['initial_relative_eval_l2'] <= threshold
                for r in complete if finite(r, 'initial_relative_eval_l2')))
    return dict(attempted=len(rows), complete=len(complete),
                targets=sorted({r['target'] for r in rows}), seeds=sorted({r['seed'] for r in rows}),
                statistics={key: stats(complete, key) for key in FIELDS}, thresholds=thresholds,
                training_error_decreased=sum(r['relative_l2'] < r['initial_relative_l2'] for r in complete),
                evaluation_error_decreased=sum(r['relative_eval_l2'] < r['initial_relative_eval_l2'] for r in complete
                                               if finite(r, 'relative_eval_l2') and finite(r, 'initial_relative_eval_l2')),
                endpoints_better_than_repaired=sum(r['relative_l2_minus_repaired'] < 0 for r in complete
                                                   if finite(r, 'relative_l2_minus_repaired')),
                initially_above_lambda025_labels=sum(int(r['initially_above_lambda025']) for r in complete),
                learned_new_lambda025_labels=sum(int(r['learned_new_hits_lambda025']) for r in complete),
                branches_with_new_lambda025_labels=sum(r['learned_new_hits_lambda025'] > 0 for r in complete))


def run(args):
    rows = load(args.endpoints)
    groups = defaultdict(list)
    for row in rows:
        if finite(row, 'direct_fine_signed_travel') and finite(row, 'compensation_signed_travel'):
            row['effective_signed_travel'] = row['direct_fine_signed_travel']+row['compensation_signed_travel']
        key = tuple(row[k] for k in ('cohort', 'width', 'start', 'eta', 'requested_steps', 'reference', 'scale'))
        groups[key].append(row)
    records = [dict(group=key, **group_facts(group)) for key, group in groups.items()]
    controls = load(args.step_controls) if args.step_controls and args.step_controls.exists() else []
    accepted_controls = [r for r in controls if r.get('either_failed') is False
                         and r.get('equal_completed_flow_time') is True
                         and r.get('input_hash_match') is True and r.get('initial_state_match') is True]
    matched = dict(records=len(controls), valid_matched_records=len(accepted_controls),
                   slope_difference=stats(accepted_controls, 'endpoint_slope_difference'),
                   relative_to_displacement=stats(accepted_controls, 'relative_to_coarse_displacement'))
    sources = [args.endpoints]+([args.step_controls] if controls else [])
    result = dict(groups=records, step_controls=matched,
                  sources={str(path): digest(path) for path in sources}, helper_sha256=digest(Path(__file__)),
                  conventions=dict(training_hits='Every-update first-hit counter; 0 means already met after intervention',
                                   evaluation_hits='Endpoint and post-repair observations only; no exact evaluation first-hit claim',
                                   scale_counts='Counts labels separately from branches; initial occupancy includes original and injected labels',
                                   failed_branches='Excluded from complete-case event claims; first-hit history can be censored by nonfinite termination'))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.write_text(json.dumps(clean(result), indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--endpoints', type=Path, required=True)
    parser.add_argument('--step-controls', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
