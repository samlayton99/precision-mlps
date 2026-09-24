"""Compact numerical facts for manual interpretation of the dilation panel."""
import csv
import json
from pathlib import Path

import numpy as np

root = Path('/workspace/junmiaoh/experiments/precision-mlps/mechanism-0365cd3-20260923/evidence/dilation_b52df33/analysis')
rows = list(csv.DictReader((root/'endpoints.csv').open()))
states = list(csv.DictReader((root/'states.csv').open()))

def summary(selected, name):
    values = [float(row[name]) for row in selected if row.get(name) not in ('', None)]
    return [float(np.min(values)), float(np.median(values)), float(np.max(values))] if values else None

facts = {}
for panel in ('late', 'wide_N512', 'wide_N1024'):
    for arm in ('original', 'repaired', 's125_primary', 's2_primary', 's125_inverse', 's2_inverse'):
        chosen = [r for r in rows if r['panel'].startswith(panel) and r['arm'] == arm]
        trace = [r for r in states if r['panel'].startswith(panel) and r['arm'] == arm]
        metrics = ('lambda_change_pct', 'gap_retention', 'initial_Fa_gain_repaired',
                   'Fa_gain_own_start', 'tracking_integrated_norm_to_effective',
                   'forecast_vector_relative_error', 'lambda_max', 'closure_error')
        record = {key: summary(chosen, key) for key in metrics}
        record.update(n=len(chosen), inward=sum(float(r['lambda_change']) < 0 for r in chosen),
            initially_outward=sum(float(r['effective_initial_signed_mean_rate']) > 0 for r in chosen),
            finally_outward=sum(float(r['effective_signed_mean_rate']) > 0 for r in chosen),
            outward_reinforced=sum(float(r['effective_signed_mean_rate']) > max(0., float(r['effective_initial_signed_mean_rate'])) for r in chosen),
            generated_q23_initial_inward=sum(float(r['q2_generated_signed_rate_initial'])+float(r['q3_generated_signed_rate_initial']) < 0 for r in chosen),
            initial_injected_hits=sum(int(r['injected_newly_above_lambda025']) for r in chosen),
            new_gd_hits=sum(int(r['newly_hit_lambda025']) for r in chosen),
            sampled_tracking_ratio=summary(trace, 'tracking_to_effective_norm'))
        signed_ratios = [abs(float(r['tracking_signed_travel'])/float(r['lambda_change']))
                         for r in chosen if abs(float(r['lambda_change'])) > 1e-14]
        record['signed_tracking_to_net_scale_change'] = (
            [float(np.min(signed_ratios)), float(np.median(signed_ratios)), float(np.max(signed_ratios))]
            if signed_ratios else None)
        facts[panel+'/'+arm] = record
target_details = [{key: row[key] for key in ('panel', 'target', 'seed', 'arm', 'lambda_change_pct',
                  'initial_Fa_gain_repaired', 'Fa_gain_own_start', 'effective_initial_signed_mean_rate',
                  'effective_signed_mean_rate', 'tracking_integrated_norm_to_effective',
                  'q2_generated_signed_rate_initial', 'q3_generated_signed_rate_initial',
                  'q2_target_signed_rate_initial', 'q3_target_signed_rate_initial')}
                  for row in rows if row['arm'] == 's2_primary' and row['panel'] != 'halfstep_run']
halves = [r for r in csv.DictReader((root/'halfstep.csv').open()) if float(r['flow_time']) == 40.]
half_summary = {key: summary(halves, key) for key in ('slope_endpoint_relative_to_primary_motion',
                                                     'half_minus_primary_change_pct')}
result = dict(range_order='min, median, max', panels=facts, twofold_primary_cases=target_details,
              final_halfstep=half_summary)
(root/'interpretation_facts.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(dict(panels=facts, final_halfstep=half_summary), indent=2))
