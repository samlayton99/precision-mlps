"""Audit saved numerical artifacts; emit data only, never report prose."""
import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

from experiments.expD36_frozen_gamma_probe.cap_analyze import collect

parser = argparse.ArgumentParser()
parser.add_argument('--root', type=Path, default=Path('/private/tmp/gamma-cap-evidence'))
parser.add_argument('--accounting', type=Path, default=Path('/private/tmp/gamma-cap-slurm-final.tsv'))
parser.add_argument('--output', type=Path, default=Path('/private/tmp/gamma-cap-run-audit.json'))
args = parser.parse_args()
root = args.root
summary = collect(root)
trained = [c for c in summary['cases'] if 'training' in c]
assert all('reference' in c for c in trained)
assert not summary['bound_violations']
assert all(not c['incomplete'] for c in summary['coverage'].values())
reached = [r for r in summary['forecast_checks'] if r['difference'] is not None]
assert all(r['difference'] == 0 for r in reached)
unresolved_reached = [r for r in summary['forecast_checks']
                      if r['actual'] >= 0 and r['predicted'] is None]
assert not unresolved_reached
rows = list(csv.DictReader(args.accounting.open(), delimiter='\t'))
assert all(r['State'] not in ['RUNNING', 'PENDING', 'COMPLETING'] for r in rows)
gpu_seconds = sum(int(r['ElapsedRaw']) * int(m.group(1))
                  for r in rows if (m := re.search(r'gres/gpu=(\d+)', r['AllocTRES'])))
assert gpu_seconds <= 36000
searches = []
for path in sorted(root.glob('search_round*_cases.json')):
    cases = [json.loads((root/'cases'/name/'meta.json').read_text())
             for name in json.loads(path.read_text())]
    for cap in sorted({r['cap'] for r in cases}):
        group = [r for r in cases if r['cap'] == cap]
        valid = [r for r in group if r['screen_hits']['0.01'][0] is not None]
        best = min(valid, key=lambda r: r['screen_hits']['0.01'][0], default=None)
        searches.append(dict(round=int(path.stem.split('round')[1].split('_')[0]),
            cap=cap, candidates=len(group), best_case=best['id'] if best else None,
            best_forecast=best['screen_hits']['0.01'][0] if best else None))
result = dict(
    cases=len(summary['cases']), trained_cases=len(trained),
    certificates=len(summary['certificates']), coverage=summary['coverage'],
    forecast_comparisons=len(summary['forecast_checks']),
    reached_forecast_comparisons=len(reached), exact_hit_agreements=len(reached),
    unresolved_reached=unresolved_reached,
    reached_by_tolerance=dict(Counter(str(r['epsilon']) for r in reached)),
    certificate_comparisons=len(summary['certificate_checks']),
    held_out_certificate_comparisons=sum(r['held_out'] for r in summary['certificate_checks']),
    bound_violations=summary['bound_violations'],
    maximum_curve_absolute_difference=max(c['reference']['max_curve_absolute_difference'] for c in trained),
    eta_L_range=[min(c['reference']['eta_L'] for c in trained),
                 max(c['reference']['eta_L'] for c in trained)],
    allocated_gpu_seconds=gpu_seconds, allocated_gpu_hours=gpu_seconds/3600,
    authorized_gpu_hours=10, jobs=len(rows), searches=sorted(searches, key=lambda r:(r['round'],r['cap'])),
    provenance_overrides=[
        dict(job=798, saved_source_commit='local', actual_snapshot='cap-code-e7cced9',
             explanation='Submission omitted PROBE_SOURCE_COMMIT; immutable deployed snapshot identifies source.'),
        dict(job=807, reused_interval_proof_job=784, reused_snapshot='cap-code-74cba14',
             explanation='Primary cap-16 smallest-budget proof was copied unchanged into the one-budget confirmation.')])
args.output.write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ['searches','provenance_overrides']},indent=2))
