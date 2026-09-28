"""Resolve relative versus absolute Adam tracking effects from scalar evidence."""
import hashlib
import io
import json
from pathlib import Path

import modal

app = modal.App('d34-adam-tracking-scale')
image = modal.Image.debian_slim(python_version='3.12').pip_install('matplotlib==3.10.3')


@app.function(image=image, cpu=1, memory=(512, 1024), timeout=180)
def analyze(payload: str):
    import csv
    import math
    import resource
    import statistics
    import time

    started = time.monotonic()
    rows = []
    for source in csv.DictReader(io.StringIO(payload)):
        if source['optimizer'] != 'adam':
            continue
        assert source['complete'] == 'True'
        w = int(float(source['width']))
        h = 2 / {177: 128, 705: 512, 1409: 1024}[w]
        r = {k: source[k] for k in ('cohort', 'target')}
        r.update(width=w, seed=int(float(source['seed'])))
        for k in ('lambda_start', 'lambda_end', 'A_fine', 'A_tracking', 'A_defect',
                  'A_unresolved', 'A_change', 'tracking_slope_activity_share',
                  'M_start', 'M_end', 'C6_start', 'C6_end', 'effective_count_start',
                  'balanced_raw_access_median', 'balanced_adaptive_access_median'):
            r[k] = float(source[k])
        r['A_start'] = w * (r['lambda_start'] / h)**2
        r['slope_rms_start'] = r['lambda_start'] / h
        r['slope_rms_end'] = r['lambda_end'] / h
        r['rms_change_percent'] = 100 * (r['lambda_end'] / r['lambda_start'] - 1)
        r['tracking_over_fine_percent'] = 100 * r['A_tracking'] / r['A_fine']
        for q in ('fine', 'tracking', 'defect', 'change', 'unresolved'):
            r[q + '_percent_start_A'] = 100 * r['A_' + q] / r['A_start']
            r[q + '_delta_lambda_squared'] = h*h/w * r['A_' + q]
        observed = r['lambda_end']**2 - r['lambda_start']**2
        accounted = sum(r[q + '_delta_lambda_squared']
                        for q in ('fine', 'tracking', 'defect', 'unresolved'))
        scale = max(r['lambda_start']**2, abs(observed))
        r['closure_relative'] = abs(accounted - observed) / scale
        assert r['closure_relative'] < 1e-9
        for end in ('start', 'end'):
            r['raw_sensitivity_bound_' + end] = (
                math.sqrt(6)/2 * math.sqrt(r['C6_' + end]) * r['M_' + end]**1.5 / w)
        r['adaptive_to_raw_access_median_ratio'] = (
            r['balanced_adaptive_access_median'] / r['balanced_raw_access_median'])
        rows.append(r)
    assert len(rows) == 50
    wide = [r for r in rows if r['cohort'] == 'wide']
    assert len(wide) == 24
    facts = {}
    for label, selected in (
        ('wide', wide),
        ('wide_gaussian', [r for r in wide if r['target'] == 'gauss_left']),
        ('wide_non_gaussian', [r for r in wide if r['target'] != 'gauss_left']),
        ('narrow', [r for r in rows if r['width'] == 177]),
    ):
        fields = ('tracking_over_fine_percent', 'tracking_percent_start_A',
                  'fine_percent_start_A', 'rms_change_percent',
                  'tracking_delta_lambda_squared', 'raw_sensitivity_bound_start',
                  'adaptive_to_raw_access_median_ratio', 'closure_relative')
        facts[label] = {'n': len(selected)}
        for key in fields:
            values = [r[key] for r in selected]
            facts[label][key] = dict(min=min(values), median=statistics.median(values), max=max(values))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    targets = ('moment5', 'mixed_sine', 'gauss_left', 'bump_right', 'step_right', 'kink_abs')
    labels = ('Degree 5', 'Mixed sine', 'Gaussian', 'Bump', 'Step', 'Kink')
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8), layout='constrained')
    for panel, (key, ylabel) in enumerate((
        ('tracking_over_fine_percent', 'Tracking / effective fine contribution (%)'),
        ('tracking_percent_start_A', 'Tracking contribution / starting slope energy (%)'),
    )):
        ax = axes[panel]
        for r in wide:
            offset = (-.14 if r['width'] == 705 else .14) + (-.04 if r['seed'] == 30 else .04)
            ax.scatter(targets.index(r['target']) + offset, r[key],
                       color='#0072B2' if r['width'] == 705 else '#D55E00',
                       marker='o' if r['seed'] == 30 else 's', s=33)
        ax.axhline(0, color='gray', lw=.8)
        ax.set_xticks(range(6), labels, rotation=25)
        ax.set_ylabel(ylabel)
        ax.grid(axis='y', alpha=.2)
    axes[0].axhline(-100, color='gray', ls=':', lw=1)
    axes[0].set_title('Relative to fine-driven growth')
    axes[1].set_title('Relative to the existing slope scale')
    handles = [Line2D([], [], ls='', marker='o', color=c, label=f'Width {w}')
               for w, c in ((705, '#0072B2'), (1409, '#D55E00'))]
    handles += [Line2D([], [], ls='', marker=m, color='black', label=f'Seed {s}')
                for s, m in ((30, 'o'), (31, 's'))]
    fig.legend(handles=handles, loc='outside lower center', ncol=4)
    fig.suptitle('Adam, updates 25k–125k: signed tracking contribution to slope energy')
    png = io.BytesIO()
    fig.savefig(png, format='png', dpi=180)
    plt.close(fig)
    csv_output = io.StringIO()
    writer = csv.DictWriter(csv_output, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    receipt = dict(platform='Modal CPU', input_sha256=hashlib.sha256(payload.encode()).hexdigest(),
                   seconds=time.monotonic()-started, memory_hard_limit_mib=1024,
                   peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                   cases=len(rows), maximum_closure_relative=max(r['closure_relative'] for r in rows))
    return {'tracking_scale.csv': csv_output.getvalue().encode(),
            'facts.json': (json.dumps(facts, indent=2)+'\n').encode(),
            'execution.json': (json.dumps(receipt, indent=2)+'\n').encode(),
            'tracking_relative_absolute.png': png.getvalue()}


@app.local_entrypoint()
def main(source: str, output: str):
    destination = Path(output)
    if destination.exists():
        raise ValueError('Use a fresh output path')
    path = Path(source)
    if path.stat().st_size > 1024**2:
        raise ValueError('Expected only the small scalar summary')
    artifacts = analyze.remote(path.read_text())
    destination.mkdir(parents=True)
    for name, data in artifacts.items():
        (destination / name).write_bytes(data)
    print(f'Saved {len(artifacts)} scalar/plot artifacts to {destination}')
