"""Build a browsable index and compact numerical summary after the spectrum audit."""
import csv
import html
import json
import numpy as np
from run import OUT, load

records=load()
assert len(records)==216
names=list(dict.fromkeys(r['target'] for r in records))
assert len(names)==24
floor=float(records[0]['resolution_floor'])
with (OUT/'all_checkpoint_percentiles.csv').open('w') as f:
    writer=csv.writer(f)
    writer.writerow(['function','initialization_multiplier','learning_rate','step',
                     *[f'p{p:02}' for p in range(0,101,10)],'unresolved_fraction'])
    for r in records:
        for step,q,unresolved in zip(r['steps'],r['quantiles'],r['unresolved_fraction']):
            writer.writerow([r['target'],r['initialization_multiplier'],r['learning_rate'],int(step),*q,unresolved])

def number(v):return 'Unresolved' if v<=floor else f'{v:.2e}'

lines=['# Readout spectrum: 24 functions','',
       'All 88,344 saved checkpoints were analyzed (216 runs × 409 checkpoints). No new training.',
       '',r'Rates are $\rho_i=\sigma_i^2/\sigma_{\max}^2$ across the 197 readout modes, including output bias.',
       'The figures show the 10th–90th percentiles. CSV files also include the minimum and maximum.',
       '',f'Values at or below {floor:.3e} are numerically unresolved; plots place them on the bottom line.',
       '', 'Final checkpoint: ranges across all 24 functions and three learning rates within each initialization.',
       '', '| Initialization | Unresolved modes | 90th-percentile rate |',
       '|---|---:|---:|']
for init in [1,2,4]:
    rows=[r for r in records if r['initialization_multiplier']==init]
    u=[float(r['unresolved_fraction'][-1]) for r in rows]
    p=[float(r['quantiles'][-1,9]) for r in rows]
    lines.append(f'| {init}x | {min(u):.1%}–{max(u):.1%} | {number(min(p))}–{number(max(p))} |')
lines+=['','These are spectral percentiles, not uncertainty intervals or percentiles of error energy.',
        'They show whether slow readout directions persist. They do not alone measure how much residual error occupies those directions.',
        '', '[Overview](figures/spectrum_overview.png) · [All 24 function figures](figures/all_24_functions.pdf) · [All checkpoint percentiles](all_checkpoint_percentiles.csv)','']
(OUT/'results.md').write_text('\n'.join(lines))
cards='\n'.join(f'<a href="figures/{n}.png"><h2>{html.escape(n.replace("_"," "))}</h2><img loading="lazy" src="figures/{n}.png" alt="{html.escape(n)} spectral percentiles"></a>' for n in names)
(OUT/'index.html').write_text('''<!doctype html><meta charset="utf-8"><title>Readout Spectrum: 24 functions</title>
<style>body{font:16px system-ui;margin:32px auto;max-width:1400px;color:#222;padding:0 24px}h1{font-size:28px}h2{font-size:18px}img{width:100%}a{color:inherit}nav a{color:#2463a0;margin-right:20px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(430px,1fr));gap:24px}.grid a{text-decoration:none;border:1px solid #ddd;padding:16px;border-radius:8px}</style>
<h1>Readout Spectrum: 24 functions</h1><p>216 existing runs · 409 checkpoints each · 197 readout modes</p>
<p>Lines show spectral percentiles from 10th to 90th. Values below numerical resolution appear on the bottom line.</p>
<nav><a href="figures/all_24_functions.pdf">All figures (PDF)</a><a href="all_checkpoint_percentiles.csv">Every checkpoint (CSV)</a><a href="results.md">Numerical summary</a></nav>
<img src="figures/spectrum_overview.png" alt="Overview of 24 functions"><div class="grid">'''+cards+'</div>')
print('\n'.join(lines))
