"""Audit an aggregate feedback/travel bootstrap using cached scalar CSVs.

The fitted response constant is retrospective. Prefix extrapolation is a
separate diagnostic, and no sampled integral is an interval certificate.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.integrate import quad

from . import population_accumulated_audit as prior
from . import population_feedback_budget as feedback


def integrated_travel(t, speed):
    """Integrate travel exactly for the piecewise-linear speed interpolant."""
    t, speed = np.asarray(t, float), np.asarray(speed, float)
    travel = prior.cumulative(t, speed)
    dt = np.diff(t)
    pieces = travel[:-1]*dt + dt**2*(2*speed[:-1]+speed[1:])/6
    return travel, np.r_[0., np.cumsum(pieces)]


def critical_response(time, f0, d0, i0):
    """r=1 bootstrap threshold with C(t)=2*sqrt(I0)*t."""
    if min(time, f0, i0) <= 0 or d0 < 0:
        raise ValueError("Invalid initial quantities")
    clock_rate = 2*math.sqrt(i0)
    def integrand(t):
        h = math.expm1(2*d0*t)/(2*d0) if d0 else t
        return math.sqrt(clock_rate*t*h)
    # Huge baseline amplification already makes this sufficient test useless.
    if 2*d0*time > 600:
        return 0.
    integral = quad(integrand, 0., time, epsabs=1e-12, epsrel=1e-10)[0]
    return 1/(math.e*f0*integral)


def audit(t, f, rate, concentration):
    t, f, rate, concentration = map(lambda x: np.asarray(x, float),
                                   (t, f, rate, concentration))
    if t[0] != 0 or np.any(f <= 0) or np.any(rate < 0) or np.any(concentration <= 0):
        raise ValueError("Require initial time zero and finite positive path quantities")
    b = prior.cumulative(t, rate)
    travel, travel_integral = integrated_travel(t, concentration**.25*f)
    excess = np.maximum(0., b-rate[0]*t)
    required = excess[1:]/travel_integral[1:]
    k = float(max(required))
    critical = critical_response(t[-1], f[0], rate[0], concentration[0])
    clock = prior.cumulative(t, np.sqrt(concentration))
    clock_ratio = float(max(clock[1:]/(2*math.sqrt(concentration[0])*t[1:])))
    gain = k/critical if critical else math.inf
    prefix = t[1:] <= min(40., t[-1])+1e-10
    prefix_k = float(max(required[prefix]))
    result = dict(time=t[-1], snapshots=len(t), f0=f[0], d0=rate[0], I0=concentration[0],
                  feedback=b[-1], travel=travel[-1], required_K=k, critical_K=critical,
                  loop_gain=gain, concentration_ratio=clock_ratio,
                  sampled_bootstrap_pass=bool(gain < 1 and clock_ratio <= 1),
                  prefix_time=min(40., t[-1]), prefix_K=prefix_k,
                  force_ratio=f[-1]/f[0])
    for factor in (1, 2, 4):
        result[f'prefix_K_factor{factor}_covers'] = bool(k <= factor*prefix_k+1e-12)
    result['prefix_K_multiplier_required'] = k/prefix_k if prefix_k else (0. if k == 0 else math.inf)
    return result


def dense_paths(source):
    groups = defaultdict(list)
    with source.open(newline='') as stream:
        for raw in csv.DictReader(stream):
            groups[(raw['target'], raw['kind'], float(raw['dt']))].append(raw)
    for (target, kind, dt), rows in groups.items():
        rows.sort(key=lambda r: float(r['time']))
        values = {k: [float(r[k]) for r in rows] for k in ('time', 'f', 'rate', 'I')}
        yield dict(dataset=source.parent.name, target=target, kind=kind, dt=dt,
                   width=705, start=20000, seed=30, arm='original',
                   **audit(values['time'], values['f'], values['rate'], values['I']))


def plot(rows, output):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    original = [r for r in rows if r['kind'] == 'archived_gd' and r['width'] == 705
                and r['start'] == 20000 and r['arm'] == 'original' and r['time'] == 40]
    targets = sorted({r['target'] for r in original})
    for r in original:
        axes[0].scatter(targets.index(r['target']), r['loop_gain'], color='#266c9c', s=25)
    axes[0].set_xticks(range(len(targets)), targets, rotation=75, ha='right', fontsize=8)
    axes[0].set_title('23 targets, two seeds: 20k further GD updates')
    dense = [r for r in rows if r['dataset'] == 'feedback_flow_100k'
             and r['kind'] == 'effective' and r['dt'] == .01]
    for j, r in enumerate(dense):
        axes[1].scatter(j, r['loop_gain'], color='#266c9c', s=40)
    axes[1].set_xticks(range(len(dense)), [r['target'] for r in dense], rotation=40, ha='right')
    axes[1].set_title('Six targets: 100k update-equivalent effective flow')
    for ax in axes:
        ax.axhline(1, color='#b23c32', linestyle='--', label='Sufficient-test threshold')
        ax.set_yscale('symlog', linthresh=.001)
        ax.set_ylabel('Retrospective response / allowed response')
        ax.grid(alpha=.15)
    axes[0].set_ylim(-.00006, 1.6)
    axes[1].set_ylim(-.00006, max(1.6, 1.6*max(r['loop_gain'] for r in dense)))
    axes[1].legend(fontsize=8)
    fig.savefig(output/'feedback_loop.png', dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', action='append', type=Path, default=[])
    parser.add_argument('--dense-source', action='append', type=Path, default=[])
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    for path in feedback.read_paths(args.source):
        first = path[0]
        # Natural reference branches only: this tests persistence, not the
        # previously audited response to explicit large parameter injections.
        if first['arm'] != 'original':
            continue
        values = audit([r['flow_time'] for r in path], [r['F_norm'] for r in path],
                       [r['directional_curvature_rate_bound'] for r in path],
                       [r['I_F'] for r in path])
        rows.append(dict({k: first[k] for k in (*prior.IDENTITY, 'width', 'start')},
                         kind='archived_gd', **values))
    for source in args.dense_source:
        rows.extend(dense_paths(source))
    prior.write_csv(args.output/'branches.csv', rows)
    groups = defaultdict(list)
    for row in rows:
        groups[(row['dataset'], row['kind'], row.get('dt', 0), row['width'], row['start'], row['time'])].append(row)
    facts = []
    for key, group in sorted(groups.items()):
        facts.append(dict(group=key, count=len(group), passes=sum(r['sampled_bootstrap_pass'] for r in group),
                          max_gain=max(r['loop_gain'] for r in group),
                          max_concentration_ratio=max(r['concentration_ratio'] for r in group),
                          prefix_factor_coverage={str(factor): sum(r[f'prefix_K_factor{factor}_covers'] for r in group)
                                                  for factor in (1, 2, 4)}))
    (args.output/'facts.json').write_text(json.dumps(prior.clean(dict(
        scope='Retrospective sampled response audit; no continuous-time or GD certificate',
        groups=facts)), indent=2)+'\n')
    plot(rows, args.output)
    print(json.dumps(prior.clean(facts), indent=2))


if __name__ == '__main__':
    main()
