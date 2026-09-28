"""Audit Theorem 14's feedback assumptions on cached scalar trajectories.

All integrations are retrospective interpolants, not certified upper budgets.
No parameter archives or JAX dependency are needed. Writes evidence, not prose.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np

from . import population_accumulated_audit as prior

LEVELS = ('directional', 'split', 'signed', 'weighted', 'structural')
FIELDS = ('directional_curvature_rate_bound', 'geometry_rate', 'compensation_rate',
          'residual_relaxation_rate', 'log_force_rate', 'weighted_curvature_rate_bound',
          'structural_curvature_rate_bound', 'generated_rate', 'target_rate')


@lru_cache(maxsize=2)
def quadrature_nodes(order):
    return np.polynomial.legendre.leggauss(order)


def exp_budget_integral(t, rate, order=32):
    """Integrate exp(2*B) where rate is piecewise linear and B its integral.

    Gauss--Legendre integration evaluates the scalar interpolant. It is not
    quadrature certification of the unknown network path. Log-space summation
    avoids intermediate overflow; an infinite result means a vacuous bound.
    """
    t, rate = np.asarray(t, float), np.asarray(rate, float)
    if np.any(rate < 0):
        raise ValueError('Feedback upper rate must be nonnegative')
    B = prior.cumulative(t, rate)
    nodes, weights = quadrature_nodes(order)
    logH = np.full(len(t), -np.inf)
    for k, dt in enumerate(np.diff(t)):
        offset = (nodes + 1) * dt / 2
        partial = B[k] + rate[k] * offset + (rate[k+1]-rate[k]) * offset**2 / (2*dt)
        pivot = 2*max(partial)
        log_piece = math.log(dt/2) + pivot + math.log(np.dot(weights, np.exp(2*partial-pivot)))
        logH[k+1] = np.logaddexp(logH[k], log_piece)
    with np.errstate(over='ignore'):
        H = np.exp(logH)
    return B, H


def constant_budget_integral(t, rate):
    t = np.asarray(t, float)
    with np.errstate(over='ignore'):
        return np.expm1(2*rate*t)/(2*rate) if rate > 0 else t.copy()


def bounds(first, B, H, C):
    f0, y0, m4, width = (first[k] for k in ('F_norm', 'fine_norm', 'M4', 'width'))
    with np.errstate(over='ignore', invalid='ignore'):
        dissipation = f0**2 * np.asarray(H)
        travel4 = np.sqrt(np.asarray(C)*dissipation)
        moment = (m4**.25 + travel4)**4
        fine_floor = np.sqrt(np.maximum(0., y0*y0-2*dissipation)) / first['target_norm']
        capacity_floor = np.maximum(0., first['target_fine_norm']-prior.CAPACITY_CONSTANT*moment/width) / first['target_norm']
        force = np.exp(math.log(f0)+np.asarray(B))
    return dict(B=np.asarray(B), H=np.asarray(H), C=np.asarray(C), force_envelope=force,
                dissipation_envelope=dissipation, weighted_travel_envelope=travel4,
                M4_envelope=moment, energy_relative_floor=fine_floor,
                combined_relative_floor=np.maximum(fine_floor, capacity_floor))


def read_paths(sources):
    paths = defaultdict(list)
    for source in sources:
        with source.open(newline='') as stream:
            for raw in csv.DictReader(stream):
                if raw.get('role') != 'trajectory':
                    continue
                if raw['reinforcement_status'] != 'finite':
                    raise ValueError('Unresolved archived state: '+raw['source'])
                row = prior.normalize(raw, source.parent.name)
                for key in FIELDS:
                    row[key] = float(raw['reinforcement_'+key])
                    if not math.isfinite(row[key]):
                        raise ValueError('Nonfinite feedback: '+key)
                paths[tuple(row[k] for k in prior.IDENTITY)].append(row)
    return [sorted(group, key=lambda r: r['flow_time']) for group in paths.values()]


def audit(path):
    first, last = path[0], path[-1]
    t = np.array([r['flow_time'] for r in path])
    if t[0] != 0:
        raise ValueError('A fixed initial state is required')
    C = prior.cumulative(t, np.sqrt([r['I_F'] for r in path]))
    geometry, compensation = (np.array([r[k+'_rate'] for r in path]) for k in ('geometry', 'compensation'))
    rates = dict(directional=np.array([r['directional_curvature_rate_bound'] for r in path]),
                 split=np.maximum(geometry, 0)+np.maximum(compensation, 0),
                 signed=np.maximum(geometry+compensation, 0),
                 weighted=np.array([r['weighted_curvature_rate_bound'] for r in path]),
                 structural=np.array([r['structural_curvature_rate_bound'] for r in path]))
    if np.any(rates['signed'] > rates['split']+1e-12) or np.any(rates['split'] > rates['directional']+1e-12):
        raise ValueError('Feedback bound ordering failed')
    base = {k: first[k] for k in (*prior.IDENTITY, 'width', 'start', 'eta', 'h', 'scale', 'reference')}
    base.update(horizon=last['horizon'], flow_time=t[-1], snapshots=len(path),
                initial_force=first['F_norm'], initial_M4=first['M4'], initial_fine_norm=first['fine_norm'],
                initial_relative_error=first['relative_l2'], final_relative_error=last['relative_l2'],
                final_M4_ratio=last['M4']/first['M4'], force_ratio=last['F_norm']/first['F_norm'],
                clock=C[-1], clock_over_initial_rate=C[-1]/(t[-1]*math.sqrt(first['I_F'])))
    endpoints, states = [], []
    for level, rate in rates.items():
        B, H = exp_budget_integral(t, rate)
        value = bounds(first, B, H, C)
        B_check, H_check = exp_budget_integral(t, rate, order=64)
        finite = np.isfinite(H) & np.isfinite(H_check) & (H_check > 0)
        quadrature_relative = np.max(np.abs(H[finite]/H_check[finite]-1)) if finite.any() else None
        log_excess = np.log([r['F_norm']/first['F_norm'] for r in path])-B
        for i, row in enumerate(path):
            states.append(dict(base, level=level, horizon=row['horizon'], flow_time=t[i],
                rate=rate[i], observed_force=row['F_norm'], observed_M4=row['M4'],
                observed_relative_error=row['relative_l2'],
                **{k: v[i] for k,v in value.items()}))
        required = (first['fine_norm']**2-(.01*first['target_norm'])**2)/2
        spare = .5*(math.log(required)-math.log(value['dissipation_envelope'][-1])) if required > 0 and value['dissipation_envelope'][-1] > 0 else None
        endpoint = dict(base, level=level, budget=B[-1], initial_rate=rate[0],
            **{k: v[-1] for k,v in value.items() if k not in ('B','C')},
            final_M4_envelope_ratio=value['M4_envelope'][-1]/first['M4'],
            energy_floor_over_observed=value['energy_relative_floor'][-1]/last['relative_l2'],
            maximum_log_force_excess=float(max(log_excess)),
            maximum_output_floor_excess=max(value['combined_relative_floor'][i]-r['relative_l2'] for i,r in enumerate(path)),
            spare_feedback_budget_for_one_percent=spare,
            quadrature_32_vs_64_relative=quadrature_relative,
            budget_over_initial_rate=B[-1]/(rate[0]*t[-1]) if rate[0] > 0 else None,
            prefix_budget_over_initial_rate=max(B[1:]/(rate[0]*t[1:])) if rate[0] > 0 else None,
            relaxation_integral=prior.cumulative(t, [-r['residual_relaxation_rate'] for r in path])[-1])
        # Initial-state-only feedback allowances; factors are a sensitivity
        # family specified before this audit, not fitted from its outcomes.
        for factor in (1, 2, 4, 8):
            Hb = constant_budget_integral(t, factor*rate[0])
            assumed = bounds(first, factor*rate[0]*t, Hb, C)
            endpoint[f'factor{factor}_sampled_premise'] = bool(np.all(B <= factor*rate[0]*t+1e-12))
            endpoint[f'factor{factor}_energy_relative_floor'] = assumed['energy_relative_floor'][-1]
        endpoints.append(endpoint)
    return states, endpoints


def summarize(endpoints):
    groups = defaultdict(list)
    for row in endpoints:
        groups[(row['width'], row['start'], row['eta'], row['horizon'], row['arm'], row['level'])].append(row)
    metrics = ('budget', 'force_ratio', 'energy_relative_floor', 'combined_relative_floor',
               'energy_floor_over_observed', 'final_M4_envelope_ratio', 'final_M4_ratio',
               'maximum_log_force_excess', 'maximum_output_floor_excess',
               'spare_feedback_budget_for_one_percent', 'budget_over_initial_rate',
               'prefix_budget_over_initial_rate', 'quadrature_32_vs_64_relative')
    result = []
    for key, rows in sorted(groups.items()):
        item = dict(zip(('width','start','eta','horizon','arm','level'), key))
        item.update(trajectories=len(rows), targets=sorted({r['target'] for r in rows}),
                    energy_floor_above_one_percent=sum(r['energy_relative_floor'] > .01 for r in rows),
                    combined_floor_above_one_percent=sum(r['combined_relative_floor'] > .01 for r in rows),
                    nonfinite_H=sum(not math.isfinite(r['H']) for r in rows))
        item.update({k: prior.stats([r[k] for r in rows]) for k in metrics})
        for factor in (1,2,4,8):
            item[f'factor{factor}_sampled_premise_count'] = sum(r[f'factor{factor}_sampled_premise'] for r in rows)
            item[f'factor{factor}_energy_floor_count'] = sum(r[f'factor{factor}_energy_relative_floor'] > .01 for r in rows)
        result.append(item)
    return result


def plots(endpoints, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    original = [r for r in endpoints if r['arm']=='original' and r['width']==705]
    targets = sorted({r['target'] for r in original})
    fig, axes = plt.subplots(1,3,figsize=(14,9),sharey=True,constrained_layout=True)
    colors = {'directional':'#167a81','split':'#ad6b25','signed':'#735090'}
    for i,target in enumerate(targets):
        for level in colors:
            group = sorted([r for r in original if r['target']==target and r['level']==level],key=lambda r:r['seed'])
            for j,row in enumerate(group):
                y=i+(j-.5)*.16
                for ax,key in zip(axes,('budget','energy_relative_floor','budget_over_initial_rate')):
                    if row[key] is not None:
                        ax.scatter(row[key],y,c=colors[level],s=22,marker={'directional':'o','split':'x','signed':'+'}[level],label=level if i==j==0 else None)
                if level=='directional':
                    axes[1].scatter(row['final_relative_error'],y,s=42,facecolors='none',edgecolors='black')
    axes[0].set_yticks(range(len(targets)),targets); axes[0].invert_yaxis()
    for ax,label in zip(axes,('Accumulated feedback B','Energy error floor; observed error is open','B(T) / (initial rate × T)')):
        ax.set(xscale='log',xlabel=label)
    axes[0].set_xscale('symlog',linthresh=1e-5)
    axes[2].set_xscale('symlog',linthresh=.01)
    axes[1].axvline(.01,color='black',ls='--',lw=1)
    axes[2].axvline(1,color='black',ls='--',lw=1)
    axes[0].legend(fontsize=8)
    fig.suptitle('All 23 targets, width 705, two seeds: 20k further GD updates\nRetrospective effective-flow budgets; no GD certificate',fontsize=12)
    fig.savefig(output/'all_target_feedback.png',dpi=180); plt.close(fig)

    fig,axes=plt.subplots(1,3,figsize=(14,4.8),constrained_layout=True)
    for j,width in enumerate((177,705,1409)):
        group=[r for r in endpoints if r['arm']=='original' and r['width']==width and r['level']=='directional']
        pos=j+np.linspace(-.14,.14,len(group))
        for ax,key in zip(axes,('energy_floor_over_observed','final_M4_envelope_ratio','spare_feedback_budget_for_one_percent')):
            vals=np.array([r[key] if r[key] is not None else np.nan for r in group])
            axes_value=np.where(np.isfinite(vals),vals,np.nan)
            ax.scatter(pos,axes_value,s=17,alpha=.6)
            ax.set_xticks(range(3),('177\nage 600k','705\nage 20k','1409\nage 20k'))
    axes[0].set(ylabel='Energy floor / actual error',ylim=(-.02,1.06))
    axes[1].set(yscale='log',ylabel='Fourth-moment upper bound / initial moment')
    axes[2].set(ylabel='Additional feedback budget before 1% floor is lost')
    axes[2].axhline(0,color='black',ls='--',lw=1)
    fig.suptitle('Directional feedback without favorable signs: what the theorem retains after 20k\nDifferent restart ages are not a controlled width comparison',fontsize=12)
    fig.savefig(output/'feedback_usefulness.png',dpi=180); plt.close(fig)

    long=[r for r in endpoints if r['panel']=='wide_N512_long' and r['arm']=='repaired']
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),constrained_layout=True)
    labels=sorted({r['target'] for r in long})
    for level,color in colors.items():
        group=sorted([r for r in long if r['level']==level],key=lambda r:r['target'])
        for ax,key in zip(axes,('budget','energy_relative_floor')):
            ax.plot(range(len(group)),[r[key] for r in group],'.',label=level,color=color)
            ax.set_xticks(range(len(labels)),labels,rotation=20)
    axes[0].set_yscale('symlog',linthresh=.001)
    axes[0].set(ylabel='Accumulated feedback B at 100k')
    axes[1].set(yscale='log',ylabel='Relative energy error floor at 100k')
    axes[1].axhline(.01,color='black',ls='--',lw=1)
    axes[0].legend()
    fig.suptitle('Six-target width-705 long continuations: a finite-interval premise test')
    fig.savefig(output/'long_feedback.png',dpi=180); plt.close(fig)


def run(args):
    rows,endpoints=[],[]
    for path in read_paths(args.source):
        states,summary=audit(path); rows.extend(states); endpoints.extend(summary)
    args.output.mkdir(parents=True,exist_ok=False)
    prior.write_csv(args.output/'states.csv',rows)
    prior.write_csv(args.output/'trajectories.csv',endpoints)
    facts=dict(scope='Theorem 14 at GD scalar histories; interpolated budgets, not continuous or discrete certification',
               sources={str(p):prior.digest(p) for p in args.source},helper_sha256=prior.digest(__file__),
               branches=len(endpoints)//len(LEVELS),states=len(rows)//len(LEVELS),
               levels=LEVELS,initial_rate_factors=[1,2,4,8],groups=summarize(endpoints))
    (args.output/'facts.json').write_text(json.dumps(prior.clean(facts),indent=2,allow_nan=False)+'\n')
    plots(endpoints,args.output)
    print(json.dumps({k:facts[k] for k in ('branches','states','levels')}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,action='append',required=True)
    parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args())
