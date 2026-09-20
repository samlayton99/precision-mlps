"""Explicit readout-update policies for the requested freezing interventions.

All geometry steps are ordinary raw-(a,b) GD. A readout can receive GD, a
numerical coefficient solve, or no update. Evaluation refits are separate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
import yaml
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.expD26_freeze_and_readout_spectrum import freeze as previous

HERE = Path(__file__).resolve().parent
RESULTS = ROOT / 'results/checkpoint_D_optimizers/expD27_readout_information'
FIELDS = previous.FIELDS
POLICIES = ('gd', 'solve', 'frozen')


def config():
    return yaml.safe_load((HERE / 'config.yaml').read_text())


def state_at(trajectory, step):
    return {key: trajectory[key][step].copy() for key in FIELDS}


def add_metrics(trajectory):
    a = trajectory['a']
    trajectory['steps'] = np.arange(len(a))
    trajectory['mean_gamma'] = np.mean(np.abs(a), axis=1)
    trajectory['mean_gamma_change'] = np.mean(np.abs(np.abs(a)-np.abs(a[0])), axis=1)
    trajectory['mean_gamma_travel'] = np.r_[0., np.cumsum(np.mean(np.abs(np.diff(np.abs(a), axis=0)), axis=1))]
    trajectory['readout_norm'] = np.linalg.norm(trajectory['v'], axis=1)
    return trajectory


def train(initial, x, y, updates, rate, policy, rcond=1e-13):
    """Record solved states, including zero, for policy='solve'.

    The SVD is detached: geometry receives the partial gradient with the
    current solved readout fixed. No readout GD occurs for that policy.
    """
    if policy not in POLICIES:
        raise ValueError(f'Unknown readout policy: {policy}')
    params = {key: torch.nn.Parameter(torch.tensor(initial[key], dtype=torch.float64),
                  requires_grad=(key != 'v' or policy == 'gd')) for key in FIELDS}
    opt = torch.optim.SGD([v for v in params.values() if v.requires_grad], lr=rate)
    tx, ty = torch.tensor(x, dtype=torch.float64), torch.tensor(y, dtype=torch.float64)
    records = {key: [] for key in FIELDS}
    records.update(train_loss=[], solve_rank=[], solve_stationarity=[])
    metadata = dict(policy=policy, requested_updates=int(updates), failed=False,
                    failure_step=None, failure_reason=None, solve_count=0, solve_seconds=0.)
    for step in range(updates+1):
        if not all(torch.isfinite(value).all().item() for value in params.values()):
            metadata.update(failed=True, failure_step=step, failure_reason='nonfinite parameters')
            break
        rank, stationarity = -1, np.nan
        if policy == 'solve':
            state = {key: value.detach().numpy().copy() for key, value in params.items()}
            start = time.perf_counter()
            try:
                coefficients, info = previous.solve_readout(state, x, y, rcond)
            except np.linalg.LinAlgError as exc:
                metadata.update(failed=True, failure_step=step, failure_reason=str(exc))
                break
            metadata['solve_seconds'] += time.perf_counter()-start
            if not np.all(np.isfinite(coefficients)):
                metadata.update(failed=True, failure_step=step, failure_reason='nonfinite solved readout')
                break
            with torch.no_grad():
                params['v'].copy_(torch.from_numpy(coefficients))
            rank, stationarity = info['rank'], info['stationarity']
            metadata['solve_count'] += 1
        opt.zero_grad(set_to_none=True)
        hidden = torch.tanh(tx[:, None]*params['a']+params['b'])
        residual = hidden @ params['v'][:-1]+params['v'][-1]-ty
        loss = .5*torch.mean(residual.square())
        if not torch.isfinite(loss).item():
            metadata.update(failed=True, failure_step=step, failure_reason='nonfinite loss')
            break
        for key in FIELDS:
            records[key].append(params[key].detach().numpy().copy())
        records['train_loss'].append(loss.item())
        records['solve_rank'].append(rank)
        records['solve_stationarity'].append(stationarity)
        if step < updates:
            loss.backward()
            if not all(p.grad is None or torch.isfinite(p.grad).all().item() for p in params.values()):
                metadata.update(failed=True, failure_step=step, failure_reason='nonfinite gradient')
                break
            opt.step()
    if not records['train_loss']:
        raise FloatingPointError(f'No finite initial state: {metadata}')
    trajectory = add_metrics({key: np.asarray(values) for key, values in records.items()})
    if policy == 'frozen':
        np.testing.assert_array_equal(trajectory['v'], np.broadcast_to(initial['v'], trajectory['v'].shape))
    metadata['completed_updates'] = len(trajectory['steps'])-1
    return trajectory, metadata


def branch_from(reference, freeze_step, x, y, cfg):
    initial = state_at(reference, freeze_step)
    continuation, metadata = train(initial, x, y, cfg['frozen_steps'], cfg['learning_rate'], 'frozen')
    keys = (*FIELDS, 'train_loss', 'solve_rank', 'solve_stationarity')
    branch = add_metrics({key: np.concatenate((reference[key][:freeze_step], continuation[key])) for key in keys})
    # The marker itself is the reference state, including its completed solve.
    branch['solve_rank'][freeze_step] = reference['solve_rank'][freeze_step]
    branch['solve_stationarity'][freeze_step] = reference['solve_stationarity'][freeze_step]
    branch['freeze_step'] = np.array(freeze_step)
    for key in FIELDS:
        np.testing.assert_array_equal(branch[key][:freeze_step+1], reference[key][:freeze_step+1])
    np.testing.assert_array_equal(branch['v'][freeze_step:], np.broadcast_to(initial['v'], branch['v'][freeze_step:].shape))
    if len(continuation['steps']) > 1:
        if freeze_step+1 < len(reference['steps']):
            # Both arms see exactly the same parameters at the marker. Their
            # first geometry update must agree, even though the reference
            # subsequently updates/solves its readout and the branch does not.
            for key in ('a', 'b'):
                np.testing.assert_array_equal(branch[key][freeze_step+1], reference[key][freeze_step+1])
        gradients = previous.numpy_gradients(initial, x, y)
        # Full NumPy reevaluation exposes cancellation sensitivity in large-v
        # fits. Check update algebra separately with the identical forward
        # values used by training, without using autodiff for the formula.
        with torch.no_grad():
            tx = torch.tensor(x, dtype=torch.float64)
            av = torch.tensor(initial['a'], dtype=torch.float64)
            bv = torch.tensor(initial['b'], dtype=torch.float64)
            vv = torch.tensor(initial['v'], dtype=torch.float64)
            hidden = torch.tanh(tx[:,None]*av+bv)
            residual = hidden@vv[:-1]+vv[-1]-torch.tensor(y, dtype=torch.float64)
        hn, rn = hidden.numpy(), residual.numpy()
        common = rn[:,None]*(1-hn**2)*initial['v'][:-1]
        same_forward = dict(a=np.mean(x[:,None]*common,axis=0), b=np.mean(common,axis=0))
        roundoff_weight = np.abs(rn[:,None]*initial['v'][:-1])*(1+hn**2)
        metadata['first_update_numpy_discrepancy'] = 0.
        metadata['first_update_same_forward_discrepancy'] = 0.
        for key in ('a', 'b'):
            actual = branch[key][freeze_step+1]
            expected = initial[key]-cfg['learning_rate']*same_forward[key]
            discrepancy = float(np.max(np.abs(actual-expected)))
            backend_difference = float(np.max(np.abs(actual-(initial[key]-cfg['learning_rate']*gradients[key]))))
            metadata['first_update_numpy_discrepancy'] = max(metadata['first_update_numpy_discrepancy'],backend_difference)
            metadata['first_update_same_forward_discrepancy'] = max(metadata['first_update_same_forward_discrepancy'],discrepancy)
            absolute_terms = roundoff_weight*(np.abs(x[:,None]) if key=='a' else 1.)
            # Conservative componentwise floating-point accumulation bound;
            # includes the derivative subtraction 1-h^2 and the SGD update.
            tolerance = 64*np.finfo(float).eps*(len(x)+8)*(np.abs(initial[key])+cfg['learning_rate']*np.mean(absolute_terms,axis=0)+1)
            if np.any(np.abs(actual-expected)>tolerance):
                raise AssertionError(f'Frozen {key} update disagrees with same-forward analytic derivative')
    if metadata['failure_step'] is not None:
        metadata['failure_step'] += freeze_step
    return branch, metadata


def solved_qi_reference(target, cfg):
    initial = previous.original.initial_state(f"gamma_{cfg['qi_gamma']}", cfg)
    x = previous.original.midpoint_grid(cfg['n_train'])
    y = previous.original.matched.target_values(target, x, cfg)
    coefficients, info = previous.solve_readout(initial, x, y, cfg['readout_rcond'])
    initial['v'] = coefficients
    return initial, info


def transfer_state(solved_qi, cfg):
    reset = cfg['transfer_reset']
    if reset == 'gamma_1':
        initial = previous.original.initial_state('gamma_1', cfg)
    elif reset == 'xavier':
        initial = previous.original.initial_state('xavier', cfg)
    else:
        raise ValueError('The poor-geometry reset needs to be specified before running.')
    initial['v'] = solved_qi['v'].copy()
    return initial


def noisy_qi_state(cfg):
    initial = previous.original.initial_state(f"gamma_{cfg['qi_gamma']}", cfg)
    if cfg['noise_kind'] != 'multiplicative_scale' or cfg['noise_level'] is None:
        raise ValueError('The geometry-noise definition needs to be specified before running.')
    rng = np.random.default_rng([cfg['seed'], cfg['resolution'], 27])
    factor = 1+float(cfg['noise_level'])*rng.normal(size=len(initial['a']))
    if np.any(factor <= 0):
        raise ValueError('Specified scale noise produced nonpositive scales; no clipping is applied.')
    old_centers = -initial['b']/initial['a']
    initial['a'] *= factor
    initial['b'] *= factor
    np.testing.assert_allclose(-initial['b']/initial['a'], old_centers, rtol=1e-14, atol=1e-14)
    return initial, factor


def validate_design(cfg):
    if cfg['transfer_reset'] is None or cfg['noise_kind'] is None or cfg['noise_level'] is None:
        raise ValueError('Geometry reset and noise choices are pending; no experiment started.')


VARIANTS = {
    'solve_then_freeze': '1. Solve the readout after every geometry step, then freeze',
    'qi_transfer': '2. QI-solved readout; reset gamma to 1 and freeze from step zero',
    'qi_zero': '3. QI geometry, zero readout; joint GD, then freeze',
    'noisy_qi_zero': '4. Noisy QI geometry, zero readout; joint GD, then freeze',
}


def run_cases(variant, target, cfg):
    x = previous.original.midpoint_grid(cfg['n_train'])
    y = previous.original.matched.target_values(target, x, cfg)
    metadata = dict(variant=variant, target=target, trajectories={})
    if variant == 'qi_transfer':
        teacher, teacher_info = solved_qi_reference(target, cfg)
        metadata['teacher_info'] = {key: float(value) for key, value in teacher_info.items()}
        initial = transfer_state(teacher, cfg)
        trajectory, info = train(initial, x, y, cfg['frozen_steps'], cfg['learning_rate'], 'frozen')
        trajectory['freeze_step'] = np.array(0)
        metadata['trajectories']['frozen'] = info
        metadata['teacher_state'] = {key: value.tolist() for key, value in teacher.items()}
        return {'frozen': trajectory}, metadata
    if variant == 'solve_then_freeze':
        initial = previous.original.initial_state('xavier', cfg)
        policy = 'solve'
    elif variant == 'qi_zero':
        initial = previous.original.initial_state(f"gamma_{cfg['qi_gamma']}", cfg)
        policy = 'gd'
    elif variant == 'noisy_qi_zero':
        initial, factors = noisy_qi_state(cfg)
        metadata['noise_factors'] = factors.tolist()
        policy = 'gd'
    else:
        raise ValueError(variant)
    total = max(cfg['freeze_steps'])+cfg['frozen_steps']
    reference, info = train(initial, x, y, total, cfg['learning_rate'], policy, cfg['readout_rcond'])
    cases = {'reference': reference}
    metadata['trajectories']['reference'] = info
    for step in cfg['freeze_steps']:
        if step >= len(reference['steps']):
            metadata['trajectories'][f'freeze_{step}'] = dict(failed=True, failure_step=step,
                                                           failure_reason='reference failed before branch')
            continue
        branch, info = branch_from(reference, step, x, y, cfg)
        cases[f'freeze_{step}'] = branch
        metadata['trajectories'][f'freeze_{step}'] = info
    return cases, metadata


def diagnose(cases, target, cfg, metadata):
    """Read-only completed-state evaluation; sparse solves never enter GD."""
    x, xx = (previous.original.midpoint_grid(cfg[key]) for key in ('n_train', 'n_eval'))
    y, yy = (previous.original.matched.target_values(target, nodes, cfg) for nodes in (x, xx))
    y_norm, yy_norm = np.linalg.norm(y), np.linalg.norm(yy)
    branch_steps = np.unique(np.concatenate([previous.refit_steps(step, cfg['frozen_steps'])
                                            for step in cfg['freeze_steps']]))
    for name, case in cases.items():
        before = {key: case[key].copy() for key in FIELDS}
        count = len(case['steps'])
        if name == 'reference':
            chosen = branch_steps[branch_steps < count]
        else:
            freeze_step = int(case['freeze_step'])
            chosen = previous.refit_steps(freeze_step, cfg['frozen_steps'])
            chosen = chosen[chosen < count]
        chosen = np.unique(np.r_[chosen, count-1]).astype(int)
        case['refit_steps'] = chosen
        case['relative_l2'] = np.empty(count)
        for field in ('refit_relative_l2', 'refit_train_relative_l2', 'refit_rank',
                      'refit_readout_norm', 'refit_stationarity'):
            case[field] = np.empty(len(chosen))
        case['refit_v'] = np.empty((len(chosen), case['v'].shape[1]))
        lookup = {int(step): i for i, step in enumerate(chosen)}
        freeze_step = int(case.get('freeze_step', -1))
        for step in range(count):
            copy_prefix = name.startswith('freeze_') and step <= freeze_step
            if copy_prefix:
                case['relative_l2'][step] = cases['reference']['relative_l2'][step]
            else:
                Ae = previous.original.design(xx, case['a'][step], case['b'][step])
                case['relative_l2'][step] = np.linalg.norm(Ae@case['v'][step]-yy)/yy_norm
            if step in lookup:
                index = lookup[step]
                if copy_prefix:
                    refindex = int(np.searchsorted(cases['reference']['refit_steps'], step))
                    for key in ('refit_relative_l2', 'refit_train_relative_l2', 'refit_rank',
                                'refit_readout_norm', 'refit_stationarity', 'refit_v'):
                        case[key][index] = cases['reference'][key][refindex]
                else:
                    state = state_at(case, step)
                    coefficients, info = previous.solve_readout(state, x, y, cfg['readout_rcond'])
                    case['refit_relative_l2'][index] = np.linalg.norm(Ae@coefficients-yy)/yy_norm
                    A = previous.original.design(x, state['a'], state['b'])
                    case['refit_train_relative_l2'][index] = np.linalg.norm(A@coefficients-y)/y_norm
                    case['refit_v'][index] = coefficients
                    for field in ('rank', 'readout_norm', 'stationarity'):
                        case['refit_'+field][index] = info[field]
        for key in FIELDS:
            np.testing.assert_array_equal(before[key], case[key])
        if not np.all(np.isfinite(case['relative_l2'])):
            raise FloatingPointError('Nonfinite independent evaluation: '+name)
        print(f"  {name}: error={case['relative_l2'][-1]:.6g}, mean gamma={case['mean_gamma'][-1]:.6g}, "
              f"refit={case['refit_relative_l2'][-1]:.6g}", flush=True)
    if 'teacher_state' in metadata:
        teacher = {key: np.array(value) for key, value in metadata['teacher_state'].items()}
        Ae = previous.original.design(xx, teacher['a'], teacher['b'])
        metadata['teacher_eval_error'] = float(np.linalg.norm(Ae@teacher['v']-yy)/yy_norm)


def save(cases, metadata, cfg):
    folder = RESULTS/'data'
    folder.mkdir(parents=True, exist_ok=True)
    path = folder/f"{metadata['variant']}__{metadata['target']}.npz"
    payload = dict(config_json=json.dumps(cfg), metadata_json=json.dumps(metadata))
    payload.update({f'{name}__{field}': value for name, case in cases.items() for field, value in case.items()})
    temporary = path.with_suffix('.tmp.npz')
    np.savez_compressed(temporary, **payload)
    temporary.replace(path)


def load(variant, target):
    path = RESULTS/'data'/f'{variant}__{target}.npz'
    with np.load(path, allow_pickle=False) as data:
        cfg = json.loads(str(data['config_json']))
        metadata = json.loads(str(data['metadata_json']))
        cases = {}
        for key in data.files:
            if '__' in key:
                name, field = key.split('__', 1)
                cases.setdefault(name, {})[field] = data[key]
    return cases, metadata, cfg


def axis_range(axes, values, *, logarithmic=False):
    from matplotlib.ticker import FuncFormatter, MaxNLocator, NullFormatter
    values = np.asarray(values)
    lo, hi = float(np.min(values)), float(np.max(values))
    if logarithmic:
        log_span = np.log10(max(hi,1e-16)/max(lo,1e-16))
        pad_factor = 10**max(.065,.055*log_span)
        lower, upper = max(lo/pad_factor, 1e-16), max(hi*pad_factor, 2e-16)
        for ax in axes:
            ax.set_yscale('log')
            ax.set_ylim(lower, upper)
            if upper/lower < 5:
                ticks = MaxNLocator(nbins=5).tick_values(lower, upper)
                ax.set_yticks(ticks[(ticks >= lower)&(ticks <= upper)])
                ax.yaxis.set_major_formatter(FuncFormatter(lambda v, pos: f'{v:.5g}'))
                ax.yaxis.set_minor_formatter(NullFormatter())
    else:
        pad = max((hi-lo)*.14, abs((hi+lo)/2)*1e-7, 1e-12)
        for ax in axes:
            ax.set_ylim(lo-pad, hi+pad)
            ax.ticklabel_format(axis='y', style='plain', useOffset=False)


def plot_branches(variant, target):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    cases, metadata, cfg = load(variant, target)
    reference = cases['reference']
    colors = plt.colormaps['viridis']([.15,.37,.6,.83])
    title = VARIANTS[variant]
    reference_label = 'Continue geometry GD + readout solves' if variant == 'solve_then_freeze' else 'Continue joint GD'
    fig, axes = plt.subplots(2,4,figsize=(17.5,8.5),dpi=170,sharey='row')
    for column, (freeze_step, color) in enumerate(zip(cfg['freeze_steps'], colors)):
        name = f'freeze_{freeze_step}'
        endpoint = freeze_step+cfg['frozen_steps']
        for row, field in enumerate(('relative_l2', 'mean_gamma')):
            ax = axes[row,column]
            use = reference['steps'] <= endpoint
            ax.plot(reference['steps'][use], reference[field][use], '--', color='.45', lw=2)
            if name in cases:
                case = cases[name]
                ax.plot(case['steps'], case[field], color=color, lw=1.7)
            else:
                ax.text(.5,.5,'Reference failed before freeze',transform=ax.transAxes,ha='center')
            failure = metadata['trajectories'].get(name, {})
            if failure.get('failed') and name in cases:
                ax.plot(cases[name]['steps'][-1], cases[name][field][-1], 'x', color='red', ms=7)
            ax.axvline(freeze_step,color='.2',ls=':',lw=1.2)
            ax.set_xlim(0,max(cfg['freeze_steps'])+cfg['frozen_steps'])
            ax.set_xticks([0,150,300,450,650])
            ax.grid(alpha=.2)
            if row == 0:
                ax.set_title(f'Freeze after {freeze_step} steps',fontsize=12,pad=12)
            else:
                ax.set_xlabel('Total geometry-GD steps')
    errors = np.concatenate([c['relative_l2'] for c in cases.values()])
    gammas = np.concatenate([c['mean_gamma'] for c in cases.values()])
    log_gamma = bool(gammas.max()/max(gammas.min(),1e-30) > 30)
    axis_range(axes[0], errors, logarithmic=True)
    axis_range(axes[1], gammas, logarithmic=log_gamma)
    axes[0,0].set_ylabel(r'Relative $L_2$ error (log scale)',fontsize=12)
    axes[1,0].set_ylabel(r'Mean scale $\overline{\gamma}=\frac{1}{m}\sum_j|a_j|$'+('\n(log scale)' if log_gamma else ''),fontsize=12)
    fig.suptitle(previous.LABELS[target]+': '+title,fontsize=17,y=.985)
    fig.legend([Line2D([],[],color='#258b8e',lw=2),Line2D([],[],color='.45',ls='--',lw=2),Line2D([],[],color='.2',ls=':')],
               ['Frozen-readout branch',reference_label,'Readout freezes'],loc='upper center',bbox_to_anchor=(.5,.94),ncol=3,frameon=False,fontsize=11)
    start_note = 'Step 0 already has a solved readout; later solves stop at the marker.' if variant == 'solve_then_freeze' else 'Zero initial readout; joint GD until the marker. The first geometry update is exactly zero.'
    if variant == 'noisy_qi_zero':
        start_note += f" Initial scales: independent {100*cfg['noise_level']:g}% normal multiplicative noise; centers preserved."
    fig.text(.5,.066,start_note+'\n'+f"Geometry GD rate {cfg['learning_rate']}; {cfg['frozen_steps']} updates after freezing, including output bias in the frozen readout. "
             f"All {reference['a'].shape[1]} neurons enter mean gamma.\n"+
             f"Errors: {cfg['n_eval']:,} independent midpoints on [−1,1]; training: {cfg['n_train']:,}. Axes are shared within each row; no refit enters the displayed branch after its marker.",ha='center',fontsize=9.5,linespacing=1.55)
    fig.subplots_adjust(left=.08,right=.985,top=.82,bottom=.18,hspace=.27,wspace=.15)
    folder = RESULTS/'figures'/variant
    folder.mkdir(parents=True,exist_ok=True)
    fig.savefig(folder/f'{target}.png')
    plt.close(fig)


def plot_refits(variant, cfg):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import LogLocator, LogFormatterMathtext
    colors = plt.colormaps['viridis']([.15,.37,.6,.83])
    fig, axes = plt.subplots(4,4,figsize=(17.5,13.8),dpi=170,sharex=True,sharey='row')
    for row, target in enumerate(cfg['targets']):
        cases, metadata, _ = load(variant,target)
        ref = cases['reference']
        clean_qi_error = None
        if variant == 'noisy_qi_zero':
            clean_cases, _, _ = load('qi_zero', target)
            clean_qi_error = clean_cases['reference']['refit_relative_l2'][0]
        for column,(freeze_step,color) in enumerate(zip(cfg['freeze_steps'],colors)):
            ax = axes[row,column]
            use = ref['refit_steps'] <= freeze_step+cfg['frozen_steps']
            ax.plot(ref['refit_steps'][use],ref['refit_relative_l2'][use],'--',color='.45',marker='.',ms=3,lw=1.6)
            name = f'freeze_{freeze_step}'
            if name in cases:
                case=cases[name]
                ax.plot(case['refit_steps'],case['refit_relative_l2'],color=color,marker='o',ms=2.5,lw=1.4)
            ax.axhline(ref['refit_relative_l2'][0],color='.7',ls=':',lw=1)
            if clean_qi_error is not None:
                ax.axhline(clean_qi_error,color='.2',ls='-.',lw=1.1)
            ax.axvline(freeze_step,color='.3',ls=':',lw=1)
            ax.set_xlim(0,max(cfg['freeze_steps'])+cfg['frozen_steps'])
            ax.set_xticks([0,150,300,450,650]);ax.grid(alpha=.2)
            if row==0:ax.set_title(f'Freeze at step {freeze_step}',pad=12)
            if column==0:ax.set_ylabel(previous.LABELS[target]+'\nRefitted relative $L_2$')
            if row==3:ax.set_xlabel('Total geometry-GD steps')
        vals=np.concatenate([case['refit_relative_l2'] for case in cases.values()])
        axis_range(axes[row],vals,logarithmic=True)
        if variant in ('qi_zero','noisy_qi_zero'):
            # Keep numerical-floor jitter in proportion; for the noisy case
            # show the full distance back to the clean-QI reference.
            for ax in axes[row]:
                ax.set_ylim(1e-16,1e-12 if variant=='qi_zero' else 1e-8)
                ax.yaxis.set_major_locator(LogLocator(base=10,numticks=5))
                ax.yaxis.set_major_formatter(LogFormatterMathtext())
    subtitle = 'Refitted geometry remains near numerical precision' if variant=='qi_zero' else 'Does the geometry support a better refitted approximation?'
    fig.suptitle(VARIANTS[variant]+'\n'+subtitle,fontsize=18,y=.985)
    handles=[Line2D([],[],color='#258b8e',marker='o',ms=3),Line2D([],[],color='.45',ls='--'),Line2D([],[],color='.7',ls=':')]
    labels=['Frozen branch, LS evaluation','Continued reference, LS evaluation','Initial geometry, LS evaluation']
    if variant=='noisy_qi_zero':
        handles.append(Line2D([],[],color='.2',ls='-.'));labels.append('Clean QI geometry, LS evaluation')
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.93),ncol=len(handles),frameon=False,fontsize=10)
    fig.subplots_adjust(top=.85,bottom=.13,left=.095,right=.985,hspace=.27,wspace=.15)
    fig.text(.5,.035,'Each marker uses an independent diagnostic readout solve on a saved state; it does not alter that trajectory.\n'
             'The training readout follows the stated variant. Relative SVD cutoff 10⁻¹³; common independent evaluation grid.\n'
             'Small error changes with large solved coefficients may reflect numerical cancellation. Each function row shares its axes.',ha='center',fontsize=10,linespacing=1.45)
    folder=RESULTS/'figures'/variant;folder.mkdir(parents=True,exist_ok=True)
    fig.savefig(folder/'refitted_geometry.png');plt.close(fig)


def plot_transfer(cfg):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    fig, axes=plt.subplots(2,4,figsize=(17.5,8.5),dpi=170,sharex=True,sharey='row')
    all_errors,all_gammas=[],[]
    for col,target in enumerate(cfg['targets']):
        cases,metadata,_=load('qi_transfer',target);case=cases['frozen']
        axes[0,col].plot(case['steps'],case['relative_l2'],color='#258b8e',lw=1.8)
        axes[0,col].plot(case['refit_steps'],case['refit_relative_l2'],'--',color='#7352a2',marker='.',ms=3)
        axes[1,col].plot(case['steps'],case['mean_gamma'],color='#258b8e',lw=1.8)
        axes[0,col].set_title(previous.LABELS[target],pad=12)
        for row in range(2):
            axes[row,col].set_xlim(0,cfg['frozen_steps']);axes[row,col].grid(alpha=.2)
            axes[row,col].axvline(0,color='.3',ls=':')
        axes[1,col].set_xlabel('Geometry-GD steps')
        all_errors.extend([case['relative_l2'],case['refit_relative_l2']]);all_gammas.append(case['mean_gamma'])
    axis_range(axes[0],np.concatenate(all_errors),logarithmic=True)
    axis_range(axes[1],np.concatenate(all_gammas),logarithmic=False)
    axes[0,0].set_ylabel('Relative $L_2$ error (log scale)',fontsize=12)
    axes[1,0].set_ylabel(r'Mean scale $\overline{\gamma}=\frac{1}{m}\sum_j|a_j|$',fontsize=12)
    fig.suptitle('2. Keep the QI-solved readout, reset gamma to 1, and train geometry',fontsize=20,y=.98)
    fig.legend([Line2D([],[],color='#258b8e',lw=2),Line2D([],[],color='#7352a2',ls='--')],
               ['Actual fixed-readout error / mean gamma','Evaluation-only readout refit'],loc='upper center',bbox_to_anchor=(.5,.93),ncol=2,frameon=False,fontsize=12)
    fig.subplots_adjust(left=.08,right=.985,top=.82,bottom=.17,hspace=.28,wspace=.15)
    fig.text(.5,.055,f"Readout solved at QI gamma={cfg['qi_gamma']:g}, then copied unchanged to gamma=1 at the same centers. "
             f"No training readout updates or solves during these {cfg['frozen_steps']} steps.\n"
             f"Raw slopes and biases train at {cfg['learning_rate']}; centers may move after initialization. "
             f"{cfg['n_train']:,} training / {cfg['n_eval']:,} independent evaluation samples on [−1,1].\n"
             'Dashed errors use separate diagnostic readouts only; they do not modify the fixed training coefficients. Shared axes within each row.',ha='center',fontsize=10,linespacing=1.6)
    folder=RESULTS/'figures'/'qi_transfer';folder.mkdir(parents=True,exist_ok=True)
    fig.savefig(folder/'all_targets.png');plt.close(fig)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--variants',nargs='+',choices=list(VARIANTS),default=list(VARIANTS))
    parser.add_argument('--plot-only',action='store_true')
    parser.add_argument('--resume',action='store_true',help='Reuse completed data only when its configuration matches exactly.')
    args=parser.parse_args();cfg=config();validate_design(cfg)
    torch.set_num_threads(cfg['threads']);torch.set_default_dtype(torch.float64)
    with threadpool_limits(limits=cfg['threads']):
        for variant in args.variants:
            for target in cfg['targets']:
                cached=RESULTS/'data'/f'{variant}__{target}.npz'
                reuse=args.plot_only or (args.resume and cached.exists())
                if reuse:
                    _,_,saved_cfg=load(variant,target)
                    if saved_cfg != cfg:raise ValueError('Cached configuration mismatch: '+str(cached))
                if not reuse:
                    started=time.perf_counter()
                    print(f'{variant}/{target}: starting training',flush=True)
                    cases,metadata=run_cases(variant,target,cfg)
                    metadata['training_seconds']=time.perf_counter()-started
                    started=time.perf_counter();diagnose(cases,target,cfg,metadata)
                    metadata['diagnostic_seconds']=time.perf_counter()-started
                    save(cases,metadata,cfg)
                if variant != 'qi_transfer':plot_branches(variant,target)
            if variant == 'qi_transfer':plot_transfer(cfg)
            else:plot_refits(variant,cfg)
            print('Completed '+variant,flush=True)


if __name__ == '__main__':
    main()
