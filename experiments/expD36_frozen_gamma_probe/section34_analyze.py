"""Select one joint recipe per optimizer and preserve full-horizon evidence."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def relative_error(p, x, y, width):
    if not np.all(np.isfinite(p)):
        return float('inf')
    a, b, c = p[:-1].reshape(3, width)
    return float(np.linalg.norm(np.tanh(x[:, None]*a+b)@c+p[-1]-y)/np.linalg.norm(y))


def json_safe(value):
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def binned_traces(array, recipe, bins=1800):
    """Retain seed trajectories and per-bin extrema, including brief spikes."""
    edges = np.unique(np.linspace(0, len(array), min(bins, len(array))+1, dtype=int))
    steps = edges[1:]-1
    typical, low, high = [], [], []
    for start, stop in zip(edges[:-1], edges[1:]):
        block = np.asarray(array[start:stop, :, recipe])
        typical.append(np.median(block, axis=0))
        low.append(np.min(block, axis=0))
        high.append(np.max(block, axis=0))
    # Endpoint markers have exact values, rather than trailing-bin averages.
    return dict(steps=steps, median=np.asarray(typical), low=np.asarray(low),
                high=np.asarray(high), endpoint=np.asarray(array[-1, :, recipe]))


def analyze(base, runs, output, export=None):
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((base/'manifest.json').read_text())
    data = np.load(base/'input.npz')
    width, h = manifest['width'], manifest['spacing']
    rows, sources, selected, traces, learned = [], {}, {}, {}, []
    for optimizer, run in runs:
        key = str(run.resolve())
        meta = json.loads((run/'metadata.json').read_text())
        state = np.load(run/'state.npz')
        p = state['p']; horizon = int(state['count'])
        errors = np.load(run/'relative_error.npy', mmap_mode='r')
        rms = np.load(run/'slope_rms.npy', mmap_mode='r')
        snapshots = np.load(run/'parameter_checkpoints.npy', mmap_mode='r')
        steps = np.load(run/'checkpoint_steps.npy')
        assert meta['width'] == width and errors.shape[0] == horizon+1
        assert steps[-1] == horizon and p.shape[1] == len(meta['recipes'])
        np.testing.assert_array_equal(snapshots[-1], p)
        sources[key] = (meta, p, errors, rms, snapshots, steps)
        for ri, recipe in enumerate(meta['recipes']):
            vals = [relative_error(q, data['validation_x'], data['validation_target'], width) for q in p[:, ri]]
            train = [relative_error(q, data['train_x'], data['target'], width) for q in p[:, ri]]
            finite = np.isfinite(train)
            np.testing.assert_allclose(np.asarray(train)[finite], errors[-1, :, ri][finite], atol=3e-11, rtol=1e-10)
            rows.append(dict(optimizer=optimizer, run=key, recipe_index=ri, **recipe,
                             horizon=horizon, validation_errors=vals,
                             median_validation_error=float(np.median(vals)), train_errors=train,
                             all_seeds_finite=bool(np.all(np.isfinite(vals)))))
    horizons = {r['horizon'] for r in rows}
    if len(horizons) != 1:
        raise ValueError('All comparison candidates must have the same training horizon')
    horizon = horizons.pop()
    for optimizer in dict.fromkeys(o for o, _ in runs):
        candidates = [r for r in rows if r['optimizer'] == optimizer and r['all_seeds_finite']]
        if not candidates:
            raise ValueError(f'No recipe has finite endpoints for all seeds: {optimizer}')
        chosen = min(candidates, key=lambda r: r['median_validation_error'])
        selected[optimizer] = dict(chosen)
        meta, params, errors, rms, snapshots, steps = sources[chosen['run']]
        ri = chosen['recipe_index']
        chosen['selected'] = True
        selected[optimizer]['eval_errors'] = [relative_error(q, data['eval_x'], data['eval_target'], width) for q in params[:, ri]]
        for metric, array, scale in [('error', errors, 1.), ('rms', rms, h)]:
            values = binned_traces(array, ri)
            for field, arr in values.items():
                traces[f'{optimizer}_{metric}_{field}'] = arr if field == 'steps' else arr*scale
        slopes = np.abs(snapshots[:, :, ri, :width])*h
        traces[f'{optimizer}_checkpoint_steps'] = steps
        traces[f'{optimizer}_lambda_q99'] = np.quantile(slopes, .99, axis=-1)
        traces[f'{optimizer}_lambda_max'] = np.max(slopes, axis=-1)
        traces[f'{optimizer}_lambda_rms_checkpoints'] = np.sqrt(np.mean(slopes**2, axis=-1))
        np.testing.assert_allclose(traces[f'{optimizer}_lambda_rms_checkpoints'], rms[steps, :, ri]*h, atol=2e-15, rtol=2e-13)
        for step in sorted({0, min(200000, horizon), min(600000, horizon), horizon}):
            if step not in steps:
                raise ValueError(f'Missing required snapshot {step}')
            si = int(np.flatnonzero(steps == step)[0])
            for seed, p in enumerate(snapshots[si, :, ri]):
                for factor in ([1., 4., 16.] if step == horizon else [1.]):
                    a, b, c = p[:-1].reshape(3, width)
                    learned.append((a*factor, b*factor, dict(name=f'{optimizer}_seed{seed}_step{step}_scale{factor:g}',
                        optimizer=optimizer, family='learned' if factor == 1 else 'slope_intervention',
                        seed=seed, snapshot_step=step, slope_multiplier=factor, gamma=None,
                        lambda_rms=float(np.sqrt(np.mean((factor*a*h)**2))),
                        source_error=relative_error(p, data['train_x'], data['target'], width),
                        joint_schedule=chosen['schedule'], joint_learning_rate=chosen['learning_rate'])))
        # The extension trigger compares equal windows near the start/end of the final fifth.
        window = max(1, horizon//100)
        first = int(.8*horizon)
        early = np.median(errors[max(0,first-window):first, :, ri], axis=0)
        late = np.median(errors[-window:, :, ri], axis=0)
        selected[optimizer]['final_fifth_error_decrease'] = float(1-np.median(late)/np.median(early))
        qi = int(np.argmin(abs(steps-first)))
        q99 = traces[f'{optimizer}_lambda_q99']
        selected[optimizer]['final_fifth_q99_increase'] = float(np.median(q99[-1])/np.median(q99[qi])-1)
    # Preserve per-schedule selection to assess constant-rate late progress even if cosine wins.
    schedule_best = {}
    for optimizer in selected:
        for schedule in sorted({r['schedule'] for r in rows if r['optimizer'] == optimizer}):
            candidates = [r for r in rows if r['optimizer'] == optimizer and r['schedule'] == schedule and r['all_seeds_finite']]
            if not candidates:
                continue
            chosen = min(candidates, key=lambda r:r['median_validation_error'])
            _, _, errors, _, snapshots, steps = sources[chosen['run']]
            ri = chosen['recipe_index']; window = max(1, horizon//100); first = int(.8*horizon)
            early = np.median(errors[max(0,first-window):first, :, ri], axis=0)
            late = np.median(errors[-window:, :, ri], axis=0)
            qi = int(np.argmin(abs(steps-first)))
            q99 = np.quantile(np.abs(snapshots[[qi,-1], :, ri, :width])*h, .99, axis=-1)
            schedule_best[f'{optimizer}_{schedule}'] = dict(chosen,
                final_fifth_error_decrease=float(1-np.median(late)/np.median(early)),
                final_fifth_q99_increase=float(np.median(q99[-1])/np.median(q99[0])-1))
    for key, best in schedule_best.items():
        rates = [r['learning_rate'] for r in rows if r['optimizer'] == best['optimizer'] and r['schedule'] == best['schedule']]
        best['rate_at_search_boundary'] = bool(best['learning_rate'] in (min(rates), max(rates)))
    summary = dict(horizon=horizon, width=width, spacing=h, candidates=rows, selected=selected,
                   best_by_schedule=schedule_best,
                   selection='One recipe per optimizer minimizing median final 4096-midpoint validation error across all seeds; requires finite endpoints in every seed.',
                   evaluation='8192-midpoint resolution check, not an untouched test set.',
                   display='Within-bin median per seed with exact min/max; full training horizon retained.')
    (output/'summary.json').write_text(json.dumps(json_safe(summary), indent=2, allow_nan=False)+'\n')
    np.savez_compressed(output/'selected_traces.npz', **traces)
    plot_joint(traces, selected, horizon, output)
    plot_candidates(rows, sources, output, horizon)
    if export is not None:
        export.mkdir(parents=True, exist_ok=True)
        a = np.array([r[0] for r in learned]); b = np.array([r[1] for r in learned])
        features = np.concatenate((np.tanh(data['train_x'][None,:,None]*a[:,None,:]+b[:,None,:]), np.ones((len(a),len(data['train_x']),1))), axis=-1)
        arrays = {k:data[k] for k in ['target','train_x','validation_x','validation_target','eval_x','eval_target']}
        np.savez_compressed(export/'input.npz', **arrays, a=a, b=b, features=features)
        copied = dict(manifest, geometries=[r[2] for r in learned], joint_recipe_selection=summary['selection'])
        copied['input_sha256'] = hashlib.sha256((export/'input.npz').read_bytes()).hexdigest()
        (export/'manifest.json').write_text(json.dumps(copied, indent=2)+'\n')
    print(json.dumps(json_safe(dict(selected=selected, best_by_schedule=schedule_best)), indent=2))


def style():
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'legend.frameon':False,'pdf.fonttype':42,'ps.fonttype':42})


def plot_joint(traces, selected, horizon, output):
    style(); fig, axes = plt.subplots(1,2,figsize=(10,3.7),layout='constrained')
    colors = {'adam':'#0072B2','gd':'#D55E00'}
    for optimizer in selected:
        color = colors.get(optimizer,'#009E73'); label = optimizer.upper() if optimizer=='gd' else optimizer.title()
        t = traces[f'{optimizer}_error_steps']/1e6
        y = traces[f'{optimizer}_error_median']
        axes[0].plot(t,np.median(y,axis=1),color=color,label=label)
        axes[0].fill_between(t,np.min(traces[f'{optimizer}_error_low'],axis=1),np.max(traces[f'{optimizer}_error_high'],axis=1),color=color,alpha=.13,lw=0)
        for name, ls in [('rms','-'),('q99','--')]:
            if name == 'rms':
                ts=traces[f'{optimizer}_rms_steps'];ys=traces[f'{optimizer}_rms_median']
            else:
                ts=traces[f'{optimizer}_checkpoint_steps'];ys=traces[f'{optimizer}_lambda_q99']
            axes[1].plot(ts/1e6,np.median(ys,axis=1),color=color,ls=ls,label=f'{label}, '+('RMS' if name=='rms' else '99th percentile'))
            axes[1].fill_between(ts/1e6,np.min(ys,axis=1),np.max(ys,axis=1),color=color,alpha=.09,lw=0)
    axes[1].axhline(.25,color='#444444',ls=':',lw=1,label='Uniform reference λ = 0.25')
    axes[0].set_ylabel('Relative output L2 error');axes[1].set_ylabel('Slope × reference spacing, λ')
    for ax in axes:
        ax.set_yscale('log');ax.set_xlim(0,horizon/1e6);ax.set_xlabel('Joint-training updates (millions)');ax.grid(alpha=.15);ax.legend(fontsize=8)
    for suffix in ['png','pdf']:fig.savefig(output/f'joint_error_and_slopes.{suffix}',dpi=220)
    plt.close(fig)


def plot_candidates(rows,sources,output,horizon):
    style(); optimizers=list(dict.fromkeys(r['optimizer'] for r in rows))
    fig,axes=plt.subplots(len(optimizers),2,figsize=(12,3.4*len(optimizers)),squeeze=False,layout='constrained')
    for oi,optimizer in enumerate(optimizers):
        for si,schedule in enumerate(['constant','cosine']):
            ax=axes[oi,si]
            candidates=[r for r in rows if r['optimizer']==optimizer and r['schedule']==schedule]
            for row,color in zip(candidates,plt.cm.viridis(np.linspace(.05,.9,len(candidates)))):
                errors=sources[row['run']][2];values=binned_traces(errors,row['recipe_index'],bins=800)
                ax.plot(values['steps']/1e6,np.median(values['median'],axis=1),color=color,label=f"η₀ = {row['learning_rate']:g}")
                for seed in range(errors.shape[1]):
                    ax.plot(values['steps']/1e6,values['median'][:,seed],color=color,alpha=.15,lw=.6)
            ax.set_title(f'{optimizer.upper()} · {schedule}');ax.set_yscale('log');ax.set_xlim(0,horizon/1e6)
            diverged = sum(not r['all_seeds_finite'] for r in candidates)
            if diverged:
                ax.text(.98,.98,f'{diverged} recipes have nonfinite endpoints',transform=ax.transAxes,ha='right',va='top',fontsize=8,bbox=dict(facecolor='white',alpha=.8,edgecolor='none'))
            ax.set_xlabel('Updates (millions)');ax.set_ylabel('Relative output L2 error');ax.legend(fontsize=7,ncol=2);ax.grid(alpha=.15)
    for suffix in ['png','pdf']:fig.savefig(output/f'all_joint_recipes.{suffix}',dpi=180)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',type=Path,required=True);parser.add_argument('--joint-run',action='append',required=True,help='optimizer=directory; may repeat optimizer for rate-expansion batches')
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--export-learned',type=Path)
    args=parser.parse_args();runs=[(s.split('=',1)[0].lower(),Path(s.split('=',1)[1])) for s in args.joint_run]
    with threadpool_limits(limits=2):analyze(args.base,runs,args.output,args.export_learned)


if __name__=='__main__':main()
