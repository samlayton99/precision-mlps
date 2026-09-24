"""Analyze the matched Adam feature assay without selecting checkpoints or traces."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def feature(a, b, x):
    return np.column_stack((np.tanh(x[:, None]*a+b), np.ones(len(x))))


def envelope(ax, values, color, label=None, alpha=1., bins=600, linewidth=1.5):
    """Plot raw sampled values with full within-bin extrema, not smoothed loss."""
    edges = np.unique(np.linspace(0, len(values), min(bins, len(values))+1).astype(int))
    starts, stops = edges[:-1], edges[1:]
    center = (starts+stops-1)/2
    lo = np.array([np.min(values[a:b]) for a,b in zip(starts,stops)])
    hi = np.array([np.max(values[a:b]) for a,b in zip(starts,stops)])
    ax.fill_between(center/1000, lo, hi, color=color, alpha=.10*alpha, linewidth=0)
    ax.plot(starts/1000, np.asarray(values)[starts], color=color, alpha=alpha,
            label=label, lw=linewidth)
    ax.plot((len(values)-1)/1000, values[-1], '.', color=color, ms=3)


def setup(ax, horizon, ylabel='Relative output error'):
    ax.set_yscale('log')
    ax.set_xlim(0, horizon/1000)
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.set_xlabel('Readout updates (thousands)')
    ax.set_ylabel(ylabel)
    ax.grid(alpha=.15)
    ax.spines[['top','right']].set_visible(False)


def save(fig, path):
    fig.savefig(path, dpi=180, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def scalar(value):
    return float(value) if np.isfinite(value) else None


def analyze(input_path, manifest_path, run_path, output):
    output.mkdir(parents=True, exist_ok=True)
    data = np.load(input_path)
    manifest = json.loads(manifest_path.read_text())
    geometries = manifest['geometries']
    metadata = json.loads((run_path/'metadata.json').read_text())
    recipes = metadata['recipes']
    errors = np.load(run_path/'relative_error.npy', mmap_mode='r')
    weights = np.load(run_path/'state.npz')['w']
    checkpoints = np.load(run_path/'readout_checkpoints.npy', mmap_mode='r')
    steps = np.load(run_path/'checkpoint_steps.npy')
    horizon, G, R = len(errors)-1, len(geometries), len(recipes)
    assert errors.shape == (horizon+1,G,R)
    assert weights.shape == (G, data['features'].shape[-1], R)
    assert checkpoints.shape == (len(steps), G, R, weights.shape[1])
    assert steps[0] == 0 and steps[-1] == horizon
    np.testing.assert_allclose(checkpoints[-1], weights.transpose(0,2,1), rtol=0, atol=0)
    y = data['target']; norm = np.linalg.norm(y)
    validation = np.zeros((G,R)); evaluation = np.zeros((G,R)); direct = np.zeros((G,R))
    for gi in range(G):
        direct[gi] = np.linalg.norm(data['features'][gi]@weights[gi]-y[:,None],axis=0)/norm
        for name, destination in [('validation',validation), ('eval',evaluation)]:
            x, target = data[name+'_x'], data[name+'_target']
            phi = feature(data['a'][gi], data['b'][gi], x)
            destination[gi] = np.linalg.norm(phi@weights[gi]-target[:,None],axis=0)/np.linalg.norm(target)
    finite = np.isfinite(direct)&np.isfinite(errors[-1])
    np.testing.assert_array_equal(np.isfinite(direct),np.isfinite(errors[-1]))
    np.testing.assert_allclose(direct[finite],errors[-1][finite],rtol=0,atol=1e-12)
    selected = {}
    endpoints = []
    schedules = list(dict.fromkeys(r['schedule'] for r in recipes))
    for gi, geom in enumerate(geometries):
        for schedule in schedules:
            indices = [i for i,r in enumerate(recipes) if r['schedule']==schedule]
            scores = np.where(np.isfinite(validation[gi,indices]),validation[gi,indices],np.inf)
            if not np.any(np.isfinite(scores)):
                raise ValueError(f'All rates failed for geometry {gi}, {schedule}')
            selected[gi,schedule] = indices[int(np.argmin(scores))]
        for ri, rec in enumerate(recipes):
            trace = errors[:,gi,ri]
            crossings = {}
            for tol in [.1,.03,.01,.003,.001]:
                hits = np.flatnonzero(trace<=tol)
                crossings[str(tol)] = int(hits[0]) if len(hits) else None
            endpoints.append(dict(geometry_index=gi, **geom, recipe_index=ri, **rec,
                train_error=scalar(direct[gi,ri]), validation_error=scalar(validation[gi,ri]),
                eval_error=scalar(evaluation[gi,ri]),
                selected=ri==selected[gi,rec['schedule']], first_raw_crossings=crossings))
    summary = dict(horizon=horizon,selection='One fixed learning rate per geometry and schedule, chosen by final validation error; no checkpoint selection or pointwise best curve.',
        error_definition='norm(prediction-target)/norm(target)',
        max_direct_train_error_gap=float(np.max(np.abs(direct[finite]-errors[-1][finite]))),
        geometry_count=G,recipes=recipes,endpoints=endpoints)
    plt.rcParams.update({'font.size':11,'axes.titlesize':12,'axes.labelsize':11,
                         'legend.fontsize':9,'font.family':'DejaVu Sans'})
    uniform = sorted([i for i,g in enumerate(geometries) if g['family']=='uniform'],key=lambda i:geometries[i]['gamma'])
    palette = plt.get_cmap('viridis')(np.linspace(.05,.85,len(uniform)))
    fig, axes = plt.subplots(2,3,figsize=(16,8),layout='constrained')
    for row,schedule in enumerate(['cosine','constant']):
        ax = axes[row,0]
        rates = sorted(set(r['learning_rate'] for r in recipes if r['schedule']==schedule))
        t = np.linspace(0,horizon,501)
        for rate in rates:
            factor = .5*(1+np.cos(np.pi*t/horizon)) if schedule=='cosine' else np.ones_like(t)
            ax.plot(t/1000,rate*factor,label=f'{rate:g}')
        ax.set(xlabel='Readout updates (thousands)',ylabel='Learning rate',title=f'{schedule.title()}: full horizon',xlim=(0,horizon/1000))
        ax.set_yscale('symlog',linthresh=min(rates)/100)
        ax.set_ylim(0,max(rates)*1.15)
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.grid(alpha=.15); ax.legend(title='Initial rate',ncol=2)
        for col, mode in [(1,'shared'),(2,'selected')]:
            ax = axes[row,col]
            for gi,color in zip(uniform,palette):
                ri = selected[gi,schedule] if mode=='selected' else next(i for i,r in enumerate(recipes) if r['schedule']==schedule and np.isclose(r['learning_rate'],.002))
                g = geometries[gi]
                label = f"γ={g['gamma']:g} (λ={g['lambda_rms']:.3g})"
                if mode=='selected': label += f"; η₀={recipes[ri]['learning_rate']:g}"
                envelope(ax,errors[:,gi,ri],color,label)
            setup(ax,horizon)
            ax.set_title('Uniform centers: '+('shared η₀=0.002' if mode=='shared' else 'validation-selected rate'))
            ax.legend(loc='best')
    fig.suptitle('Matched frozen readouts: schedule and slope comparison\nLines are raw errors; shaded bands retain within-bin minima and maxima',fontsize=15)
    save(fig,output/'uniform_schedule_comparison.png')
    learned = [i for i,g in enumerate(geometries) if g['family']=='learned']
    snapshots = sorted(set(geometries[i]['snapshot_step'] for i in learned))
    colors = plt.get_cmap('plasma')(np.linspace(.1,.8,len(snapshots)))
    reference = next(i for i in uniform if np.isclose(geometries[i]['lambda_rms'],.25))
    reference_label = f"Uniform centers, γ={geometries[reference]['gamma']:.3g} (λ=.25)"
    fig,axes = plt.subplots(1,2,figsize=(13,5),layout='constrained')
    for stage,color in zip(snapshots,colors):
        indices = sorted([i for i in learned if geometries[i]['snapshot_step']==stage],key=lambda i:geometries[i]['seed'])
        traces = np.stack([errors[:,gi,selected[gi,'cosine']] for gi in indices],axis=1)
        for j in range(len(indices)):
            envelope(axes[0],traces[:,j],color,alpha=.23,linewidth=.7)
        envelope(axes[0],np.median(traces,axis=1),color,f'Frozen after {stage:,} joint updates',linewidth=2)
    ri = selected[reference,'cosine']
    envelope(axes[0],errors[:,reference,ri],'black',reference_label,linewidth=2)
    setup(axes[0],horizon); axes[0].set_title('Readout reset: cosine, selected rates'); axes[0].legend()
    seeds = sorted(set(geometries[i]['seed'] for i in learned))
    for seed in seeds:
        ix = [next(i for i in learned if geometries[i]['seed']==seed and geometries[i]['snapshot_step']==stage) for stage in snapshots]
        vals = [evaluation[i,selected[i,'cosine']] for i in ix]
        axes[1].plot(range(len(ix)),vals,'o-',label=f'Seed {seed}',alpha=.85)
    axes[1].axhline(evaluation[reference,ri],color='black',ls='--',label='Uniform λ=.25')
    axes[1].set_xticks(range(len(snapshots)),[f'{s:,}' for s in snapshots])
    axes[1].set(xlabel='Joint-training updates before freezing',ylabel='Final independent-grid relative error',yscale='log',title=f'Feature acquisition under identical {horizon:,}-update assays')
    axes[1].grid(alpha=.15); axes[1].legend(ncol=2)
    fig.suptitle('Did joint training acquire useful features?\nEvery assay resets readout and Adam state to zero; all five seeds retained',fontsize=15)
    save(fig,output/'learned_feature_acquisition.png')
    # Target harmonics are orthonormal on this midpoint grid. The remainder
    # retains any error created outside the three target directions.
    x=data['train_x']; q=np.sqrt(2)*np.sin(x[:,None]*np.array([2,6,14])[None,:]*np.pi)
    np.testing.assert_allclose(q.T@q/len(x),np.eye(3),rtol=0,atol=2e-14)
    selected_final = [i for i in learned if geometries[i]['snapshot_step']==max(snapshots)]
    components={}; max_decomposition_gap=0.
    for gi in selected_final+[reference]:
        ri=selected[gi,'cosine']
        residual=data['features'][gi]@checkpoints[:,gi,ri,:].T-y[:,None]
        energy=(q.T@residual)**2/(len(x)*norm**2)
        total=np.sum(residual**2,axis=0)/norm**2
        remainder=total-energy.sum(axis=0)
        assert np.min(remainder)>-1e-12
        components[gi]=np.vstack([energy,np.maximum(remainder,0)])
        max_decomposition_gap=max(max_decomposition_gap,float(np.max(abs(components[gi].sum(axis=0)-total))))
        np.testing.assert_allclose(np.sqrt(total),errors[steps,gi,ri],rtol=0,atol=1e-12)
    summary['harmonic_decomposition_max_gap']=max_decomposition_gap
    summary['target_harmonic_energy_fractions']=((q.T@y)**2/(len(x)*norm**2)).tolist()
    fig,axes=plt.subplots(1,3,figsize=(16,4.9),layout='constrained')
    for gi in selected_final:
        g=geometries[gi]; ri=selected[gi,'cosine']
        axes[0].scatter(g['source_error'],direct[gi,ri],s=45,label=f"Seed {g['seed']}")
    values=[geometries[i]['source_error'] for i in selected_final]+[direct[i,selected[i,'cosine']] for i in selected_final]
    lo,hi=min(values)*.7,max(values)*1.4
    axes[0].plot([lo,hi],[lo,hi],'k--',alpha=.5)
    axes[0].set(xscale='log',yscale='log',xlabel='Original joint-training endpoint error',ylabel='Reset-readout endpoint error',title='Final learned dictionary: training history',xlim=(lo,hi),ylim=(lo,hi))
    axes[0].set_xticks([.02,.05,.1],['0.02','0.05','0.10'])
    axes[0].set_yticks([.02,.05,.1],['0.02','0.05','0.10'])
    axes[0].minorticks_off()
    axes[0].grid(alpha=.15); axes[0].legend()
    comp_colors=['#3b528b','#21918c','#d28c16','#777777']
    labels=['Target: 1 cycle per unit x','Target: 3 cycles per unit x','Target: 7 cycles per unit x','Other output directions']
    for ax,comp,title in [(axes[1],np.mean(np.stack([components[i] for i in selected_final]),axis=0),'Final learned dictionaries: mean over seeds'),(axes[2],components[reference],reference_label)]:
        for ci,(color,label) in enumerate(zip(comp_colors,labels)):
            ax.plot(steps/1000,comp[ci],color=color,label=label,lw=1.7)
        ax.plot(steps/1000,comp.sum(axis=0),color='black',ls='--',lw=1.2,label='Total squared relative error')
        setup(ax,horizon,'Squared error / target squared norm')
        ax.set_title(title); ax.legend()
    fig.suptitle('Where does the remaining output error live?\nCosine, fixed validation-selected rates; harmonic curves use saved actual readout states',fontsize=15)
    save(fig,output/'residual_harmonic_diagnostics.png')
    fig,axes=plt.subplots(1,3,figsize=(14,4.3),layout='constrained')
    for gi,color in zip(uniform,palette):
        envelope(axes[0],errors[:,gi,selected[gi,'cosine']],color,
                 f"λ={geometries[gi]['lambda_rms']:g}")
    setup(axes[0],horizon)
    axes[0].set_title('A  Uniform centers: effect of slope')
    axes[0].legend(ncol=2,fontsize=8.5)
    for stage,color in zip(snapshots,colors):
        ix=[i for i in learned if geometries[i]['snapshot_step']==stage]
        values=np.stack([errors[:,i,selected[i,'cosine']] for i in ix],axis=1)
        envelope(axes[1],np.median(values,axis=1),color,
                 'Initial features' if stage==0 else f'After {stage//1000}k joint updates')
        edges=np.unique(np.r_[np.arange(0,horizon+1,250),horizon+1])
        lower=[values[a:b].min() for a,b in zip(edges[:-1],edges[1:])]
        upper=[values[a:b].max() for a,b in zip(edges[:-1],edges[1:])]
        axes[1].fill_between((edges[:-1]+edges[1:]-1)/2000,lower,upper,
                             color=color,alpha=.12,lw=0)
    envelope(axes[1],errors[:,reference,selected[reference,'cosine']],'black','Uniform λ=.25')
    setup(axes[1],horizon)
    axes[1].set_title('B  Acquired features: restart the readout')
    axes[1].legend(fontsize=8.5)
    for gi in selected_final:
        g=geometries[gi]
        axes[2].plot([0,1],[g['source_error'],direct[gi,selected[gi,'cosine']]],'o-',
                     lw=1.5,label=f"Seed {g['seed']}")
    axes[2].axhline(direct[reference,selected[reference,'cosine']],color='black',ls='--',
                    lw=1.2,label='Uniform λ=.25 readout')
    axes[2].set(yscale='log',ylabel='Relative output error',xlim=(-.2,1.2),
                title='C  Does a fresh readout remove the error?')
    axes[2].set_xticks([0,1],['Joint endpoint','Fresh readout\n200k updates'])
    axes[2].grid(alpha=.15);axes[2].spines[['top','right']].set_visible(False)
    axes[2].legend(fontsize=8.5,ncol=2)
    fig.suptitle('512 hidden units total · matched target and samples · full-horizon cosine readout schedules',fontsize=12)
    save(fig,output/'matched_adam_summary.png')
    np.savez_compressed(output/'harmonic_diagnostics.npz',steps=steps,geometry_indices=np.array(list(components)),components=np.stack(list(components.values())))
    (output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(output),max_direct_error_gap=summary['max_direct_train_error_gap'],geometry_count=G)))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['input','manifest','run','output']: p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args(); analyze(a.input,a.manifest,a.run,a.output)
