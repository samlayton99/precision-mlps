"""Actual fp64 GD on center samples, with separate spectral diagnostics.

Analytic backward passes use PyTorch and are checked against autograd. No
readout solves, momentum, clipping, or acceptance tests enter training.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.linalg import svd
import torch
from threadpoolctl import threadpool_limits
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.expD24_gd_residual_spectrum.run import uniform_geometry
from experiments.expD24_gd_residual_spectrum.first_steps_refit import target_values

HERE = Path(__file__).resolve().parent
RESULTS = ROOT / 'results/checkpoint_D_optimizers/expD30_center_sampled_gd'
SCHEDULES = ('baseline', 'large_constant', 'warmup_cosine')
MODES = ('fixed', 'joint')
LABELS = ('Sine', 'Mixed sine', 'Runge', 'Gaussian envelope')
COLORS = ('#440154', '#21918c', '#e0ba24')


def setup(cfg):
    centers, h, _ = uniform_geometry(cfg['resolution'], cfg['halo'])
    x = centers[(centers >= cfg['domain'][0]) & (centers <= cfg['domain'][1])].copy()
    assert len(x) == cfg['resolution'] + 1
    y = np.column_stack([target_values(t, x, cfg) for t in cfg['targets']])
    a = np.full(len(centers), cfg['gamma'])
    b = -a * centers
    return x, y, centers, h, a, b


def feature(x, a, b):
    return np.column_stack((np.tanh(x[:, None] * a + b), np.ones(len(x))))


def spectral_data(cfg, x, y, a, b):
    A = feature(x, a, b) / np.sqrt(len(x))
    U, s, Vh = svd(A, full_matrices=False)
    alpha = U.T @ (y / np.sqrt(len(x)))
    assert np.all(s > cfg['readout_rcond'] * s[0]), 'Center interpolation is not numerically full row rank.'
    coef = Vh.T @ (alpha / s[:, None])
    return dict(singular_values=s, alpha=alpha, weakest_left_vectors=U[:, -2:], initial_ls_coefficients=coef,
                initial_ls_train_loss=.5 * np.mean((feature(x, a, b) @ coef-y)**2, axis=0),
                peak_rate=np.array(cfg['spectral_factor'] / s[0]**2),
                stability_limit=np.array(2 / s[0]**2))


def learning_rates(cfg, peak, name):
    steps = cfg['steps']
    if name == 'baseline':
        return np.full(steps, cfg['base_rate'])
    if name == 'large_constant':
        return np.full(steps, peak)
    if name != 'warmup_cosine':
        raise ValueError(name)
    w = min(cfg['warmup_steps'], steps-2)
    rates = np.empty(steps)
    rates[:w] = np.linspace(cfg['base_rate'], peak, w)
    phase = np.linspace(0, np.pi, steps-w)
    floor = cfg['final_rate_fraction'] * peak
    rates[w:] = floor + .5 * (peak-floor) * (1+np.cos(phase))
    return rates


@torch.jit.script
def fixed_chunk(H: torch.Tensor, y: torch.Tensor, v: torch.Tensor,
                rates: torch.Tensor):
    losses = torch.empty((len(rates), y.shape[1]), dtype=y.dtype)
    for k in range(len(rates)):
        residual = H @ v-y
        losses[k] = .5 * torch.mean(residual * residual, dim=0)
        v = v - rates[k] * (H.t() @ residual) / y.shape[0]
    return v, losses


@torch.jit.script
def joint_chunk(x: torch.Tensor, y: torch.Tensor, a: torch.Tensor,
                b: torch.Tensor, v: torch.Tensor, rates: torch.Tensor):
    # Target is batch dimension: each target is an independent network.
    losses = torch.empty((len(rates), y.shape[0]), dtype=y.dtype)
    xx = x.unsqueeze(0).unsqueeze(-1)
    for k in range(len(rates)):
        hidden = torch.tanh(xx * a.unsqueeze(1) + b.unsqueeze(1))
        residual = torch.bmm(hidden, v[:, :-1].unsqueeze(-1)).squeeze(-1) + v[:, -1:] - y
        losses[k] = .5 * torch.mean(residual * residual, dim=1)
        common = residual.unsqueeze(-1) * (1-hidden*hidden) * v[:, :-1].unsqueeze(1)
        ga = torch.mean(common * xx, dim=1)
        gb = torch.mean(common, dim=1)
        gc = torch.mean(hidden * residual.unsqueeze(-1), dim=1)
        gd = torch.mean(residual, dim=1, keepdim=True)
        a = a-rates[k] * ga
        b = b-rates[k] * gb
        v = v-rates[k] * torch.cat((gc, gd), dim=1)
    return a, b, v, losses


def snapshot_steps(cfg):
    steps = cfg['steps']
    return np.unique(np.r_[np.arange(min(steps, 12)+1),
        np.geomspace(1, steps, cfg['snapshot_count']).astype(int),
        np.arange(0, steps+1, 10000), steps]).astype(int)


def predict_torch(x, a, b, v):
    return torch.bmm(torch.tanh(x[None, :, None]*a[:, None, :]+b[:, None, :]),
                     v[:, :-1, None]).squeeze(-1)+v[:, -1:]


def train(cfg, mode, name, spectral, data_dir):
    x, y, centers, h, initial_a, initial_b = setup(cfg)
    rate = learning_rates(cfg, float(spectral['peak_rate']), name)
    n_target = y.shape[1]
    tx, ty = torch.tensor(x), torch.tensor(y.T.copy())
    a = torch.tensor(np.tile(initial_a, (n_target, 1)))
    b = torch.tensor(np.tile(initial_b, (n_target, 1)))
    v = torch.zeros((n_target, len(initial_a)+1), dtype=torch.float64)
    H = torch.cat((torch.tanh(tx[:, None]*a[0]+b[0]), torch.ones((len(x), 1), dtype=torch.float64)), dim=1)
    snapshots = snapshot_steps(cfg)
    lo, hi = cfg['domain']
    xx = lo+(np.arange(cfg['n_eval'])+.5)*(hi-lo)/cfg['n_eval']
    yy = np.column_stack([target_values(t, xx, cfg) for t in cfg['targets']])
    txx, tyy = torch.tensor(xx), torch.tensor(yy.T.copy())
    losses = np.empty((cfg['steps']+1, n_target))
    records = {key: [] for key in ('eval_relative', 'mean_gamma', 'mean_gamma_displacement',
               'mean_center_displacement', 'readout_norm', 'max_gamma_displacement')}
    start = time.perf_counter()
    reported = -1
    previous = 0
    for step in snapshots:
        if step > previous:
            rr = torch.tensor(rate[previous:step])
            if mode == 'fixed':
                vv, ll = fixed_chunk(H, ty.t(), v.t().contiguous(), rr)
                v = vv.t().contiguous()
            elif mode == 'joint':
                a, b, v, ll = joint_chunk(tx, ty, a, b, v, rr)
            else:
                raise ValueError(mode)
            losses[previous:step] = ll.numpy()
        residual = predict_torch(tx, a, b, v)-ty
        losses[step] = (.5*torch.mean(residual**2, dim=1)).numpy()
        if not np.all(np.isfinite(losses[previous:step+1])):
            raise FloatingPointError(f'Nonfinite loss in {mode}/{name} at or before {step}')
        er = torch.linalg.vector_norm(predict_torch(txx, a, b, v)-tyy, dim=1) / torch.linalg.vector_norm(tyy, dim=1)
        records['eval_relative'].append(er.numpy())
        gammas = np.abs(a.numpy())
        center_change = np.abs(-b.numpy()/a.numpy()-centers)
        records['mean_gamma'].append(gammas.mean(axis=1))
        records['mean_gamma_displacement'].append(np.abs(gammas-cfg['gamma']).mean(axis=1))
        records['max_gamma_displacement'].append(np.abs(gammas-cfg['gamma']).max(axis=1))
        records['mean_center_displacement'].append(center_change.mean(axis=1))
        records['readout_norm'].append(torch.linalg.vector_norm(v, dim=1).numpy())
        if step//20000 != reported or step == cfg['steps']:
            relative = np.sqrt(2*losses[step]/np.mean(y*y, axis=0))
            print(f'{mode}/{name} step {step}: train relative {relative}; elapsed {time.perf_counter()-start:.1f}s', flush=True)
            reported = step//20000
        previous = step
    if mode == 'fixed':
        np.testing.assert_array_equal(a.numpy(), np.tile(initial_a, (n_target, 1)))
        np.testing.assert_array_equal(b.numpy(), np.tile(initial_b, (n_target, 1)))
    final_refit_train, final_refit_eval, final_refit_norm, final_singular_values = [], [], [], []
    for t in range(n_target):
        Phi = feature(x, a[t].numpy(), b[t].numpy())
        U, s, Vh = svd(Phi/np.sqrt(len(x)), full_matrices=False)
        keep = s > cfg['readout_rcond']*s[0]
        coef = Vh[keep].T @ ((U[:, keep].T @ (y[:, t]/np.sqrt(len(x))))/s[keep])
        final_refit_train.append(.5*np.mean((Phi@coef-y[:, t])**2))
        final_refit_eval.append(np.linalg.norm(feature(xx, a[t].numpy(), b[t].numpy())@coef-yy[:, t])/np.linalg.norm(yy[:, t]))
        final_refit_norm.append(np.linalg.norm(coef))
        final_singular_values.append(s)
    out = dict(config_json=np.array(json.dumps(cfg)), mode=np.array(mode), schedule=np.array(name),
        x=x, y=y, centers=centers, h=np.array(h), rates=rate, train_loss=losses,
        snapshot_steps=snapshots, **{k: np.array(vv) for k, vv in records.items()},
        final_a=a.numpy(), final_b=b.numpy(), final_v=v.numpy(),
        final_refit_train_loss=np.array(final_refit_train), final_refit_eval_relative=np.array(final_refit_eval),
        final_refit_norm=np.array(final_refit_norm), final_singular_values=np.array(final_singular_values),
        elapsed_seconds=np.array(time.perf_counter()-start))
    np.savez_compressed(data_dir/f'{mode}_{name}.npz', **out)
    return out


def constant_prediction(s, alpha, rate, steps):
    q = np.abs(1-rate*s*s)
    with np.errstate(divide='ignore'):
        logs = np.log(q)
    decay = np.exp(2*np.asarray(steps)[:, None]*logs[None, :])
    decay[np.asarray(steps)==0] = 1.
    return .5*decay @ (alpha*alpha)


def threshold_steps(s, alpha, rate, norm2, tolerance):
    threshold = .5*norm2*tolerance**2
    low, high = 0, 1
    def val(k):
        return constant_prediction(s, alpha[:, None], rate, np.array([k]))[0, 0]
    while val(high) > threshold:
        high *= 2
        if high > 10**13:
            return np.inf
    while high-low > 1:
        mid = (low+high)//2
        if val(mid) <= threshold:
            high = mid
        else:
            low = mid
    return int(high)


def validate_predictions(spectral, trajectories):
    discrepancies = {}
    for name in SCHEDULES[:2]:
        data = trajectories[('fixed', name)]
        norm2 = np.mean(data['y']**2, axis=0)
        worst = 0.
        for start in range(0, len(data['train_loss']), 4096):
            steps = np.arange(start, min(start+4096, len(data['train_loss'])))
            predicted = constant_prediction(spectral['singular_values'], spectral['alpha'], data['rates'][0], steps)
            actual_rel = np.sqrt(2*data['train_loss'][steps]/norm2)
            predicted_rel = np.sqrt(2*predicted/norm2)
            np.testing.assert_allclose(actual_rel, predicted_rel, rtol=2e-7, atol=2e-11)
            worst = max(worst, float(np.max(abs(actual_rel-predicted_rel))))
        discrepancies[name] = worst
    return discrepancies


def plot_all(cfg, spectral, trajectories, figures):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
    peak = float(spectral['peak_rate'])
    names = (r'Constant $\eta=0.002$', f'Constant η = {peak:.5f}', 'Warmup + cosine decay')
    handles = [Line2D([], [], color=c, lw=2, label=n) for c,n in zip(COLORS,names)]
    fig, axes = plt.subplots(4, 2, figsize=(14, 15), sharex=True, sharey=True, dpi=150)
    fig.subplots_adjust(left=.085, right=.985, top=.885, bottom=.08, hspace=.25, wspace=.14)
    fig.suptitle('Gamma 128 · training samples exactly at the initial centers', y=.98, fontsize=19)
    fig.text(.5,.947,'Zero readout · 129 fixed training points · 177 tanh neurons · 200,000 actual GD updates',ha='center')
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5,.928), ncol=3, frameon=False)
    for t, target in enumerate(LABELS):
        for m, mode in enumerate(MODES):
            ax = axes[t,m]
            endpoints=[]
            for color, name in zip(COLORS, SCHEDULES):
                d = trajectories[(mode,name)]
                # Preserve first 100 samples then use dense log/time subsampling for file size.
                ids=np.unique(np.r_[np.arange(101),np.geomspace(100,cfg['steps'],2500).astype(int),cfg['steps']])
                ax.plot(ids,np.maximum(d['train_loss'][ids,t],1e-33),color=color,lw=1.6)
                endpoints.append(np.sqrt(2*d['train_loss'][-1,t]/np.mean(d['y'][:,t]**2)))
            floor=spectral['initial_ls_train_loss'][t]
            ax.set_title(f'{target} — '+('fixed geometry' if mode=='fixed' else 'joint geometry + readout'),fontsize=12)
            ax.set_xscale('symlog',linthresh=10);ax.set_yscale('log')
            ax.set_xlim(0,cfg['steps']);ax.set_ylim(1e-16,1)
            ax.grid(which='major',alpha=.18)
            text='Final relative L₂: '+ ' / '.join(f'{e:.2e}' for e in endpoints)
            text+=f'\nInitial LS loss: {floor:.1e} (below displayed range)'
            ax.text(.03,.06,text,transform=ax.transAxes,fontsize=9,
                    bbox=dict(facecolor='white',alpha=.85,edgecolor='none'))
            if m==0: ax.set_ylabel(r'Training loss $L=\frac{1}{2n}\sum_i e_i^2$')
            if t==3: ax.set_xlabel('GD update number (linear through 10; logarithmic thereafter)')
    fig.text(.5,.026,'Axes show the actual GD losses; the much smaller least-squares references are stated numerically in each panel.'
             '\nEndpoint relative errors follow the legend order. Learning-rate schedules use the same rate for all trainable parameters.',ha='center',fontsize=10)
    fig.savefig(figures/'training_convergence.png');plt.close(fig)

    fig, axes_grid=plt.subplots(2,2,figsize=(14,11.5),dpi=150)
    axes=axes_grid.ravel()
    fig.subplots_adjust(left=.09,right=.97,top=.86,bottom=.16,wspace=.28,hspace=.40)
    fig.suptitle('Interpolation is feasible; ordinary GD still sees weak matrix directions',y=.975,fontsize=18)
    s=spectral['singular_values'];ax=axes[0]
    ax.plot(np.arange(1,len(s)+1),s,color='#21918c',lw=2)
    ax.set_yscale('log');ax.set_xlabel('Singular-value index (largest first)');ax.set_ylabel(r'$\sigma_j(A_0)$')
    ax.set_title(f'Full row rank: {len(s)} / {len(s)}\nCondition number {s[0]/s[-1]:,.0f}')
    ax=axes[2]; tt=np.r_[0,np.geomspace(1,2e9,1200)]
    norms=np.mean(trajectories[('fixed','baseline')]['y']**2,axis=0)
    pp=np.sqrt(2*constant_prediction(s,spectral['alpha'],peak,tt)/norms)
    cmap=plt.get_cmap('viridis')
    for t,label in enumerate(LABELS):ax.plot(tt,np.maximum(pp[:,t],1e-16),label=label,color=cmap(.1+.8*t/3),lw=1.7)
    ax.axvline(cfg['steps'],ls=':',color='.5',label='Actual training horizon')
    ax.set_xscale('log');ax.set_yscale('log');ax.set_xlim(1,2e9);ax.set_ylim(1e-14,1)
    ax.set_xlabel('Predicted GD updates');ax.set_ylabel('Training relative L₂');ax.set_title('Fixed geometry · larger constant rate\nExact-arithmetic prediction')
    ax.legend(frameon=False,fontsize=9,loc='lower left')
    ax=axes[3]
    for c,n in zip(COLORS,SCHEDULES):
        rr=trajectories[('fixed',n)]['rates'];ids=np.unique(np.r_[np.arange(min(1001,len(rr))),np.linspace(0,len(rr)-1,700).astype(int)])
        ax.plot(ids,rr[ids],color=c,lw=1.8)
    ax.axhline(float(spectral['stability_limit']),color='.5',ls=':',lw=1)
    ax.set_xlabel('GD update number');ax.set_ylabel(r'Learning rate $\eta_k$');ax.set_title('Three prescribed schedules\nDotted: fixed-geometry stability limit')
    ax.set_xticks([0,50000,100000,150000,200000], ['0','50,000','100,000','150,000','200,000'])
    ax=axes[1]
    x=trajectories[('fixed','baseline')]['x']
    for j,c,label in ((-1,'#440154','Weakest direction'),(-2,'#21918c','Second weakest')):
        ax.plot(x,spectral['weakest_left_vectors'][:,j],color=c,lw=1,marker='.',ms=2,label=label)
    ax.set_xlabel('Training sample position xᵢ');ax.set_ylabel(r'Unit-length left singular vector $u_j(x_i)$')
    ax.set_title('Weak directions alternate across neighboring samples\nThese are matrix modes, not a Fourier transform')
    ax.legend(frameon=False,fontsize=9,loc='upper center',bbox_to_anchor=(.5,-.18),ncol=2)
    for ax in axes:ax.grid(alpha=.18)
    fig.text(.5,.102,r'$A_0=[\tanh(128(x_i-z_j)),1]/\sqrt{129},\qquad r_{k+1}=(I-\eta_k A_0A_0^T)r_k$',ha='center',fontsize=14)
    fig.text(.5,.035,'Prediction uses target loadings in every singular direction. It is not additional executed training or a prediction for joint GD.'
             '\nThe 1,000-step warmup starts at 0.002; cosine decay ends at 1% of the peak. No momentum or coefficient solves enter training.',ha='center',fontsize=10)
    fig.savefig(figures/'spectrum_and_schedules.png');plt.close(fig)

    fig,axes=plt.subplots(2,4,figsize=(17,8),dpi=150,sharex='row',sharey='row')
    fig.subplots_adjust(left=.065,right=.985,top=.85,bottom=.17,wspace=.2,hspace=.35)
    fig.suptitle('Fitting the centers versus fitting between them',y=.975,fontsize=19)
    fig.text(.5,.923,'Larger constant rate · solid: joint GD · dashed: fixed geometry · dotted gray: initial-geometry LS interpolation',ha='center',fontsize=11)
    lo,hi=cfg['domain'];xx=lo+(np.arange(cfg['n_eval'])+.5)*(hi-lo)/cfg['n_eval']
    yy=np.column_stack([target_values(t,xx,cfg) for t in cfg['targets']])
    x,y,z,h,a,b=setup(cfg)
    lspred=feature(xx,a,b)@spectral['initial_ls_coefficients']
    for t,label in enumerate(LABELS):
        for mode,ls in zip(MODES,('--','-')):
            d=trajectories[(mode,'large_constant')]
            pred=feature(xx,d['final_a'][t],d['final_b'][t])@d['final_v'][t]
            axes[0,t].plot(xx,pred-yy[:,t],color='#21918c',ls=ls,lw=1.2)
            steps=d['snapshot_steps']
            train=np.sqrt(2*d['train_loss'][steps,t]/np.mean(y[:,t]**2))
            axes[1,t].plot(steps,train,color='#440154',ls=ls,lw=1.6)
            axes[1,t].plot(steps,d['eval_relative'][:,t],color='#21918c',ls=ls,lw=1.6)
        axes[0,t].plot(xx,lspred[:,t]-yy[:,t],color='.5',ls=':',lw=1)
        axes[0,t].set_title(label);axes[0,t].set_xlabel('x');axes[0,t].axhline(0,color='.7',lw=.6)
        axes[1,t].set_xscale('symlog',linthresh=10);axes[1,t].set_yscale('log');axes[1,t].set_xlabel('GD update number')
        axes[1,t].set_ylim(1e-12,2);axes[1,t].set_xlim(0,cfg['steps'])
        for ax in axes[:,t]:ax.grid(alpha=.18)
    axes[0,0].set_ylabel('Final residual f(x) − target(x)')
    axes[1,0].set_ylabel('Relative L₂ error')
    fig.text(.5,.065,'Bottom row — purple: 129 training centers; teal: 8,192 independent midpoint samples.'
             '\nThe original samples remain fixed when learned centers move. High training accuracy need not imply high between-center accuracy.',ha='center',fontsize=11)
    fig.savefig(figures/'between_centers.png');plt.close(fig)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--plot-only',action='store_true');parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();cfg=yaml.safe_load((HERE/'config.yaml').read_text())
    torch.set_default_dtype(torch.float64);torch.set_num_threads(cfg['threads'])
    data_dir=RESULTS/'data';figures=RESULTS/'figures'
    data_dir.mkdir(parents=True,exist_ok=True);figures.mkdir(parents=True,exist_ok=True)
    with threadpool_limits(limits=cfg['threads']):
        x,y,z,h,a,b=setup(cfg);spectral=spectral_data(cfg,x,y,a,b)
        all_data={}
        for mode in MODES:
            for schedule in SCHEDULES:
                path=data_dir/f'{mode}_{schedule}.npz'
                if args.plot_only or (args.resume and path.exists()):
                    with np.load(path,allow_pickle=False) as saved:data={k:saved[k] for k in saved.files}
                    if json.loads(str(data['config_json']))!=cfg:raise ValueError('Saved configuration mismatch')
                else:data=train(cfg,mode,schedule,spectral,data_dir)
                all_data[(mode,schedule)]=data
        checks=validate_predictions(spectral,all_data)
        thresholds=np.array([1e-2,1e-4,1e-8,1e-12])
        predicted=np.array([[threshold_steps(spectral['singular_values'],spectral['alpha'][:,t],
                    float(spectral['peak_rate']),np.mean(y[:,t]**2),tol) for tol in thresholds] for t in range(y.shape[1])])
        spectral.update(prediction_thresholds=thresholds,predicted_steps=predicted,validation_json=np.array(json.dumps(checks)))
        np.savez_compressed(data_dir/'spectrum.npz',**spectral)
        plot_all(cfg,spectral,all_data,figures)
        print('Prediction checks:',checks,flush=True)
        print('Predicted threshold steps:',predicted,flush=True)
    print(f'Complete: {RESULTS}',flush=True)


if __name__=='__main__':main()
