"""Five explanatory figures from saved D24 data. No training or readout solves.

Run with --figure 1 ... 5 to render a subset; the default renders all five.
The animation combines an analytic whole-line spectrum and absolute band errors.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.integrate import simpson
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.expD24_gd_residual_spectrum import direct_scale_test as direct
from experiments.expD24_gd_residual_spectrum import four_way_comparison as four
from experiments.expD24_gd_residual_spectrum import gamma_comparison as gamma_exp
from experiments.expD24_gd_residual_spectrum import readout_comparison as readout
from experiments.expD24_gd_residual_spectrum import spectrum, whole_line as whole

BASE = gamma_exp.RESULTS.parent
OUT = BASE / "story_figures"
BAND_COLORS = ["#2776b5", "#db7825", "#249868"]
BAND_LABELS = [r"Low: $|\omega|/\pi<4$", r"Middle: $4\leq|\omega|/\pi<10$", r"High: $|\omega|/\pi\geq10$"]
GAMMAS = [1., 4., 16., 64.]


def graphics():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 11, "axes.labelsize": 12, "axes.titlesize": 13,
                         "legend.fontsize": 11, "savefig.facecolor": "white"})
    return plt


def time_axis(ax, log=False):
    if log:
        ax.set_xscale("symlog", linthresh=2)
        ax.set_xticks([0, 2, 10, 100, 2000], labels=["0", "2", "10", "100", "2000"])
    else:
        ax.set_xticks([0, 500, 1000, 1500, 2000])
    ax.set_xlim(0, 2000)
    ax.grid(alpha=.18)


def gamma_axis(ax):
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator([.5, 1, 4, 16, 64, 256]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlim(.5, 256)
    ax.grid(alpha=.18)


def load():
    cases, cfg = readout.load_cases(gamma_exp.RESULTS / "data/data.npz")
    by_gamma = {(c["target"], gamma_exp.gamma_of(c)): c for c in cases if c["method"] == "gd"}
    w_cases, w_cfg = whole.load_data(BASE / "whole_line/data.npz")
    matched, m_cfg = four.load(four.RESULTS / "data/comparison.npz")
    with np.load(direct.RESULTS / "data/controls.npz") as f:
        controls = {k: f[k] for k in f.files}
    with np.load(direct.RESULTS / "data/pairings.npz") as f:
        pairs = {k: f[k] for k in f.files}
    with np.load(direct.RESULTS / "data/matched_residual.npz") as f:
        probe = {k: f[k] for k in f.files}
    return dict(cases=by_gamma, cfg=cfg, whole_cases=w_cases, whole_cfg=w_cfg,
                matched=matched, matched_cfg=m_cfg, controls=controls, pairs=pairs, probe=probe)


def band_errors(saved):
    """Low/mid integrals from cached analytic spectra; high is the energy complement.

    The complement includes frequencies outside the displayed [0,32] range.
    Check against independent Gaussian frequency quadrature at three states.
    """
    cfg = saved["whole_cfg"]
    steps = spectrum.frame_steps(cfg["steps"])
    k = np.linspace(0, cfg["max_mode"], cfg["frequency_points"])
    normalization = whole.target_energy(cfg)
    fq, fw = readout.frequency_quadrature(64)
    masks = [fq < 4, (fq >= 4) & (fq < 10), fq >= 10]
    values = []
    discrepancies = []
    for case in saved["whole_cases"]:
        power = np.abs(case["spectra"])**2
        low = simpson(power[:, k <= 4], x=k[k <= 4], axis=1)
        middle = simpson(power[:, (k >= 4) & (k <= 10)], x=k[(k >= 4) & (k <= 10)], axis=1)
        total = 2*case["loss"][steps]
        bands = np.column_stack((low, middle, total-low-middle))
        assert bands.min() > -1e-10
        np.testing.assert_allclose(bands.sum(axis=1), total, atol=1e-14, rtol=1e-13)
        for i in (0, len(steps)//2, len(steps)-1):
            params = {key: case["parameters"][key][i] for key in ("a", "b", "v")}
            E = whole.model_spectrum(params, np.pi*fq)-whole.target_spectrum(np.pi*fq, cfg)
            independent = np.array([np.sum(fw[m]*abs(E[m])**2) for m in masks])
            discrepancies.append(float(np.max(abs(independent-bands[i]))/normalization))
            np.testing.assert_allclose(bands[i]/normalization, independent/normalization, atol=2e-7, rtol=2e-6)
        values.append(bands/normalization)
    return np.asarray(values), max(discrepancies)


def figure1(saved, bands):
    from PIL import Image
    plt = graphics()
    cfg = saved["whole_cfg"]
    cases = saved["whole_cases"]
    steps = np.asarray(spectrum.frame_steps(cfg["steps"]))
    k = np.linspace(0, cfg["max_mode"], cfg["frequency_points"])
    fig, axes = plt.subplots(4, 2, figsize=(13, 12.7), dpi=135, sharex="col", sharey="col")
    spectrum_lines, band_lines, total_lines, cursors, labels = [], [], [], [], []
    for row, case in enumerate(cases):
        ax, right = axes[row]
        ax.plot(k, abs(case["spectra"][0]), color="#b4b9be", lw=1.4, ls="--")
        line, = ax.plot(k, abs(case["spectra"][0]), color="#21262c", lw=1.8)
        spectrum_lines.append(line)
        ax.set(xlim=(0, 32), ylim=(0, .57), xticks=[0, 4, 8, 12, 16, 24, 32])
        ax.set_ylabel(rf"$\gamma_0={case['initial_gamma']:g}$"+"\n"+r"$|\widehat e_t(\omega)|$")
        ax.grid(alpha=.18)
        labels.append(ax.text(.5, 1.035, "", transform=ax.transAxes, ha="center", fontsize=11))
        lines = []
        for j, color in enumerate(BAND_COLORS):
            right.plot(steps, bands[row, :, j], color=color, alpha=.18, lw=1.5)
            l, = right.plot([], [], color=color, lw=2.0)
            lines.append(l)
        band_lines.append(lines)
        total = bands[row].sum(axis=1)
        right.plot(steps, total, color="#333333", ls=":", alpha=.25, lw=1.2)
        l, = right.plot([], [], color="#333333", ls=":", lw=1.7)
        total_lines.append(l)
        cursors.append(right.axvline(0, color="#888888", lw=.8, alpha=.7))
        right.set(yscale="log", ylim=(1e-8, 1.15), ylabel=r"Band error $\mathcal{E}_b(t)$")
        time_axis(right, log=True)
        if row == 3:
            ax.set_xlabel(r"Frequency $\omega/\pi$")
            right.set_xlabel("GD step t (log spacing after 2)")
    axes[0, 0].set_title("Residual spectrum: initial (gray) and current (black)", pad=37)
    axes[0, 1].set_title("Error remaining in each frequency band", pad=37)
    title = fig.suptitle("", fontsize=19, y=.982)
    fig.text(.5, .947, "Gaussian-envelope mixed sine on the whole line · ordinary GD · initial gamma varies; width and learning rate stay fixed",
             ha="center", fontsize=10.5)
    handles = [plt.Line2D([], [], color=c, lw=2) for c in BAND_COLORS]
    handles.append(plt.Line2D([], [], color="#333333", ls=":", lw=2))
    fig.legend(handles, BAND_LABELS+[r"Total $E(t)^2$"], loc="upper center", bbox_to_anchor=(.5, .923), ncol=4, frameon=False)
    fig.subplots_adjust(left=.085, right=.98, top=.843, bottom=.135, hspace=.36, wspace=.26)
    fig.text(.5, .072, r"Right: $\mathcal{E}_b(t)=\|e_b(t)\|_{L_2}^2/\|y\|_{L_2}^2$; the colored values sum to the total squared relative error.",
             ha="center", fontsize=12)
    fig.text(.5, .030, "The denominator is fixed, so a falling curve means less error in that band—not merely a smaller share.\n"
             "Exact whole-line spectra; no Fourier window or taper. The high band includes the tail beyond the displayed frequency range.",
             ha="center", fontsize=10.5)
    durations = spectrum.frame_durations(len(steps), 32.)
    destination = OUT / "01_frequency_learning.gif"
    partial = destination.with_suffix(".partial.gif")
    palette = None
    with partial.open("wb") as stream:
        for i, step in enumerate(steps):
            title.set_text(f"1 · Which frequencies are fitted first?   GD step {step:,} / 2,000")
            for row, case in enumerate(cases):
                spectrum_lines[row].set_ydata(abs(case["spectra"][i]))
                labels[row].set_text(f"Current relative L₂ error: {case['relative_l2'][step]:.4f}")
                for j, l in enumerate(band_lines[row]):
                    l.set_data(steps[:i+1], bands[row, :i+1, j])
                total_lines[row].set_data(steps[:i+1], bands[row, :i+1].sum(axis=1))
                cursors[row].set_xdata([step, step])
            fig.canvas.draw()
            rgb = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
            if i in (0, len(steps)//2, len(steps)-1):
                rgb.save(Path('/tmp')/f'd24_story_animation_{i}.png')
            palette = spectrum.write_gif_frame(stream, rgb, palette, int(durations[i]))
            if i % 20 == 0: print('Animation frame',i+1,'/',len(steps),flush=True)
        stream.write(b';')
    plt.close(fig)
    partial.replace(destination)
    with Image.open(destination) as im:
        assert im.n_frames == len(steps)
        duration = 0
        for i in range(im.n_frames):
            im.seek(i); im.load(); duration += im.info['duration']
        assert duration == 32000


def figure2(saved):
    plt = graphics()
    d = saved['probe']
    fig, axes = plt.subplots(1, 3, figsize=(18, 7.5), dpi=160)
    k = np.linspace(0, 32, 4097)
    colors = plt.cm.viridis([.12, .38, .65, .88])
    for gamma, color in zip(GAMMAS, colors):
        value = abs(direct.tangent_spectrum(np.pi*k, np.array([gamma]), np.array([0.]))[:, 0])
        axes[0].semilogy(k, np.maximum(value,1e-16), color=color, lw=2, label=rf'$\gamma={gamma:g}$')
    axes[0].set(xlim=(0, 32), ylim=(1e-12, 1), xlabel=r'Frequency $\omega/\pi$',
                ylabel=r'Tangent magnitude $|\widehat\psi_\gamma(\omega)|$')
    axes[0].set_xticks([0,8,16,24,32]); axes[0].grid(alpha=.18)
    for mode, grad, color in zip(d['modes'], d['gradient'], BAND_COLORS):
        axes[1].loglog(d['gamma'], abs(grad), color=color, lw=2.2, label=rf'$e_{{{mode:g}}}$: Gaussian $\times\sin({mode:g}\pi x)$')
        axes[2].semilogx(d['gamma'], abs(grad)/max(abs(grad)), color=color, lw=2.2)
    for ax in axes[1:]:
        gamma_axis(ax); ax.set_xlabel(r'Neuron scale $\gamma$ in $\tanh(\gamma x)$')
    axes[1].set(ylim=(1e-32,1), ylabel=r'Actual gradient magnitude $G_k(\gamma)$')
    axes[2].set(ylim=(-.02,1.05), ylabel=r'$G_k(\gamma)\,/\,\max_{\gamma\prime}G_k(\gamma\prime)$')
    axes[2].set_yticks([0,.25,.5,.75,1])
    for ax,title in zip(axes,['Tangent: bandwidth and magnitude','Fixed residual: actual gradient','Same gradient, divided by its own peak']):
        ax.set_title(title, pad=18, fontsize=13)
    fig.suptitle('2 · Increasing gamma changes both spectral reach and gradient size', fontsize=21,y=.98)
    fig.text(.5,.926,r'Center $z=0$ and readout $c=1$ stay fixed. No training. Each residual is fixed throughout its gamma sweep.',ha='center',fontsize=12)
    fig.text(.5,.874,r'$e_k(x)=-C_k e^{-x^2/(2\sigma^2)}\sin(k\pi x),\quad \sigma=0.4,\quad \|e_k\|_{L_2(\mathbb{R})}=1$',ha='center',fontsize=15)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.21,.825),ncol=2,frameon=False)
    fig.legend(*axes[1].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.68,.825),ncol=3,frameon=False,fontsize=10.5)
    fig.subplots_adjust(left=.067,right=.987,top=.655,bottom=.255,wspace=.31)
    fig.text(.5,.131,r'$\psi_\gamma(x)=x\,\mathrm{sech}^2(\gamma x),\qquad G_k(\gamma)=\left|\int_{\mathbb{R}}e_k(x)\psi_\gamma(x)\,dx\right|=|\partial_\gamma L_{\mathbb{R}}|$',ha='center',fontsize=17)
    fig.text(.5,.042,'Left legend varies the tangent scale; middle/right legend varies the fixed residual function.\n'
             'The right panel compares where each residual produces its strongest gradient. Equal peak heights do not mean equal absolute gradients.',ha='center',fontsize=11)
    fig.savefig(OUT/'02_scale_sensitivity.png'); plt.close(fig)


def figure3(saved):
    plt = graphics()
    targets = ['sine','sine_mixture','runge']
    colors = plt.cm.viridis([.12,.38,.65,.88])
    fig,axes = plt.subplots(3,2,figsize=(13,11.6),dpi=160,sharex='col',sharey='col')
    for row,target in enumerate(targets):
        final,refit=[],[]
        for gamma,color in zip(GAMMAS,colors):
            case=saved['cases'][target,gamma]
            losses=saved['controls'][f'{target}_{gamma:g}__frozen_loss']
            error=np.sqrt(losses/case['loss'][0])
            axes[row,0].plot(np.arange(len(losses)),error,color=color,lw=2)
            final.append(error[-1]);refit.append(case['views']['gd_refit']['relative_l2'][0])
        axes[row,0].set(yscale='log',ylim=(.009,1.1),ylabel=readout.LABELS[target]+'\nRelative L₂ error')
        time_axis(axes[row,0])
        axes[row,1].loglog(GAMMAS,final,'o-',color='#30343a',lw=2,label='Readout GD after 2,000 steps')
        axes[row,1].loglog(GAMMAS,refit,'s-',color='#9b4da6',lw=2,label='Least-squares readout at initialization')
        axes[row,1].set(xlim=(.8,80),ylim=(1e-16,1.3),ylabel='Relative error (log scale)',xticks=GAMMAS)
        axes[row,1].set_xticklabels(['1','4','16','64'])
        axes[row,1].axvline(16,color='#888888',ls=':',lw=1)
        axes[row,1].grid(alpha=.18)
        if row==2:
            axes[row,0].set_xlabel('Readout-GD step t')
            axes[row,1].set_xlabel(r'Fixed initial scale $\gamma_0$ (log scale)')
    axes[0,0].set_title('Freeze geometry; train only the readout',pad=17)
    axes[0,1].set_title('Faster GD need not mean a lower refitted error',pad=17)
    fig.suptitle('3 · Large initial gamma can help readout GD even when approximation worsens',fontsize=19,y=.983)
    fig.legend([plt.Line2D([],[],color=c,lw=2) for c in colors],[rf'$\gamma_0={g:g}$' for g in GAMMAS],loc='upper center',bbox_to_anchor=(.27,.944),ncol=4,frameon=False)
    fig.legend(*axes[0,1].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.74,.944),ncol=1,frameon=False)
    fig.subplots_adjust(top=.835,bottom=.20,left=.09,right=.98,wspace=.3,hspace=.26)
    fig.text(.5,.079,'All three targets use the same finite interval [−1,1], fixed uniform centers, zero initial readout, and readout-GD rate 0.002.\n'
             'Left: geometry stays exactly fixed throughout training. Right: the same initial geometry is evaluated with a separately solved readout.',ha='center',fontsize=11)
    fig.text(.5,.026,r'$N=128$, 177 neurons; $h=1/64$. Dotted line: $\gamma=16$, $\lambda=0.25$ (QI reference). At $\gamma=64$, $\lambda=1$. '
             '\nGD errors use training quadrature; LS errors use 32,768 independent evaluation nodes and SVD cutoff 10⁻¹³.',ha='center',fontsize=10.5)
    fig.savefig(OUT/'03_readout_vs_approximation.png');plt.close(fig)


def figure4(saved):
    plt=graphics()
    arms=list(four.ARMS);labels=['Xavier',r'$\gamma_0=1$',r'$\gamma_0=4$',r'$\gamma_0=16$']
    colors=['#555b65',*plt.cm.viridis([.12,.48,.85])]
    fig,axes=plt.subplots(4,3,figsize=(16,13),dpi=155,sharex=True,sharey='col')
    for row,target in enumerate(four.TARGETS):
        for arm,color in zip(arms,colors):
            case=saved['matched'][f'{target}__{arm}'];t=case['steps'];E=case['errors']
            motion=np.mean(abs(abs(case['a'])-abs(case['a'][0])),axis=1)
            axes[row,0].plot(t,np.maximum(motion,1e-10),color=color,lw=1.8)
            axes[row,1].plot(t,E[:,1]-E[:,0],color=color,lw=1.8)
            axes[row,2].plot(t,E[:,3]-E[:,2],color=color,lw=1.8)
        axes[row,0].set(yscale='log',ylim=(1e-8,.05),ylabel=readout.LABELS[target]+'\n'+r'Mean $|\Delta\gamma|$')
        axes[row,1].set_ylabel(r'$\Delta E_{\rm GD}$')
        axes[row,2].set_ylabel(r'$\Delta E_{\rm LS}$')
        for ax in axes[row]:
            time_axis(ax)
            if row==3:ax.set_xlabel('Ordinary-GD step t')
        for ax in axes[row,1:]:
            ax.set_yscale('symlog',linthresh=1e-6)
            ax.set_ylim(-.0003,.3)
            ax.set_yticks([-1e-4,-1e-6,0,1e-6,1e-4,1e-2,1e-1])
            ax.axhspan(-1e-6,1e-6,color='#aaaaaa',alpha=.08,zorder=0)
            ax.axhline(0,color='#999999',lw=.8)
            for line in ax.lines:
                values=np.asarray(line.get_ydata())
                assert np.all((values>=-.0003)&(values<=.3)), 'Error-benefit curve would be clipped'
    for ax,title in zip(axes[0],['How far does gamma move?','Does the ordinary GD fit improve?','Does the separately refitted geometry improve?']):ax.set_title(title,pad=17,fontsize=12)
    fig.suptitle('4 · Geometry motion: changes in the GD fit and the refitted approximation',fontsize=20,y=.987)
    fig.legend([plt.Line2D([],[],color=c,lw=2) for c in colors],labels,loc='upper center',bbox_to_anchor=(.5,.953),ncol=4,frameon=False,fontsize=12)
    fig.text(.5,.906,'Middle and right share the same signed-log scale: positive = improvement, negative = deterioration; linear within ±10⁻⁶.',ha='center',fontsize=11)
    fig.subplots_adjust(top=.838,bottom=.185,left=.08,right=.986,wspace=.34,hspace=.34)
    fig.text(.5,.084,r'Left: $W^{-1}\sum_j|\gamma_j(t)-\gamma_j(0)|$. Middle: $E_{\rm frozen,GD}(t)-E_{\rm free,GD}(t)$. '
             '\n'+r'Right: $E_{\rm initial,LS}-E_{\rm current,LS}(t)$. Both error reductions use identical relative-$L_2$ units and axis limits.',ha='center',fontsize=12)
    fig.text(.5,.028,'Matched finite-domain runs for all four targets: [−1,1], 1,024 training and 32,768 evaluation nodes; GD rate 0.002; 2,000 steps.\n'
             'Readout solves occur after training, including step zero. Zero gamma displacement is omitted on the log axis; tiny refit changes need not be meaningful.\n'
             'These baseline settings do not establish that scale learning cannot improve approximation at other learning rates.',ha='center',fontsize=10.5)
    fig.savefig(OUT/'04_geometry_motion_and_benefit.png');plt.close(fig)


def figure5(saved):
    plt=graphics()
    fig,axes=plt.subplots(2,4,figsize=(18,10),dpi=155,sharex=True,sharey='row')
    for col,gamma in enumerate(GAMMAS):
        key=f'{gamma:g}__';d=saved['pairs']
        t=np.array(saved['cases']['gaussian_envelope',gamma]['snapshot_steps'])
        band_g=np.mean(abs(d[key+'band_gradient']),axis=-1)
        total=np.mean(abs(d[key+'gradient']),axis=-1);upper=d[key+'envelope'].mean(axis=-1)
        np.testing.assert_allclose(d[key+'band_gradient'].sum(axis=1),d[key+'gradient'],rtol=1e-10,atol=1e-14)
        assert np.all(total <= upper+1e-14)
        for j,color in enumerate(BAND_COLORS):
            axes[0,col].plot(t,d[key+'band_energy'][:,j],color=color,lw=2,marker='o',ms=3)
            axes[1,col].plot(t[1:],np.maximum(band_g[1:,j],1e-16),color=color,lw=2,marker='o',ms=3)
        axes[1,col].plot(t[1:],total[1:],color='#20252a',lw=1.8)
        axes[1,col].plot(t[1:],upper[1:],color='#85858b',ls='--',lw=1.8)
        axes[0,col].set(ylim=(-.025,1.025),yticks=[0,.25,.5,.75,1],title=rf'Initial $\gamma_0={gamma:g}$')
        axes[1,col].set(yscale='log',ylim=(8e-17,1e-3),yticks=[1e-16,1e-12,1e-8,1e-4])
        axes[1,col].set_title(f'Final actual / bound = {total[-1]/upper[-1]:.0%}',pad=13,fontsize=11)
        for ax in axes[:,col]:time_axis(ax,log=True)
        axes[1,col].set_xlabel('GD step t (log spacing after 2)',fontsize=10)
    axes[0,0].set_ylabel('Fraction of remaining error energy')
    axes[1,0].set_ylabel('Mean absolute gradient\n(log scale)')
    fig.suptitle('5 · The frequency band containing the error need not supply the scale gradient',fontsize=21,y=.985)
    fig.text(.5,.940,'Same whole-line Gaussian trajectories as Figure 1. Each column fixes the initial gamma; time runs horizontally.',ha='center',fontsize=12)
    handles=[plt.Line2D([],[],color=c,lw=2) for c in BAND_COLORS]+[plt.Line2D([],[],color='#20252a',lw=2),plt.Line2D([],[],color='#85858b',ls='--',lw=2)]
    fig.legend(handles,BAND_LABELS+['Actual total gradient','Magnitude-only upper bound'],loc='upper center',bbox_to_anchor=(.5,.91),ncol=5,frameon=False)
    fig.text(.5,.857,'TOP: where the remaining error sits. BOTTOM: each band’s scale-gradient contribution, including the current readout coefficients.',ha='center',fontsize=12)
    fig.subplots_adjust(top=.794,bottom=.235,left=.077,right=.985,hspace=.40,wspace=.17)
    fig.text(.5,.149,r'$g_{jb}(t)=\frac{c_j(t)}{2\pi}\,\mathrm{Re}\!\int_{\omega\in b}\widehat e_t(\omega)\,\overline{\widehat\psi_{j,t}(\omega)}\,d\omega$',ha='center',fontsize=17)
    fig.text(.5,.097,r'Bottom colors: $W^{-1}\sum_j|g_{jb}|$. Black: $W^{-1}\sum_j|\sum_b g_{jb}|$. '
             'Gray: replace the Fourier product by its magnitude before integrating.',ha='center',fontsize=12)
    fig.text(.5,.038,'Signed contributions are integrated before taking absolute values; colored gradient curves need not add to the black curve.\n'
             'The top fractions sum to 1; they do not show the absolute error size (Figure 1 does). A smaller actual/bound ratio means more phase cancellation.\n'
             'Section-2 centered-scale derivatives are measured here; ordinary raw-slope GD uses a different derivative. Values below 10⁻¹⁶ are displayed at the floor.',ha='center',fontsize=10.5)
    fig.savefig(OUT/'05_frequency_gradient_pairing.png');plt.close(fig)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--figure',type=int,nargs='+',default=[1,2,3,4,5]);args=parser.parse_args()
    OUT.mkdir(exist_ok=True);(OUT/'data').mkdir(exist_ok=True)
    saved=load();checks={}
    # Figures 1 and 5 must describe the very same whole-line trajectories.
    frame_steps=spectrum.frame_steps(2000)
    for wc in saved['whole_cases']:
        case=saved['cases']['gaussian_envelope',wc['initial_gamma']]
        indices=[frame_steps.index(t) for t in case['snapshot_steps']]
        for key in ('a','b','v'):
            np.testing.assert_array_equal(wc['parameters'][key][indices],case['parameters'][key])
    checks['animation_and_pairing_states_identical']=True
    with threadpool_limits(limits=2):
        if 1 in args.figure:
            bands,error=band_errors(saved)
            np.savez_compressed(OUT/'data/band_errors.npz',band_errors=bands,steps=spectrum.frame_steps(2000))
            checks['band_energy_max_absolute_normalized_discrepancy']=error
            figure1(saved,bands)
        for number,fn in [(2,figure2),(3,figure3),(4,figure4),(5,figure5)]:
            if number in args.figure:fn(saved);print('Rendered figure',number,flush=True)
    sources=[BASE/'whole_line/data.npz',gamma_exp.RESULTS/'data/data.npz',four.RESULTS/'data/comparison.npz',
             direct.RESULTS/'data/controls.npz',direct.RESULTS/'data/pairings.npz',direct.RESULTS/'data/matched_residual.npz']
    manifest_path=OUT/'data/provenance.json'
    manifest=json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest.update(checks)
    manifest['sources']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    manifest['training_repeated']=False
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    print('Finished; source trajectories were read only.',flush=True)


if __name__=='__main__':main()
