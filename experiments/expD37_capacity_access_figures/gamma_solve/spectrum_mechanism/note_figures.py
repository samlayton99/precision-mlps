"""Note figures from saved spectral experiments, with a small independent audit.

No training, fitting, interpolation, or existing output replacement. The
center-integral step forecast is not the finite-ratio lower-bound theorem.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/precisionmlps-mpl")
import numpy as np
from scipy import linalg as la
from threadpoolctl import threadpool_limits
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter, LogLocator, NullFormatter

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SOURCE = ROOT / "results/checkpoint_D_optimizers/expD37_capacity_access_figures/gamma_solve/spectrum_mechanism"
OUT = SOURCE / "note_figures"
COLORS = plt.colormaps["viridis"]([.07, .34, .62, .88])
WIDTHS = [64, 128, 256, 512]
DIVISORS = [32, 16, 8, 4]


def load(name):
    return json.loads((SOURCE / "data" / name).read_text())


def save(fig, name):
    fig.savefig(OUT / name, dpi=240, facecolor="white")
    plt.close(fig)


def clean(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=.16, linewidth=.6)
    ax.set_axisbelow(True)


def gamma_axis(ax, values):
    ax.set_xscale("log", base=2)
    ax.set_xticks(values)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(NullFormatter())


def independent_audit(steps, bands, transitions):
    """Direct B construction + alternate SVD; tests records, not roundoff bounds."""
    h = 2 / steps["N"]
    halo = steps["halo_per_side"]
    centers = -1 + np.arange(-halo, steps["N"] + halo + 1) * h
    x = np.linspace(-1, 1, steps["m"])
    y = (np.sin(2*np.pi*x) + .5*np.sin(6*np.pi*x) + .25*np.sin(10*np.pi*x))/np.sqrt(len(x))
    audit = []
    for record, solved in zip(steps["rows"], bands["rows"]):
        gamma = record["gamma"]
        b = np.column_stack([np.ones(len(x)), np.tanh(gamma*(x[:, None]-centers))])/np.sqrt(len(x))
        u, s, vt = la.svd(b, full_matrices=False, lapack_driver="gesdd")
        p = (u.T@y)**2/(y@y)
        null = float(la.norm(y-u@(u.T@y))**2/(y@y))
        rates = .5*(s/s[0])**2
        def error_squared(n):
            return float(p@np.exp(2*n*np.log1p(-rates)) + null)
        lo, hi = 0, 1
        while error_squared(hi) > .01**2:
            lo, hi = hi, hi*2
        while hi-lo > 1:
            mid = (lo+hi)//2
            if error_squared(mid) > .01**2:
                lo = mid
            else:
                hi = mid
        keep = solved["retained_readout_modes"]
        v = vt[:keep].T@((u[:, :keep].T@y)/s[:keep])
        explicit_error = float(la.norm(b@v-y)/la.norm(y))
        saved_n = record["finite_tanh_spectral_steps"]
        audit.append(dict(gamma=gamma, alternate_svd_steps=hi,
                          saved_steps=saved_n,
                          relative_step_difference=(hi-saved_n)/saved_n,
                          explicit_readout_error_alternate_svd=explicit_error,
                          explicit_readout_error_saved=solved["explicit_resolved_readout_relative_error"]))
        assert abs(hi-saved_n)/saved_n < 1e-6
        assert abs(explicit_error-solved["explicit_resolved_readout_relative_error"]) < 1e-9
    resolved = [r for r in transitions["records"] if r["lower_bound"] is not None]
    assert all(r["lower_bound"] <= r["actual_ratio"]*(1+1e-5) for r in resolved)
    assert all(abs(r["bound_fraction"]-r["lower_bound"]/r["actual_ratio"]) < 1e-12 for r in resolved)
    assert all(abs(r["improvement_bound"]-r["lower_bound"]/r["start_ratio"]) < 1e-8*max(1,r["improvement_bound"]) for r in resolved)
    return dict(alternate_svd="gesdd; original used gesvd", rows=audit,
                resolved_transition_records_checked=len(resolved),
                interval_certified=False, gd_executed=False)


def step_figure(data, bands):
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.7), gridspec_kw={"width_ratios": [1.12, 1, 1]})
    gs = np.array([r["gamma"] for r in data["rows"]])
    actual = np.array([r["finite_tanh_spectral_steps"] for r in data["rows"]])
    forecast = np.array([r["integral_quadrature"][-1]["steps"] for r in data["rows"]])
    ax = axes[0]
    ax.plot(gs, actual, color="#222222", lw=1.9, marker="o", ms=4)
    ax.plot(gs, forecast, color="#1768a5", lw=1.2, ls="--", marker="o", ms=8,
            markerfacecolor="none", markeredgewidth=1.4)
    ax.set_yscale("log")
    ax.set(xlabel=r"Slope $\gamma$", ylabel="Steps to 1% relative error", ylim=(4e3, 1e13), title="(a) Predicted and actual steps")
    gamma_axis(ax, gs)
    ax.set_yticks([1e4, 1e6, 1e8, 1e10, 1e12])
    labels=[r"$1.01\!\times\!10^{12}$", r"$1.97\!\times\!10^7$", "66,209", "18,272"]
    for x, n, label in zip(gs, actual, labels):
        ax.annotate(label, (x, n), xytext=(0, 9), textcoords="offset points", ha="center", fontsize=9)
    ax.set_xlim(3.1, 89)
    ax = axes[1]
    deviation = 100*(forecast/actual-1)
    ax.bar(np.arange(4), deviation, width=.6, color=COLORS, edgecolor="none")
    ax.axhline(0, color=".35", lw=.8)
    for j, d in enumerate(deviation):
        label = "0%" if d == 0 else f"{d:+.4f}%"
        ax.annotate(label, (j, d), xytext=(0, 6 if d>=0 else -14), textcoords="offset points", ha="center", fontsize=9)
    ax.set(xticks=np.arange(4), xticklabels=["4", "8", "16", "64"], xlabel=r"Slope $\gamma$",
           ylabel="Prediction error in step count (%)", ylim=(-.39, .24), title="(b) Forecast error")
    ax = axes[2]
    feasible = np.array([r["explicit_resolved_readout_relative_error"] for r in bands["rows"]])
    ax.plot(gs, feasible, color="#3b826c", marker="o", lw=1.8, ms=5)
    ax.axhline(.01, color=".35", ls=":", lw=1.1)
    ax.text(4, .014, "Requested GD error: 1%", fontsize=9, color=".25")
    ax.annotate(r"$5.85\times10^{-4}$", (4, feasible[0]), xytext=(7, -19), textcoords="offset points", fontsize=9)
    ax.set_yscale("log")
    gamma_axis(ax, gs)
    ax.set(xlabel=r"Slope $\gamma$", ylabel="Explicit readout relative error", ylim=(6e-8,.04), xlim=(3.1,89), title="(c) Accurate readouts exist")
    for ax in axes:
        clean(ax)
    handles=[Line2D([],[],color="#222222",marker="o",lw=1.8,ms=4,label="Actual: finite tanh spectrum"),
             Line2D([],[],color="#1768a5",marker="o",mfc="none",ls="--",lw=1.2,ms=7,label="Prediction: center-integral spectrum")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.37,1.005), ncol=2, frameon=False, fontsize=10)
    fig.subplots_adjust(left=.066,right=.99,bottom=.20,top=.77,wspace=.46)
    save(fig,"step_counts_and_capacity.png")


def learning_curves(data):
    curves = np.load(SOURCE/"data/step_prediction_check.npz")
    fig, ax = plt.subplots(figsize=(7.5,3.8))
    for row, color in zip(data["rows"], COLORS):
        g = int(row["gamma"])
        keep=(curves["steps"]>=1)&(curves["steps"]<=2*row["finite_tanh_spectral_steps"])
        ax.loglog(curves["steps"][keep],curves[f"gamma{g}_finite"][keep],color=color,lw=2)
        ax.loglog(curves["steps"][keep],curves[f"gamma{g}_integral"][keep],color=color,lw=1.2,ls="--")
    ax.axhline(.01,color=".35",lw=1,ls=":")
    ax.text(2,.0115,"1% error",fontsize=10)
    ax.set(xlabel="GD steps",ylabel="Training relative L2 error",ylim=(.004,1.1),xlim=(1,2e12))
    clean(ax)
    handles=[Line2D([],[],color=c,lw=2,label=rf"$\gamma={int(r['gamma'])}$") for r,c in zip(data["rows"],COLORS)]
    fig.legend(handles=handles,loc="upper center",bbox_to_anchor=(.52,.99),ncol=4,frameon=False)
    fig.subplots_adjust(left=.10,right=.98,bottom=.17,top=.82)
    save(fig,"predicted_learning_curves.png")


def doubling_figure(data):
    fig, axes = plt.subplots(2,4,figsize=(11.7,5.0),sharex=True,
                             gridspec_kw={"height_ratios":[2,1]})
    for col,n in enumerate(WIDTHS):
        for d,color in zip(DIVISORS,COLORS):
            rows=[r for r in data["records"] if r["N"]==n and r["divisor"]==d and r["doubling"]]
            x=np.array([r["start_gamma"] for r in rows])
            actual=np.array([np.nan if r["actual_ratio"] is None else r["actual_ratio"] for r in rows])
            lower=np.array([np.nan if not r["lower_bound"] else r["lower_bound"] for r in rows])
            frac=np.array([np.nan if r["bound_fraction"] is None else r["bound_fraction"] for r in rows])
            axes[0,col].plot(x,actual,color=color,lw=1.6,marker=".",ms=2.5)
            axes[0,col].plot(x,lower,color=color,lw=1.5,ls="--",marker=".",ms=2.5)
            axes[1,col].plot(x,frac,color=color,lw=1.5,marker=".",ms=2.5)
        axes[0,col].set_title(rf"$N={n}$",pad=8)
        axes[0,col].set(yscale="log",ylim=(1e-26,1),yticks=[1e-25,1e-20,1e-15,1e-10,1e-5,1])
        axes[1,col].set(ylim=(0,1.05),yticks=[0,.5,1],xlabel=r"Starting $\gamma_a$")
        axes[1,col].axhline(1,color=".5",ls=":",lw=.7)
        for ax in axes[:,col]:
            gamma_axis(ax,[2,4,8,16,32,64]); ax.set_xlim(2,64); clean(ax)
        if col:
            for ax in axes[:,col]: ax.tick_params(labelleft=False)
    axes[0,0].set_ylabel(r"Final ratio $\lambda_i(2\gamma_a)/\lambda_1(2\gamma_a)$")
    axes[1,0].set_ylabel("Bound / actual")
    handles=[Line2D([],[],color=c,lw=2,label=rf"$i=N/{d}$") for d,c in zip(DIVISORS,COLORS)]
    handles += [Line2D([],[],color=".2",lw=1.8,label="Actual"),Line2D([],[],color=".2",ls="--",lw=1.8,label="Lower bound")]
    fig.legend(handles=handles,loc="upper center",bbox_to_anchor=(.54,.995),ncol=6,frameon=False,fontsize=10)
    fig.subplots_adjust(left=.095,right=.99,bottom=.12,top=.855,hspace=.20,wspace=.14)
    save(fig,"doubling_ratios.png")


def alignment_figure():
    data=json.loads((SOURCE.parent/"data/default.json").read_text())["comparison"]
    gammas=np.array(data["gammas"])
    maximum=data["max_steps"]
    fig,axes=plt.subplots(1,2,figsize=(9.5,3.8),sharex=True,sharey=True)
    for ax,variant,color,title,label in zip(axes,["fixed_p","fixed_rates"],["#1768a5","#ce6518"],
        ["(a) Change ratios; fix target weights","(b) Change target weights; fix ratios"],
        [r"Only $\lambda_i/\lambda_1$ changes",r"Only $p_i$ changes"]):
        lo=np.array([maximum if v is None else v for v in data[f"steps_{variant}_lower"]])
        hi=np.array([maximum if v is None else v for v in data[f"steps_{variant}_upper"]])
        ax.fill_between(gammas,lo,hi,color=color,alpha=.15,lw=0)
        for kind,c,ls,leg in [("actual","#222222","-","Actual: both change"),(variant,color,"--",label)]:
            vals=np.array([np.nan if v is None else v for v in data[f"steps_{kind}"]])
            ax.plot(gammas,vals,color=c,ls=ls,lw=1.8,marker=".",ms=3,label=leg)
            mask=(gammas>=2)&np.isnan(vals)
            ax.scatter(gammas[mask],np.full(mask.sum(),maximum),marker="x",color=c,s=20)
        ax.axvline(8,color=".5",ls=":",lw=.8)
        ax.set(yscale="log",ylim=(1e3,3e15),xlim=(2,128),xlabel=r"Slope $\gamma$",title=title)
        gamma_axis(ax,[2,4,8,16,32,64,128]); clean(ax)
    axes[0].set_ylabel("Steps to 1% relative L2 error")
    handles = [Line2D([],[],color="#222222",lw=1.8,label="Actual: both change"),
               Line2D([],[],color="#1768a5",ls="--",lw=1.8,label=r"Only $\lambda_i/\lambda_1$ changes"),
               Line2D([],[],color="#ce6518",ls="--",lw=1.8,label=r"Only $p_i$ changes")]
    fig.legend(handles=handles,loc="upper center",bbox_to_anchor=(.55,.995),ncol=3,frameon=False,fontsize=10)
    fig.subplots_adjust(left=.095,right=.99,bottom=.18,top=.78,wspace=.17)
    save(fig,"ratios_versus_target_weights.png")


def kernel_figure():
    original=np.load(SOURCE/"data/kernel_gamma_comparison.npz")
    x, centers = original["x"], original["centers"]
    gammas = np.array([1., 4., 8., 16.])
    matrices, audit = [], []
    for gamma in gammas:
        phi = np.column_stack((np.ones(len(x)), np.tanh(gamma*(x[:, None]-centers))))
        b = phi/np.sqrt(len(x))
        k = b@b.T
        direct = (1 + phi[:, 1:]@phi[:, 1:].T)/len(x)
        symmetry = float(np.max(np.abs(k-k.T)))
        normalization = float(np.max(np.abs(k-direct)))
        assert symmetry <= 1e-14 and normalization <= 1e-14
        matrices.append(k)
        audit.append(dict(gamma=float(gamma), symmetry_max_abs=symmetry,
                          explicit_one_over_m_formula_max_abs=normalization,
                          bias_column_squared_norm=float(b[:, 0]@b[:, 0])))
    matrices = np.asarray(matrices)
    np.savez_compressed(OUT/"kernel_gamma_comparison.npz", gammas=gammas,
                        x=x, centers=centers, kernels=matrices)
    (OUT/"kernel_audit.json").write_text(json.dumps(dict(N_intervals=128,
        tanh_neurons=len(centers), m=len(x), gamma=gammas.tolist(),
        construction="B=[1,tanh(gamma*(x-c))]/sqrt(m); K=B@B.T",
        color_scale="shared viridis across gamma=1,4,8,16", rows=audit), indent=2)+"\n")
    fig,axes=plt.subplots(1,4,figsize=(9.5,2.75),sharex=True,sharey=True)
    for ax,gamma,k in zip(axes,gammas,matrices):
        im=ax.imshow(k,cmap="viridis",vmin=matrices.min(),vmax=matrices.max(),origin="upper",
                     interpolation="nearest",extent=(-1,1,1,-1),aspect="equal")
        ax.set(title=rf"$\gamma={gamma:g}$",xlabel=r"Sample $x_b$",xticks=[-1,0,1],yticks=[-1,0,1])
    axes[0].set_ylabel(r"Sample $x_a$")
    fig.subplots_adjust(left=.07,right=.89,bottom=.21,top=.87,wspace=.14)
    cb=fig.add_axes([.918,.23,.014,.60])
    fig.colorbar(im,cax=cb,label=r"$(K_\gamma)_{ab}$")
    save(fig,"finite_kernels.png")


def manifest(steps,bands,transitions,audit):
    rows=[]
    for r,b in zip(steps["rows"],bands["rows"]):
        rows.append(dict(gamma=r["gamma"],actual_steps=r["finite_tanh_spectral_steps"],
                         predicted_steps=r["integral_quadrature"][-1]["steps"],
                         relative_prediction_error=r["integral_quadrature"][-1]["relative_step_error"],
                         explicit_readout_error=b["explicit_resolved_readout_relative_error"],
                         strongest_saved_necessary_step_bound=max(v["necessary_steps_to_one_percent"] for v in b["bands"])))
    (OUT/"figure_data.json").write_text(json.dumps(dict(step_rows=rows,audit=audit,
        step_geometry={k:steps[k] for k in ["N","W","m","halo_per_side","integration_bounds","target","learning_rate"]},
        doubling_geometries=transitions["geometries"],interval_certified=False,gd_executed=False),indent=2)+"\n")
    text="""# Note figure manifest

All PNGs use saved calculations except the kernel heatmap, which directly
recomputes B_gamma B_gamma^T at gamma=1,4,8,16 on the saved geometry.
No smoothing, fitting, new training, or change of experimental geometry was used. Thin joining lines
connect actual sample values. Source code is `note_figures.py`.

| Figure | What it establishes | What it does not establish |
|---|---|---|
| `ratios_versus_target_weights.png` | At 1% error in the default mixed-sine experiment, changing eigenvalue ratios while freezing rank-indexed target weights reproduces the main orders of magnitude in step-count improvement. | This is a rank-paired counterfactual, not eigenvector tracking or a theorem that target alignment is irrelevant. Shading represents unresolved-direction sensitivity, not a confidence interval. |
| `finite_kernels.png` | Actual finite kernels at gamma=1,4,8,16, on identical geometry and a shared viridis color scale. | Kernel images do not themselves establish improved normalized eigenvalues. |
| `doubling_ratios.png` | Direct ratios and lower bounds for gamma_a -> 2 gamma_a, at all four preselected width/rank configurations. Bottom row exposes looseness. | Not all lower bounds exceed starting ratios; the old-space restriction and denominator upper bound can be conservative. Corrections are measured using the new finite kernel. |
| `step_counts_and_capacity.png` | Finite-center integral spectra forecast actual finite-tanh spectral first-hit counts accurately, while explicit readouts already attain below 1% error. | These are not counts from executed training and are not an independent consequence of the ratio lower-bound theorem. No rigorous error enclosure is inferred from the observed prediction errors. |
| `predicted_learning_curves.png` | The entire finite-tanh and center-integral spectral curves agree closely until 1% acquisition. | Agreement does not certify unresolvable machine-precision tail dynamics. |

## Geometry and normalization

Default mixed-sine experiment: 128 interior intervals (h=1/64), 153 tanh
neurons plus a bias, 263 uniformly spaced samples on [-1,1], 12 halo centers
per side, center-cell interval [-1.1953125,1.1953125]. Target is
sin(2 pi x) + 0.5 sin(6 pi x) + 0.25 sin(10 pi x). B is divided by sqrt(m);
loss is half the mean squared error. Initial readout is zero. Both the finite
and center-integral predictions use eta=0.5/lambda_max(K_finite).

The width sweep's N denotes **interior intervals**, not total tanh neurons.
For N=64,128,256,512 the total neuron counts are 81,153,289,559 and m is
131,263,521,1031. Halo per side is ceil(sqrt(N)). Rank colors use
i=N/32,N/16,N/8,N/4. Caption this convention explicitly to avoid conflict
with theoretical notation in which N may denote total neurons.

## Step-count evidence

The center integral is evaluated with 16 Gauss-Legendre points per center
cell. Results agree closely with the independently saved 8-point calculation.
These approximate the finite center sum; they are not the collaborator's
periodic Fourier construction. Eigenvalues are obtained from feature SVDs,
which resolve small rates more accurately than diagonalizing an explicitly
formed Gram matrix. Error is evaluated by the exact fixed-matrix spectral
formula, with log1p used for tiny learning rates. Integer first-hit counts
are found by bracketing and binary search; no fitted trajectory rates occur.

| Gamma | Actual finite spectral steps | Integral prediction | Prediction error | Explicit readout error |
|---|---:|---:|---:|---:|
"""
    for r in rows:
        text+=f"| {r['gamma']:g} | {r['actual_steps']:,} | {r['predicted_steps']:,} | {100*r['relative_prediction_error']:+.5f}% | {r['explicit_readout_error']:.5e} |\n"
    text+="""
At gamma 4, a saved positive-rate band with eta*lambda in (1e-14,1e-11]
contains 0.02735465123 of target squared norm. It yields a necessary-step
bound of 280,573,583,534 for 1% error. This is separate from eigenvalue-ratio
LOWER bounds: a lower bound on a new eigenvalue does not yield a necessary
training delay. The saved explicit readout has norm 4625.807 and uses 30
resolved modes; its relative error is 5.85000788e-4. Main count is 1.0076e12.

## Numerical audit and limitations

An independent direct feature construction with the alternate `gesdd` SVD
recomputes all four finite spectral counts and all four explicit readout
errors. Results and relative differences are in `figure_data.json`. Every
resolved transition record was also checked for lower<=actual and arithmetic
consistency of its tightness/gain metrics. This audits the numbers; it is
not interval certification of rounding errors.

Unresolved starting eigenspaces are omitted from lower-bound curves. The
saved filter requires both the singular value and cutoff gap to exceed
10*eps*max(shape)*s1; additional compressed-matrix/correction consistency
checks are applied. Blank lower-bound regions are not zero bounds.

## Sources

- `../data/step_prediction_check.json` and `.npz`: step counts and learning curves.
- `../data/slow_band_check.json`: explicit readout errors and necessary-step bands.
- `../data/finite_ratio_transitions.json`: all doubling ratios and bounds.
- `../../data/default.json`: rank-paired target-weight controls, gamma0=8.
- `kernel_gamma_comparison.npz`: actual kernel matrices at gamma=1,4,8,16, recomputed from the sample/center arrays in `../data/kernel_gamma_comparison.npz`. The earlier file is unchanged.
- `kernel_audit.json`: direct symmetry and 1/m normalization checks for these four matrices.
"""
    (OUT/"MANIFEST.md").write_text(text)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,
                        "axes.labelsize":11,"axes.titlesize":11,"xtick.labelsize":9,
                        "ytick.labelsize":9,"axes.titlepad":10,"axes.linewidth":.7})
    steps=load("step_prediction_check.json")
    bands=load("slow_band_check.json")
    transitions=load("finite_ratio_transitions.json")
    with threadpool_limits(1):
        audit=independent_audit(steps,bands,transitions)
    step_figure(steps,bands)
    learning_curves(steps)
    doubling_figure(transitions)
    alignment_figure()
    kernel_figure()
    manifest(steps,bands,transitions,audit)
    print(json.dumps(audit,indent=2))
    print(f"Saved five figures, provenance memo, and audit data to {OUT}")


if __name__ == "__main__":
    main()
