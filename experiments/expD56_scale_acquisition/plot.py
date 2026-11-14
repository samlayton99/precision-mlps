"""Paper-style figures from measured checkpoints or explicitly synthetic previews."""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/precisionmlps-mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, LogLocator, MaxNLocator, NullFormatter, NullLocator
import numpy as np
import yaml

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.expD56_scale_acquisition.model import TARGETS, initial_parameters, midpoint_grid, target

HERE=Path(__file__).resolve().parent
OUT=ROOT/"results/checkpoint_D_optimizers/expD56_scale_acquisition"
# Learning-rate families; each function gets a consistent shade within its family.
COLORS={.01:np.array(matplotlib.colors.to_rgba("#0072B2")),
        .001:np.array(matplotlib.colors.to_rgba("#E69F00")),
        .0001:np.array(matplotlib.colors.to_rgba("#A04DA8"))}


def function_shades(lr, count=16):
    base=COLORS[lr][:3]
    light=.35*base+.65*np.ones(3)
    dark=.85*base
    return np.array([np.r_[(1-t)*light+t*dark,1.] for t in np.linspace(0,1,count)])


def apply_style():
    plt.rcParams.update({"font.family":"DejaVu Sans", "mathtext.fontset":"dejavusans",
                         "font.size":7, "axes.titlesize":9.2, "axes.labelsize":8,
                         "xtick.labelsize":7, "ytick.labelsize":7,
                         "pdf.fonttype":42, "ps.fonttype":42})


def style_axis(ax):
    for spine in ("top","right"): ax.spines[spine].set_visible(False)
    for spine in ("left","bottom"): ax.spines[spine].set_linewidth(.9)
    ax.tick_params(which="major",length=3,width=.8,pad=3)
    ax.tick_params(which="minor",length=1.5,width=.4)
    ax.set_axisbelow(True)
    ax.grid(axis="y",which="major",color="#D8D8D8",linewidth=.45)


def cohort_summary(curves, allow_partial=False):
    """Equal function weighting, not neuron pooling or confidence intervals."""
    array=np.asarray(curves)
    if array.ndim!=2 or (array.shape[0]<2 if allow_partial else array.shape[0]!=16) or not np.all(np.isfinite(array)):
        raise ValueError("Require 16 finite trajectories, or at least two for an explicit partial cohort")
    return np.median(array,axis=0), *np.quantile(array,[.025,.975],axis=0)


def neuron_slope_summary(snapshots, width):
    """Within-run absolute-slope distribution, independently at each checkpoint."""
    snapshots=np.asarray(snapshots)
    if snapshots.ndim!=2 or snapshots.shape[1]!=3*width+1:
        raise ValueError("Expected one full parameter vector per checkpoint")
    slopes=np.abs(snapshots[:,:width])
    low,median,high=np.quantile(slopes,[.01,.5,.99],axis=1)
    return {"gamma_neuron_median":median,"gamma_neuron_p01":low,"gamma_neuron_p99":high}


def dummy_data(config):
    """Synthetic layout-only curves. Never written into the empirical runs directory."""
    rng=np.random.default_rng(56030)
    steps=np.unique(np.r_[0,np.geomspace(1,config["steps"],180).astype(int),np.linspace(0,config["steps"],301).astype(int)])
    out={}
    names=[n for group in config["targets"].values() for n in group]
    traits=rng.uniform(.08,3.3,16)
    for multiplier in config["initialization_multipliers"]:
        theta=initial_parameters(config["width"],config["seed"],multiplier).numpy()
        g0=np.sqrt(np.mean(theta[:config["width"]]**2))
        for lr in config["learning_rates"]:
            t=steps/config["steps"]
            for i,name in enumerate(names):
                activity=(lr/.001)**.25*traits[i]/np.sqrt(multiplier)
                rise=(.2*(1-np.exp(-40*t))+.8*np.log1p(5*t)/np.log(6))
                gamma=g0*(1+activity*rise)
                error=.02+.95*np.exp(-(1+activity)*np.log1p(100*t))
                low,median,high=np.quantile(np.abs(theta[:config["width"]]),[.01,.5,.99])
                out[(multiplier,lr,name)]={"step":steps,"gamma_rms":gamma,"eval_rel_l2":error,
                    "gamma_neuron_median":median*gamma/g0,"gamma_neuron_p01":low*gamma/g0,
                    "gamma_neuron_p99":high*gamma/g0}
    return out, {}


def measured_data(results):
    data,provenance={},{}
    for path in sorted((results/"runs").glob("*/checkpoint.npz")):
        status=json.loads((path.parent/"status.json").read_text())
        contents=path.read_bytes()
        with np.load(io.BytesIO(contents)) as saved:
            keys=saved["keys"].tolist()
            history=saved["history"]
            rows={key:history[:,i] for i,key in enumerate(keys)}
            snapshots=saved["snapshots"]
            if len(snapshots)!=len(history): raise ValueError("Parameter/history checkpoint mismatch")
            rows.update(neuron_slope_summary(snapshots,status["width"]))
        data[(status["initialization_multiplier"],status["learning_rate"],status["target"])]=rows
        provenance[str(path.relative_to(ROOT))]=hashlib.sha256(contents).hexdigest()
    return data,provenance


def common_curves(data, config, multiplier, lr, field, names=None):
    if names is None: names=[n for group in config["targets"].values() for n in group]
    if not names: return None
    rows=[data.get((multiplier,lr,n)) for n in names]
    if any(row is None for row in rows): return None
    # A complete cohort over a common observed interval. Never extrapolate missing runs.
    end=min(row["step"][-1] for row in rows)
    steps=rows[0]["step"][rows[0]["step"]<=end]
    curves=np.array([np.interp(steps,row["step"],row[field]) for row in rows])
    return steps,curves


def aggregate_figure(config,data,destination,dummy,field,partial=False,slope_scale="linear"):
    fig,axes=plt.subplots(1,3,figsize=(9.4,3.25),sharex=True,sharey=True)
    fig.subplots_adjust(left=.075,right=.985,bottom=.24,top=.78,wspace=.20)
    distribution=field=="gamma_distribution"
    gamma=field in ("gamma_rms","gamma_distribution")
    reference=config["quill_reference"]["lambda"]*config["quill_reference"]["interior_intervals"]/2
    titles=["a) Default Initialization", "b) 2x Initialization", "c) 4x Initialization"]
    all_values=[]
    aggregate={}
    names=[n for group in config["targets"].values() for n in group]
    cohort_names=names
    if partial:
        cohort_names=[name for name in names if all(
            (multiplier,lr,name) in data and data[(multiplier,lr,name)]["step"][-1]>=config["steps"]
            for multiplier in config["initialization_multipliers"] for lr in config["learning_rates"])]
    individual_records={}
    for ax,multiplier,title in zip(axes,config["initialization_multipliers"],titles):
        style_axis(ax)
        ax.set_title(title,fontweight="bold",pad=7)
        plotted=0
        final_medians=[]
        for lr in config["learning_rates"]:
            color=COLORS[lr]
            for name,shade in zip(names,function_shades(lr,len(names))):
                row=data.get((multiplier,lr,name))
                if row is None: continue
                if distribution:
                    median,lo,hi=(row[k] for k in ("gamma_neuron_median","gamma_neuron_p01","gamma_neuron_p99"))
                    ax.fill_between(row["step"],lo,hi,color=shade,alpha=.025,lw=0,zorder=1)
                    ax.plot(row["step"],hi,color=shade,alpha=.65,lw=.45,ls=(0,(1,3)),dash_capstyle="round",zorder=2)
                    ax.plot(row["step"],lo,color=shade,alpha=.65,lw=.45,ls=(0,(1,3)),dash_capstyle="round",zorder=2)
                    ax.plot(row["step"],median,color=shade,alpha=1.,lw=.60,zorder=3)
                    if row["step"][-1]>=config["steps"]:
                        final_medians.append(float(median[-1]))
                    all_values.extend(lo.tolist()+hi.tolist())
                    individual_records[f"init{multiplier:g}/lr{lr:.0e}/{name}"]={
                        "step":row["step"].tolist(),"median":median.tolist(),
                        "p1":lo.tolist(),"p99":hi.tolist(),"neurons":config["width"]}
                    plotted+=1
                    continue
                ax.plot(row["step"],row[field],color=shade,alpha=.40,lw=.35,zorder=2)
                all_values.extend(row[field].tolist())
                individual_records[f"init{multiplier:g}/lr{lr:.0e}/{name}"]={
                    "step":row["step"].tolist(),"values":row[field].tolist()}
                plotted+=1
            if distribution: continue
            cohort=common_curves(data,config,multiplier,lr,field,names=cohort_names)
            if cohort is None or len(cohort_names)<2: continue
            steps,curves=cohort
            if not np.all(np.isfinite(curves)):
                ax.text(.03,.05,"Nonfinite diagnostics: inspect run status",transform=ax.transAxes,fontsize=6)
                continue
            median,lo,hi=cohort_summary(curves,allow_partial=partial or len(cohort_names)!=16)
            ax.fill_between(steps,lo,hi,color=color,alpha=.16,lw=0,zorder=1)
            ax.plot(steps,median,color=color,alpha=1.,lw=.60,zorder=3)
            aggregate[f"init{multiplier:g}/lr{lr:.0e}"]={"step":steps.tolist(),"median":median.tolist(),
                                                       "p2.5":lo.tolist(),"p97.5":hi.tolist(),
                                                       "functions":len(cohort_names),"target_names":cohort_names}
        if gamma: ax.axhline(reference,color="#333333",ls="--",lw=1.25,zorder=4)
        if distribution and len(final_medians)==len(names)*len(config["learning_rates"]):
            panel_median=float(np.median(final_medians))
            ax.annotate(f"Median: {panel_median:.2f}",xy=(.97,panel_median),
                        xycoords=("axes fraction","data"),xytext=(0,2.5),textcoords="offset points",
                        ha="right",va="bottom",fontsize=7,color="#222222",zorder=5,
                        bbox={"facecolor":"white","edgecolor":"none","alpha":.85,"pad":1.0})
        if not plotted: ax.text(.5,.5,"Waiting for all 16 functions",ha="center",transform=ax.transAxes,color="#777777")
        shown_steps=config.get("plot_end_step",config["steps"])
        ax.set(xscale="linear",xlim=(0,shown_steps),yscale=slope_scale if gamma else "log",
               xlabel=config.get("optimizer_label","GD")+" step")
        ax.set_xticks(np.linspace(0,shown_steps,5))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x,_:f"{x:g}" if x<1000 else f"{x/1000:g}k"))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.tick_params(labelleft=True)
        if gamma and slope_scale=="log":
            subs=(1.,2.,5.) if distribution else (1.,1.5,2.,3.,5.,7.5)
            ax.yaxis.set_major_locator(LogLocator(base=10,subs=subs,numticks=100))
            ax.yaxis.set_major_formatter(FuncFormatter(lambda y,_:f"{y:g}"))
            ax.yaxis.set_minor_locator(NullLocator())
        elif gamma:
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5,steps=[1,2,2.5,5,10]))
            ax.yaxis.set_minor_locator(NullLocator())
        else:
            ax.yaxis.set_major_locator(LogLocator(base=10,numticks=8))
            ax.yaxis.set_minor_locator(LogLocator(base=10,subs=(2.,5.),numticks=50))
            ax.yaxis.set_minor_formatter(NullFormatter())
    if gamma:
        ymax=max([20.,reference*1.12]+[v*1.08 for v in all_values if np.isfinite(v)])
        if slope_scale=="log":
            positives=[v for v in all_values if np.isfinite(v) and v>0]
            axes[0].set_ylim(min([.45]+[v*.9 for v in positives]),max(24.,ymax))
        else:
            axes[0].set_ylim(0,ymax)
        axes[0].set_ylabel(r"$\gamma_i$ distribution (98% interval)" if distribution else r"RMS slope $\gamma_i$")
        axes[-1].annotate(rf"QUILL: $\gamma = {reference:g}$",xy=(.98,reference),
                          xycoords=("axes fraction","data"),xytext=(0,5),
                          textcoords="offset points",ha="right",fontsize=7,color="#444444")
    else:
        vals=[v for v in all_values if np.isfinite(v) and v>0]
        axes[0].set_ylim(min([1e-4]+vals)*.7,max([1.2]+vals)*1.2)
        axes[0].set_ylabel(r"Relative $L^2$ error")
    title=config.get("figure_title","Scale Acquisition Ablation: 16 functions") if gamma else "Fitting error ablation"
    fig.suptitle(title,x=.53,y=.96,fontsize=12,fontweight="bold")
    handles=[Line2D([],[],color=COLORS[lr],lw=.60,label=rf"$\eta=10^{{{int(np.log10(lr))}}}$")
             for lr in config["learning_rates"]]
    if distribution and config.get("show_median_legend",len(names)!=16):
        handles += [Line2D([],[],color="#444444",lw=.60,label="Median")]
    if gamma: handles += [Line2D([],[],color="#333333",ls="--",lw=1.25,label="QUILL")]
    fig.legend(handles=handles,loc="lower center",bbox_to_anchor=(.53,.045),ncol=len(handles),frameon=False,
               fontsize=7,handlelength=1.6,handletextpad=.45,columnspacing=1.35)
    if dummy:
        fig.text(.985,.01,"Dummy data",ha="right",fontsize=6.5,color="#777777")
    elif partial:
        expected=len(names)*len(config["initialization_multipliers"])*len(config["learning_rates"])
        label=f"Partial data · {len(data)}/{expected} runs available" if distribution else f"Partial data · bands: {len(cohort_names)}/{len(names)} complete functions"
        if distribution and "plot_end_step" in config:
            label=f"Partial data · {config['plot_end_step']/1000:g}k / {config['steps']/1000:g}k steps"
        fig.text(.985,.01,label,
                 ha="right",fontsize=6.5,color="#777777")
    name="scale_acquisition" if gamma else "fitting_error"
    if dummy: name+="_DUMMY"
    elif partial: name+="_PARTIAL"
    fig.savefig(destination/f"{name}.png",dpi=300,bbox_inches="tight",pad_inches=.12)
    fig.savefig(destination/f"{name}.pdf",bbox_inches="tight",pad_inches=.12)
    plt.close(fig)
    (destination/f"{name}_displayed.json").write_text(json.dumps({"synthetic":dummy,"partial":partial,
        "field":field,"distribution_axis":"neurons within each function/run" if distribution else "functions",
        "absolute_slopes":distribution,"curves":aggregate,"individual_trajectories":individual_records})+"\n")


def target_gallery(config,destination):
    x=midpoint_grid(config["eval_points"])
    xt=midpoint_grid(config["train_points"])
    names=[n for group in config["targets"].values() for n in group]
    colors={"easy":"#0072B2","medium":"#B37600","hard":"#D55E00"}
    fig,axes=plt.subplots(4,4,figsize=(10.1,7.4),sharex=True)
    fig.subplots_adjust(top=.87,bottom=.09,left=.06,right=.98,hspace=.67,wspace=.30)
    for ax,name in zip(axes.flat,names):
        difficulty,label,_=TARGETS[name]
        y=target(name,x)/np.sqrt(np.mean(target(name,xt)**2))
        style_axis(ax)
        ax.plot(x,y,color=colors[difficulty],lw=1.3)
        ax.axhline(0,color="#aaaaaa",lw=.4,zorder=0)
        ax.set_title(label,fontsize=9,pad=5)
        ax.text(.02,.93,difficulty,transform=ax.transAxes,fontsize=6.5,color=colors[difficulty],va="top")
        ax.set_xlim(-1,1)
        ax.set_xticks([-1,0,1])
        low,high=np.min(y),np.max(y)
        ax.set_ylim(low-.12*(high-low),high+.25*(high-low))
    fig.suptitle("The 16 target functions",x=.52,y=.97,fontsize=14,fontweight="bold")
    fig.text(.52,.925,"Unit training-grid RMS on [−1, 1]; no mean subtraction. Difficulty groups are provisional.",
             ha="center",fontsize=8,color="#555555")
    fig.text(.52,.025,"Actual target functions—not dummy training results.  Pₖ denotes the degree-k Legendre polynomial.",
             ha="center",fontsize=8,color="#555555")
    fig.savefig(destination/"target_functions.png",dpi=250,bbox_inches="tight",pad_inches=.12)
    fig.savefig(destination/"target_functions.pdf",bbox_inches="tight",pad_inches=.12)
    plt.close(fig)


def individual_figures(config,data,destination):
    names=[n for group in config["targets"].values() for n in group]
    reference=config["quill_reference"]["lambda"]*config["quill_reference"]["interior_intervals"]/2
    for multiplier in config["initialization_multipliers"]:
        peaks=[float(np.max(row["gamma_neuron_p99"])) for (scale,_,_),row in data.items()
               if scale==multiplier and np.all(np.isfinite(row["gamma_neuron_p99"]))]
        upper=max([20.]+[peak*1.08 for peak in peaks])
        fig,axes=plt.subplots(4,4,figsize=(10.1,7.1),sharex=True,sharey=True)
        fig.subplots_adjust(left=.07,right=.98,bottom=.12,top=.88,hspace=.46,wspace=.20)
        for ax,name in zip(axes.flat,names):
            style_axis(ax)
            for lr in config["learning_rates"]:
                row=data.get((multiplier,lr,name))
                if row is not None:
                    ax.fill_between(row["step"],row["gamma_neuron_p01"],row["gamma_neuron_p99"],
                                    color=COLORS[lr],alpha=.12,lw=0)
                    ax.plot(row["step"],row["gamma_neuron_median"],color=COLORS[lr],lw=.60)
            ax.axhline(reference,color="#444444",ls="--",lw=.8)
            ax.set_title(TARGETS[name][1],fontsize=8,pad=5)
            ax.set(yscale="linear",ylim=(0,upper),xscale="linear",xlim=(0,config["steps"]))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda x,_:f"{x:g}" if x<1000 else f"{x/1000:g}k"))
            ax.set_xticks([0,config["steps"]/2,config["steps"]])
            ax.xaxis.set_minor_locator(NullLocator())
        fig.suptitle(f"Slope distributions · first-layer initialization ×{multiplier:g}",fontsize=12,fontweight="bold")
        handles=[Line2D([],[],color=COLORS[lr],lw=1.5,label=rf"$\eta={lr:g}$") for lr in config["learning_rates"]]
        handles.append(Line2D([],[],color="#444444",ls="--",label="QUILL reference"))
        fig.legend(handles=handles,loc="lower center",bbox_to_anchor=(.52,.025),ncol=4,frameon=False)
        fig.savefig(destination/f"per_function_init{multiplier:g}.png",dpi=250,bbox_inches="tight")
        plt.close(fig)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--dummy",action="store_true")
    parser.add_argument("--slope-scale",choices=("linear","log"),default="linear")
    parser.add_argument("--partial",action="store_true",help="Show available data; aggregate only functions completed across every setting")
    parser.add_argument("--results",type=Path,default=OUT/"sweep")
    args=parser.parse_args()
    if args.dummy and args.partial: parser.error("Choose dummy or partial, not both")
    args.results=args.results.resolve()
    apply_style()
    if args.dummy:
        config=yaml.safe_load((HERE/"config.yaml").read_text())
        data,provenance=dummy_data(config)
        destination=OUT/"preview"
    else:
        config=json.loads((args.results/"manifest.json").read_text())["config"]
        data,provenance=measured_data(args.results)
        destination=args.results/"figures"
        if args.partial: destination=destination/"partial"
    if args.slope_scale=="log": destination=destination/"log"
    destination.mkdir(parents=True,exist_ok=True)
    for field in ("gamma_distribution","eval_rel_l2"):
        aggregate_figure(config,data,destination,args.dummy,field,args.partial,args.slope_scale)
    target_gallery(config,destination)
    if not args.dummy: individual_figures(config,data,destination)
    (destination/"figure_provenance.json").write_text(json.dumps({"synthetic":args.dummy,"partial":args.partial,
        "source_sha256":provenance,"renderer_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "style_reference":"expD55_qi_adam_drift: DejaVu Sans, learning-rate colors, thin axes, bottom legend",
        "scale_figure_title":"Scale Acquisition Ablation: 16 functions","slope_y_axis":args.slope_scale+"; gamma_i distribution (98% interval)",
        "step_x_axis":"linear, from step zero",
        "rate_colors":{str(lr):color.tolist() for lr,color in COLORS.items()},
        "function_shades":{str(lr):function_shades(lr).tolist() for lr in COLORS},
        "interval_alpha":.16,"individual_alpha":.40,"individual_linewidth":.35,
        "median_alpha":1.,"median_linewidth":.60,
        "slope_distribution":{"within":"196 neurons per function/run", "quantity":"absolute first-layer weights",
            "line":"median", "band":"1st to 99th percentile", "band_alpha":.025,
            "across_function_aggregation":False},
        "aggregation":"Median and linearly interpolated 2.5th/97.5th percentiles across equally weighted functions; partial mode restricts the cohort to functions complete in all nine settings.",
        "smoothing":"none","seed_count":1},indent=2)+"\n")
    print(destination)


if __name__=="__main__": main()
