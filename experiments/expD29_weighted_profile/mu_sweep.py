"""Sweep mu in theta <- theta - eta(mu DF_tau + DG_tau).

Reuse the original mu=1 and mu=1000 runs without copying their data. New
weights use the identical training function, initial states, and base rate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from threadpoolctl import threadpool_limits
import yaml

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from experiments.expD29_weighted_profile import run as base

RESULTS=base.RESULTS/"mu_sweep"


def config():
    extra=yaml.safe_load((Path(__file__).parent/"mu_sweep.yaml").read_text())
    return base.config()|dict(weights=extra["mu_values"])


def case_path(arm,target,mu):
    name=f"{arm}__{target}__{mu}.npz"
    original=base.RESULTS/"data"/name
    return original if original.exists() and mu in base.config()["weights"] else RESULTS/"data"/name


def read_case(path,cfg):
    original=path.parent==base.RESULTS/"data"
    stored_cfg=base.config() if original else cfg
    # The only allowed difference in the reused cases is the sweep membership.
    assert {k:v for k,v in stored_cfg.items() if k!="weights"}=={k:v for k,v in cfg.items() if k!="weights"}
    return base.load(path,stored_cfg)


def audit(cases,cfg):
    e=base.previous
    x=e.previous.midpoint_grid(cfg["n_train"])
    xe=e.previous.midpoint_grid(cfg["n_eval"])
    summary=[];cutoffs=[]
    for (arm,target,mu),c in cases.items():
        reference=cases[arm,target,1]
        for key in ("a","b","v"):
            np.testing.assert_array_equal(c[key][0],reference[key][0])
        assert int(c["saved_steps"][-1])==int(c["step"][-1]),"Terminal state does not match recorded endpoint"
        state={k:c[k][-1] for k in ("a","b","v")}
        y=e.previous.matched.target_values(target,x,cfg)
        ye=e.previous.matched.target_values(target,xe,cfg)
        features=e.hidden(state["a"],state["b"],x)
        eval_features=e.hidden(state["a"],state["b"],xe)
        A=np.c_[features,np.ones(len(x))]/np.sqrt(len(x))
        yn=y/np.sqrt(len(x))
        actual=eval_features@state["v"][:-1]+state["v"][-1]
        primary=None
        for cutoff in (1e-12,1e-13,1e-14):
            profile=e.profiled_matrix(A,yn,cutoff)
            fitted=eval_features@profile["vstar"][:-1]+profile["vstar"][-1]
            record=dict(arm=arm,target=target,mu=mu,step=int(c["step"][-1]),rcond=cutoff,
                        F=profile["F"],rank=profile["rank"],coefficient_norm=float(np.linalg.norm(profile["vstar"])),
                        refit_eval_relative_l2=float(np.linalg.norm(fitted-ye)/np.linalg.norm(ye)))
            cutoffs.append(record)
            if cutoff==cfg["readout_rcond"]: primary=record
        sensitive=json.loads(str(c["audits_json"]))
        checked=[r for r in sensitive if r["update_norm"]>0]
        row=dict(arm=arm,target=target,mu=mu,status=str(c["status"]),step=int(c["step"][-1]),
                 L=float(c["L"][-1]),F=float(c["F"][-1]),G=float(c["G"][-1]),
                 actual_eval_relative_l2=float(np.linalg.norm(actual-ye)/np.linalg.norm(ye)),
                 refit_eval_relative_l2=primary["refit_eval_relative_l2"],
                 coefficient_norm=primary["coefficient_norm"],rank=primary["rank"],
                 gamma_initial=float(c["mean_gamma"][0]),gamma_final=float(c["mean_gamma"][-1]),
                 gamma_max=float(np.max(abs(state["a"]))),gamma_displacement=float(c["gamma_displacement"][-1]),
                 geometry_displacement=float(c["geometry_displacement"][-1]),
                 unresolved_audits=sum(not r["gF_resolved"] for r in sensitive),audits=len(sensitive),
                 maximum_relative_update_variation=max((r["amplified_variation"]/r["update_norm"] for r in checked),default=0.),
                 seconds=float(c["seconds"]))
        summary.append(row)
    (RESULTS/"data").mkdir(parents=True,exist_ok=True)
    (RESULTS/"data"/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    (RESULTS/"data"/"terminal_cutoff_audit.json").write_text(json.dumps(cutoffs,indent=2)+"\n")
    return summary


def plot(cases,cfg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import ScalarFormatter,LogLocator,MaxNLocator,NullFormatter

    figdir=RESULTS/"figures";figdir.mkdir(parents=True,exist_ok=True)
    mus=cfg["weights"]
    colors=dict(zip(mus,plt.cm.viridis(np.linspace(.03,.9,len(mus)))))
    for arm in cfg["arms"]:
        fig,axes=plt.subplots(3,4,figsize=(18.5,11.4),dpi=160,sharex=True)
        # Use a signed log gamma-change axis only when the sweep produces a
        # large range, preserving the sign and revealing smaller mu curves.
        peaks=[float(np.max(abs(cases[arm,t,mu]["mean_gamma"]-cases[arm,t,mu]["mean_gamma"][0])))
               for t in cfg["targets"] for mu in mus]
        positive=np.array([p for p in peaks if p>1e-14])
        signed_log=len(positive)>0 and positive.max()/positive.min()>50
        linear_threshold=float(positive.min()/10) if signed_log else None
        for col,target in enumerate(cfg["targets"]):
            baseline=cases[arm,target,1]
            for mu in mus:
                c=cases[arm,target,mu]
                for row,key in enumerate(("L","F","mean_gamma")):
                    valid=np.isfinite(c[key])
                    if row<2:valid &= c[key]>0
                    if row==1:
                        valid &= np.isin(c["step"],baseline["saved_steps"]) | (c["step"]==c["step"][-1])
                    values=c[key] if row!=2 else c[key]-c[key][0]
                    axes[row,col].plot(c["step"][valid],values[valid],color=colors[mu],
                                       lw=2 if mu==1 else 1.6,ls="--" if mu==1 else "-")
                    if str(c["status"])!="complete" and np.any(valid):
                        last=np.flatnonzero(valid)[-1]
                        axes[row,col].plot(c["step"][last],values[last],"x",color=colors[mu],ms=7)
            axes[0,col].set_title(base.previous.LABELS[target],fontsize=14,pad=14)
            for row in (0,1):
                ax=axes[row,col];ax.set_yscale("log")
                low,high=ax.get_ylim()
                if high/low<10:
                    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
                    ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False,useMathText=True))
                    ax.yaxis.set_minor_formatter(NullFormatter())
                else:ax.yaxis.set_major_locator(LogLocator(base=10,numticks=5))
            if signed_log:
                axes[2,col].set_yscale("symlog",linthresh=linear_threshold,linscale=.7)
            else:
                formatter=ScalarFormatter(useOffset=False,useMathText=True);formatter.set_powerlimits((-3,3))
                axes[2,col].yaxis.set_major_formatter(formatter)
            axes[2,col].set_xlabel("GD updates",fontsize=11)
            if arm=="qi_zero":
                axes[1,col].text(.5,.08,"Numerical-floor fluctuations",transform=axes[1,col].transAxes,
                                 ha="center",fontsize=9,color=".35")
            for row in range(3):
                ax=axes[row,col];ax.set_xlim(0,cfg["steps"]);ax.set_xticks([0,100,200,300,400,500])
                ax.grid(alpha=.17);ax.spines[["top","right"]].set_visible(False);ax.tick_params(labelsize=9)
                ax.margins(y=.14)
                if row==2 and signed_log:
                    low,high=ax.get_ylim()
                    powers=10.**np.arange(np.ceil(np.log10(linear_threshold)),
                                          np.ceil(np.log10(max(abs(low),abs(high))))+1)
                    ticks=sorted([-v for v in powers if low<=-v<=high]+[0.]+[v for v in powers if low<=v<=high])
                    ax.set_yticks(ticks)
        axes[0,0].set_ylabel("Actual loss $L$\n(log scale)",fontsize=12)
        axes[1,0].set_ylabel("Refitted loss $F_\\tau$\n(log scale)",fontsize=12)
        axes[2,0].set_ylabel("Mean gamma change\n"+r"$\overline{\gamma}_t-\overline{\gamma}_0$"+
                             ("\n(signed log scale)" if signed_log else ""),fontsize=12)
        fig.suptitle(base.previous.ARMS[arm]+"\nSweep the approximation-gradient weight μ",fontsize=18,y=.985)
        labels=[r"$\mu=1$ (ordinary GD)" if mu==1 else rf"$\mu=10^{{{int(np.log10(mu))}}}$" for mu in mus]
        fig.legend([Line2D([],[],color=colors[mu],lw=2,ls="--" if mu==1 else "-") for mu in mus],labels,
                   loc="upper center",bbox_to_anchor=(.5,.914),ncol=len(mus),frameon=False,fontsize=12)
        fig.subplots_adjust(left=.085,right=.98,top=.845,bottom=.14,hspace=.27,wspace=.32)
        initial=cases[arm,cfg["targets"][0],1]["mean_gamma"][0]
        scale_note=f"Gamma-change axes: signed log, linear within ±{linear_threshold:.2g}." if signed_log else "Gamma-change axes are linear."
        fig.text(.5,.058,
                 "Geometry step: −η(μ DFτ + DGτ); readout remains ordinary GD. η=0.002; 500 updates; all other settings matched.\n"
                 f"Mean initial γ={initial:.7g}. Each panel fits its own vertical range. {scale_note}\n"
                 "Fτ uses the existing cutoff 10⁻¹³σ₁ and its local derivative; cutoff crossings and numerical uncertainty remain relevant.\n"
                 "Fτ curves share diagnostic steps; all weighted-step values are saved. An × marks a stopped run's last valid point. No clipping or rate retuning.",
                 ha="center",va="center",fontsize=9.5)
        fig.savefig(figdir/f"{arm}.png");plt.close(fig)


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--plot-only",action="store_true")
    parser.add_argument("--mu",nargs="+",type=int);parser.add_argument("--arms",nargs="+")
    parser.add_argument("--targets",nargs="+");args=parser.parse_args();cfg=config()
    torch.set_default_dtype(torch.float64);torch.set_num_threads(cfg["threads"])
    cases={}
    with threadpool_limits(limits=cfg["threads"]):
        for arm in args.arms or cfg["arms"]:
            for target in args.targets or cfg["targets"]:
                for mu in args.mu or cfg["weights"]:
                    path=case_path(arm,target,mu)
                    if path.exists():c=read_case(path,cfg)
                    elif args.plot_only:continue
                    else:
                        start=time.perf_counter();c=base.train(target,arm,mu,cfg)
                        c["seconds"]=np.array(time.perf_counter()-start);base.save(c,cfg,path)
                    cases[arm,target,mu]=c
        if len(cases)==len(cfg["arms"])*len(cfg["targets"])*len(cfg["weights"]):
            if not args.plot_only:audit(cases,cfg)
            plot(cases,cfg)
        else:print(f"Subset available: {len(cases)} cases.",flush=True)


if __name__=="__main__":main()
