"""Matched ordinary GD versus theta-gradient 1000 DF_tau + DG_tau."""
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

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from experiments.expD28_loss_gradient_decomposition import run as previous

HERE = Path(__file__).resolve().parent
RESULTS = ROOT / "results/checkpoint_D_optimizers/expD29_weighted_profile"


def config():
    return yaml.safe_load((HERE/"config.yaml").read_text())


def add_profile_gradient(parameters, gF, weight):
    """DL+(weight-1)DF = weight*DF+DG. Never modify readout gradients."""
    width = parameters["a"].numel()
    with torch.no_grad():
        parameters["a"].grad.add_(torch.as_tensor((weight-1)*gF[:width]))
        parameters["b"].grad.add_(torch.as_tensor((weight-1)*gF[width:]))


def train(target, arm, weight, cfg):
    x = previous.previous.midpoint_grid(cfg["n_train"])
    y = previous.previous.matched.target_values(target,x,cfg)
    initial = previous.initial_state(arm,cfg)
    p = {k:torch.nn.Parameter(torch.tensor(v,dtype=torch.float64)) for k,v in initial.items()}
    optimizer = torch.optim.SGD(p.values(),lr=cfg["learning_rate"])
    tx,ty = torch.tensor(x),torch.tensor(y)
    chosen = set(previous.snapshots(cfg["steps"],cfg["diagnostic_snapshots"]))
    audit_steps = {0,1,10,50,100,250,cfg["steps"]}
    chosen |= {s for s in audit_steps if s<=cfg["steps"]}
    history = []
    states = {k:[] for k in p}
    saved_steps, audits = [], []
    status = "complete"
    last_recorded_state = None
    for step in range(cfg["steps"]+1):
        optimizer.zero_grad(set_to_none=True)
        prediction = torch.tanh(tx[:,None]*p["a"]+p["b"])@p["v"][:-1]+p["v"][-1]
        loss = .5*torch.mean((prediction-ty).square())
        if not torch.isfinite(loss):
            status = f"nonfinite loss at step {step}"
            break
        loss.backward()
        state = {k:v.detach().numpy() for k,v in p.items()}
        ordinary = np.r_[p["a"].grad.numpy(),p["b"].grad.numpy()].copy()
        row = dict(step=step,L=float(loss.detach()),F=np.nan,G=np.nan,H=np.nan,
                   mean_gamma=float(np.mean(abs(state["a"]))),
                   gamma_displacement=float(np.mean(abs(abs(state["a"])-abs(initial["a"])))),
                   geometry_displacement=float(np.linalg.norm(np.r_[state["a"]-initial["a"],state["b"]-initial["b"]])),
                   readout_norm=float(np.linalg.norm(state["v"])),rank=np.nan,
                   profile_coefficient_norm=np.nan,gL_norm=float(np.linalg.norm(ordinary)),
                   gF_norm=np.nan,gG_norm=np.nan,update_gradient_norm=float(np.linalg.norm(ordinary)))
        diag = None
        if weight!=1 or step in chosen:
            try:
                diag = previous.one_diagnostic(state,x,y,cfg["readout_rcond"])
            except np.linalg.LinAlgError:
                status = f"SVD failed at step {step}"
                break
            np.testing.assert_allclose(ordinary,diag["gL"],rtol=1e-8,atol=2e-13)
            row.update(F=diag["F"],G=row["L"]-diag["F"],H=row["L"]+(weight-1)*diag["F"],
                       rank=diag["rank"],profile_coefficient_norm=diag["vstar_norm"],
                       gF_norm=float(np.linalg.norm(diag["gF"])),gG_norm=float(np.linalg.norm(ordinary-diag["gF"])))
            if weight!=1:
                add_profile_gradient(p,diag["gF"],weight)
                row["update_gradient_norm"] = float(np.linalg.norm(np.r_[p["a"].grad.numpy(),p["b"].grad.numpy()]))
        if not all(torch.isfinite(v.grad).all() for v in p.values()):
            status = f"nonfinite gradient at step {step}"
            break
        history.append(row)
        last_recorded_state = {k:state[k].copy() for k in p}
        if step in chosen:
            saved_steps.append(step)
            for k in p:
                states[k].append(state[k].copy())
        if step in audit_steps:
            alternatives = [previous.one_diagnostic(state,x,y,cfg["readout_rcond"],driver="gesdd"),
                            previous.one_diagnostic(state,x,y,cfg["readout_rcond"],backend="torch")]
            variation = max(np.linalg.norm(diag["gF"]-d["gF"]) for d in alternatives)
            ranks_agree = all(d["rank"]==diag["rank"] for d in alternatives)
            audits.append(dict(step=step,gF_variation=float(variation),F_variation=max(abs(diag["F"]-d["F"]) for d in alternatives),
                               ranks_agree=bool(ranks_agree),gF_resolved=bool(ranks_agree and row["gF_norm"]>cfg["resolution_factor"]*variation),
                               amplified_variation=float((weight-1)*variation),
                               update_norm=row["update_gradient_norm"]))
        if step in (0,100,250,cfg["steps"]):
            print(f"{arm}/{target}/weight={weight}, step {step}: L={row['L']:.5g}, F={row['F']:.5g}, mean gamma={row['mean_gamma']:.6g}",flush=True)
        if step<cfg["steps"]:
            optimizer.step()
    if not history:
        raise FloatingPointError(status)
    # Preserve the last valid endpoint if a larger weight stops between
    # scheduled snapshots. A stopped trajectory must not be labeled step 500.
    if saved_steps[-1] != history[-1]["step"]:
        saved_steps.append(history[-1]["step"])
        for k in p:
            states[k].append(last_recorded_state[k])
    case = {key:np.asarray([row[key] for row in history]) for key in history[0]}
    case.update({k:np.asarray(v) for k,v in states.items()})
    case.update(saved_steps=np.asarray(saved_steps),audits_json=np.array(json.dumps(audits)),
                target=np.array(target),arm=np.array(arm),weight=np.array(weight),status=np.array(status))
    return case


def save(case,cfg,path):
    path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,config_json=json.dumps(cfg),**case)


def load(path,cfg):
    with np.load(path,allow_pickle=False) as data:
        if json.loads(str(data["config_json"]))!=cfg:
            raise ValueError(f"Configuration mismatch: {path}")
        return {key:data[key] for key in data.files if key!="config_json"}


def plot(cases,cfg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import ScalarFormatter,LogLocator,MaxNLocator,NullFormatter

    figdir=RESULTS/"figures";figdir.mkdir(parents=True,exist_ok=True)
    colors={1:"#2868ae",1000:"#c45b21"}
    for arm in cfg["arms"]:
        fig,axes=plt.subplots(3,4,figsize=(18,11.2),dpi=160,sharex=True)
        for col,target in enumerate(cfg["targets"]):
            for weight in cfg["weights"]:
                c=cases[arm,target,weight]
                for row,key in enumerate(("L","F","mean_gamma")):
                    valid=np.isfinite(c[key])
                    if row<2: valid &= c[key]>0
                    if row==1:
                        valid &= np.isin(c["step"],cases[arm,target,1]["saved_steps"])
                    values=c[key] if row!=2 else c[key]-c[key][0]
                    axes[row,col].plot(c["step"][valid],values[valid],color=colors[weight],
                                       lw=2.5 if weight==1 else 1.8,ls="--" if weight==1000 else "-")
                if str(c["status"])!="complete":
                    axes[0,col].text(.97,.06,str(c["status"]),transform=axes[0,col].transAxes,
                                     ha="right",fontsize=9,color=colors[weight])
            baseline=cases[arm,target,1];modified=cases[arm,target,1000]
            axes[0,col].set_title(previous.LABELS[target],fontsize=14,pad=14)
            for row in (0,1):
                axes[row,col].set_yscale("log")
                low,high=axes[row,col].get_ylim()
                if high/low<10:
                    axes[row,col].yaxis.set_major_locator(MaxNLocator(nbins=4))
                    axes[row,col].yaxis.set_major_formatter(ScalarFormatter(useOffset=False,useMathText=True))
                    axes[row,col].yaxis.set_minor_formatter(NullFormatter())
                else:
                    axes[row,col].yaxis.set_major_locator(LogLocator(base=10,numticks=5))
            formatter=ScalarFormatter(useOffset=False,useMathText=True);formatter.set_powerlimits((-3,3))
            axes[2,col].yaxis.set_major_formatter(formatter)
            axes[2,col].set_xlabel("GD updates",fontsize=11)
            ranks=modified["rank"][np.isfinite(modified["rank"])]
            if ranks.min()!=ranks.max():
                axes[1,col].text(.97,.96,f"Retained rank: {int(ranks.min())} ↔ {int(ranks.max())}",
                                 transform=axes[1,col].transAxes,ha="right",va="top",fontsize=9,color=colors[1000])
            if arm=="qi_zero":
                axes[1,col].text(.5,.1,"Numerical floor: fluctuations are not\nresolved approximation improvement",
                                 transform=axes[1,col].transAxes,ha="center",fontsize=9,color=".35")
            for row in range(3):
                ax=axes[row,col]
                ax.set_xlim(0,cfg["steps"])
                ax.set_xticks([0,100,200,300,400,500])
                ax.grid(alpha=.18)
                ax.spines[["top","right"]].set_visible(False)
                ax.tick_params(labelsize=9)
                ax.margins(y=.18)
        axes[0,0].set_ylabel(r"Actual loss $L=\frac{1}{2}\,\mathrm{mean}(e^2)$",fontsize=12)
        axes[1,0].set_ylabel(r"Numerical approximation loss $F_\tau$",fontsize=12)
        axes[2,0].set_ylabel(r"Change in mean scale $\overline{\gamma}_t-\overline{\gamma}_0$",fontsize=12)
        fig.suptitle(previous.ARMS[arm]+"\nGive the approximation gradient 1,000× weight",fontsize=18,y=.985)
        fig.legend([Line2D([],[],color=colors[w],lw=2,ls="--" if w==1000 else "-") for w in cfg["weights"]],
                   [r"Ordinary GD: $DF_\tau+DG_\tau$",r"Weighted: $1000\,DF_\tau+DG_\tau$"],
                   loc="upper center",bbox_to_anchor=(.5,.91),ncol=2,frameon=False,fontsize=12)
        fig.subplots_adjust(left=.08,right=.98,top=.85,bottom=.135,hspace=.27,wspace=.28)
        fig.text(.5,.055,
                 "500 updates at η=0.002; 1,024 identical samples in [−1,1]; N=128, 177 neurons, seed 0. Each panel fits its own vertical range.\n"
                 f"Initial mean γ={cases[arm,cfg['targets'][0],1]['mean_gamma'][0]:.7g}. Fτ is plotted at identical saved steps in both runs; all weighted-step values are retained in data.\n"
                 "Both runs update the current readout by ordinary GD. Weighted geometry follows H=L+999Fτ; solved coefficients are never installed.\n"
                 "Fτ uses the same SVD cutoff 10⁻¹³σ₁ and retained-subspace derivative as the previous experiment. No clipping, retuning, or gradient gating.",
                 ha="center",va="center",fontsize=10)
        fig.savefig(figdir/f"{arm}.png")
        plt.close(fig)


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--plot-only",action="store_true")
    parser.add_argument("--arms",nargs="+");parser.add_argument("--targets",nargs="+")
    args=parser.parse_args();cfg=config()
    torch.set_default_dtype(torch.float64);torch.set_num_threads(cfg["threads"])
    cases={}
    with threadpool_limits(limits=cfg["threads"]):
        for arm in args.arms or cfg["arms"]:
            for target in args.targets or cfg["targets"]:
                for weight in cfg["weights"]:
                    path=RESULTS/"data"/f"{arm}__{target}__{weight}.npz"
                    if path.exists(): c=load(path,cfg)
                    elif args.plot_only: continue
                    else:
                        start=time.perf_counter();c=train(target,arm,weight,cfg)
                        c["seconds"]=np.array(time.perf_counter()-start);save(c,cfg,path)
                    cases[arm,target,weight]=c
        if len(cases)==len(cfg["arms"])*len(cfg["targets"])*len(cfg["weights"]):plot(cases,cfg)
        else:print("Subset saved.",flush=True)


if __name__=="__main__": main()
