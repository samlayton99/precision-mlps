"""Small paired-nudge diagnostic: 128 neurons per state in two regimes.

At each joint-GD state, independently step geometry with readout frozen and
readout with geometry frozen. Plot signed changes of parameter magnitudes:
X=|c| (|a_geometry_next|-|a|), Y=|a| (|c_readout_next|-|c|).
The same-state one-step gradients equal the respective joint-GD gradients.
"""
from pathlib import Path
import argparse
import json
import sys

import numpy as np
import torch
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from experiments.expD26_freeze_and_readout_spectrum import freeze as base
from experiments.expD26_freeze_and_readout_spectrum import late_freeze as late

RESULTS=late.RESULTS
KEYS=("a","b","v")


def independent_fork(state,x,y,eta,frozen_keys):
    p={k:torch.nn.Parameter(torch.tensor(state[k]),requires_grad=k not in frozen_keys) for k in KEYS}
    optimizer=torch.optim.SGD([v for v in p.values() if v.requires_grad],lr=eta)
    prediction=torch.tanh(x[:,None]*p["a"]+p["b"])@p["v"][:-1]+p["v"][-1]
    (.5*(prediction-y).square().mean()).backward()
    optimizer.step()
    return {k:v.detach().numpy().copy() for k,v in p.items()}


def fit(x,y):
    x,y=x.ravel(),y.ravel()
    xc,yc=x-x.mean(),y-y.mean()
    slope=float((xc@yc)/(xc@xc))
    intercept=float(y.mean()-slope*x.mean())
    r=float((xc@yc)/np.sqrt((xc@xc)*(yc@yc)))
    residual=y-(slope*x+intercept)
    r2=float(1-(residual@residual)/(yc@yc))
    return dict(slope=slope,intercept=intercept,pearson_r=r,R_squared=r2,
                direction_agreement=float(np.mean(np.sign(x)==np.sign(y))),
                rms_geometry=float(np.sqrt(np.mean(x*x))),rms_readout=float(np.sqrt(np.mean(y*y))))


def sample(initial,offset,cfg):
    x=base.original.midpoint_grid(cfg["n_train"])
    y=base.original.matched.target_values("runge",x,cfg)
    tx,ty=torch.tensor(x),torch.tensor(y)
    p={k:torch.nn.Parameter(torch.tensor(initial[k])) for k in KEYS}
    optimizer=torch.optim.SGD(p.values(),lr=cfg["learning_rate"])
    selected=np.arange(cfg["halo"],cfg["halo"]+128)
    records=[]
    maximum_check=0.
    total=cfg.get("nudge_steps",200)
    for step in range(total+1):
        optimizer.zero_grad(set_to_none=True)
        prediction=torch.tanh(tx[:,None]*p["a"]+p["b"])@p["v"][:-1]+p["v"][-1]
        loss=.5*(prediction-ty).square().mean()
        loss.backward()
        state={k:v.detach().numpy().copy() for k,v in p.items()}
        gradients={k:v.grad.detach().numpy().copy() for k,v in p.items()}
        if step:
            # These probes leave the training parameters untouched. Both
            # use this same state and residual, with the opposite block fixed.
            geometry={k:state[k]-cfg["learning_rate"]*gradients[k] if k in ("a","b") else state[k].copy() for k in KEYS}
            readout={k:state[k]-cfg["learning_rate"]*gradients[k] if k=="v" else state[k].copy() for k in KEYS}
            if step in (1,100,200,total):
                for expected,frozen in ((geometry,{"v"}),(readout,{"a","b"})):
                    actual=independent_fork(state,tx,ty,cfg["learning_rate"],frozen)
                    for k in KEYS:
                        discrepancy=float(np.max(abs(actual[k]-expected[k])))
                        maximum_check=max(maximum_check,discrepancy)
                        np.testing.assert_allclose(actual[k],expected[k],rtol=2e-15,atol=3e-15)
                        if k in frozen:
                            np.testing.assert_array_equal(actual[k],state[k])
            gamma=abs(state["a"][selected])
            coeff=abs(state["v"][selected])
            delta_gamma=abs(geometry["a"][selected])-gamma
            delta_coeff=abs(readout["v"][selected])-coeff
            # A first-order expansion in a*x, keeping each current bias:
            # dL/da ~= c sech²(b) <x e>; dL/dc ~= tanh(b)<e> + a sech²(b)<x e>.
            residual=(prediction-ty).detach().numpy()
            e0=float(np.mean(residual));e1=float(np.mean(x*residual))
            a,b,c=(state[k][selected] for k in ("a","b","v"))
            q=1-np.tanh(b)**2
            ga_model=c*q*e1
            gc_model=np.tanh(b)*e0+a*q*e1
            ga=gradients["a"][selected];gc=gradients["v"][selected]
            records.append(dict(step=step,actual_step=offset+step,
                                X=coeff*delta_gamma,Y=gamma*delta_coeff,
                                gamma=gamma,coefficient=coeff,
                                delta_gamma=delta_gamma,delta_coefficient=delta_coeff,
                                residual_mean=e0,residual_first_moment=e1,
                                slope_model_relative_error=float(np.linalg.norm(ga-ga_model)/max(np.linalg.norm(ga),1e-300)),
                                readout_model_relative_error=float(np.linalg.norm(gc-gc_model)/max(np.linalg.norm(gc),1e-300)),
                                relative_error=float(torch.sqrt(2*loss.detach()/torch.mean(ty.square())))))
        if step and step%1000==0:
            print(f"window offset {offset}, step {step}/{total}, E={records[-1]['relative_error']:.6g}",flush=True)
        if step<total:
            optimizer.step()
    data={k:np.asarray([r[k] for r in records]) for k in records[0]}
    data["selected_neurons"]=selected
    report=fit(data["X"],data["Y"])
    report.update(points=int(data["X"].size),start_actual_step=int(data["actual_step"][0]),
                  end_actual_step=int(data["actual_step"][-1]),
                  first_relative_error=float(data["relative_error"][0]),
                  final_relative_error=float(data["relative_error"][-1]),
                  independent_step_max_discrepancy=maximum_check)
    return data,report


def plot(cases,reports,stem="nudge_correlation",raw=False,joint=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.lines import Line2D
    from matplotlib.ticker import ScalarFormatter

    total=len(cases["early"]["step"])
    extended=total>200
    rows=2 if extended else 1
    fig,axes=plt.subplots(rows,2,figsize=(14.7,11.6 if extended else 7.2),dpi=180,squeeze=False)
    norm=Normalize(1,total)
    for row in range(rows):
      for col,(name,title) in enumerate(zip(("early","fitted"),("Xavier start","Fitted nonlinear Runge · γ ≈ 16"))):
        ax=axes[row,col]
        full=cases[name]
        mask=np.ones(total,dtype=bool) if row==0 else full["step"]>total-1000
        xkey,ykey=("delta_gamma","delta_coefficient") if raw else ("X","Y")
        c={"X":full[xkey][mask],"Y":full[ykey][mask],"step":full["step"][mask]}
        r=reports[name] if row==0 else fit(c["X"],c["Y"])
        x,y=c["X"].ravel(),c["Y"].ravel()
        color=np.repeat(c["step"],128)
        sc=ax.scatter(x,y,c=color,cmap="viridis",norm=norm,s=1 if extended else 2,alpha=.45 if extended else .55,linewidths=0,rasterized=True)
        low,high=float(x.min()),float(x.max())
        grid=np.linspace(low,high,200)
        ax.plot(grid,r["slope"]*grid+r["intercept"],color=".1",lw=1.7)
        ax.plot(grid,grid,color=".55",ls="--",lw=1)
        ax.axhline(0,color=".8",lw=.7);ax.axvline(0,color=".8",lw=.7)
        for axis in (ax.xaxis,ax.yaxis):
            formatter=ScalarFormatter(useOffset=False,useMathText=True)
            formatter.set_powerlimits((0,0));axis.set_major_formatter(formatter)
        window=f"All {total:,} steps" if row==0 else "Last 1,000 steps · zoomed axes"
        ax.set_title(title+" · "+window+f"\nr = {r['pearson_r']:.3f}     R² = {r['R_squared']:.3f}     slope = {r['slope']:.3g}",fontsize=11.5 if extended else 13,pad=17)
        xlabel=r"Geometry nudge: $\Delta|\gamma_j|$" if raw else r"Geometry nudge: $|c_j|\,\Delta|\gamma_j|$"
        ylabel=r"Readout nudge: $\Delta|c_j|$" if raw else r"Readout-based prediction: $|\gamma_j|\,\Delta|c_j|$"
        if joint:
            xlabel=r"Actual gamma change: $|\gamma_j^{(k)}|-|\gamma_j^{(k-1)}|$"
            ylabel=r"Actual readout change: $|c_j^{(k)}|-|c_j^{(k-1)}|$"
        ax.set_xlabel(xlabel+("" if joint else "\n(readout frozen)"),fontsize=12,labelpad=10)
        ax.set_ylabel(ylabel+("" if joint else "\n(geometry frozen)"),fontsize=12,labelpad=10)
        ax.grid(alpha=.15);ax.spines[["top","right"]].set_visible(False)
        ax.tick_params(labelsize=10)
        ax.margins(x=.07,y=.09)
    title="Raw geometry and readout nudges — no coefficient multipliers" if raw else "Do rescaled geometry and readout nudges correlate?"
    if joint:title="Ordinary joint GD — all parameters update at every step"
    fig.suptitle(f"{title}\n{128*total:,} pairs per full-window panel · 128 neurons × {total:,} training states",fontsize=17,y=.985)
    fig.legend(handles=[Line2D([],[],color=".1",lw=1.7,label="Best linear fit, with intercept"),
                        Line2D([],[],color=".55",ls="--",label="Equal numerical deltas: y = x" if raw else "Equal rescaled nudges: y = x")],
               loc="upper center",bbox_to_anchor=(.48,.925 if extended else .884),ncol=2,frameon=False,fontsize=11)
    fig.subplots_adjust(left=.09,right=.875,top=.85 if extended else .755,bottom=.18 if extended else .245,wspace=.34,hspace=.53)
    color_axis=fig.add_axes([.905,.30,.015,.42])
    cb=fig.colorbar(sc,cax=color_axis)
    cb.set_ticks([1]+[int(total*t) for t in (.25,.5,.75,1)]);cb.set_label(f"Step within the {total:,}-step window",fontsize=10,labelpad=9)
    cb.solids.set_alpha(1)
    fitted_start=int(cases["fitted"]["actual_step"][0]);fitted_end=int(cases["fitted"]["actual_step"][-1])
    if joint:
        fig.text(.485,.06 if extended else .085,
                 "Each point uses the actual before/after parameter values from one simultaneous GD update; no block is frozen.\n"
                 "Positive = growth in magnitude; negative = shrinkage. Raw deltas, no coefficient multipliers. All axes are linear.\n"
                 f"Xavier: completed updates 1–{total}. Fitted Runge: completed updates {fitted_start}–{fitted_end}, starting near 1% error.\n"
                 "Slopes, hidden biases, readouts, and output bias all train at η=0.002. Scatter shows the same 128 selected neuron slots.\n"
                 "Viridis shows completed update number within each window. Each axis has its own range and printed power of ten.",
                 ha="center",va="center",fontsize=9)
        fig.savefig(RESULTS/(stem+".png"));plt.close(fig)
        return
    fig.text(.485,.06 if extended else .085,
             "Positive = growth in magnitude; negative = shrinkage. All axes are linear; each axis has its own range and power-of-ten multiplier.\n"
             "Both one-step probes start from the same parameters; the main trajectory continues ordinary joint GD at η=0.002.\n"
             f"Xavier window: Runge, updates 1–{total}. Fitted window: Runge, updates {fitted_start}–{fitted_end}, beginning near 1% error.\n"
             "γ=|a| for the trained tanh(ax+b) model; geometry probes update a and b. Fixed non-halo slots 24–151; all 177 neurons train.\n"
             "Color denotes the current joint-training state, not the duration of a frozen branch. These are paired instantaneous nudges.",
             ha="center",va="center",fontsize=9)
    fig.savefig(RESULTS/(stem+".png"))
    plt.close(fig)


def plot_evolution(cases,stem):
    """Separate time evolution from a pooled scatter dominated by early steps."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import ScalarFormatter

    total=len(cases["early"]["step"])
    fig,axes=plt.subplots(3,2,figsize=(14,10.5),dpi=170,sharex=True)
    purple,green="#482878","#21918c"
    temporal={}
    for col,name in enumerate(("early","fitted")):
        c=cases[name];t=c["step"]
        per_step=[fit(x,y) for x,y in zip(c["X"],c["Y"])]
        r=np.array([d["pearson_r"] for d in per_step])
        nonzero=(c["X"]!=0)&(c["Y"]!=0)
        agrees=(np.sign(c["X"])==np.sign(c["Y"]))&nonzero
        agreement=agrees.sum(axis=1)/nonzero.sum(axis=1)
        gx=np.array([d["rms_geometry"] for d in per_step]);gy=np.array([d["rms_readout"] for d in per_step])
        axes[0,col].plot(t,c["relative_error"],color=".2",lw=2)
        axes[1,col].plot(t,r,color=".2",lw=2)
        axes[1,col].plot(t,agreement,color=".5",lw=1.5,ls="--")
        axes[1,col].set_ylim(-.05,1.02)
        axes[2,col].plot(t,gx/gx[0],color=purple,lw=2)
        axes[2,col].plot(t,gy/gy[0],color=green,lw=2)
        axes[2,col].set_xlabel("Step within this training window",fontsize=11)
        axes[0,col].set_title("Xavier start" if col==0 else "Already fitted nonlinear Runge",fontsize=14,pad=14)
        for row in range(3):
            ax=axes[row,col];ax.set_xlim(0,total);ax.grid(alpha=.18)
            ax.spines[["top","right"]].set_visible(False)
            ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False,useMathText=True))
            ax.margins(y=.10)
        temporal[name]=dict(first_r=float(r[0]),last_r=float(r[-1]),
                            first_nonzero_direction_agreement=float(agreement[0]),
                            last_nonzero_direction_agreement=float(agreement[-1]),
                            first_error=float(c["relative_error"][0]),last_error=float(c["relative_error"][-1]),
                            last_geometry_rms_fraction=float(gx[-1]/gx[0]),last_readout_rms_fraction=float(gy[-1]/gy[0]),
                            tail=fit(c["X"][-1000:],c["Y"][-1000:]),
                            first_200_slope_model_max_error=float(np.max(c["slope_model_relative_error"][:200])),
                            first_200_readout_model_max_error=float(np.max(c["readout_model_relative_error"][:200])))
    axes[0,0].set_ylabel("Training relative $L_2$ error\n(linear scale)",fontsize=12)
    axes[1,0].set_ylabel("Pearson correlation / fraction\nagreeing in growth or shrinkage",fontsize=12)
    axes[2,0].set_ylabel("RMS rescaled nudge\ndivided by its own initial RMS",fontsize=12)
    fig.suptitle("How the nudge relationship evolves over training\nEach time point summarizes 128 paired one-step probes",fontsize=17,y=.985)
    handles=[Line2D([],[],color=".2",lw=2,label="Pearson correlation across neurons"),
             Line2D([],[],color=".5",lw=1.5,ls="--",label="Fraction agreeing in sign (nonzero pairs)"),
             Line2D([],[],color=purple,lw=2,label="Geometry nudge strength / its initial strength"),
             Line2D([],[],color=green,lw=2,label="Readout nudge strength / its initial strength")]
    fig.legend(handles=handles,loc="upper center",bbox_to_anchor=(.5,.908),ncol=2,frameon=False,fontsize=11)
    fig.subplots_adjust(left=.12,right=.97,top=.825,bottom=.13,hspace=.27,wspace=.24)
    fig.text(.52,.05,
             "All axes are linear. Correlation is recomputed at each step; it is not the correlation of the entire pooled cloud.\n"
             "The two bottom curves each start at one. Their equal starting heights do not mean equal absolute nudge sizes.\n"
             "Same Runge targets, initial states, η=0.002, and opposite-coefficient rescaling as the scatter plots. No frozen branch is carried forward.",
             ha="center",va="center",fontsize=9.5)
    fig.savefig(RESULTS/(stem+"_evolution.png"));plt.close(fig)
    (RESULTS/"data"/(stem+"_evolution.json")).write_text(json.dumps(temporal,indent=2)+"\n")


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--steps",type=int,default=200)
    parser.add_argument("--data-only",action="store_true")
    parser.add_argument("--raw-deltas",action="store_true",help="Plot the saved signed changes in magnitudes without coefficient multipliers.")
    args=parser.parse_args()
    cfg=late.config()
    if args.steps!=200:cfg["nudge_steps"]=args.steps
    torch.set_default_dtype(torch.float64);torch.set_num_threads(cfg["threads"])
    stem="nudge_correlation"+(f"_{args.steps}" if args.steps!=200 else "")
    path=RESULTS/"data"/(stem+".npz")
    summary_path=RESULTS/"data"/(stem+"_summary.json")
    meta=cfg|dict(diagnostic_steps=args.steps,selected_slots=[24,151],nudge_definition="signed magnitude changes, multiplied by the other coefficient magnitude")
    with threadpool_limits(limits=cfg["threads"]):
        if path.exists():
            with np.load(path) as source:
                assert json.loads(str(source["config_json"]))==meta
                cases={name:{k.split("__",1)[1]:source[k] for k in source.files if k.startswith(name+"__")} for name in ("early","fitted")}
            reports=json.loads(summary_path.read_text())
        else:
            early=base.original.initial_state("xavier",cfg)
            with np.load(RESULTS/"data"/"late_freeze_runge.npz") as source:
                fitted={k:source["warmup__"+k][-1] for k in KEYS}
                offset=int(source["warmup__step"][-1])
            cases,reports={},{}
            for name,initial,start in (("early",early,0),("fitted",fitted,offset)):
                cases[name],reports[name]=sample(initial,start,cfg)
            if args.steps>200:
                # Extending the duration must reproduce the existing prefix.
                with np.load(RESULTS/"data"/"nudge_correlation.npz") as reference:
                    for name in cases:
                        for key in ("X","Y","gamma","coefficient","relative_error"):
                            np.testing.assert_array_equal(cases[name][key][:200],reference[name+"__"+key])
            np.savez_compressed(path,config_json=json.dumps(meta),
                                **{f"{name}__{k}":v for name,c in cases.items() for k,v in c.items()})
            summary_path.write_text(json.dumps(reports,indent=2)+"\n")
        if not args.data_only:
            if args.raw_deltas:
                raw_reports={name:fit(c["delta_gamma"],c["delta_coefficient"]) for name,c in cases.items()}
                for name,c in cases.items():
                    # These are the exact stored deltas underlying the previous plot.
                    np.testing.assert_array_equal(c["X"],c["coefficient"]*c["delta_gamma"])
                    np.testing.assert_array_equal(c["Y"],c["gamma"]*c["delta_coefficient"])
                    raw_reports[name]["tail"]=fit(c["delta_gamma"][-1000:],c["delta_coefficient"][-1000:])
                plot(cases,raw_reports,stem+"_raw",raw=True)
                (RESULTS/"data"/(stem+"_raw_summary.json")).write_text(json.dumps(raw_reports,indent=2)+"\n")
                print(json.dumps(raw_reports,indent=2))
            else:
                plot(cases,reports,stem)
                if args.steps>200:plot_evolution(cases,stem)
        print(json.dumps(reports,indent=2))


if __name__=="__main__":main()
