"""Ordinary-GD trajectories with offline loss/geometry-gradient decomposition.

F_tau is the least-squares residual outside the retained left-singular space.
Its derivative includes the motion of that space, including the correction
to the untruncated VarPro envelope gradient. G_tau is defined as L-F_tau.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import scipy.linalg as sla
import torch
from threadpoolctl import threadpool_limits
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.expD24_gd_residual_spectrum import four_way_comparison as previous

HERE = Path(__file__).resolve().parent
RESULTS = ROOT / "results/checkpoint_D_optimizers/expD28_loss_gradient_decomposition"
LABELS = {"sine": "Sine", "sine_mixture": "Mixed sine", "runge": "Runge",
          "gaussian_envelope": "Gaussian envelope"}
ARMS = {"xavier": "Xavier", "scaled_xavier": "Scaled Xavier · centers preserved · mean λ₀ = 0.25",
        "qi_zero": "QI geometry · γ₀ = 16 · zero readout"}


def config():
    return yaml.safe_load((HERE / "config.yaml").read_text())


def snapshots(steps, count):
    if steps <= count:
        return np.arange(steps + 1)
    early = np.arange(21)
    return np.unique(np.r_[early, np.rint(np.geomspace(21, steps, count-len(early))).astype(int), steps])


def initial_state(arm, cfg):
    arrays, _ = previous.matched.rc.original.initial_arrays(
        cfg["resolution"], cfg["seed"], arm, lambda_star=cfg["lambda_star"], halo=cfg["halo"])
    return dict(a=arrays["a"], b=arrays["b"], v=np.r_[arrays["c"], arrays["d"]])


def hidden(a, b, x, backend="numpy"):
    s = x[:, None] * a + b
    if backend == "torch":
        return torch.tanh(torch.tensor(s)).numpy()
    return np.tanh(s)


def matrix_gradient_to_geometry(matrix_gradient, h, x):
    """A=[tanh(a*x+b),1]/sqrt(n); return derivatives in theta=(a,b)."""
    weighted = matrix_gradient[:, :-1] * (1-h*h) / np.sqrt(len(x))
    return np.r_[np.sum(weighted*x[:, None], axis=0), np.sum(weighted, axis=0)]


def profiled_matrix(A, y, rcond, driver="gesvd"):
    """F_tau and its exact local matrix derivative on a separated SVD branch.

    Let I denote retained and D discarded thin-SVD indices, alpha=U.T@y.
    The ordinary envelope matrix gradient is r_tau v_tau.T. Moving the
    retained *right* singular space adds the two cross-subspace terms below.
    Null-space directions outside the thin U are already in r_tau.
    """
    U, s, Vh = sla.svd(A, full_matrices=False, check_finite=False, lapack_driver=driver)
    cutoff = rcond*s[0] if len(s) else 0.
    rank = int(np.sum(s > cutoff))
    alpha = U.T@y
    if rank:
        fitted = U[:, :rank]@alpha[:rank]
        vstar = Vh[:rank].T@(alpha[:rank]/s[:rank])
    else:
        fitted = np.zeros_like(y)
        vstar = np.zeros(A.shape[1])
    rstar = fitted-y
    envelope = np.outer(rstar, vstar)
    correction = np.zeros_like(A)
    if 0 < rank < len(s):
        ratio = s[rank:, None]/s[None, :rank]
        denominator = 1-ratio*ratio
        B = -(alpha[rank:, None]*alpha[None, :rank]/s[None, :rank])*ratio**2/denominator
        C = -(alpha[:rank, None]*alpha[None, rank:]/s[:rank, None])*ratio.T/denominator.T
        correction = U[:, rank:]@B@Vh[:rank] + U[:, :rank]@C@Vh[rank:]
    return dict(F=.5*float(rstar@rstar), matrix_gradient=envelope+correction,
                envelope_matrix_gradient=envelope, vstar=vstar, rstar=rstar,
                rank=rank, singular_values=s, cutoff=cutoff,
                cutoff_margin=float(np.min(abs(s-cutoff))/s[0]) if len(s) else 0.)


def one_diagnostic(state, x, y, rcond, driver="gesvd", backend="numpy"):
    a, b, v = (state[k] for k in ("a", "b", "v"))
    h = hidden(a, b, x, backend)
    A = np.c_[h, np.ones(len(x))]/np.sqrt(len(x))
    yn = y/np.sqrt(len(x))
    r = A@v-yn
    profile = profiled_matrix(A, yn, rcond, driver)
    gL = matrix_gradient_to_geometry(np.outer(r, v), h, x)
    gF = matrix_gradient_to_geometry(profile["matrix_gradient"], h, x)
    envelope = matrix_gradient_to_geometry(profile["envelope_matrix_gradient"], h, x)
    gG = gL-gF
    norms = np.array([np.linalg.norm(g) for g in (gL, gF, gG)])
    cosine = float(np.dot(gF, gG)/(norms[1]*norms[2])) if norms[1]*norms[2] > 0 else np.nan
    cosine = np.clip(cosine, -1, 1)
    gap_vector = A@v-(profile["rstar"]+yn)
    return dict(L=.5*float(r@r), F=profile["F"], G=.5*float(r@r)-profile["F"],
                gL=gL, gF=gF, gG=gG, envelope_gradient=envelope, norms=norms,
                cosine=cosine, rank=profile["rank"], vstar_norm=np.linalg.norm(profile["vstar"]),
                refit_train_relative_l2=float(np.linalg.norm(A@profile["vstar"]-yn)/np.linalg.norm(yn)),
                singular_values=profile["singular_values"], cutoff_margin=profile["cutoff_margin"],
                orthogonal_split_error=abs(.5*float(r@r)-profile["F"]-.5*float(gap_vector@gap_vector)))


def train(target, arm, cfg):
    """No solver or spectral computation is called during training."""
    x = previous.midpoint_grid(cfg["n_train"])
    y = previous.matched.target_values(target, x, cfg)
    initial = initial_state(arm, cfg)
    p = {key: torch.nn.Parameter(torch.tensor(v, dtype=torch.float64)) for key, v in initial.items()}
    optimizer = torch.optim.SGD(p.values(), lr=cfg["learning_rate"])
    tx, ty = torch.tensor(x), torch.tensor(y)
    chosen = snapshots(cfg["steps"], cfg["diagnostic_snapshots"])
    choose = set(chosen)
    states = {key: [] for key in p}
    gradients = []
    losses, means = [], []
    for step in range(cfg["steps"]+1):
        optimizer.zero_grad(set_to_none=True)
        f = torch.tanh(tx[:, None]*p["a"]+p["b"])@p["v"][:-1]+p["v"][-1]
        loss = .5*torch.mean((f-ty).square())
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite loss: {target}/{arm}/{step}")
        loss.backward()
        losses.append(float(loss.detach()))
        means.append(float(p["a"].detach().abs().mean()))
        if step in choose:
            for key in p:
                states[key].append(p[key].detach().numpy().copy())
            gradients.append(np.r_[p["a"].grad.numpy(), p["b"].grad.numpy()])
        if step < cfg["steps"]:
            optimizer.step()
    return {key: np.asarray(value) for key, value in states.items()} | dict(
        steps=chosen, train_loss=np.asarray(losses), mean_gamma=np.asarray(means),
        training_geometry_gradient=np.asarray(gradients), target=np.array(target), arm=np.array(arm))


def diagnose(case, cfg):
    target, arm = str(case["target"]), str(case["arm"])
    x = previous.midpoint_grid(cfg["n_train"])
    y = previous.matched.target_values(target, x, cfg)
    xe = previous.midpoint_grid(cfg["n_eval"])
    ye = previous.matched.target_values(target, xe, cfg)
    records = []
    validation = []
    for i, step in enumerate(case["steps"]):
        state = {k: case[k][i] for k in ("a", "b", "v")}
        primary = one_diagnostic(state, x, y, cfg["readout_rcond"])
        alternatives = [one_diagnostic(state, x, y, cfg["readout_rcond"], driver="gesdd"),
                        one_diagnostic(state, x, y, cfg["readout_rcond"], backend="torch")]
        errorF = max(np.linalg.norm(primary["gF"]-d["gF"]) for d in alternatives)
        errorG = max(np.linalg.norm(primary["gG"]-d["gG"]) for d in alternatives)
        agreement = all(d["rank"] == primary["rank"] for d in alternatives)
        derivative_resolved = bool(agreement and primary["norms"][1] > cfg["resolution_factor"]*errorF)
        cosine_resolved = bool(derivative_resolved and primary["norms"][2] > cfg["resolution_factor"]*errorG
                               and np.isfinite(primary["cosine"]))
        all_cosines = np.array([primary["cosine"]]+[d["cosine"] for d in alternatives])
        primary.update(gradient_F_uncertainty=errorF, gradient_G_uncertainty=errorG,
                       rank_agreement=agreement, derivative_resolved=derivative_resolved,
                       cosine_resolved=cosine_resolved,
                       cosine_low=np.nanmin(all_cosines) if np.isfinite(all_cosines).any() else np.nan,
                       cosine_high=np.nanmax(all_cosines) if np.isfinite(all_cosines).any() else np.nan,
                       envelope_correction_norm=np.linalg.norm(primary["gF"]-primary["envelope_gradient"]))
        # Independent original training gradient; numerical profile cannot affect it.
        difference = float(np.max(abs(primary["gL"]-case["training_geometry_gradient"][i])))
        np.testing.assert_allclose(primary["gL"], case["training_geometry_gradient"][i], rtol=1e-9, atol=2e-14)
        norm_identity = abs(primary["norms"][0]**2 - (primary["norms"][1]**2+primary["norms"][2]**2
                         + 2*np.dot(primary["gF"],primary["gG"])))
        pred = hidden(state["a"],state["b"],xe)@state["v"][:-1]+state["v"][-1]
        primary["eval_relative_l2"] = np.linalg.norm(pred-ye)/np.linalg.norm(ye)
        primary["cancellation_ratio"] = primary["norms"][0]/sum(primary["norms"][1:]) if sum(primary["norms"][1:]) else np.nan
        validation.append([difference, norm_identity])
        records.append(primary)
        if i in (0, len(case["steps"])-1) or i % 40 == 0:
            print(f"{target}/{arm}: diagnostic {i+1}/{len(case['steps'])}, step {step}, "
                  f"norms {primary['norms']}, cosine {primary['cosine']:.3g}, resolved={cosine_resolved}", flush=True)
    for key in records[0]:
        case[key] = np.asarray([d[key] for d in records])
    case["validation"] = np.asarray(validation)
    return case


def save(case, cfg, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp.npz")
    np.savez_compressed(temp, config_json=json.dumps(cfg), **case)
    temp.replace(path)


def load(path, cfg):
    with np.load(path, allow_pickle=False) as f:
        if json.loads(str(f["config_json"])) != cfg:
            raise ValueError(f"Configuration differs: {path}")
        return {k:f[k] for k in f.files if k != "config_json"}


def plot(cases, cfg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import LogLocator

    colors = ["#2868ae", "#138878", "#c77622"]
    figures = RESULTS/"figures"
    figures.mkdir(parents=True, exist_ok=True)
    def log_limits(values):
        positive = np.concatenate([np.asarray(v).ravel() for v in values])
        positive = positive[np.isfinite(positive) & (positive > 0)]
        return float(positive.min()/2), float(positive.max()*2)

    for arm in cfg["arms"]:
        arm_cases = [cases[target,arm] for target in cfg["targets"]]
        # QI's profiled loss/gradient have already been checked to be at the
        # numerical floor. Label their off-axis values instead of allocating
        # thirty decades to them and compressing the actual training curves.
        floor_only = arm == "qi_zero"
        loss_keys = ("L","G") if floor_only else ("L","F","G")
        loss_bottom, loss_top = log_limits([c[key] for c in arm_cases for key in loss_keys])
        norm_values = ([c["norms"][c["steps"]>0][:,[0,2]] for c in arm_cases] if floor_only
                       else [c["norms"] for c in arm_cases])
        norm_bottom, norm_top = log_limits(norm_values)
        fig, axes = plt.subplots(3,4,figsize=(19,12.2),dpi=160,sharex=True,sharey="row")
        for col, target in enumerate(cfg["targets"]):
            c=cases[target,arm]; step=c["steps"]
            axes[0,col].set_title(LABELS[target],fontsize=15,pad=15)
            for j,key in enumerate(("L","F","G")):
                axes[0,col].plot(step,np.maximum(c[key],1e-32),color=colors[j],lw=2.6 if j==0 else 1.7,
                                 ls="--" if j==2 else "-")
            for j in range(3):
                values=np.where(c["norms"][:,j]>0,c["norms"][:,j],np.nan)
                if floor_only:
                    values=np.where(values>=norm_bottom,values,np.nan)
                axes[1,col].plot(step,values,color=colors[j],lw=2.6 if j==0 else 1.7,ls="--" if j==2 else "-")
            if floor_only:
                axes[0,col].text(.97,.86,f"max Fτ ≈ {np.max(c['F']):.1e}\n(numerical floor; below view)",
                                 transform=axes[0,col].transAxes,ha="right",fontsize=10,color=colors[1])
                axes[1,col].text(.97,.07,f"max ‖DFτ‖ ≈ {np.max(c['norms'][:,1]):.1e}\n(unresolved; below view)",
                                 transform=axes[1,col].transAxes,ha="right",fontsize=10,color=colors[1])
            if c["norms"][0,0]==0:
                axes[1,col].plot(0,norm_bottom,"v",mfc="white",mec=colors[0],ms=5,clip_on=False)
                axes[1,col].text(.025,.025,"DL(0) = 0",transform=axes[1,col].transAxes,color=colors[0],fontsize=9)
            # A dotted segment explicitly marks an unresolved numerical DF estimate.
            uncertain=~c["derivative_resolved"]
            axes[1,col].plot(step,np.where(uncertain,c["norms"][:,1],np.nan),
                            color="white",lw=2.8,zorder=3)
            axes[1,col].plot(step,np.where(uncertain,c["norms"][:,1],np.nan),
                            color=colors[1],lw=1.7,ls=":",zorder=4)
            axes[1,col].plot(step,cfg["resolution_factor"]*c["gradient_F_uncertainty"],
                            color=".55",ls=":",lw=.8,alpha=.8)
            valid=c["cosine_resolved"]
            axes[2,col].plot(step,np.where(valid,c["cosine"],np.nan),color="#7353a6",lw=1.6)
            axes[2,col].fill_between(step,c["cosine_low"],c["cosine_high"],where=valid,color="#7353a6",alpha=.16)
            axes[2,col].axhline(0,color=".45",lw=.8,ls="--")
            # Small gray marks show exactly which saved states are unresolved.
            axes[2,col].plot(step[~valid],np.full(np.sum(~valid),-1.055),"|",color=".55",ms=5,clip_on=False)
            if not np.any(valid):
                axes[2,col].text(.5,.68,"DF is numerically unresolved\ncosine is not reported",transform=axes[2,col].transAxes,
                                 ha="center",va="center",fontsize=11,color=".4")
            elif np.any(~valid):
                axes[2,col].text(.98,.98,f"{np.sum(~valid)}/{len(valid)} states unresolved",transform=axes[2,col].transAxes,
                                 ha="right",va="top",fontsize=9,color=".45")
            axes[0,col].set(yscale="log",ylim=(loss_bottom,loss_top))
            axes[1,col].set(yscale="log",ylim=(norm_bottom,norm_top))
            axes[0,col].yaxis.set_major_locator(LogLocator(base=10,numticks=7))
            axes[1,col].yaxis.set_major_locator(LogLocator(base=10,numticks=7))
            axes[2,col].set(ylim=(-1.1,1.1),yticks=[-1,-.5,0,.5,1],xlabel="GD updates")
            for row in range(3):
                ax=axes[row,col]
                ax.set(xlim=(0,cfg["steps"]),xticks=[0,500,1000,1500,2000])
                ax.grid(alpha=.16)
                ax.spines[["top","right"]].set_visible(False)
                ax.tick_params(labelsize=10)
        axes[0,0].set_ylabel(r"Loss: $\frac{1}{2}\,\mathrm{mean}(e^2)$",fontsize=13)
        axes[1,0].set_ylabel(r"Geometry-gradient norm $\|D_\theta(\cdot)\|_2$",fontsize=13)
        axes[2,0].set_ylabel(r"$\cos(DF_\tau,DG_\tau)$",fontsize=13)
        fig.suptitle(ARMS[arm]+"\nOrdinary GD: approximation floor versus unfinished readout",fontsize=19,y=.985)
        handles=[Line2D([],[],color=c,lw=2,ls="--" if j==2 else "-") for j,c in enumerate(colors)]
        fig.legend(handles,[r"$L$: actual loss",r"$F_\tau$: numerical LS floor",
                            r"$G_\tau=L-F_\tau$: readout gap"],loc="upper center",
                   bbox_to_anchor=(.5,.914),ncol=3,frameon=False,fontsize=12)
        fig.subplots_adjust(left=.075,right=.985,top=.85,bottom=.145,hspace=.32,wspace=.15)
        gradient_handles = handles + [Line2D([],[],color=".55",lw=1.2,ls=":")]
        fig.legend(gradient_handles,
                   [r"$\|DL\|_2$: actual-loss gradient",
                    r"$\|DF_\tau\|_2$: LS-floor gradient",
                    r"$\|DG_\tau\|_2$: readout-gap gradient",
                    "Numerical-resolution threshold\n(10 × numerical variation in DFτ)"],
                   loc="lower center",bbox_to_anchor=(.53,axes[1,0].get_position().y1+.006),
                   ncol=4,frameon=False,fontsize=11.5,columnspacing=1.6)
        fig.text(.5,.068,
                 "All four targets train on the same 1,024 midpoints in [−1,1]; N=128, 177 neurons, halo 24 per side; rate 0.002. All readout solves are offline.\n"
                 "θ=(all raw slopes, all biases); readout coordinates excluded. Fτ uses SVD cutoff 10⁻¹³σ₁; DFτ includes the changing retained-subspace derivative.\n"
                 "Dotted DF: unresolved estimate. Gray dotted norm: 10× variation across SVD drivers / activation rounding; gray cosine ticks: omitted unresolved states.\n"
                 "Y limits fit each initialization; function columns share scales. QI numerical-floor values are labeled below view. Middle row shows norms, not squared norms.",
                 ha="center",va="center",fontsize=10)
        fig.savefig(figures/f"{arm}.png")
        plt.close(fig)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--plot-only",action="store_true")
    parser.add_argument("--targets",nargs="+")
    parser.add_argument("--arms",nargs="+")
    args=parser.parse_args()
    cfg=config()
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(cfg["threads"])
    cases={}
    with threadpool_limits(limits=cfg["threads"]):
        for arm in args.arms or cfg["arms"]:
            for target in args.targets or cfg["targets"]:
                path=RESULTS/"data"/f"{target}__{arm}.npz"
                if path.exists():
                    case=load(path,cfg)
                elif args.plot_only:
                    continue
                else:
                    start=time.perf_counter()
                    case=train(target,arm,cfg)
                    case["training_seconds"]=np.array(time.perf_counter()-start)
                    start=time.perf_counter()
                    case=diagnose(case,cfg)
                    case["diagnostic_seconds"]=np.array(time.perf_counter()-start)
                    save(case,cfg,path)
                cases[target,arm]=case
        if len(cases)==len(cfg["targets"])*len(cfg["arms"]):
            plot(cases,cfg)
        else:
            print("Subset saved; render final figures after all twelve cases are available.",flush=True)


if __name__=="__main__":
    main()
