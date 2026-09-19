"""Detached normalization evidence and figures; never edits a training state."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import svd

from . import core, diagnostics, difference_analysis as old, difference_training as training
from . import parameter_scale as campaign, ratio_analysis, run

LABELS={"scaled":"Collective normalization", "parameter_scale":"Parameter-scale normalization"}


def next_update(cp, g, case, gc, gl):
    scale=g.d if case["coordinates"]=="scaled" else g.alpha
    eta=case["eta"]
    if case["optimizer"]=="gd":
        return -eta*scale**2*gc, -eta*gl
    count=int(cp["adam_count"])+1
    steps=[]
    eps_r=1e-8*scale/g.d
    for block,grad,eps in (("readout",scale*gc,eps_r),("slope",gl,1e-8)):
        mu=.9*cp[f"adam_{block}_mu"]+.1*grad
        nu=.999*cp[f"adam_{block}_nu"]+.001*grad**2
        steps.append(-eta*(mu/(1-.9**count))/(np.sqrt(nu/(1-.999**count))+eps))
    return scale*steps[0],steps[1]


def probe(g, cp, case, samples=16):
    x=np.linspace(-1,1,samples*g.n+1);y=core.target(x,"sine",np);root=np.sqrt(len(x))
    c,gamma=cp["c"],cp["gamma"]
    a=diagnostics.features(x,g.centers,gamma)/root;r=a@c-y/root
    ref=a*g.d
    ur,sr,vr=svd(ref,full_matrices=False,lapack_driver="gesdd")
    scale=g.d if case["coordinates"]=="scaled" else g.alpha
    b=a*scale
    un,sn,vn=(ur,sr,vr) if case["coordinates"]=="scaled" else svd(b,full_matrices=False,lapack_driver="gesdd")
    distances=x[:,None]-g.centers;e=np.exp(-2*np.abs(distances*gamma))
    j=c[1:]*distances*4*e/(1+e)**2/(g.h*root)
    gc,gl=a.T@r,j.T@r
    dc,dl=next_update(cp,g,case,gc,gl)
    bounds,rb,_=diagnostics.band_residuals(r)
    band_c,band_l=rb@a,rb@j
    refits=[];cutoff_forces={}
    xv=diagnostics.midpoint_grid(32768)
    for cutoff in old.CUTOFFS:
        keep=sr>cutoff*sr[0];u=ur[:,keep]
        fitted=g.d*(vr[keep].T@((u.T@(y/root))/sr[keep]))
        rr=a@fitted-y/root
        rv=diagnostics.prediction(xv,g.centers,fitted,gamma)-core.target(xv,"sine",np)
        gp,gn,leak,closure=old.projected_forces(j,r,u)
        refits.append(dict(cutoff=cutoff,rank=int(keep.sum()),mse=float(rr@rr),midpoint_mse=float(np.mean(rv**2)),
                           coefficient_norm=float(np.linalg.norm(fitted)),parallel_force_norm=float(np.linalg.norm(gp)),
                           perpendicular_force_norm=float(np.linalg.norm(gn)),projection_leakage=float(leak),closure=float(closure)))
        cutoff_forces[cutoff]=(gp,gn,u)
    gp,gn,u=cutoff_forces[1e-12]
    regions={}
    for name,mask in {"all":np.ones(g.width,bool),**g.masks}.items():
        direction=np.where(mask,np.sign(gamma),0.);direction/=max(np.linalg.norm(direction),1.)
        tangent=j@direction;perp=tangent-u@(u.T@tangent)
        regions[name]=dict(parallel_force=float(-gp@direction),perpendicular_force=float(-gn@direction),
                           tangent_norm=float(np.linalg.norm(tangent)),perpendicular_tangent_norm=float(np.linalg.norm(perp)),
                           signed_actual_lambda_step=float(direction@dl),
                           lambda_quantiles=np.quantile(np.abs(g.h*gamma[mask]),[0,.1,.5,.9,1]).tolist(),
                           weight_over_allowance_rms=float(np.sqrt(np.mean((c[1:][mask]/g.alpha[1:][mask])**2))),
                           step_over_allowance_rms=float(np.sqrt(np.mean((dc[1:][mask]/g.alpha[1:][mask])**2))))
    _,pieces,budget=ratio_analysis.update_budget(x,y,g.centers,g.h,c,gamma,dc,dl)
    modal=un.T@r;motion=un.T@(a@dc)
    record=dict(mse=float(r@r),refits=refits,regions=regions,
                readout_linear_mse_change_by_region={"bias":float(2*gc[0]*dc[0]),
                    **{name:float(2*gc[1:][mask]@dc[1:][mask]) for name,mask in g.masks.items()}},
                native_sigma_max=float(sn[0]),reference_sigma_max=float(sr[0]),
                frozen_readout_gd_stability_ceiling=float(2/sn[0]**2),bias_gd_stability_ceiling=float(2/scale[0]**2),
                native_retained_rank=int(np.sum(sn>1e-12*sn[0])),reference_retained_rank=int(np.sum(sr>1e-12*sr[0])),
                fourier_closure=float(abs(np.sum(rb**2)-r@r)),gradient_closure=float(np.linalg.norm(gl-gp-gn)),
                update_budget={k:np.asarray(v).tolist() for k,v in budget.items()},
                actual_update_modal_fractions=[dict(threshold=t,residual_below=float(np.sum(modal[sn<t*sn[0]]**2)/max(r@r,1e-300)),
                    readout_update_below=float(np.sum(motion[sn<t*sn[0]]**2)/max(np.sum((a@dc)**2),1e-300))) for t in (.1,1e-3,1e-4,1e-6)])
    if case["optimizer"]=="adam":
        record["adam_epsilon_dominated_fraction"]={block:float(np.mean(cp[f"adam_{block}_sqrt_v_over_epsilon"]<1)) for block in ("readout","slope")}
    arrays=dict(singular_native=sn,singular_reference=sr,residual_coefficients=modal,readout_update_coefficients=motion,
                band_bounds=bounds,band_mse=np.sum(rb**2,axis=1),band_readout_linear_mse_change=2*band_c@dc,
                band_geometry_linear_mse_change=2*band_l@dl,band_gradient_physical_readout=band_c,band_gradient_lambda=band_l,
                band_bias_linear_mse_change=2*band_c[:,0]*dc[0],
                gradient_lambda_parallel=gp,gradient_lambda_perpendicular=gn,
                gradient_c=gc,gradient_lambda=gl,next_delta_c=dc,next_delta_lambda=dl,c=c,gamma=gamma)
    return record,arrays,(un,sn,vr,ur,sr)


def dense_audit(folder,g,case,end,dest):
    dense=ratio_analysis.read_dense(folder,end)
    x=np.linspace(-1,1,16*g.n+1);root=np.sqrt(len(x));y=core.target(x,"sine",np)
    a0=diagnostics.features(x,g.centers,dense["gamma"][0])/root
    scale=g.d if case["coordinates"]=="scaled" else g.alpha
    un,sn,_=svd(a0*scale,full_matrices=False,lapack_driver="gesdd")
    ur,sr,_=(un,sn,None) if case["coordinates"]=="scaled" else svd(a0*g.d,full_matrices=False,lapack_driver="gesdd")
    kept=ur[:,sr>1e-12*sr[0]]
    rng=np.random.default_rng(391)
    indices=np.arange(16)*128+rng.integers(0,128,16)
    rows=[]
    for index in indices:
        c,gamma,dc,dl=(dense[k][index] for k in ("c","gamma","delta_c","delta_lambda"))
        r,pieces,budget=ratio_analysis.update_budget(x,y,g.centers,g.h,c,gamma,dc,dl)
        a=diagnostics.features(x,g.centers,gamma)/root
        distances=x[:,None]-g.centers;e=np.exp(-2*np.abs(distances*gamma))
        j=c[1:]*distances*4*e/(1+e)**2/(g.h*root)
        bounds,rb,_=diagnostics.band_residuals(r/root)
        gc,gl=rb@a,rb@j
        gp,gn,leak,closure=old.projected_forces(j,r/root,kept)
        rows.append(dict(step=dense["step"][index],**budget,band_mse=np.sum(rb**2,axis=1),
                         readout_geometry_function_cosine=float(pieces[0]@pieces[1]/max(np.linalg.norm(pieces[0])*np.linalg.norm(pieces[1]),1e-300)),
                         band_readout_linear_mse_change=2*gc@dc,band_geometry_linear_mse_change=2*gl@dl,
                         band_bias_linear_mse_change=2*gc[:,0]*dc[0],
                         readout_linear_by_region=np.array([2*gc[:,0].sum()*dc[0]]+
                             [2*gc[:,1:][:,mask].sum(axis=0)@dc[1:][mask] for mask in g.masks.values()]),
                         singular_residual_coefficients=un.T@(r/root),singular_readout_update_coefficients=un.T@(pieces[0]/root),
                         gradient_lambda_parallel=gp,gradient_lambda_perpendicular=gn,projection_leakage=leak,
                         gradient_closure=max(closure,np.linalg.norm(gc.sum(axis=0)-dense["gradient_c"][index]),
                                              np.linalg.norm(gl.sum(axis=0)-dense["gradient_lambda"][index]))))
    arrays={k:np.stack([r[k] for r in rows]) for k in rows[0]}
    run.save_arrays(dest/f"dense_{end}.npz",**arrays,singular_values=sn,band_bounds=bounds)
    run.save_arrays(dest/f"dense_parameters_{end}.npz",**{k:dense[k] for k in ("step","c","gamma")})
    motion={}
    for label,delta in (("w",dense["delta_c"][:,1:]),("lambda",dense["delta_lambda"])):
        norms=np.linalg.norm(delta,axis=1)
        cos=np.sum(delta[:-1]*delta[1:],axis=1)/np.maximum(norms[:-1]*norms[1:],1e-300)
        ratio=np.linalg.norm(delta[:-1]+delta[1:],axis=1)/np.maximum(norms[:-1]+norms[1:],1e-300)
        motion[label]=dict(adjacent_cosine_median=float(np.median(cos)),two_step_net_ratio_median=float(np.median(ratio)),
                           mean_step_rms=float(np.sqrt(np.mean(delta**2,axis=1)).mean()),
                           net_change_rms=float(np.sqrt(np.mean(delta.sum(axis=0)**2))))
    direction=np.sign(dense["gamma"][:,g.core])/np.sqrt(g.core.sum())
    growth_force=-np.sum(direction*dense["gradient_lambda"][:,g.core],axis=1)
    growth_step=np.sum(direction*dense["delta_lambda"][:,g.core],axis=1)
    growth=dict(mean_force=float(growth_force.mean()),mean_actual_step=float(growth_step.mean()),
                force_quantiles=np.quantile(growth_force,[.1,.5,.9]).tolist(),
                actual_step_quantiles=np.quantile(growth_step,[.1,.5,.9]).tolist(),
                fraction_positive_force=float(np.mean(growth_force>0)),fraction_positive_step=float(np.mean(growth_step>0)))
    return dict(end=end,sampled_steps=dense["step"][indices].tolist(),motion=motion,
                core_growth=growth,
                mean_readout_geometry_function_cosine=float(arrays["readout_geometry_function_cosine"].mean()),
                mean_readout_linear_by_region=dict(zip(["bias",*g.masks],arrays["readout_linear_by_region"].mean(axis=0).tolist())),
                mean_mse_change=arrays["mse_change"].mean(axis=0).tolist(),
                mean_linear_mse_change=arrays["linear_mse_change"].mean(axis=0).tolist(),
                mean_quadratic_cost=arrays["quadratic_mse_cost"].mean(axis=0).tolist(),
                maximum_prediction_closure=float(arrays["closure_max"].max()),maximum_gradient_closure=float(arrays["gradient_closure"].max()))


def analyze_case(task):
    root,output,case,end=task;key=training.case_key(case);folder=root/key;dest=output/key
    dest.mkdir(parents=True,exist_ok=True);g=core.geometry(case["n"])
    history,bounds=old.history(folder,end)
    run.save_arrays(dest/"history.npz",**history,band_bounds=bounds,centers=g.centers,alpha=g.alpha)
    trace=old.read_trace(folder,end);blocks=2*trace[:end//20000*20000,0].reshape(-1,20000)
    run.save_arrays(dest/"window_mse.npz",step=20000*(np.arange(len(blocks))+1),mean=blocks.mean(axis=1),
                    quantiles=np.quantile(blocks,[0,.1,.5,.9,1],axis=1))
    record=dict(case=case,end=end,checkpoints={},source_hashes={},dense=[])
    steps=sorted({0,100,1000,20000,100000,end})
    for step in steps:
        file=folder/f"checkpoint_{step:09d}.npz"
        if step>end or not file.exists():continue
        with np.load(file) as cp: stats,arrays,_=probe(g,cp,case)
        record["checkpoints"][str(step)]=stats;record["source_hashes"][str(step)]=hashlib.sha256(file.read_bytes()).hexdigest()
        run.save_arrays(dest/f"spectrum_{step}.npz",**arrays)
        run.write_json(dest/"mechanism.json",record)
    for dense_end in (2048,end):
        audit=dense_audit(folder,g,case,dense_end,dest)
        mse=2*trace[dense_end-2048:dense_end,0]
        changes=np.diff(mse)
        audit["consecutive_loss_changes"]=dict(intervals=len(changes),mean=float(changes.mean()),
            mean_absolute=float(np.abs(changes).mean()),net=float(mse[-1]-mse[0]),
            quantiles=np.quantile(changes,[.1,.5,.9]).tolist())
        record["dense"].append(audit)
    with np.load(folder/f"checkpoint_{end:09d}.npz") as cp:
        record["doubled_grid"],arrays,_=probe(g,cp,case,samples=32)
    run.save_arrays(dest/"doubled_grid.npz",**arrays)
    residual_rms=np.sqrt(2*trace[:,0]);record["residual_reduction_events"]={}
    for reduction in (10,100):
        hits=np.flatnonzero(residual_rms<=residual_rms[0]/reduction)
        if len(hits):
            step=int(hits[0]);near=int(history["step"][np.searchsorted(history["step"],step,side="right")-1])
            record["residual_reduction_events"][str(reduction)]=dict(first_update=step,preceding_checkpoint=near,
                checkpoint_lambda_median=float(np.median(np.abs(history["lambda"][history["step"]==near][:,g.core]))))
        else:record["residual_reduction_events"][str(reduction)]=None
    record["physical_update_scales"]=dict(readout=(case["eta"]*(g.d if case["coordinates"]=="scaled" else g.alpha)**(2 if case["optimizer"]=="gd" else 1)).tolist(),
         gamma=case["eta"]/g.h**(2 if case["optimizer"]=="gd" else 1),physical_readout_epsilon=(1e-8/g.d).tolist(),physical_gamma_epsilon=1e-8*g.h)
    record["late_mean_mse"]=float(blocks[-1].mean());record["late_range"]=np.quantile(blocks[-1],[0,.1,.5,.9,1]).tolist()
    record["previous_mean_mse"]=float(blocks[-2].mean());record["core_lambda_median"]=float(np.median(np.abs(history["lambda"][-1,g.core])))
    run.write_json(dest/"mechanism.json",record)
    print(json.dumps(dict(analyzed=key,end=end)),flush=True)
    return record


def figures(output):
    rows=json.loads((output/"sweep_summary.json").read_text())
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout="constrained")
    for ax,opt in zip(axes,campaign.OPTIMIZERS):
        for k,coord in enumerate(campaign.MAPS):
            group=sorted([r for r in rows if r["n"]==512 and r["seed"]==0 and r["optimizer"]==opt and r["coordinates"]==coord],key=lambda r:r["eta"])
            valid=[r for r in group if r["eligible"]]
            ax.loglog([r["eta"] for r in valid],[r["window_mean_mse"] for r in valid],"o-",label=LABELS[coord])
            failed=[r["eta"] for r in group if r["failed_update"]]
            ax.scatter(failed,np.full(len(failed),1.5+k),marker="x",color=f"C{k}")
        ax.set(xlabel="Constant shared learning rate",ylabel="Mean MSE, updates 80k–100k",title=opt.upper()+"; N=512, seed 0")
        ax.text(.02,.02,"× = numerical failure; height is not MSE",transform=ax.transAxes,fontsize=8);ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.savefig(output/"rate_sweep.png",dpi=150);plt.close(fig)
    records=[json.loads(p.read_text()) for p in sorted(output.glob("*_N*/mechanism.json"))]
    if not records:return
    records.sort(key=lambda r:(r["case"]["optimizer"],r["case"]["n"],r["case"]["seed"],campaign.MAPS.index(r["case"]["coordinates"])))
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout="constrained")
    for record in records:
        c=record["case"];ax=axes[campaign.OPTIMIZERS.index(c["optimizer"]),(512,1024).index(c["n"])]
        with np.load(output/training.case_key(c)/"window_mse.npz") as w:
            color="C0" if c["coordinates"]=="scaled" else "C1"
            ax.loglog(w["step"],w["mean"],color=color,ls="-" if c["seed"]==0 else "--",label=f'{LABELS[c["coordinates"]]}, s{c["seed"]}, eta={c["eta"]:g}')
            ax.fill_between(w["step"],w["quantiles"][1],w["quantiles"][3],color=color,alpha=.08)
        ax.set(xlabel="Updates",ylabel="Training MSE: 20k mean",title=f'{c["optimizer"].upper()}, N={c["n"]}');ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.savefig(output/"training_progress.png",dpi=160);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout="constrained")
    for record in records:
        c=record["case"];ax=axes[campaign.OPTIMIZERS.index(c["optimizer"]),(512,1024).index(c["n"])]
        with np.load(output/training.case_key(c)/"history.npz") as h:
            core_mask=np.abs(h["centers"])<=1
            q=np.quantile(np.abs(h["lambda"][:,core_mask]),[.1,.5,.9],axis=1)
            color="C0" if c["coordinates"]=="scaled" else "C1"
            ax.semilogx(np.maximum(h["step"],1),q[1],color=color,ls="-" if c["seed"]==0 else "--",label=f'{LABELS[c["coordinates"]]}, s{c["seed"]}')
            ax.fill_between(np.maximum(h["step"],1),q[0],q[2],color=color,alpha=.06)
        ax.set(xlabel="Updates",ylabel="Core |lambda|: median and 10–90%",title=f'{c["optimizer"].upper()}, N={c["n"]}')
    for ax in axes.flat:
        ax.axhline(.25,color=".4",lw=1,ls=":",label="Construction reference 0.25")
        ax.set_yscale("symlog",linthresh=1e-4);ax.set_ylim(bottom=0);ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.savefig(output/"geometry_progress.png",dpi=160);plt.close(fig)
    for record in records:
        case,end=record["case"],record["end"];dest=output/training.case_key(case)
        with np.load(dest/f"spectrum_{end}.npz") as s,np.load(dest/f"dense_{end}.npz") as d:
            fig,axes=plt.subplots(2,3,figsize=(16,8),layout="constrained")
            for name in ("native","reference"):
                value=s[f"singular_{name}"];axes[0,0].semilogy(value/value[0],label=name)
            axes[0,0].set(xlabel="Index",ylabel="Relative singular value",title="Readout spectrum");axes[0,0].legend()
            bounds=d["band_bounds"];labels=["DC" if lo==0 else str(lo) if hi-lo==1 else f"{lo}–{hi-1}" for lo,hi in bounds]
            energy=d["band_mse"].mean(axis=0)
            axes[0,1].bar(np.arange(len(energy)),energy);axes[0,1].set_yscale("log");axes[0,1].set(ylabel="Residual MSE",title="16 stratified late-window states")
            axes[0,2].bar(np.arange(len(energy)),100*energy/energy.sum());axes[0,2].set(ylabel="Percent of residual MSE",title="Same bands, normalized")
            rel=d["singular_values"]/d["singular_values"][0];keep=rel>1e-14
            for field,label in (("singular_residual_coefficients","Residual"),("singular_readout_update_coefficients","Readout update")):
                axes[1,0].loglog(rel[keep],np.mean(d[field]**2,axis=0)[keep],".",label=label)
            axes[1,0].set(xlabel="Relative singular value",ylabel="Squared modal coefficient",title="Fixed window-start native basis");axes[1,0].legend()
            for field,label in (("band_readout_linear_mse_change","Readout"),("band_geometry_linear_mse_change","Geometry")):
                axes[1,1].plot(np.mean(d[field],axis=0),"o-",label=label)
            axes[1,1].set_yscale("symlog",linthresh=1e-18);axes[1,1].set(ylabel="Signed linear MSE change",title="Negative values reduce MSE");axes[1,1].legend()
            steps=sorted(map(int,record["checkpoints"]))
            for field,label in (("parallel_force","Shared with readout"),("perpendicular_force","Outside retained span")):
                axes[1,2].plot(steps,[record["checkpoints"][str(t)]["regions"]["core"][field] for t in steps],"o-",label=label)
            axes[1,2].set_xscale("symlog",linthresh=100);axes[1,2].set_yscale("symlog",linthresh=1e-14)
            axes[1,2].set(xlabel="Updates",ylabel="Signed core scale-growth force",title="Positive favors increasing |lambda|");axes[1,2].legend(fontsize=8)
            for ax in (axes[0,1],axes[0,2],axes[1,1]):ax.set_xticks(np.arange(len(labels)),labels,rotation=60,ha="right",fontsize=7)
            for ax in axes.flat:ax.grid(alpha=.15)
            fig.suptitle(f'{case["optimizer"].upper()}; {LABELS[case["coordinates"]]}; N={case["n"]}; seed {case["seed"]}; eta={case["eta"]:g}; update {end:,}')
            fig.savefig(dest/"mechanism.png",dpi=140);plt.close(fig)
        with np.load(dest/"history.npz") as h:
            fig,axes=plt.subplots(2,1,figsize=(12,7),sharex=True,layout="constrained")
            for step in sorted({0,100000,end}):
                hits=np.flatnonzero(h["step"]==step)
                if not len(hits):continue
                k=hits[0]
                axes[0].plot(h["centers"],h["c"][k,1:],".",ms=2,label=f'Update {step:,}')
                axes[1].plot(h["centers"],h["gamma"][k],".",ms=2,label=f'Update {step:,}')
            for ax in axes:
                ax.axvspan(h["centers"][0],-1,color=".94");ax.axvspan(1,h["centers"][-1],color=".94")
                ax.set_yscale("symlog",linthresh=.01 if ax==axes[0] else .1);ax.grid(alpha=.15);ax.legend()
            axes[0].set(ylabel="Physical signed readout w")
            axes[1].set(xlabel="Fixed physical center",ylabel="Physical signed gamma")
            fig.suptitle(f'{case["optimizer"].upper()}; {LABELS[case["coordinates"]]}; seed {case["seed"]}; eta={case["eta"]:g}')
            fig.savefig(dest/"parameter_snapshots.png",dpi=150);plt.close(fig)


def animations(output, optimizer_choice=None, seed_choice=None, workers=4):
    if optimizer_choice is None:
        with ProcessPoolExecutor(max_workers=min(workers,4)) as pool:
            futures=[pool.submit(animations,output,opt,seed) for opt in campaign.OPTIMIZERS for seed in (0,1)]
            for future in futures:future.result()
        return
    from matplotlib.animation import FuncAnimation,FFMpegWriter
    records=json.loads((output/"summary.json").read_text())
    for optimizer in campaign.OPTIMIZERS:
        for seed in (0,1):
            if (optimizer,seed)!=(optimizer_choice,seed_choice):continue
            pair=[]
            for coord in campaign.MAPS:
                candidates=[r for r in records if (r["case"]["optimizer"],r["case"]["coordinates"],r["case"]["n"],r["case"]["seed"])==(optimizer,coord,512,seed)]
                if len(candidates)==1:pair.append(candidates[0])
            if len(pair)!=2:continue
            for late in (False,True):
                histories=[]
                for record in pair:
                    folder=output/training.case_key(record["case"])
                    with np.load(folder/"history.npz") as h:history=dict(h)
                    if late:
                        with np.load(folder/f'dense_parameters_{record["end"]}.npz') as d:
                            history={"centers":history["centers"],**{k:d[k][-256:] for k in ("step","c","gamma")}}
                    histories.append(history)
                common=np.intersect1d(histories[0]["step"],histories[1]["step"])
                if late:frames=common;fps=20
                else:
                    common=[s for s in common if s<1000 or s<=300000 and s%10000==0 or s>300000 and s%100000==0 or s==common[-1]]
                    frames=np.repeat(common,[6 if s<=300000 else 1 for s in common]);fps=6
                fig,axes=plt.subplots(2,2,figsize=(13,7),sharex=True,layout="constrained")
                lines=[];bias=[]
                for col,(record,h) in enumerate(zip(pair,histories)):
                    for row,field in enumerate(("c","gamma")):
                        values=h[field][:,1:] if field=="c" else h[field]
                        if late:values=values-values[0]
                        bound=max(float(np.max(np.abs(values)))*1.1,1e-9)
                        line,=axes[row,col].plot(h["centers"],values[0],".",ms=3);lines.append((line,row,col,field))
                        axes[row,col].set(ylim=(-bound,bound),ylabel=("Change in " if late else "")+("physical w" if row==0 else "physical gamma"))
                        axes[row,col].set_yscale("symlog",linthresh=max(bound/1000,1e-15) if late else .01 if row==0 else .1)
                        if late:
                            exponent=np.floor(np.log10(bound))
                            ticks=10.**np.arange(exponent-2,exponent+1)
                            axes[row,col].set_yticks(np.r_[-ticks[::-1],0.,ticks])
                        axes[row,col].axvspan(h["centers"][0],-1,color=".93");axes[row,col].axvspan(1,h["centers"][-1],color=".93")
                        axes[row,col].grid(alpha=.15)
                    c=record["case"]
                    axes[0,col].set_title(f'{LABELS[c["coordinates"]]}; shared eta={c["eta"]:g}')
                    axes[1,col].set_xlabel("Fixed physical center")
                    bias.append(axes[0,col].text(.02,.98,"",transform=axes[0,col].transAxes,va="top"))
                title=fig.suptitle("")
                def update(index):
                    step=int(frames[index])
                    for line,row,col,field in lines:
                        h=histories[col];k=np.searchsorted(h["step"],step)
                        value=h[field][k]-h[field][0] if late else h[field][k]
                        line.set_ydata(value[1:] if field=="c" else value)
                        if field=="c":bias[col].set_text(f'Bias b={h["c"][k,0]:+.5g}')
                    title.set_text(f'{optimizer.upper()}; seed {seed}; update {step:,}; actual saved states\n'+
                                   (f'Consecutive updates; changes since {histories[0]["step"][0]:,}' if late else 'First 300k: one checkpoint/second; later: six/second'))
                movie=FuncAnimation(fig,update,frames=len(frames),interval=1000/fps,repeat=False)
                name=f'{optimizer}_seed_{seed}'+("_late" if late else "")
                movie.save(output/f'{name}.mp4',writer=FFMpegWriter(fps=fps,codec="libx264",bitrate=1400,extra_args=["-threads","1"]),dpi=100)
                update(len(frames)//2);fig.savefig(output/f'{name}_middle.png',dpi=100);plt.close(fig)
                run.write_json(output/f'{name}_animation.json',dict(cases=[training.case_key(r["case"]) for r in pair],steps=[int(s) for s in frames],fps=fps))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,required=True);parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--cases",type=Path);parser.add_argument("--end",type=int,default=100000)
    parser.add_argument("--workers",type=int,default=4);parser.add_argument("--summary-only",action="store_true")
    parser.add_argument("--figures-only",action="store_true")
    parser.add_argument("--movies",action="store_true")
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    if args.movies:animations(args.output,workers=args.workers);return
    if args.figures_only:figures(args.output);return
    rows=campaign.table(args.root)
    for row in rows:
        if row["eligible"]:
            g=core.geometry(row["n"])
            with np.load(args.root/row["key"]/"checkpoint_000100000.npz") as cp:
                row["core_abs_lambda_median_at_100k"]=float(np.median(np.abs(cp["lambda"][g.core])))
                row["physical_readout_norm_at_100k"]=float(np.linalg.norm(cp["c"]))
    run.write_json(args.output/"sweep_summary.json",rows)
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with (args.output/"sweep_summary.csv").open("w") as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    if not args.summary_only:
        cases=json.loads(args.cases.read_text())
        tasks=[(args.root,args.output,c,args.end) for c in cases if (args.root/training.case_key(c)/f"checkpoint_{args.end:09d}.npz").exists()
               and not json.loads((args.root/training.case_key(c)/"latest.json").read_text())["failed_update"]]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:records=list(pool.map(analyze_case,tasks))
        run.write_json(args.output/"summary.json",records)
    figures(args.output)


if __name__=="__main__":main()
