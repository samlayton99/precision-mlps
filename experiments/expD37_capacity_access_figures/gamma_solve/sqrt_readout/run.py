"""Actual-tanh spectral GD versus parallel frozen-readout Adam on a square root.

Separate commands let CPU spectral evaluation and CUDA Adam run concurrently.
The problem archive is shared verbatim between machines. No kernel approximation,
whitening, center training, or scale training enters either arm.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np


def prepare(out, domain):
    out.mkdir(parents=True, exist_ok=True)
    lo, hi = (-1., 1.) if domain == "mapped" else (0., 1.)
    N, m, halo = 128, 263, 12
    h = (hi-lo)/N
    centers = lo+h*np.arange(-halo, N+halo+1)
    x = np.linspace(lo, hi, m)
    y = np.sqrt(np.maximum((x-lo)/(hi-lo), 0))
    gammas = np.array([1., 4., 8., 16., 32., 64.])
    hidden = np.tanh(gammas[:,None,None]*(x[None,:,None]-centers[None,None,:]))
    phi = np.concatenate([np.ones((len(gammas),m,1)), hidden], axis=2)
    cfg = dict(target="sqrt((x+1)/2)" if domain == "mapped" else "sqrt(x)",
               domain=[lo,hi], N=N, m=m, W=len(centers), halo_per_side=halo,
               h=h, gammas=gammas.tolist(), lambdas=(h*gammas).tolist(),
               steps=20000, dtype="float64", initial_readout="zero",
               geometry="frozen, uniform centers, common gamma", coordinates="raw",
               samples="uniform including endpoints", metric="training relative L2 norm",
               loss="0.5*mean((Phi*v-y)^2)", gd_step="0.5/lambda_max(K)",
               adam_lr=.001, adam_betas=[.9,.999], adam_eps=1e-8,
               adam_weight_decay=0., adam_schedule="constant")
    np.savez_compressed(out/"problem.npz", x=x, centers=centers, y=y, phi=phi, gammas=gammas)
    (out/"config.json").write_text(json.dumps(cfg,indent=2)+"\n")
    print(json.dumps(cfg), flush=True)


def load(out):
    with np.load(out/"problem.npz") as f:
        data = dict(f)
    cfg=json.loads((out/"config.json").read_text())
    return cfg,data


def gd(out):
    from scipy.linalg import svd
    from threadpoolctl import threadpool_limits
    cfg,data=load(out)
    y=data["y"]/np.sqrt(cfg["m"])
    ynorm=np.linalg.norm(y)
    steps=np.arange(cfg["steps"]+1)
    long_steps=np.unique(np.r_[0,np.arange(101),np.geomspace(1,1e9,1800).astype(np.int64)])
    short_curves=[]; long_curves=[]; singular=[]; target_weights=[]; rows=[]
    with threadpool_limits(1):
        for gamma,phi in zip(data["gammas"],data["phi"]):
            J=phi/np.sqrt(cfg["m"])
            u,s,vt=svd(J,full_matrices=False,lapack_driver="gesvd",check_finite=False)
            a=u.T@y
            null=y-u@a
            p=np.r_[(a/ynorm)**2, np.dot(null,null)/ynorm**2]
            rates=np.r_[.5*(s/s[0])**2,0.]
            logs=np.log1p(-rates)
            def error(ns):
                ns=np.atleast_1d(ns)
                return np.sqrt(np.exp(2*ns[:,None]*logs[None,:])@p)
            lo,hi=0,10**15
            hit=None
            if error(hi)[0] <= .01:
                while hi-lo>1:
                    mid=(hi+lo)//2
                    if error(mid)[0] <= .01: hi=mid
                    else: lo=mid
                hit=hi
            # Capacity reference uses an explicit numerical SVD cutoff, not the
            # unconstrained nullspace estimate of a near-rank-deficient matrix.
            keep=s > 1e-14*s[0]
            coefficient=vt[keep].T@(a[keep]/s[keep])
            floor=float(np.linalg.norm(J@coefficient-y)/ynorm)
            short=error(steps); long=error(long_steps)
            short_curves.append(short);long_curves.append(long)
            singular.append(s);target_weights.append(p)
            rows.append(dict(gamma=float(gamma),eta=float(.5/s[0]**2),
                first_hit_one_percent=hit,final_error=float(short[-1]),
                numerical_lstsq_relative_error=floor,lstsq_rcond=1e-14,
                spectral_zero_error=float(short[0])))
    np.savez_compressed(out/"gd.npz", steps=steps, errors=np.array(short_curves).T,
                        long_steps=long_steps,long_errors=np.array(long_curves).T,
                        singular_values=np.array(singular),target_weights=np.array(target_weights))
    (out/"gd_summary.json").write_text(json.dumps(rows,indent=2)+"\n")
    print(json.dumps(rows,indent=2),flush=True)


def verify(out):
    """Check the spectral trajectory and batched analytic Adam gradient."""
    import torch
    from scipy.linalg import svd
    from threadpoolctl import threadpool_limits
    torch.set_num_threads(1)
    cfg,data=load(out)
    with threadpool_limits(1):
        J=data["phi"][3]/np.sqrt(cfg["m"])
        y=data["y"]/np.sqrt(cfg["m"])
        u,s,_=svd(J,full_matrices=False)
        eta=.5/s[0]**2
        v=np.zeros(J.shape[1]);r=-y.copy()
        for _ in range(200): v-=eta*(J.T@r);r=J@v-y
        a=u.T@y
        predicted=-(u@(a*np.exp(200*np.log1p(-eta*s*s))))-(y-u@a)
        gd_error=float(np.linalg.norm(r-predicted)/np.linalg.norm(y))
        assert gd_error<1e-12
    B=torch.from_numpy(data["phi"]/np.sqrt(cfg["m"]))
    Y=torch.from_numpy(y)[None,:,None]
    manual=torch.nn.Parameter(torch.zeros((len(B),B.shape[2],1),dtype=torch.float64))
    automatic=torch.nn.Parameter(torch.zeros_like(manual))
    opts=[torch.optim.Adam([p],lr=cfg["adam_lr"],betas=cfg["adam_betas"],
                          eps=cfg["adam_eps"],foreach=False) for p in [manual,automatic]]
    for _ in range(30):
        with torch.no_grad(): manual.grad=B.transpose(1,2)@(B@manual-Y)
        opts[0].step()
        opts[1].zero_grad()
        (.5*(B@automatic-Y).square().sum()).backward()
        opts[1].step()
    difference=float(torch.max(abs(manual-automatic)).detach())
    assert difference < 1e-13
    result=dict(gd_200_step_relative_residual_vector_difference=gd_error,
                adam_30_step_max_coefficient_difference_vs_autograd=difference)
    (out/"verification.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result),flush=True)


def adam(out, device):
    import torch
    cfg,data=load(out)
    torch.set_num_threads(1)
    device=torch.device(device)
    if device.type=="cuda":
        torch.backends.cuda.matmul.allow_tf32=False
    B=torch.tensor(data["phi"]/np.sqrt(cfg["m"]),dtype=torch.float64,device=device)
    BT=B.transpose(1,2).contiguous()
    y=torch.tensor(data["y"]/np.sqrt(cfg["m"]),dtype=torch.float64,device=device)[None,:,None]
    ynorm=torch.linalg.vector_norm(y)
    v=torch.nn.Parameter(torch.zeros((len(B),B.shape[2],1),dtype=torch.float64,device=device))
    optimizer=torch.optim.Adam([v],lr=cfg["adam_lr"],betas=cfg["adam_betas"],
                               eps=cfg["adam_eps"],weight_decay=0.,foreach=False)
    history=torch.empty((cfg["steps"]+1,len(B)),dtype=torch.float64,device=device)
    if device.type=="cuda": torch.cuda.synchronize(device)
    start=time.perf_counter()
    with torch.no_grad():
        for step in range(cfg["steps"]):
            residual=B@v-y
            history[step]=torch.linalg.vector_norm(residual,dim=(1,2))/ynorm
            v.grad=BT@residual
            optimizer.step()
        history[-1]=torch.linalg.vector_norm(B@v-y,dim=(1,2))/ynorm
    if device.type=="cuda": torch.cuda.synchronize(device)
    seconds=time.perf_counter()-start
    errors=history.cpu().numpy()
    assert np.isfinite(errors).all()
    np.savez_compressed(out/"adam.npz",steps=np.arange(cfg["steps"]+1),errors=errors,
                        coefficients=v.detach().cpu().numpy(),gammas=data["gammas"])
    # Optimizer state is retained so the requested horizon can be extended.
    torch.save(dict(step=cfg["steps"],coefficients=v.detach(),optimizer=optimizer.state_dict(),
                    config=cfg),out/"adam_checkpoint.pt")
    rows=[]
    for j,gamma in enumerate(data["gammas"]):
        hits=np.flatnonzero(errors[:,j]<=.01)
        rows.append(dict(gamma=float(gamma),first_hit_one_percent=int(hits[0]) if len(hits) else None,
                         final_error=float(errors[-1,j]),best_error=float(errors[:,j].min()),
                         best_step=int(errors[:,j].argmin())))
    result=dict(device=str(device),device_name=torch.cuda.get_device_name(device) if device.type=="cuda" else "CPU",
                torch_version=torch.__version__,seconds=seconds,rows=rows,
                problem_sha256=hashlib.sha256((out/"problem.npz").read_bytes()).hexdigest(),
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out/"adam_summary.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2),flush=True)


def plot(out):
    os.environ.setdefault("MPLCONFIGDIR","/tmp/precisionmlps-mpl")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    cfg,data=load(out)
    with np.load(out/"gd.npz") as f: g=dict(f)
    with np.load(out/"adam.npz") as f: a=dict(f)
    adam_device=json.loads((out/"adam_summary.json").read_text())["device_name"]
    assert np.array_equal(g["steps"],a["steps"])
    colors=plt.colormaps["viridis"](np.linspace(.05,.87,len(data["gammas"])))
    handles=[Line2D([0],[0],color=c,lw=2,label=rf"$\gamma={v:g}$") for c,v in zip(colors,data["gammas"])]
    ymin=10**np.floor(np.log10(min(g["errors"].min(),a["errors"].min())*.7))
    ymax=max(1.15,1.1*a["errors"].max())
    fig,axes=plt.subplots(1,2,figsize=(12.5,5.1),sharex=True,sharey=True)
    for ax,curves,title in zip(axes,[g["errors"],a["errors"]],
                              ["GD: actual tanh spectrum",rf"Adam: executed on {adam_device}, learning rate $10^{{-3}}$"]):
        for j,color in enumerate(colors): ax.plot(g["steps"],curves[:,j],color=color,lw=1.35)
        ax.axhline(.01,color="black",ls="--",lw=1)
        ax.text(19000,.0115,"1%",ha="right",fontsize=10)
        ax.set_xscale("symlog",linthresh=10,linscale=.5)
        ax.set_yscale("log")
        ax.set(xlim=(0,cfg["steps"]),ylim=(ymin,ymax),xlabel="Step (logarithmic after step 10)",title=title)
        ax.grid(alpha=.16)
    axes[0].set_ylabel("Training relative L2 error")
    target_label=r"$f(x)=\sqrt{(x+1)/2}$ on $[-1,1]$" if cfg["domain"][0]<0 else r"$f(x)=\sqrt{x}$ on $[0,1]$"
    fig.suptitle(target_label+" | frozen geometry, zero readout",y=.99,fontsize=13)
    fig.legend(handles=handles,loc="upper center",bbox_to_anchor=(.5,.925),ncol=len(handles),frameon=False)
    fig.text(.5,.025,f"{cfg['W']} tanh neurons + bias; {cfg['m']} endpoint-inclusive samples; FP64; raw curves, no best-so-far filtering.",ha="center",fontsize=9)
    fig.subplots_adjust(left=.085,right=.98,bottom=.17,top=.76,wspace=.16)
    fig.savefig(out.parent/"gd_vs_adam_20k.png",dpi=180,bbox_inches="tight")
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,5.1))
    for j,color in enumerate(colors): ax.plot(g["long_steps"],g["long_errors"][:,j],color=color,lw=1.7)
    ax.axhline(.01,color="black",ls="--",lw=1)
    ax.text(8e8,.0115,"1%",ha="right")
    ax.axvline(cfg["steps"],color=".4",ls=":",lw=1)
    ax.text(cfg["steps"]*1.2,.8,"20,000 steps",rotation=90,va="top",color=".4")
    ax.set_xscale("symlog",linthresh=10,linscale=.5);ax.set_yscale("log")
    low=10**np.floor(np.log10(g["long_errors"].min()*.7))
    ax.set(xlim=(0,1e9),ylim=(low,1.15),xlabel="GD step (logarithmic after step 10)",ylabel="Training relative L2 error")
    ax.grid(alpha=.16)
    fig.suptitle("GD from the actual tanh spectrum — "+target_label,y=.99,fontsize=12)
    fig.legend(handles=handles,loc="upper center",bbox_to_anchor=(.5,.925),ncol=len(handles),frameon=False)
    fig.subplots_adjust(left=.11,right=.98,bottom=.15,top=.78)
    fig.savefig(out.parent/"gd_spectral_long_range.png",dpi=180,bbox_inches="tight")
    plt.close(fig)
    print(out.parent/"gd_vs_adam_20k.png",flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("action",choices=["prepare","gd","verify","adam","plot"])
    parser.add_argument("--data",type=Path,required=True)
    parser.add_argument("--domain",choices=["mapped","positive"],default="mapped")
    parser.add_argument("--device",default="cuda:4")
    args=parser.parse_args()
    if args.action=="prepare": prepare(args.data,args.domain)
    elif args.action=="adam": adam(args.data,args.device)
    else: globals()[args.action](args.data)
