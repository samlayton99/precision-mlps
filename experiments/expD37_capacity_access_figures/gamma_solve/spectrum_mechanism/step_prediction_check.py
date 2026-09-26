"""Compare center-integral forecasts and finite-tanh spectral GD counts.

The integral is an approximation to the finite center sum, not the collaborator
note's periodic Fourier construction. All forecasts use the finite model's
learning rate. No polynomial approximation or fitted optimization rate is used.
"""
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/precisionmlps-mpl")
import numpy as np
from scipy.linalg import svd
from threadpoolctl import threadpool_limits
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run import design, center_quadrature, OUT
sys.path.insert(0, str(HERE.parent))
from core import Geometry, target, weights


def spectrum(J, y, eta=None):
    u, s, _ = svd(J, full_matrices=False, lapack_driver="gesvd",
                  check_finite=False)
    if eta is None:
        eta = .5 / s[0]**2
    rates = np.r_[eta * s*s, 0.]
    assert np.all((rates >= 0) & (rates < 1))
    return rates, weights(u, y), float(eta)


def error(rates, p, n):
    return float(np.sqrt(p @ np.exp(2*float(n)*np.log1p(-rates))))


def first_hit(rates, p, epsilon=.01, maximum=10**15):
    if error(rates, p, maximum) > epsilon:
        return None
    lo, hi = 0, 1
    while error(rates, p, hi) > epsilon:
        lo, hi = hi, min(2*hi, maximum)
    while hi-lo > 1:
        mid = (lo+hi)//2
        if error(rates, p, mid) <= epsilon:
            hi = mid
        else:
            lo = mid
    return hi


def main():
    geometry = Geometry()
    x, c = geometry.arrays()
    y = target(x, "mixed")/np.sqrt(len(x))
    rows = []
    arrays = {}
    sample_steps = np.unique(np.r_[0, np.geomspace(1, 2e12, 900).astype(np.int64)])
    arrays["steps"] = sample_steps
    for gamma in [4., 8., 16., 64.]:
        J, _ = design(x, c, gamma)
        rates, p, eta = spectrum(J, y)
        direct = first_hit(rates, p)
        row = dict(gamma=gamma, eta=eta, finite_tanh_spectral_steps=direct,
                   finite_tanh_error_before=error(rates, p, direct-1),
                   finite_tanh_error_at=error(rates, p, direct),
                   integral_quadrature=[])
        arrays[f"gamma{int(gamma)}_finite"] = np.array([error(rates,p,n) for n in sample_steps])
        for order in [8, 16]:
            nodes, w = center_quadrature(geometry, order)
            Ji, _ = design(x, nodes, gamma, w)
            irates, ip, _ = spectrum(Ji, y, eta)
            predicted = first_hit(irates, ip)
            row["integral_quadrature"].append(dict(order=order, steps=predicted,
                relative_step_error=(predicted-direct)/direct,
                relative_kernel_operator_error=float(np.linalg.norm(
                    Ji@Ji.T-J@J.T, 2)/(rates[0]/eta))))
            if order == 16:
                arrays[f"gamma{int(gamma)}_integral"] = np.array([error(irates,ip,n) for n in sample_steps])
        rows.append(row)
        print(json.dumps(row), flush=True)
    result = dict(N=geometry.N, W=len(c), m=len(x), halo_per_side=geometry.halo,
        target="sin(2*pi*x)+0.5*sin(6*pi*x)+0.25*sin(10*pi*x)",
        loss="0.5*mean((Phi*v-y)^2)", zero_initial_readout=True,
        learning_rate="0.5/lambda_max(K_finite) for BOTH finite and integral models",
        epsilon=.01, interval_certified=False, gd_executed=False,
        integral="[1+(1/h)*integral_a^b tanh(gamma*(x-c))*tanh(gamma*(xprime-c))dc]/m",
        integration_bounds=list(geometry.bounds),
        independent_fourier_predictor=False, rows=rows)
    (OUT/"data/step_prediction_check.json").write_text(json.dumps(result,indent=2)+"\n")
    np.savez_compressed(OUT/"data/step_prediction_check.npz", **arrays)
    plot(rows, arrays)


def plot(rows, arrays):
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.1))
    colors = plt.colormaps["viridis"]([.08,.34,.62,.88])
    ax = axes[0]
    for row, color in zip(rows,colors):
        g = int(row["gamma"])
        # Stop each curve just after its first 1% crossing.
        keep = (arrays["steps"] >= 1) & (arrays["steps"] <= 2*row["finite_tanh_spectral_steps"])
        ax.loglog(arrays["steps"][keep], arrays[f"gamma{g}_finite"][keep], color=color, lw=2)
        ax.loglog(arrays["steps"][keep], arrays[f"gamma{g}_integral"][keep], color=color, lw=1.4, ls="--")
    ax.axhline(.01,color=".35",lw=1,ls=":")
    ax.set(xlabel="GD step n", ylabel="Training relative L2 error",ylim=(.004,1.1),
           xlim=(1,2e12),title="Learning curves")
    ax.grid(alpha=.15)
    handles=[Line2D([0],[0],color="black",lw=2,label="Finite tanh spectrum"),
             Line2D([0],[0],color="black",lw=1.4,ls="--",label="Center-integral approximation")]
    ax.legend(handles=handles,loc="lower center",bbox_to_anchor=(.5,1.08),ncol=1,frameon=False,fontsize=9)
    ax = axes[1]
    xpos=np.arange(4)
    finite=[r["finite_tanh_spectral_steps"] for r in rows]
    approx=[r["integral_quadrature"][-1]["steps"] for r in rows]
    ax.plot(xpos,finite,"o-",color="#333333",label="Finite tanh spectrum")
    ax.plot(xpos,approx,"x--",color="#b85b28",label="Center-integral approximation")
    for i,(row,color) in enumerate(zip(rows,colors)):
        delta=100*row["integral_quadrature"][-1]["relative_step_error"]
        count_label = f"{finite[i]:.4e}" if finite[i] > 1e10 else f"{finite[i]:,}"
        ax.annotate(f"{count_label}\nIntegral error: {delta:+.3f}%",(i,finite[i]),
                    xytext=(0,12),textcoords="offset points",
                    ha="left" if i == 0 else "right" if i == 3 else "center",fontsize=9)
    ax.set_yscale("log")
    ax.set(xticks=xpos,xticklabels=["4","8","16","64"],xlabel="Fixed gamma",ylabel="First step reaching 1% relative L2",ylim=(4e3,1e14),title="Step counts")
    ax.grid(axis="y",alpha=.15)
    ax.legend(loc="lower center",bbox_to_anchor=(.5,1.08),frameon=False,fontsize=9)
    fig.suptitle("Mixed sine | 153 tanh neurons + bias | 263 samples | zero readout",y=.99,fontsize=12)
    gamma_handles=[Line2D([0],[0],color=co,lw=3,label=rf"$\gamma={int(ro['gamma'])}$") for ro,co in zip(rows,colors)]
    fig.legend(handles=gamma_handles,loc="lower center",bbox_to_anchor=(.5,.01),ncol=4,frameon=False)
    fig.subplots_adjust(left=.07,right=.98,bottom=.18,top=.72,wspace=.28)
    fig.savefig(OUT/"step_prediction_check.png",dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    with threadpool_limits(1):
        main()
