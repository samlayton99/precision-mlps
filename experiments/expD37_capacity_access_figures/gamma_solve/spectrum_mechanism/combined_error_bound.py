"""Evaluate the finite-ratio theorem as a full-target GD error bound.

No training and no integral-spectrum forecast. Every target weight is retained;
unresolved or zero ratio bounds get zero decay. All enclosures are numerical
FP64 evaluations of exact-arithmetic inequalities, not interval certificates.
"""
import json
import os
from collections import Counter
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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from finite_ratio_bound import main_kernel, symmetric
from run import Geometry, design, OUT
from core import target, weights

DEST = OUT / "combined_error_bound"
PAIRS = [(2., 4.), (4., 8.), (8., 16.), (16., 32.), (32., 64.), (8., 64.)]
TOL = .01
MAX_STEPS = 10**18
EPS = np.finfo(float).eps


def squared_error(ratios, p, n):
    return float(p @ np.exp(2 * float(n) * np.log1p(-ratios / 2)))


def crossing(ratios, p):
    floor = float(np.sqrt(p[ratios == 0].sum()))
    if floor >= TOL:
        return None, "nondecaying_bound_above_tolerance", floor
    if squared_error(ratios, p, MAX_STEPS) > TOL**2:
        return None, "beyond_search_limit", floor
    lo, hi = 0, 1
    while squared_error(ratios, p, hi) > TOL**2:
        lo, hi = hi, min(2 * hi, MAX_STEPS)
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if squared_error(ratios, p, mid) <= TOL**2:
            hi = mid
        else:
            lo = mid
    assert squared_error(ratios, p, hi) <= TOL**2
    assert squared_error(ratios, p, hi - 1) > TOL**2
    return hi, "crossed", floor


def monotone_lower_bounds(raw):
    # If j >= i, lambda_i >= lambda_j >= raw_j * lambda_1.
    return np.maximum.accumulate(raw[::-1])[::-1]


def run(svd_driver="gesvd"):
    geom = Geometry()
    x, centers = geom.arrays()
    y = target(x, "mixed") / np.sqrt(len(x))
    snapshots = {}
    for gamma in sorted({g for pair in PAIRS for g in pair}):
        b, _ = design(x, centers, gamma)
        u, s, _ = la.svd(b, full_matrices=False, lapack_driver=svd_driver)
        k = b @ b.T
        snapshots[gamma] = dict(b=b, u=u, s=s, eigenvalues=s*s,
            main=main_kernel(geom, gamma), p=weights(u, y),
            top_upper=float(np.max(np.sum(np.abs(k), axis=1))))
    steps = np.unique(np.r_[0, np.geomspace(1, MAX_STEPS, 1500).astype(np.int64)])
    arrays = {"steps": steps, "x": x, "centers": centers, "y": y}
    records = []
    for ga, gb in PAIRS:
        old, new = snapshots[ga], snapshots[gb]
        label = f"g{int(ga)}_to_g{int(gb)}"
        dim = len(old["s"])
        initial_threshold = 10 * EPS * max(old["b"].shape) * old["s"][0]
        u = old["u"]
        delta = new["main"] - old["main"]
        c_all = symmetric(np.diag(old["eigenvalues"]) + u.T @ delta @ u)
        ub = u.T @ new["b"]
        actual_compressed = symmetric(ub @ ub.T)
        # Same resolution policy as finite_ratio_transitions.py.
        compressed_floor = 64 * EPS * (old["eigenvalues"][0] + la.norm(delta, "fro"))
        actual_ratios = np.r_[new["eigenvalues"] / new["eigenvalues"][0], 0.]
        raw = np.zeros(dim + 1)
        subspace = np.zeros(dim + 1)
        subspace_exact_top = np.zeros(dim + 1)
        rank_rows = []
        for rank in range(1, dim + 1):
            gap = old["s"][rank-1] - (old["s"][rank] if rank < dim else 0.)
            resolved = old["s"][rank-1] > initial_threshold and gap > initial_threshold
            row = dict(rank=rank, actual_ratio=float(actual_ratios[rank-1]),
                       lower_bound=0., status="starting_subspace_unresolved")
            if not resolved:
                rank_rows.append(row)
                continue
            # Upper ceiling on what this particular old trial space can prove.
            minimum_actual = float(la.svdvals(ub[:rank])[-1]**2)
            subspace[rank-1] = minimum_actual / new["top_upper"]
            subspace_exact_top[rank-1] = minimum_actual / new["eigenvalues"][0]
            assert subspace_exact_top[rank-1] <= actual_ratios[rank-1] * (1 + 1e-5)
            c = c_all[:rank, :rank]
            ev, eq = la.eigh(c)
            row.update(main_min=float(ev[0]), compressed_resolution_floor=float(compressed_floor),
                       exact_subspace_ratio=float(subspace[rank-1]))
            if ev[0] <= compressed_floor:
                row["status"] = "compressed_matrix_unresolved"
                rank_rows.append(row)
                continue
            rem = symmetric(actual_compressed[:rank, :rank] - c)
            norm_rem = symmetric((eq.T @ rem @ eq) / np.sqrt(ev[:, None]*ev[None, :]))
            epsilon = float(np.max(np.abs(la.eigvalsh(norm_rem))))
            epsilon_check = float(np.max(np.abs(la.eigvalsh(rem, c))))
            row.update(epsilon=epsilon, epsilon_solver_difference=abs(epsilon-epsilon_check))
            if abs(epsilon-epsilon_check) > 2e-5 * max(1., epsilon):
                row["status"] = "correction_unresolved"
                rank_rows.append(row)
                continue
            numerator = max(1-epsilon, 0.) * ev[0]
            lower = numerator / new["top_upper"]
            if numerator > minimum_actual * (1+1e-5) or lower > actual_ratios[rank-1]*(1+1e-5):
                row["status"] = "inequality_unresolved"
                rank_rows.append(row)
                continue
            raw[rank-1] = lower
            row.update(lower_bound=float(lower), status="positive" if lower > 0 else "zero")
            rank_rows.append(row)
        bounded = monotone_lower_bounds(raw)
        subspace = monotone_lower_bounds(subspace)
        subspace_exact_top = monotone_lower_bounds(subspace_exact_top)
        p = new["p"]
        assert np.all(bounded <= actual_ratios*(1+1e-5))
        assert np.all(bounded <= 1.)
        actual_n, actual_status, actual_floor = crossing(actual_ratios, p)
        bound_n, bound_status, bound_floor = crossing(bounded, p)
        raw_n, _, raw_floor = crossing(raw, p)
        trial_n, _, trial_floor = crossing(subspace, p)
        trial_exact_n, _, _ = crossing(subspace_exact_top, p)
        actual_curve = np.array([squared_error(actual_ratios, p, n) for n in steps])
        bound_curve = np.array([squared_error(bounded, p, n) for n in steps])
        raw_curve = np.array([squared_error(raw, p, n) for n in steps])
        trial_curve = np.array([squared_error(subspace, p, n) for n in steps])
        assert np.min(bound_curve-actual_curve) >= -1e-12
        if bound_n is not None:
            assert bound_n >= actual_n
        zeros = np.flatnonzero(bounded == 0)
        tail = np.r_[np.cumsum(p[::-1])[::-1], 0.]
        # Smallest number of leading eigenvectors covering 1 - tolerance^2 energy.
        needed_rank = next((i for i in range(len(p)+1) if tail[i] < TOL**2), None)
        row = dict(reference_gamma=ga, gamma=gb, label=label,
            actual_steps=actual_n, actual_status=actual_status,
            bound_steps=bound_n, bound_status=bound_status,
            slowdown=None if bound_n is None else bound_n/actual_n,
            bound_error_at_actual_crossing=float(np.sqrt(squared_error(bounded,p,actual_n))) if actual_n else None,
            bound_asymptotic_error=bound_floor, actual_asymptotic_error=actual_floor,
            raw_bound_steps=raw_n, raw_bound_asymptotic_error=raw_floor,
            exact_subspace_steps=trial_n, exact_subspace_asymptotic_error=trial_floor,
            exact_subspace_exact_top_steps=trial_exact_n,
            last_positive_bound_rank=int(np.flatnonzero(bounded>0)[-1]+1) if np.any(bounded>0) else 0,
            rank_needed_for_one_percent=needed_rank,
            status_counts=dict(Counter(r["status"] for r in rank_rows)),
            lambda1=float(new["eigenvalues"][0]), lambda1_upper=new["top_upper"],
            lambda1_upper_factor=new["top_upper"]/new["eigenvalues"][0],
            minimum_curve_gap=float(np.min(bound_curve-actual_curve)), ranks=rank_rows)
        records.append(row)
        for name, array in dict(actual_ratios=actual_ratios, lower_ratios=bounded,
                raw_lower_ratios=raw, p=p, exact_subspace_ratios=subspace,
                actual_error_squared=actual_curve, bound_error_squared=bound_curve,
                raw_bound_error_squared=raw_curve, exact_subspace_error_squared=trial_curve).items():
            arrays[f"{label}_{name}"] = array
        print(json.dumps({k:v for k,v in row.items() if k != "ranks"}), flush=True)
    # Reproduce the previously saved moderate-rank bounds and direct counts.
    previous = json.loads((OUT/"data/finite_ratio_bound.json").read_text())
    checks = []
    for rec in records:
        if rec["reference_gamma"] != 8:
            continue
        for rank in [12, 20, 32]:
            match = next((r for r in previous["rows"] if r["gamma"] == rec["gamma"] and r["rank"] == rank), None)
            if match:
                now = rec["ranks"][rank-1]["lower_bound"]
                relative = abs(now-match["lower_bound"])/match["lower_bound"]
                if svd_driver == "gesvd":
                    assert relative < 2e-5
                checks.append(dict(gamma=rec["gamma"],rank=rank,relative_difference=relative))
    saved_counts = json.loads((OUT/"data/step_prediction_check.json").read_text())
    for rec in records:
        old = next((r for r in saved_counts["rows"] if r["gamma"] == rec["gamma"]), None)
        if old:
            if svd_driver == "gesvd":
                assert rec["actual_steps"] == old["finite_tanh_spectral_steps"]
    result = dict(geometry=dict(N=geom.N,m=geom.m,W=len(centers),h=geom.h,
                  halo_per_side=geom.halo),target="sin(2*pi*x)+0.5*sin(6*pi*x)+0.25*sin(10*pi*x)",
                  tolerance=TOL,maximum_steps=MAX_STEPS,learning_rate="0.5/lambda1(actual finite K)",
                  bound="sum_i p_i(gamma)*(1-lower_ratio_i/2)^(2*n)",
                  ratio_envelope="b_i=max_{j>=i} raw_bound_j; justified by eigenvalue ordering",
                  unresolved="zero bound, retain all target energy; no optimistic truncation",
                  precision="FP64 numerical evaluation; NOT interval certified",training_executed=False,
                  svd_driver=svd_driver,
                  previous_bound_checks=checks,records=records)
    DEST.joinpath("data").mkdir(parents=True,exist_ok=True)
    (DEST/"data/results.json").write_text(json.dumps(result,indent=2)+"\n")
    np.savez_compressed(DEST/"data/results.npz",**arrays)
    return result, arrays


def plot(result, arrays):
    plt.rcParams.update({"font.size":10,"axes.titlesize":12,"axes.labelsize":11})
    fig, axes = plt.subplots(2,3,figsize=(15,8.7),sharex=True,sharey=True)
    for ax, rec in zip(axes.flat,result["records"]):
        key=rec["label"]
        n=arrays["steps"][1:]
        ax.loglog(n,np.sqrt(arrays[f"{key}_actual_error_squared"][1:]),color="#31688e",lw=2)
        ax.loglog(n,np.sqrt(arrays[f"{key}_bound_error_squared"][1:]),color="#b55c00",lw=2,ls="--")
        ax.axhline(TOL,color=".35",lw=1,ls=":")
        ax.scatter([rec["actual_steps"]],[TOL],color="#31688e",s=26,zorder=5)
        if rec["bound_steps"] is not None:
            ax.scatter([rec["bound_steps"]],[TOL],facecolors="white",edgecolors="#b55c00",s=34,zorder=5)
            caption=f"Actual: {rec['actual_steps']:,} steps\nBound: {rec['bound_steps']:,} ({rec['slowdown']:.2f}x)"
        else:
            caption=f"Actual: {rec['actual_steps']:,} steps\nBound stays above {100*rec['bound_asymptotic_error']:.2f}%"
        ax.text(.98,.97,caption,transform=ax.transAxes,ha="right",va="top",fontsize=9,
                bbox=dict(facecolor="white",edgecolor="none",alpha=.9))
        ax.set_title(rf"$\gamma_0={int(rec['reference_gamma'])}\;\to\;\gamma={int(rec['gamma'])}$",pad=10)
        ax.set(xlim=(1,MAX_STEPS),ylim=(1e-4,1.2),xticks=[1,1e4,1e8,1e12,1e16],yticks=[1,1e-1,1e-2,1e-3,1e-4])
        ax.grid(which="major",alpha=.18)
    for ax in axes[-1]: ax.set_xlabel("GD step n (spectral calculation)")
    for ax in axes[:,0]: ax.set_ylabel("Relative L2 error")
    handles=[Line2D([],[],color="#31688e",lw=2,label="Actual finite-kernel residual"),
             Line2D([],[],color="#b55c00",lw=2,ls="--",label="Upper residual bound from ratio theorem"),
             Line2D([],[],color=".35",lw=1,ls=":",label="1% target")]
    fig.suptitle("Does the eigenvalue-ratio bound give a useful bound on the full error?",fontsize=16,y=.98)
    fig.text(.5,.942,"Mixed sine | 153 tanh neurons + bias | 263 samples | zero readout",ha="center")
    fig.legend(handles=handles,loc="upper center",bbox_to_anchor=(.5,.925),ncol=3,frameon=False)
    fig.text(.5,.018,"Each panel trains at its final gamma; the reference gamma only constructs the bound. Geometry stays frozen.\n"
             "Every target component is included. Unresolved ratios contribute nondecaying error.\n"
             "Dashed curves use the ratio theorem, not the earlier integral-spectrum prediction. FP64 evaluation; no interval certification.",
             ha="center",fontsize=9,color=".3")
    fig.subplots_adjust(left=.07,right=.985,top=.84,bottom=.13,hspace=.30,wspace=.17)
    fig.savefig(DEST/"combined_error_bound.png",dpi=180)
    plt.close(fig)

    fig,axes=plt.subplots(2,3,figsize=(15,8.7),sharex=True)
    for ax,rec in zip(axes.flat,result["records"]):
        key=rec["label"]
        ranks=np.arange(1,len(arrays[f"{key}_p"])+1)
        actual=arrays[f"{key}_actual_ratios"]
        bounded=arrays[f"{key}_lower_ratios"]
        mass=arrays[f"{key}_p"]
        ax.semilogy(ranks,actual,color="#31688e",lw=1.8)
        ax.semilogy(ranks,np.where(bounded>0,bounded,np.nan),color="#b55c00",lw=1.8,ls="--")
        ax.axvline(rec["last_positive_bound_rank"]+.5,color="#b55c00",lw=1,ls=":")
        other=ax.twinx()
        tail=np.cumsum(mass[::-1])[::-1]
        other.semilogy(ranks,tail,color="#35a779",lw=1.3,alpha=.85)
        other.axhline(TOL**2,color="#35a779",ls=":",lw=.8)
        other.set_ylim(1e-8,1.1)
        if ax in axes[:,2]: other.set_ylabel("Target energy at this rank and higher",color="#24855b")
        else: other.set_yticklabels([])
        ax.set(title=rf"$\gamma_0={int(rec['reference_gamma'])}\;\to\;\gamma={int(rec['gamma'])}$",
               xlim=(1,100),ylim=(1e-20,1.3),yticks=[1,1e-5,1e-10,1e-15,1e-20])
        ax.grid(alpha=.15)
    for ax in axes[-1]: ax.set_xlabel("Eigenvalue rank i")
    for ax in axes[:,0]: ax.set_ylabel(r"Normalized rate $\lambda_i/\lambda_1$")
    handles=[Line2D([],[],color="#31688e",lw=2,label="Actual ratio"),
             Line2D([],[],color="#b55c00",lw=2,ls="--",label="Lower ratio bound"),
             Line2D([],[],color="#35a779",lw=1.5,label="Remaining target energy (right axis)")]
    fig.suptitle("Which eigenvalue ranks determine whether the bound reaches 1%?",fontsize=16,y=.98)
    fig.legend(handles=handles,loc="upper center",bbox_to_anchor=(.5,.94),ncol=3,frameon=False)
    fig.text(.5,.027,"Orange vertical line: last rank with a positive lower bound. Green dotted line: 1% squared = 0.0001.\n"
             "If unresolved ranks retain more than 0.0001 target energy, this error bound cannot establish 1% accuracy.",
             ha="center",fontsize=9,color=".3")
    fig.subplots_adjust(left=.075,right=.925,top=.85,bottom=.13,hspace=.30,wspace=.18)
    fig.savefig(DEST/"bound_rank_coverage.png",dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--svd-driver", choices=["gesvd", "gesdd"], default="gesvd")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output:
        DEST = args.output
    with threadpool_limits(limits=1):
        result, arrays = run(svd_driver=args.svd_driver)
    plot(result,arrays)
