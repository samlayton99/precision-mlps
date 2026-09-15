"""Quick fp32 look (2026-09-08): pure tones, fp32 features/solve/forward, halo 32 (the realistic setting used everywhere else),
three activations, N = 64..512. Question: does the fp32 constant A_K(lambda) = 2^-23 sit on the fp32 floor left of an fp32
wall, and where are the fp32 wall and floor? Figure: figures/fp32_quick_tones.png.
Run: uv run --extra dev python experiments/expC07_lambda_energy_rule/lambda_rule/fp32_quick.py
"""
from __future__ import annotations
import json, os, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent; REPO_ROOT = HERE.parents[2]; sys.path.insert(0, str(HERE))
from rule import log_khat, constant_rule, KERNEL_ORDER  # noqa: E402
OUT = REPO_ROOT / "results" / "checkpoint_C_geometry" / "expC07_lambda_energy_rule" / "lambda_rule"
LAM = [float(f"{l:.5g}") for l in np.geomspace(0.03, 1.5, 60)]
ACTS = ["tanh", "gelu", "swish"]; WIDTHS = [64, 128, 256, 512]; HALO = 32
TONES = {"sin_1pi": 1, "cos_4pi": 4, "sin_16pi": 16}


def apply_act(z, name):
    from scipy.special import erf, expit
    if name == "gelu": return 0.5 * z * (1.0 + erf(z / np.sqrt(2.0)))
    if name == "swish": return z * expit(z)
    return np.tanh(z)


def f_of(name):
    k = TONES[name]
    return (lambda x: np.cos(k * np.pi * x)) if name.startswith("cos") else (lambda x: np.sin(k * np.pi * x))


def solve(job):
    from scipy.linalg import lstsq
    act, lam, N, fn = job["act"], job["lam"], job["N"], job["fn"]; dt = np.float32; f = f_of(fn)
    h = 2.0 / N; c = (-1.0 + np.arange(-HALO, N + HALO + 1) * h).astype(dt); g = dt(lam / h)
    x = np.linspace(-1, 1, max(2003, 2 * c.size + 3)).astype(dt)
    A = np.hstack([apply_act(g * (x[:, None] - c[None, :]), act), np.ones((x.size, 1), dt)]).astype(dt)
    sol = lstsq(A, f(x.astype(np.float64)).astype(dt), lapack_driver="gelsd")[0]
    xe = np.linspace(-1, 1, 4001).astype(dt)
    fit = (apply_act(g * (xe[:, None] - c[None, :]), act) @ sol[:-1].astype(dt) + sol[-1]).astype(np.float64)
    fe = f(xe.astype(np.float64))
    return {**job, "rel_l2": float(np.linalg.norm(fit - fe) / np.linalg.norm(fe))}


def fiber(act, k, N, lams, mmax=8):
    r = KERNEL_ORDER[act]; th = k * np.pi * 2.0 / N; out = []
    for lam in lams:
        ms = np.arange(-mmax, mmax + 1); tt = th + 2 * np.pi * ms
        lt = 2 * log_khat(act, np.abs(tt) / lam) - 2 * r * np.log(np.abs(tt)); lt -= lt.max(); v = np.exp(lt)
        out.append(np.sqrt(v[ms != 0].sum() / v.sum()))
    return np.array(out)


def main():
    rows_path = OUT / "data" / "fp32_quick_rows.json"
    if rows_path.exists():
        rows = json.loads(rows_path.read_text())
    else:
        jobs = [{"act": a, "lam": l, "N": n, "fn": t} for a in ACTS for l in LAM for n in WIDTHS for t in TONES]
        rows, t0 = [], time.time()
        with ProcessPoolExecutor(max_workers=max(1, (os.cpu_count() or 2) - 2)) as ex:
            for fut in as_completed([ex.submit(solve, j) for j in jobs]):
                rows.append(fut.result())
        rows_path.write_text(json.dumps(rows)); print(f"{len(rows)} fp32 solves in {time.time()-t0:.0f}s")
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    col = {"sin_1pi": "tab:blue", "cos_4pi": "tab:green", "sin_16pi": "tab:red"}
    fig, axes = plt.subplots(3, 4, figsize=(16, 11), sharex=True, sharey=True)
    summary = []
    for i, act in enumerate(ACTS):
        c32 = constant_rule(act, "fp32"); c64 = constant_rule(act, "fp64")
        for j, N in enumerate(WIDTHS):
            ax = axes[i, j]
            for fn in TONES:
                pts = sorted([r for r in rows if r["act"] == act and r["N"] == N and r["fn"] == fn], key=lambda r: r["lam"])
                lam = np.array([r["lam"] for r in pts]); rel = np.array([r["rel_l2"] for r in pts])
                lr = np.log(rel); sm = np.exp(np.array([np.median(lr[max(0, k-2):k+3]) for k in range(len(lr))]))
                ax.loglog(lam, rel, "-", color=col[fn], lw=1.3)
                ff = fiber(act, TONES[fn], N, LAM); ax.loglog(LAM, ff, "-.", color=col[fn], lw=0.8, alpha=0.7)
                q = 2 * TONES[fn] / N
                if q < 1 and sm.min() < 1e-2:
                    kmin = int(sm.argmin()); wall = lam[np.where(sm <= 10 * sm.min())[0].max()]
                    ok = np.where(ff <= 10 * sm.min())[0]; pw = LAM[ok.max()] if ok.size else float("nan")
                    e_c32 = float(np.exp(np.interp(np.log(c32), np.log(lam), np.log(sm))))
                    summary.append({"act": act, "N": N, "fn": fn, "q": q, "floor": float(sm.min()), "argmin": float(lam[kmin]),
                                    "wall_meas": float(wall), "wall_pred": float(pw), "const32": c32, "regret_const32": e_c32 / float(sm.min())})
            ax.axvline(c32, color="k", lw=1.3, ls=":"); ax.axvline(c64, color="0.6", lw=1.0, ls="--")
            ax.set_xlim(0.03, 1.5); ax.set_ylim(1e-8, 1e1); ax.grid(True, which="both", alpha=0.25)
            if i == 0: ax.set_title(f"N = {N}", fontsize=11)
            if j == 0: ax.set_ylabel(f"{act}\nrel $L_2$ (fp32 pipeline)", fontsize=10)
            if i == 2: ax.set_xlabel(r"$\lambda$")
    handles = [Line2D([0], [0], color=col[fn], lw=1.6, label=f"measured {fn}") for fn in TONES]
    handles += [Line2D([0], [0], color="gray", lw=0.8, ls="-.", label="fiber floor (same color)"),
                Line2D([0], [0], color="k", lw=1.3, ls=":", label=r"fp32 constant $A_K=2^{-23}$"),
                Line2D([0], [0], color="0.6", lw=1.0, ls="--", label=r"fp64 constant $A_K=2^{-52}$")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=6, frameon=False, fontsize=9)
    fig.suptitle("fp32 quick look: pure tones, fp32 features/solve/forward, halo 32", y=0.95, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93]); p = OUT / "figures" / "fp32_quick_tones.png"; fig.savefig(p, dpi=130); print("saved", p)
    (OUT / "data" / "fp32_quick_summary.json").write_text(json.dumps(summary, indent=1))
    print(f"{'act':5s} {'N':>4s} {'tone':9s} {'q':>5s} {'floor':>8s} {'argmin':>6s} {'wall':>6s} {'pred':>6s} {'c32':>5s} {'regret':>6s}")
    for s in summary:
        print(f"{s['act']:5s} {s['N']:4d} {s['fn']:9s} {s['q']:5.3f} {s['floor']:8.1e} {s['argmin']:6.3f} {s['wall_meas']:6.3f} {s['wall_pred']:6.3f} {s['const32']:5.3f} {s['regret_const32']:6.1f}")


if __name__ == "__main__":
    main()
