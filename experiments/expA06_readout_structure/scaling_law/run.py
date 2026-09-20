"""expA06/scaling_law -- what functional form governs ||v|| vs width?

Supplementary (precomputed) sub-experiment to the expA06 explorer. Sweeps the SVD
min-norm readout over activation x target x N, records the solved-coefficient norm
||v||_2 (plus max|v| and rel L2 for reference), then fits several candidate laws to
every (activation, target) norm-vs-W curve and asks: WHICH SINGLE FORM fits best
across ALL combos?

Candidate forms (fit in log10 ||v||; R^2 measured there, i.e. relative error):
  power        log y = a + b*log10(W)                (y = A * W^b)
  exponential  log y = a + b*W                       (y = A * r^W, geometric)
  stretched    log y = a + b*W^c                     (y = A * exp(beta * W^c))
  powerlog     log y = a + b*log10(W) + c*log10(lnW) (power with log correction)
  powerfloor   y = A*W^b + C                         (power decaying to a floor)
  expfloor     y = A*r^W + C                         (geometric decaying to a floor)

Sweep: N in a log grid up to --nmax (default 2048), n_train = max(2003, 2W+3)
equispaced (always overdetermined), pure min-norm lstsq (no rcond/ridge), fp64.
Rows are saved incrementally, so a partial sweep is still usable.

Run:  uv run python experiments/expA06_readout_structure/scaling_law/run.py           # sweep + fit
      uv run python ... --fit-only                                                    # refit existing rows
      uv run python ... --smoke                                                       # tiny pipeline check
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
REPO_ROOT = HERE.parents[2]
sys.path.insert(0, str(PARENT))
from app import ACTIVATIONS, PRESETS, build_phi, geometry, make_fn   # noqa: E402

OUT_DIR = REPO_ROOT / "results" / "checkpoint_A_numerics" / "expA06_readout_structure" / "scaling_law"
ROWS = OUT_DIR / "scaling_rows.json"
N_GRID = [16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048]
FN_SHORT = {PRESETS[0]: "sine", PRESETS[1]: "sine_8pi", PRESETS[2]: "runge",
            PRESETS[3]: "sine_mixture", PRESETS[4]: "exp", PRESETS[5]: "poly5",
            PRESETS[6]: "abs_cubed"}


# ----------------------------- sweep -----------------------------

def solve_case(job):
    from scipy.linalg import lstsq as sp_lstsq
    act, fn_text, N = job["act"], job["fn"], job["N"]
    f_vec = make_fn(fn_text)
    centers, gamma, halo = geometry(N, 0.25)
    W = centers.size
    x = np.linspace(-1.0, 1.0, max(2003, 2 * W + 3))
    y = f_vec(x)
    A = np.hstack([build_phi(x, gamma, centers, act), np.ones((x.size, 1))])
    sol, _, rank, sv = sp_lstsq(A, y, lapack_driver="gelsd")
    v = sol[:-1]
    xt = np.linspace(-1.0, 1.0, 4001)
    resid = build_phi(xt, gamma, centers, act) @ v + sol[-1] - f_vec(xt)
    return {"act": act, "fn": FN_SHORT.get(fn_text, fn_text), "N": N, "W": int(W),
            "norm_v": float(np.linalg.norm(v)), "max_v": float(np.abs(v).max()),
            "rel_l2": float(np.linalg.norm(resid) / np.linalg.norm(f_vec(xt))),
            "rank": int(rank), "cond": float(sv[0] / sv[rank - 1]) if rank else np.inf}


def sweep(n_grid, acts, fns, workers):
    jobs = sorted(
        [{"act": a, "fn": f, "N": n} for a in acts for f in fns for n in n_grid],
        key=lambda j: j["N"])                       # small N first -> early partials
    rows = []
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(solve_case, j) for j in jobs]
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            rows.append(r)
            ROWS.write_text(json.dumps(rows))       # incremental save
            print(f"  [{i}/{len(jobs)}] {r['act']:8s} {r['fn']:13s} N={r['N']:<5d} "
                  f"W={r['W']:<5d} ||v||={r['norm_v']:.3e} relL2={r['rel_l2']:.1e} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    return rows


# ----------------------------- fits -----------------------------

def _r2(t, t_hat):
    ss = np.sum((t - t.mean()) ** 2)
    return 1.0 - np.sum((t - t_hat) ** 2) / ss if ss > 0 else np.nan


FORMS = {}


def form(name, n_params):
    def deco(fn):
        FORMS[name] = (fn, n_params)
        return fn
    return deco


@form("power", 2)
def f_power(W, a, b):
    return a + b * np.log10(W)


@form("exponential", 2)
def f_expo(W, a, b):
    return a + b * W


@form("stretched", 3)
def f_stretch(W, a, b, c):
    return a + b * np.power(W, c)


@form("powerlog", 3)
def f_powerlog(W, a, b, c):
    return a + b * np.log10(W) + c * np.log10(np.log(W))


@form("powerfloor", 3)
def f_powerfloor(W, a, b, c):
    return np.log10(np.abs(10.0 ** a * np.power(W, b) + 10.0 ** c) + 1e-300)


@form("expfloor", 3)
def f_expfloor(W, a, b, c):
    return np.log10(np.abs(10.0 ** a * np.exp(b * W) + 10.0 ** c) + 1e-300)


@form("geolinear", 3)
def f_geolinear(W, a, b, c):
    # Sam's proposal: geometric transient + LINEAR term, y = A r^W + B W
    return np.log10(10.0 ** a * np.exp(b * W) + 10.0 ** c * W + 1e-300)


@form("geoaffine", 4)
def f_geoaffine(W, a, b, c, d):
    # geometric + linear + constant, y = A r^W + B W + C
    return np.log10(10.0 ** a * np.exp(b * W) + 10.0 ** c * W + 10.0 ** d + 1e-300)


@form("geosqrt", 3)
def f_geosqrt(W, a, b, c):
    # theory candidate: halo mass ~ sqrt(W), y = A r^W + B sqrt(W)
    return np.log10(10.0 ** a * np.exp(b * W) + 10.0 ** c * np.sqrt(W) + 1e-300)


@form("geopower", 4)
def f_geopower(W, a, b, c, d):
    # geometric + free-power tail, y = A r^W + B W^d -- lets the data pick the tail exponent
    return np.log10(10.0 ** a * np.exp(b * W) + 10.0 ** c * np.power(W, d) + 1e-300)


P0 = {"power": [0, -1], "exponential": [0, -1e-3], "stretched": [0, -1, 0.5],
      "powerlog": [0, -1, 0], "powerfloor": [1, -1, -2], "expfloor": [1, -1e-3, -2],
      "geolinear": [1, -1e-2, -4], "geoaffine": [1, -1e-2, -5, -1],
      "geosqrt": [1, -1e-2, -2], "geopower": [1, -1e-2, -3, 0.5]}


def fit_all(rows):
    combos = sorted({(r["act"], r["fn"]) for r in rows})
    results = {name: [] for name in FORMS}
    per_combo = {}
    warnings.filterwarnings("ignore")
    for act, fn in combos:
        pts = sorted([r for r in rows if r["act"] == act and r["fn"] == fn],
                     key=lambda r: r["W"])
        W = np.array([r["W"] for r in pts], float)
        t = np.log10(np.array([r["norm_v"] for r in pts], float) + 1e-300)
        if W.size < 4:
            continue
        per_combo[(act, fn)] = {}
        for name, (fmodel, npar) in FORMS.items():
            try:
                popt, _ = curve_fit(fmodel, W, t, p0=P0[name], maxfev=20000)
                r2 = _r2(t, fmodel(W, *popt))
            except Exception:
                popt, r2 = None, np.nan
            results[name].append(r2)
            per_combo[(act, fn)][name] = {"r2": None if np.isnan(r2) else float(r2),
                                          "params": None if popt is None else [float(p) for p in popt]}
    summary = {}
    for name, r2s in results.items():
        arr = np.array(r2s, float)
        ok = arr[~np.isnan(arr)]
        summary[name] = {"mean_r2": float(ok.mean()) if ok.size else None,
                         "median_r2": float(np.median(ok)) if ok.size else None,
                         "worst_r2": float(ok.min()) if ok.size else None,
                         "n_fit": int(ok.size), "n_params": FORMS[name][1]}
    return summary, per_combo


# ----------------------------- figures -----------------------------

def make_figures(rows, summary, per_combo):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    acts = sorted({r["act"] for r in rows})
    fns = sorted({r["fn"] for r in rows})
    best = max((n for n in summary if summary[n]["worst_r2"] is not None),
               key=lambda n: summary[n]["worst_r2"])

    # per-activation panels: data + best-form fit
    fig, axes = plt.subplots(1, len(acts), figsize=(3.6 * len(acts), 4.2), sharey=True)
    axes = np.atleast_1d(axes)
    cmap = plt.get_cmap("tab10")
    for ax, act in zip(axes, acts):
        for j, fn in enumerate(fns):
            pts = sorted([r for r in rows if r["act"] == act and r["fn"] == fn],
                         key=lambda r: r["W"])
            if not pts:
                continue
            W = np.array([r["W"] for r in pts], float)
            y = np.array([r["norm_v"] for r in pts], float)
            ax.plot(W, y, "o", ms=3.5, color=cmap(j))
            info = per_combo.get((act, fn), {}).get(best)
            if info and info["params"]:
                Wf = np.geomspace(W.min(), W.max(), 200)
                ax.plot(Wf, 10 ** FORMS[best][0](Wf, *info["params"]),
                        "-", lw=1.2, color=cmap(j), alpha=0.7)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_title(act); ax.set_xlabel("W")
        ax.grid(True, which="both", alpha=0.25)
    axes[0].set_ylabel(r"$\|v\|_2$ (solved readout)")
    handles = [Line2D([0], [0], color=cmap(j), marker="o", ms=4, lw=1.2, label=fn)
               for j, fn in enumerate(fns)]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.06),
               ncol=len(fns), frameon=False)
    fig.suptitle(f"expA06/scaling_law -- norm vs width, dots=data, lines=best universal "
                 f"form ('{best}')", y=1.14)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    p1 = OUT_DIR / "figures" / "scaling_curves.png"
    p1.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p1, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"  saved {p1}")

    # form comparison
    fig, ax = plt.subplots(figsize=(8, 4))
    names = sorted(FORMS, key=lambda n: -(summary[n]["worst_r2"] or -9))
    x = np.arange(len(names))
    ax.bar(x - 0.2, [summary[n]["mean_r2"] for n in names], 0.4, label="mean R2 (log-space)")
    ax.bar(x + 0.2, [summary[n]["worst_r2"] for n in names], 0.4, label="worst-combo R2")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{n}\n({summary[n]['n_params']}p)" for n in names])
    ax.axhline(1.0, color="k", lw=0.8, ls=":")
    ax.set_ylabel("R2 of log10 ||v|| fit")
    ax.set_ylim(min(0.0, min((summary[n]["worst_r2"] or 0) for n in names) - 0.05), 1.02)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=False)
    ax.set_title("Which law fits ALL activation x function curves best?", y=1.12)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    p2 = OUT_DIR / "figures" / "scaling_form_comparison.png"
    fig.savefig(p2, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"  saved {p2}")
    return best


# ----------------------------- main -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--fit-only", action="store_true")
    ap.add_argument("--nmax", type=int, default=2048)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    acts, fns, grid = ACTIVATIONS, PRESETS, [n for n in N_GRID if n <= args.nmax]
    if args.smoke:
        acts, fns, grid = ["tanh", "gelu"], PRESETS[:2], [16, 32, 64, 128]

    if args.fit_only:
        rows = json.loads(ROWS.read_text())
    else:
        os.environ.setdefault("OMP_NUM_THREADS", "2")
        print(f"sweep: {len(acts)} acts x {len(fns)} fns x {len(grid)} widths "
              f"(N up to {grid[-1]}) = {len(acts)*len(fns)*len(grid)} solves")
        rows = sweep(grid, acts, fns, args.workers)

    summary, per_combo = fit_all(rows)
    print("\nFORM COMPARISON (R2 of log10||v||, higher = better):")
    for name in sorted(FORMS, key=lambda n: -(summary[n]["worst_r2"] if summary[n]["worst_r2"] is not None else -9)):
        s = summary[name]
        fmt = lambda x: f"{x:.4f}" if x is not None else "  n/a"
        print(f"  {name:12s} ({s['n_params']}p)  mean={fmt(s['mean_r2'])}  "
              f"median={fmt(s['median_r2'])}  worst={fmt(s['worst_r2'])}  over {s['n_fit']} combos")
    best = make_figures(rows, summary, per_combo)
    (OUT_DIR / "fit_summary.json").write_text(json.dumps(
        {"summary": summary,
         "per_combo": {f"{a}|{f}": v for (a, f), v in per_combo.items()}}))
    print(f"\nbest universal form: {best}")
    print(f"  saved {OUT_DIR/'fit_summary.json'}")


if __name__ == "__main__":
    main()
