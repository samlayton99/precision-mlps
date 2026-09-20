"""Precision ladder (2026-09-08): how the lambda curve, its wall and its floor move with the working precision.

Native precisions on this machine: fp32 and fp64 only (float16 has no LAPACK solver; longdouble == float64 on arm64).
Intermediate precisions are EMULATED at p mantissa bits: features, right-hand side and solved coefficients are rounded to
p bits; the forward pass rounds every partial sum to p bits (sequential over columns); the SVD solve runs in fp64 with the
cutoff 2^-p (gelsd's rule at that precision). The solve's own internal roundoff is NOT emulated. p = 24 is checked against
the native fp32 run (fp32_quick) and p = 53 is native fp64. Tones, halo 32, three activations, N = 64..256.
Outputs: figures/precision_ladder_curves.png, figures/precision_ladder_wall.png, data/precision_ladder_*.json.
Run: uv run --extra dev python experiments/expC07_lambda_energy_rule/lambda_rule/precision_ladder.py
"""
from __future__ import annotations
import json, os, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent; REPO_ROOT = HERE.parents[2]; sys.path.insert(0, str(HERE))
from rule import log_khat, KERNEL_ORDER, LAM_GRID_RULE  # noqa: E402
OUT = REPO_ROOT / "results" / "checkpoint_C_geometry" / "expC07_lambda_energy_rule" / "lambda_rule"
LAM = [float(f"{l:.5g}") for l in np.geomspace(0.03, 1.5, 40)]
ACTS = ["tanh", "gelu", "swish"]; WIDTHS = [64, 128, 256]; HALO = 32
TONES = {"sin_1pi": 1, "cos_4pi": 4, "sin_16pi": 16}
GENERAL = {  # five non-tone targets for the second figure (same as expC09's definitions)
    "runge25": ("$1/(1+25x^2)$", lambda x: 1.0 / (1.0 + 25.0 * x ** 2)),
    "rat_off": ("$1/((x-0.5)^2+0.04)$", lambda x: 1.0 / ((x - 0.5) ** 2 + 0.04)),
    "expsin": (r"$e^{\sin 3\pi x}$", lambda x: np.exp(np.sin(3 * np.pi * x))),
    "poly4": ("$x^4-x^2$", lambda x: x ** 4 - x ** 2),
    "abs3": ("$|x|^3$", lambda x: np.abs(x) ** 3),
}
N_GENERAL = 128
PBITS = [11, 16, 24, 32, 40, 48, 53]          # 11 ~ fp16, 24 ~ fp32, 53 = fp64 (native)


def apply_act(z, name):
    from scipy.special import erf, expit
    if name == "gelu": return 0.5 * z * (1.0 + erf(z / np.sqrt(2.0)))
    if name == "swish": return z * expit(z)
    return np.tanh(z)


def f_of(name):
    if name in GENERAL:
        return GENERAL[name][1]
    k = TONES[name]
    return (lambda x: np.cos(k * np.pi * x)) if name.startswith("cos") else (lambda x: np.sin(k * np.pi * x))


def rnd(x, p):
    """Round to p mantissa bits (round-to-nearest); p = 53 is the identity."""
    if p >= 53: return np.asarray(x, dtype=np.float64)
    m, e = np.frexp(np.asarray(x, dtype=np.float64))
    return np.ldexp(np.round(m * 2.0 ** p) / 2.0 ** p, e)


def solve(job):
    from scipy.linalg import lstsq
    act, lam, N, fn, p = job["act"], job["lam"], job["N"], job["fn"], job["p"]; f = f_of(fn); eps = 2.0 ** -(p - 1)
    h = 2.0 / N; c = -1.0 + np.arange(-HALO, N + HALO + 1) * h; g = lam / h
    x = np.linspace(-1, 1, max(2003, 2 * c.size + 3))
    A = rnd(np.hstack([apply_act(g * (x[:, None] - c[None, :]), act), np.ones((x.size, 1))]), p)
    b = rnd(f(x), p)
    sol = rnd(lstsq(A, b, cond=eps, lapack_driver="gelsd")[0], p)
    xe = np.linspace(-1, 1, 4001)
    Ae = rnd(apply_act(g * (xe[:, None] - c[None, :]), act), p)
    if p >= 53:
        fit = Ae @ sol[:-1] + sol[-1]
    else:  # sequential accumulation with p-bit rounding of every partial sum
        fit = rnd(np.full(xe.size, sol[-1]), p)
        for k in range(Ae.shape[1]):
            fit = rnd(fit + rnd(Ae[:, k] * sol[k], p), p)
    fe = f(xe)
    return {**job, "rel_l2": float(np.linalg.norm(fit - fe) / np.linalg.norm(fe))}


def fiber(act, k, N, lams, mmax=8):
    r = KERNEL_ORDER[act]; th = k * np.pi * 2.0 / N; out = []
    for lam in lams:
        ms = np.arange(-mmax, mmax + 1); tt = th + 2 * np.pi * ms
        lt = 2 * log_khat(act, np.abs(tt) / lam) - 2 * r * np.log(np.abs(tt)); lt -= lt.max(); v = np.exp(lt)
        out.append(np.sqrt(v[ms != 0].sum() / v.sum()))
    return np.array(out)


def const_rule(act, eps):
    g = LAM_GRID_RULE; ok = np.where(log_khat(act, 2 * np.pi / g) - np.log(eps) <= 0)[0]
    return float(g[ok.max()]) if ok.size else float("nan")


def mark_design_point(ax, act, p, color, lam, rel):
    """Where the rule lands on the measured curve: x = lambda_const (A_K(lambda) = eps_p), y = the measured error at that
    lambda (log-interpolated on this precision's curve); draw the vertical, the horizontal through the point, and the dot."""
    eps = 2.0 ** -(p - 1); lc = const_rule(act, eps)
    y = float(np.exp(np.interp(np.log(lc), np.log(lam), np.log(rel))))
    ax.axvline(lc, color=color, lw=0.8, ls=":"); ax.axhline(y, color=color, lw=0.6, ls=":", alpha=0.7)
    ax.plot(lc, y, "o", color=color, ms=6, mec="k", mew=0.6, zorder=6)


def smooth(rel):
    lr = np.log(rel); return np.exp(np.array([np.median(lr[max(0, i-2):i+3]) for i in range(len(lr))]))


def main():
    rows_path = OUT / "data" / "precision_ladder_rows.json"
    if rows_path.exists():
        rows = json.loads(rows_path.read_text())
    else:
        jobs = [{"act": a, "lam": l, "N": n, "fn": t, "p": p} for p in PBITS for a in ACTS for l in LAM for n in WIDTHS for t in TONES]
        rows, t0 = [], time.time()
        with ProcessPoolExecutor(max_workers=max(1, (os.cpu_count() or 2) - 2)) as ex:
            futs = [ex.submit(solve, j) for j in jobs]
            for i, fut in enumerate(as_completed(futs), 1):
                rows.append(fut.result())
                if i % 1000 == 0: print(f"  [{i}/{len(jobs)}] {time.time()-t0:.0f}s", flush=True)
        rows_path.write_text(json.dumps(rows)); print(f"{len(rows)} solves in {time.time()-t0:.0f}s")

    # ---- validation of the emulation: p=24 vs native fp32 (fp32_quick rows, same tones/halo/N)
    nat = json.loads((OUT / "data" / "fp32_quick_rows.json").read_text())
    print("\nEmulation check, p=24 vs native fp32: smoothed floor ratio (emulated/native) and argmin, per cell")
    for act in ACTS:
        for N in WIDTHS:
            for fn in TONES:
                if 2 * TONES[fn] / N >= 1: continue
                e = sorted([r for r in rows if r["p"] == 24 and r["act"] == act and r["N"] == N and r["fn"] == fn], key=lambda r: r["lam"])
                n = sorted([r for r in nat if r["act"] == act and r["N"] == N and r["fn"] == fn], key=lambda r: r["lam"])
                se = smooth(np.array([r["rel_l2"] for r in e])); sn = smooth(np.array([r["rel_l2"] for r in n]))
                le = np.array([r["lam"] for r in e]); ln = np.array([r["lam"] for r in n])
                print(f"  {act:5s} N={N:3d} {fn:9s} floor emu {se.min():.1e} native {sn.min():.1e} ratio {se.min()/sn.min():5.2f} | argmin emu {le[se.argmin()]:.3f} native {ln[sn.argmin()]:.3f}")

    # ---- per-cell summary: floor, wall (10x shoulder), predicted wall, constant, argmin
    summ = []
    for p in PBITS:
        eps = 2.0 ** -(p - 1)
        for act in ACTS:
            for N in WIDTHS:
                for fn in TONES:
                    q = 2 * TONES[fn] / N
                    if q >= 1: continue
                    pts = sorted([r for r in rows if r["p"] == p and r["act"] == act and r["N"] == N and r["fn"] == fn], key=lambda r: r["lam"])
                    lam = np.array([r["lam"] for r in pts]); sm = smooth(np.array([r["rel_l2"] for r in pts]))
                    if sm.min() > 1e-2: continue
                    wall = lam[np.where(sm <= 10 * sm.min())[0].max()]
                    ff = fiber(act, TONES[fn], N, LAM); ok = np.where(ff <= 10 * sm.min())[0]; pw = LAM[ok.max()] if ok.size else float("nan")
                    summ.append({"p": p, "eps": eps, "act": act, "N": N, "fn": fn, "q": q, "floor": float(sm.min()), "floor_over_eps": float(sm.min() / eps),
                                 "argmin": float(lam[sm.argmin()]), "wall": float(wall), "wall_pred": float(pw), "const": const_rule(act, eps)})
    (OUT / "data" / "precision_ladder_summary.json").write_text(json.dumps(summ, indent=1))

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    cmap = plt.get_cmap("viridis"); pc = {p: cmap(i / (len(PBITS) - 1)) for i, p in enumerate(PBITS)}
    # ---- figure 1: curves, cos(4 pi x) at N=128 and sin(pi x) at N=256, one panel per activation, one line per precision
    cases = [("cos_4pi", 128), ("sin_1pi", 256), ("sin_16pi", 256)]
    fig, axes = plt.subplots(3, 3, figsize=(16, 12), sharex=True, sharey=True)
    for i, (fn, N) in enumerate(cases):
        for j, act in enumerate(ACTS):
            ax = axes[i, j]
            for p in PBITS:
                pts = sorted([r for r in rows if r["p"] == p and r["act"] == act and r["N"] == N and r["fn"] == fn], key=lambda r: r["lam"])
                lam_ = np.array([r["lam"] for r in pts]); rel_ = np.array([r["rel_l2"] for r in pts])
                ax.loglog(lam_, rel_, "-", color=pc[p], lw=1.3)
                mark_design_point(ax, act, p, pc[p], lam_, rel_)
            ax.loglog(LAM, fiber(act, TONES[fn], N, LAM), "k-.", lw=0.9)
            ax.set_xlim(0.03, 1.5); ax.set_ylim(1e-16, 1e1); ax.grid(True, which="both", alpha=0.25)
            if i == 0: ax.set_title(act, fontsize=12)
            if j == 0: ax.set_ylabel(f"{fn}, N={N}\nrel $L_2$", fontsize=10)
            if i == 2: ax.set_xlabel(r"$\lambda$")
    handles = [Line2D([0], [0], color=pc[p], lw=1.6, label=f"p={p} bits" + (" (native fp64)" if p == 53 else " (~fp32, emulated)" if p == 24 else " (~fp16, emulated)" if p == 11 else " (emulated)")) for p in PBITS]
    handles += [Line2D([0], [0], color="k", lw=0.9, ls="-.", label="fiber floor (exact arithmetic)"),
                Line2D([0], [0], marker="o", color="gray", lw=0, ms=6, mec="k", label=r"where the rule lands: $\lambda_{\rm const}$ from $A_K(\lambda)=\varepsilon_p$, $y$ = measured error there")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=5, frameon=False, fontsize=9)
    fig.suptitle("Precision ladder (EMULATED p < 53: inputs, coefficients, partial sums rounded; SVD cutoff 2^-p; solver internals not emulated): tones, halo 32", y=0.93, fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.91]); fig.savefig(OUT / "figures" / "precision_ladder_curves.png", dpi=130); plt.close(fig)
    # ---- figure 1b: five non-tone targets at N_GENERAL, one panel per (target, activation)
    rows_g_path = OUT / "data" / "precision_ladder_rows_general.json"
    if rows_g_path.exists():
        rows_g = json.loads(rows_g_path.read_text())
    else:
        jobs = [{"act": a, "lam": l, "N": N_GENERAL, "fn": t, "p": p} for p in PBITS for a in ACTS for l in LAM for t in GENERAL]
        rows_g, t0 = [], time.time()
        with ProcessPoolExecutor(max_workers=max(1, (os.cpu_count() or 2) - 2)) as ex:
            for fut in as_completed([ex.submit(solve, j) for j in jobs]):
                rows_g.append(fut.result())
        rows_g_path.write_text(json.dumps(rows_g)); print(f"{len(rows_g)} general-target solves in {time.time()-t0:.0f}s")
    import importlib.util
    spec = importlib.util.spec_from_file_location("c9", HERE / "run_c09_general.py"); c9 = importlib.util.module_from_spec(spec); spec.loader.exec_module(c9)
    for name, (tex, fn_) in GENERAL.items():
        c9.TARGETS[name] = (tex, fn_)
    fig, axes = plt.subplots(len(GENERAL), 3, figsize=(16, 19), sharex=True, sharey=True)
    for i, (fn, (tex, _)) in enumerate(GENERAL.items()):
        for j, act in enumerate(ACTS):
            ax = axes[i, j]
            for p in PBITS:
                pts = sorted([r for r in rows_g if r["p"] == p and r["act"] == act and r["fn"] == fn], key=lambda r: r["lam"])
                lam_ = np.array([r["lam"] for r in pts]); rel_ = np.array([r["rel_l2"] for r in pts])
                ax.loglog(lam_, rel_, "-", color=pc[p], lw=1.3)
                mark_design_point(ax, act, p, pc[p], lam_, rel_)
            pred = c9.fiber_prediction(act, fn, N_GENERAL, LAM, reach="main", halo=HALO)["err"]
            ax.loglog(LAM, pred, "k-.", lw=0.9)
            ax.set_xlim(0.03, 1.5); ax.set_ylim(1e-16, 1e1); ax.grid(True, which="both", alpha=0.25)
            if i == 0: ax.set_title(act, fontsize=12)
            if j == 0: ax.set_ylabel(f"{tex}, N={N_GENERAL}\nrel $L_2$", fontsize=10)
            if i == len(GENERAL) - 1: ax.set_xlabel(r"$\lambda$")
    handles = [Line2D([0], [0], color=pc[p], lw=1.6, label=f"p={p} bits" + (" (native fp64)" if p == 53 else " (~fp32, emulated)" if p == 24 else " (~fp16, emulated)" if p == 11 else " (emulated)")) for p in PBITS]
    handles += [Line2D([0], [0], color="k", lw=0.9, ls="-.", label="fiber projection, exact arithmetic (expC09 machinery)"),
                Line2D([0], [0], marker="o", color="gray", lw=0, ms=6, mec="k", label=r"where the rule lands: $\lambda_{\rm const}$ from $A_K(\lambda)=\varepsilon_p$, $y$ = measured error there")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=5, frameon=False, fontsize=9)
    fig.suptitle("Precision ladder 2 (EMULATED p < 53, see figure 1 caption): five non-tone targets, N = 128, halo 32", y=0.955, fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.94]); fig.savefig(OUT / "figures" / "precision_ladder_curves_2.png", dpi=130); plt.close(fig)

    # ---- figure 2: wall, argmin, constant vs eps
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    mk = {"sin_1pi": "o", "cos_4pi": "s", "sin_16pi": "^"}; sz = {64: 4, 128: 6, 256: 8}
    for ax, act in zip(axes, ACTS):
        for s in summ:
            if s["act"] != act: continue
            ax.plot(s["eps"], s["wall"], mk[s["fn"]], color="tab:blue", ms=sz[s["N"]], mfc="none", mew=1.2)
            ax.plot(s["eps"], s["argmin"], mk[s["fn"]], color="tab:blue", ms=sz[s["N"]] - 1, alpha=0.5, mec="none")
            ax.plot(s["eps"], s["wall_pred"], mk[s["fn"]], color="#d1352b", ms=sz[s["N"]] - 2, alpha=0.8, mec="none")
        ee = np.geomspace(2.0 ** -53, 2.0 ** -10, 100); ax.plot(ee, [const_rule(act, e) for e in ee], "k-", lw=1.4)
        ax.set_xscale("log"); ax.set_xlabel(r"$\varepsilon_p = 2^{-(p-1)}$"); ax.set_title(act, fontsize=12); ax.grid(alpha=0.3, which="both")
        ax.set_ylim(0, 1.6)
    axes[0].set_ylabel(r"$\lambda$")
    handles = [Line2D([0], [0], color="k", lw=1.4, label=r"constant rule $A_K(\lambda)=\varepsilon_p$"),
               Line2D([0], [0], marker="o", color="tab:blue", lw=0, mfc="none", label="measured wall (10x shoulder)"),
               Line2D([0], [0], marker="o", color="tab:blue", lw=0, alpha=0.5, label="measured argmin"),
               Line2D([0], [0], marker="o", color="#d1352b", lw=0, label="predicted wall: fiber curve at 10x the measured floor")]
    handles += [Line2D([0], [0], marker=mk[fn], color="gray", lw=0, label=fn) for fn in TONES] + [Line2D([0], [0], marker="o", color="gray", lw=0, ms=sz[N], label=f"N={N}") for N in WIDTHS]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=5, frameon=False, fontsize=9)
    fig.suptitle("Where the wall, the argmin and the constant sit as the precision changes (tones, halo 32; p < 53 emulated)", y=0.86, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.84]); fig.savefig(OUT / "figures" / "precision_ladder_wall.png", dpi=150); plt.close(fig)
    print("saved figures")
    print("\nper precision (all tones/widths/acts with a resolved minimum): median floor/eps_p, median wall/const, median wall/wall_pred, median argmin/const")
    for p in PBITS:
        S = [s for s in summ if s["p"] == p]
        if not S: continue
        r = lambda key: np.median([s[key] for s in S])
        print(f"  p={p:2d} eps={2.0**-(p-1):.1e} n={len(S):2d} | floor/eps {r('floor_over_eps'):8.0f} | wall/const {np.median([s['wall']/s['const'] for s in S]):.2f} | wall/pred {np.median([s['wall']/s['wall_pred'] for s in S if np.isfinite(s['wall_pred'])]):.2f} | argmin/const {np.median([s['argmin']/s['const'] for s in S]):.2f}")


if __name__ == "__main__":
    main()
