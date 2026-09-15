"""expC08 anchor rule, exploratory sweeps (2026-09-09; moved from expC07/lambda_rule on 2026-09-10): Sam's output-ratio aliasing anchor against measured lambda curves.

The rule.  For an activation psi with kernel K = psi^(r), width N (h = 2/N), a target with exponential representation
P = sum_j b_j e^{i omega_j x}, B = sum |b_j|, omega_M = max |omega_j| (grid frequency theta_M = 2 omega_M / N), and an
aliasing budget eps, define the first-pair output ratio

    R(lambda) = [ (theta_M/(2pi-theta_M))^r |Khat((2pi-theta_M)/lambda)| + (theta_M/(2pi+theta_M))^r |Khat((2pi+theta_M)/lambda)| ]
                / min{ |Khat(0)|, |Khat(theta_M/lambda)| }  x  1/(1 - rho_K),   rho_K = |Khat((4pi-theta_M)/lambda)| / |Khat((2pi-theta_M)/lambda)|

and lambda_anchor = sup{ lambda : B R(lambda) <= eps }.  The old rule is lambda_const: |Khat(2pi/lambda)|/|Khat(0)| = eps.
Both use eps = eps_p = 2^-(p-1) (unit roundoff at p mantissa bits; 2.2e-16 for fp64).

Targets.  Three wave mixtures a1 sin(k1 pi x) + a2 sin(k2 pi x) + a3 sin(k3 pi x) with amplitudes (1, 1/2, 1/4) at rising
frequency: B = 1.75 exactly, omega_M = k3 pi.  Three non-tones with closed-form spectra; omega_M is the two-standard-deviation
point of the spectral energy |fhat(omega)|^2 (sigma_omega = ||f'||/||f|| on the line), and B is the L^1 mass of fhat/(2 pi)
inside |omega| <= omega_M (the coefficient sum of the retained exponentials); delta = the mass outside is reported.

Two sweeps, both halo 32, lambda on a 40-point log grid in [0.03, 1.5], relative L2 on 4001 points:
  width:      native fp64, N in WIDTHS                       -> figures/anchor_rule_width.png
  precision:  N = N_PREC, p in PBITS (p < 53 emulated as in precision_ladder.py: inputs, coefficients and partial sums
              rounded to p bits, SVD cutoff 2^-p; solver internals NOT emulated)   -> figures/anchor_rule_precision.png
Predictions are written to data/anchor_rule_predictions.json before any solve runs.
Run: uv run --extra dev python experiments/expC08_anchor_rule/anchor_rule.py [--predict-only]
"""
from __future__ import annotations
import json, os, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent; REPO_ROOT = HERE.parents[1]
LR = REPO_ROOT / "experiments" / "expC07_lambda_energy_rule" / "lambda_rule"   # rule.py (frozen kernels) and run_c09_general.py live there
sys.path.insert(0, str(LR)); sys.path.insert(0, str(HERE))
import rule as _rule  # noqa: E402  (frozen: log_khat for tanh/gelu/swish, LAM_GRID_RULE)
OUT = REPO_ROOT / "results" / "checkpoint_C_geometry" / "expC08_anchor_rule"

LAM = [float(f"{l:.5g}") for l in np.geomspace(0.03, 1.5, 40)]
ACTS = ["tanh", "gelu", "swish", "gaussian"]
ORDER = {"tanh": 1, "gelu": 2, "swish": 2, "gaussian": 0}     # kernel order r; gaussian psi(x) = e^{-x^2}, K = psi
HALO = 32
WIDTHS = [32, 64, 128, 256, 512]
N_PREC = 128
PBITS = [11, 16, 24, 32, 40, 48, 53]
TWO_PI = 2 * np.pi
OMEGA_MODE = "mean" if "--omega-mean" in sys.argv else "max"   # max: omega_M = top frequency / 2 sigma; mean: |b_j|-weighted mean |omega_j|
SUF = "_mean" if OMEGA_MODE == "mean" else ""

# ----------------------------------------------------------------------------- targets
MIX = {  # name: (tex, [(amplitude, k) ...]) for sum a sin(k pi x)
    "mix_1_3_5":   (r"$\sin\pi x+\frac{1}{2}\sin3\pi x+\frac{1}{4}\sin5\pi x$",     [(1.0, 1), (0.5, 3), (0.25, 5)]),
    "mix_2_6_10":  (r"$\sin2\pi x+\frac{1}{2}\sin6\pi x+\frac{1}{4}\sin10\pi x$",   [(1.0, 2), (0.5, 6), (0.25, 10)]),
    "mix_4_12_20": (r"$\sin4\pi x+\frac{1}{2}\sin12\pi x+\frac{1}{4}\sin20\pi x$",  [(1.0, 4), (0.5, 12), (0.25, 20)]),
}
NONTONE = {
    "runge25": "$1/(1+25x^2)$",
    "gauss20": "$e^{-20x^2}$",
    "expsin":  r"$e^{\sin 3\pi x}$",
}
TEX = {**{k: v[0] for k, v in MIX.items()}, **NONTONE}
TARGET_NAMES = list(MIX) + list(NONTONE)


def f_of(name):
    if name in MIX:
        terms = MIX[name][1]
        return lambda x: sum(a * np.sin(k * np.pi * x) for a, k in terms)
    return {"runge25": lambda x: 1.0 / (1.0 + 25.0 * x ** 2),
            "gauss20": lambda x: np.exp(-20.0 * x ** 2),
            "expsin":  lambda x: np.exp(np.sin(3 * np.pi * x))}[name]


def spectral_inputs(name):
    """(B, omega_M, delta, sigma_omega, lines) for the rule.  lines = [(|b_j|, omega_j)] when the spectrum is discrete."""
    from scipy.special import iv, erf
    if name in MIX:
        terms = MIX[name][1]
        lines = [(a / 2, s * k * np.pi) for a, k in terms for s in (+1, -1)]
        if OMEGA_MODE == "mean":
            wbar = sum(a * k for a, k in terms) * np.pi / sum(a for a, _ in terms)
            return {"B": float(sum(a for a, _ in terms)), "omega_M": float(wbar), "delta": 0.0, "sigma": float("nan"), "lines": lines,
                    "how": "exact lines: B = sum of amplitudes, omega = |b_j|-weighted mean |omega_j|"}
        return {"B": float(sum(a for a, _ in terms)), "omega_M": float(max(k for _, k in terms) * np.pi),
                "delta": 0.0, "sigma": float("nan"), "lines": lines, "how": "exact lines: B = sum of amplitudes, omega_M = top frequency"}
    if OMEGA_MODE == "mean":
        return spectral_inputs_mean(name)
    if name == "runge25":   # fhat = (pi/5) e^{-|w|/5}; |fhat|^2 Laplace with scale 5/2 -> var = 2 (5/2)^2
        sig = np.sqrt(2.0) * 2.5; wM = 2 * sig
        B = 1.0 - np.exp(-wM / 5.0); delta = np.exp(-wM / 5.0)          # L1 mass of fhat/(2 pi): total 1
        return {"B": float(B), "omega_M": float(wM), "delta": float(delta), "sigma": float(sig), "lines": None,
                "how": "fhat = (pi/5) e^{-|w|/5}; omega_M = 2 sigma of |fhat|^2; B = L1 mass of fhat/2pi inside"}
    if name == "gauss20":   # fhat = sqrt(pi/20) e^{-w^2/80}; |fhat|^2 ~ e^{-w^2/40} -> var = 20
        sig = np.sqrt(20.0); wM = 2 * sig
        B = float(erf(wM / np.sqrt(80.0))); delta = 1.0 - B                  # total L1 mass 1 (= f(0))
        return {"B": B, "omega_M": float(wM), "delta": float(delta), "sigma": float(sig), "lines": None,
                "how": "fhat = sqrt(pi/20) e^{-w^2/80}; omega_M = 2 sigma of |fhat|^2; B = L1 mass of fhat/2pi inside"}
    if name == "expsin":    # e^{sin phi} = sum_n c_n e^{i n phi}, |c_n| = I_n(1), phi = 3 pi x
        n = np.arange(-40, 41); c = iv(np.abs(n), 1.0); w = 3 * np.pi * n
        sig = np.sqrt((w ** 2 * c ** 2).sum() / (c ** 2).sum()); wM = 2 * sig
        keep = np.abs(w) <= wM
        return {"B": float(c[keep].sum()), "omega_M": float(wM), "delta": float(c[~keep].sum()), "sigma": float(sig),
                "lines": [(float(ci), float(wi)) for ci, wi in zip(c[keep], w[keep])],
                "how": "|c_n| = I_n(1) at omega = 3 pi n; omega_M = 2 sigma of the line energies; B = sum of retained |c_n|"}
    raise ValueError(name)


def spectral_inputs_mean(name):
    """Mean-frequency variant: omega = |fhat|-weighted mean of |omega| (same weights as B), B = the whole L1 mass, delta = 0."""
    from scipy.special import iv
    if name == "runge25":   # |fhat| ~ e^{-|w|/5}: mean |w| = 5
        return {"B": 1.0, "omega_M": 5.0, "delta": 0.0, "sigma": float("nan"), "lines": None, "how": "fhat = (pi/5) e^{-|w|/5}; omega = |fhat|-weighted mean |w| = 5; B = total L1 mass 1"}
    if name == "gauss20":   # |fhat| ~ e^{-w^2/80}: half-normal with variance 40, mean = sqrt(40) sqrt(2/pi)
        return {"B": 1.0, "omega_M": float(np.sqrt(40.0) * np.sqrt(2 / np.pi)), "delta": 0.0, "sigma": float("nan"), "lines": None,
                "how": "fhat = sqrt(pi/20) e^{-w^2/80}; omega = |fhat|-weighted mean |w|; B = total L1 mass 1"}
    if name == "expsin":
        n = np.arange(-40, 41); c = iv(np.abs(n), 1.0); w = 3 * np.pi * n
        return {"B": float(c.sum()), "omega_M": float((np.abs(w) * c).sum() / c.sum()), "delta": 0.0, "sigma": float("nan"),
                "lines": [(float(ci), float(wi)) for ci, wi in zip(c, w)], "how": "|c_n| = I_n(1) at 3 pi n; omega = |c_n|-weighted mean |omega|; B = sum |c_n| = e"}
    raise ValueError(name)


# ----------------------------------------------------------------------------- kernels and the two rules
def log_khat(act, xi):
    xi = np.asarray(xi, dtype=float)
    if act == "gaussian":
        return -xi ** 2 / 4.0                      # psi = e^{-x^2}: Khat(xi)/Khat(0) = e^{-xi^2/4}
    return _rule.log_khat(act, xi)


def apply_act(z, act):
    from scipy.special import erf, expit
    if act == "gelu": return 0.5 * z * (1.0 + erf(z / np.sqrt(2.0)))
    if act == "swish": return z * expit(z)
    if act == "gaussian": return np.exp(-z ** 2)
    return np.tanh(z)


def log_R(act, lam, N, omega_M):
    """log of the first-pair output ratio R(lambda) with the 1/(1-rho_K) correction; nan if theta_M >= pi."""
    r = ORDER[act]; th = 2.0 * omega_M / N
    if th >= np.pi: return float("nan")
    lam = np.asarray(lam, dtype=float)
    t_minus = r * np.log(th / (TWO_PI - th)) + log_khat(act, (TWO_PI - th) / lam)
    t_plus = r * np.log(th / (TWO_PI + th)) + log_khat(act, (TWO_PI + th) / lam)
    num = np.logaddexp(t_minus, t_plus)
    den = np.minimum(0.0, log_khat(act, th / lam))              # log min{1, Khat(theta_M/lam)/Khat(0)}
    rho = np.exp(log_khat(act, (2 * TWO_PI - th) / lam) - log_khat(act, (TWO_PI - th) / lam))
    return num - den - np.log1p(-rho)


def lambda_anchor(act, N, omega_M, B, eps, grid=_rule.LAM_GRID_RULE):
    lr = log_R(act, grid, N, omega_M)
    if np.isscalar(lr) and np.isnan(lr): return float("nan")
    ok = np.where(np.log(B) + lr <= np.log(eps))[0]
    return float(grid[ok.max()]) if ok.size else float("nan")


def lambda_const(act, eps, grid=_rule.LAM_GRID_RULE):
    ok = np.where(log_khat(act, TWO_PI / grid) <= np.log(eps))[0]
    return float(grid[ok.max()]) if ok.size else float("nan")


def fiber_floor(act, name, N, lams, mmax=8):
    """Exact-arithmetic L2 floor for a discrete spectrum: sqrt( sum |b_j|^2 Lambda_r(theta_j) / sum |b_j|^2 )."""
    r = ORDER[act]; lines = spectral_inputs(name)["lines"]
    if lines is None: return None
    out = []
    for lam in lams:
        num = den = 0.0
        for b, w in lines:
            th = abs(w) * 2.0 / N
            if th == 0.0: continue                                  # the constant is exact (bias column)
            if th >= np.pi: num += b * b; den += b * b; continue     # unresolvable line: all of it is error
            ms = np.arange(-mmax, mmax + 1); tt = th + TWO_PI * ms
            lt = 2 * log_khat(act, np.abs(tt) / lam) - 2 * r * np.log(np.abs(tt)); lt -= lt.max(); v = np.exp(lt)
            num += b * b * v[ms != 0].sum() / v.sum(); den += b * b
        out.append(np.sqrt(num / den))
    return np.array(out)


# ----------------------------------------------------------------------------- the solve (as precision_ladder.py)
def rnd(x, p):
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
    else:
        fit = rnd(np.full(xe.size, sol[-1]), p)
        for k in range(Ae.shape[1]):
            fit = rnd(fit + rnd(Ae[:, k] * sol[k], p), p)
    fe = f(xe)
    return {**job, "rel_l2": float(np.linalg.norm(fit - fe) / np.linalg.norm(fe))}


def run_jobs(jobs, path):
    if path.exists(): return json.loads(path.read_text())
    rows, t0 = [], time.time()
    with ProcessPoolExecutor(max_workers=max(1, (os.cpu_count() or 2) - 2)) as ex:
        futs = [ex.submit(solve, j) for j in jobs]
        for i, fut in enumerate(as_completed(futs), 1):
            rows.append(fut.result())
            if i % 1000 == 0: print(f"  [{i}/{len(jobs)}] {time.time()-t0:.0f}s", flush=True)
    path.write_text(json.dumps(rows)); print(f"{len(rows)} solves -> {path.name} in {time.time()-t0:.0f}s", flush=True)
    return rows


def smooth(rel):
    lr = np.log(rel); return np.exp(np.array([np.median(lr[max(0, i-2):i+3]) for i in range(len(lr))]))


def curve(rows, **key):
    pts = sorted([r for r in rows if all(r[k] == v for k, v in key.items())], key=lambda r: r["lam"])
    return np.array([r["lam"] for r in pts]), np.array([r["rel_l2"] for r in pts])


def y_on_curve(lam0, lam, rel):
    if not np.isfinite(lam0): return float("nan")
    return float(np.exp(np.interp(np.log(lam0), np.log(lam), np.log(rel))))


# ----------------------------------------------------------------------------- main
def main():
    (OUT / "data").mkdir(parents=True, exist_ok=True); (OUT / "figures").mkdir(parents=True, exist_ok=True)
    spec = {n: spectral_inputs(n) for n in TARGET_NAMES}
    # spot checks of the formula against Sam's table (eps = 5e-16, tanh): mixed 2/6/10 at N=64 -> 0.18958, N=256 -> 0.25423; sine 2pi N=64 -> 0.2627
    chk = {"mix_2_6_10@64": lambda_anchor("tanh", 64, 10 * np.pi, 1.75, 5e-16), "mix_2_6_10@256": lambda_anchor("tanh", 256, 10 * np.pi, 1.75, 5e-16),
           "sin2pi@64": lambda_anchor("tanh", 64, 2 * np.pi, 1.0, 5e-16), "sin2pi@64_gelu": lambda_anchor("gelu", 64, 2 * np.pi, 1.0, 5e-16),
           "sin2pi@64_swish": lambda_anchor("swish", 64, 2 * np.pi, 1.0, 5e-16)}
    print("formula check vs Sam's table (expect 0.18958, 0.25423, 0.2627, 0.7542, 0.5263):", {k: round(v, 4) for k, v in chk.items()})
    preds = {"eps_rule": "eps_p = 2^-(p-1)", "spectral": {n: {k: v for k, v in s.items() if k != "lines"} for n, s in spec.items()},
             "width": {}, "precision": {}}
    for act in ACTS:
        for n in TARGET_NAMES:
            s = spec[n]
            for N in WIDTHS:
                preds["width"][f"{act}|{n}|{N}"] = {"anchor": lambda_anchor(act, N, s["omega_M"], s["B"], 2.0 ** -52), "const": lambda_const(act, 2.0 ** -52),
                                                   "theta_M": 2 * s["omega_M"] / N}
            for p in PBITS:
                eps = 2.0 ** -(p - 1)
                preds["precision"][f"{act}|{n}|{p}"] = {"anchor": lambda_anchor(act, N_PREC, s["omega_M"], s["B"], eps), "const": lambda_const(act, eps),
                                                        "theta_M": 2 * s["omega_M"] / N_PREC}
    (OUT / "data" / f"anchor_rule_predictions{SUF}.json").write_text(json.dumps(preds, indent=1))
    print("spectral inputs:"); [print(f"  {n:12s} B={s['B']:.4f} omega_M={s['omega_M']:.3f} ({s['omega_M']/np.pi:.2f} pi) delta={s['delta']:.3f} sigma={s['sigma']:.3f}") for n, s in spec.items()]
    if "--predict-only" in sys.argv: return

    jobs_w = [{"act": a, "lam": l, "N": N, "fn": t, "p": 53} for a in ACTS for t in TARGET_NAMES for N in WIDTHS for l in LAM]
    jobs_p = [{"act": a, "lam": l, "N": N_PREC, "fn": t, "p": p} for p in PBITS for a in ACTS for t in TARGET_NAMES for l in LAM]
    rows_w = run_jobs(jobs_w, OUT / "data" / "anchor_rule_rows_width.json")
    rows_p = run_jobs(jobs_p, OUT / "data" / "anchor_rule_rows_precision.json")

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import importlib.util
    sp = importlib.util.spec_from_file_location("c9", LR / "run_c09_general.py"); c9 = importlib.util.module_from_spec(sp); sp.loader.exec_module(c9)
    c9.log_khat = log_khat; c9.KERNEL_ORDER = ORDER          # gaussian kernel for the fiber projection of the non-tones
    cmap = plt.get_cmap("viridis")

    def fiber_curve(act, n, N):
        ff = fiber_floor(act, n, N, LAM)
        if ff is not None: return ff
        return c9.fiber_prediction(act, n, N, LAM, reach="main", halo=HALO)["err"]

    def draw(ax, lam, rel, color, lam_new, lam_old, old_style):
        ax.loglog(lam, rel, "-", color=color, lw=1.3)
        if np.isfinite(lam_old):
            ax.axvline(lam_old, **old_style)
        if np.isfinite(lam_new):
            y = y_on_curve(lam_new, lam, rel)
            ax.axvline(lam_new, color=color, lw=0.9, ls="-", alpha=0.85)
            ax.axhline(y, color=color, lw=0.5, ls=":", alpha=0.5)
            ax.plot(lam_new, y, "o", color=color, ms=6, mec="k", mew=0.6, zorder=6)
            return y
        return float("nan")

    summary = []
    # ---- figure 1: width sweep at fp64
    cw = {N: cmap(i / (len(WIDTHS) - 1)) for i, N in enumerate(WIDTHS)}
    fig, axes = plt.subplots(len(TARGET_NAMES), len(ACTS), figsize=(18, 22), sharex=True, sharey=True)
    for i, n in enumerate(TARGET_NAMES):
        for j, act in enumerate(ACTS):
            ax = axes[i, j]; lc = lambda_const(act, 2.0 ** -52)
            ax.axvline(lc, color="0.55", lw=1.0, ls=":", zorder=1)
            for N in WIDTHS:
                lam, rel = curve(rows_w, act=act, N=N, fn=n)
                la = preds["width"][f"{act}|{n}|{N}"]["anchor"]
                y = draw(ax, lam, rel, cw[N], la, float("nan"), None)
                sm = smooth(rel); wall = lam[np.where(sm <= 10 * sm.min())[0].max()]
                summary.append({"sweep": "width", "act": act, "fn": n, "N": N, "p": 53, "eps": 2.0 ** -52, "anchor": la, "const": lc, "E_anchor": y,
                                "E_const": y_on_curve(lc, lam, rel), "E_min": float(sm.min()), "argmin": float(lam[sm.argmin()]), "wall": float(wall),
                                "theta_M": 2 * spec[n]["omega_M"] / N})
            ax.set_xlim(0.03, 1.5); ax.set_ylim(1e-16, 1e1); ax.grid(True, which="both", alpha=0.25)
            if i == 0: ax.set_title(f"{act}  (r = {ORDER[act]})", fontsize=12)
            if j == 0: ax.set_ylabel(f"{TEX[n]}\nB={spec[n]['B']:.3g}, $\\omega$={spec[n]['omega_M']/np.pi:.2f}$\\pi$\nrel $L_2$", fontsize=9)
            if i == len(TARGET_NAMES) - 1: ax.set_xlabel(r"$\lambda$")
    handles = [Line2D([0], [0], color=cw[N], lw=1.6, label=f"N={N}") for N in WIDTHS]
    handles += [Line2D([0], [0], marker="o", color="gray", lw=0, ms=6, mec="k", label=r"new rule: $\lambda_{\rm anchor}$ ($B\,R\leq\varepsilon$), $y$ = measured error there"),
                Line2D([0], [0], color="gray", lw=0.9, label=r"vertical at $\lambda_{\rm anchor}$ (colour = N)"),
                Line2D([0], [0], color="0.55", lw=1.0, ls=":", label=r"old rule: $A_K(\lambda)=\varepsilon$ (N-independent)")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=4, frameon=False, fontsize=9)
    fig.suptitle(r"Anchor rule, width sweep: native fp64, $\varepsilon=2^{-52}$, halo 32; non-tones use $\omega_M$ = 2$\sigma$ of the spectral energy" if OMEGA_MODE == "max" else r"Anchor rule (MEAN frequency), width sweep: native fp64, $\varepsilon=2^{-52}$, halo 32; $\omega$ = $|b_j|$-weighted mean $|\omega_j|$, B = full coefficient sum", y=0.965, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.955]); fig.savefig(OUT / "figures" / f"anchor_rule_width{SUF}.png", dpi=130); plt.close(fig)

    # ---- figure 2: precision sweep at N_PREC
    cp = {p: cmap(i / (len(PBITS) - 1)) for i, p in enumerate(PBITS)}
    fig, axes = plt.subplots(len(TARGET_NAMES), len(ACTS), figsize=(18, 22), sharex=True, sharey=True)
    for i, n in enumerate(TARGET_NAMES):
        for j, act in enumerate(ACTS):
            ax = axes[i, j]
            for p in PBITS:
                eps = 2.0 ** -(p - 1)
                lam, rel = curve(rows_p, act=act, N=N_PREC, fn=n, p=p)
                la = preds["precision"][f"{act}|{n}|{p}"]["anchor"]; lc = preds["precision"][f"{act}|{n}|{p}"]["const"]
                y = draw(ax, lam, rel, cp[p], la, lc, dict(color=cp[p], lw=0.9, ls=":", alpha=0.35, zorder=1))
                sm = smooth(rel); wall = lam[np.where(sm <= 10 * sm.min())[0].max()]
                summary.append({"sweep": "precision", "act": act, "fn": n, "N": N_PREC, "p": p, "eps": eps, "anchor": la, "const": lc, "E_anchor": y,
                                "E_const": y_on_curve(lc, lam, rel), "E_min": float(sm.min()), "argmin": float(lam[sm.argmin()]), "wall": float(wall),
                                "theta_M": 2 * spec[n]["omega_M"] / N_PREC})
            ax.loglog(LAM, fiber_curve(act, n, N_PREC), "k-.", lw=0.9, zorder=2)
            ax.set_xlim(0.03, 1.5); ax.set_ylim(1e-16, 1e1); ax.grid(True, which="both", alpha=0.25)
            if i == 0: ax.set_title(f"{act}  (r = {ORDER[act]})", fontsize=12)
            if j == 0: ax.set_ylabel(f"{TEX[n]}\nB={spec[n]['B']:.3g}, $\\omega$={spec[n]['omega_M']/np.pi:.2f}$\\pi$\nrel $L_2$", fontsize=9)
            if i == len(TARGET_NAMES) - 1: ax.set_xlabel(r"$\lambda$")
    handles = [Line2D([0], [0], color=cp[p], lw=1.6, label=f"p={p}" + (" (fp64)" if p == 53 else " (~fp32, emul.)" if p == 24 else " (~fp16, emul.)" if p == 11 else " (emul.)")) for p in PBITS]
    handles += [Line2D([0], [0], marker="o", color="gray", lw=0, ms=6, mec="k", label=r"new rule: $\lambda_{\rm anchor}$ ($B\,R\leq\varepsilon_p$), $y$ = measured error there"),
                Line2D([0], [0], color="gray", lw=0.9, ls=":", alpha=0.6, label=r"old rule: $A_K(\lambda)=\varepsilon_p$ (dotted, faded, colour = p)"),
                Line2D([0], [0], color="k", lw=0.9, ls="-.", label="fiber floor, exact arithmetic")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=5, frameon=False, fontsize=9)
    fig.suptitle(rf"Anchor rule{' (MEAN frequency)' if OMEGA_MODE == 'mean' else ''}, precision sweep: N = {N_PREC}, halo 32, $\varepsilon_p=2^{{-(p-1)}}$; p < 53 EMULATED (inputs, coefficients, partial sums rounded; SVD cutoff $2^{{-p}}$; solver internals not emulated)", y=0.965, fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.955]); fig.savefig(OUT / "figures" / f"anchor_rule_precision{SUF}.png", dpi=130); plt.close(fig)
    (OUT / "data" / f"anchor_rule_summary{SUF}.json").write_text(json.dumps(summary, indent=1))
    print("saved figures")

    # ---- console summary
    def med(vals): v = np.array([x for x in vals if np.isfinite(x)]); return (np.median(v) if v.size else float("nan")), v.size
    for sweep in ("width", "precision"):
        print(f"\n[{sweep}] per activation: median anchor/wall, median const/wall, median E(anchor)/E_min, median E(const)/E_min, median E(anchor)/eps  (n cells with a defined anchor)")
        for act in ACTS:
            S = [s for s in summary if s["sweep"] == sweep and s["act"] == act and np.isfinite(s["anchor"])]
            aw, k = med([s["anchor"] / s["wall"] for s in S]); cw_, _ = med([s["const"] / s["wall"] for s in S])
            ea, _ = med([s["E_anchor"] / s["E_min"] for s in S]); ec, _ = med([s["E_const"] / s["E_min"] for s in S]); ee, _ = med([s["E_anchor"] / s["eps"] for s in S])
            print(f"  {act:8s} anchor/wall {aw:5.2f}  const/wall {cw_:5.2f}  E(anchor)/Emin {ea:7.1f}  E(const)/Emin {ec:7.1f}  E(anchor)/eps {ee:9.1f}  (n={k})")
        print(f"[{sweep}] cells where the anchor is past the wall (anchor/wall > 1.1):")
        for s in summary:
            if s["sweep"] == sweep and np.isfinite(s["anchor"]) and s["anchor"] / s["wall"] > 1.1:
                print(f"    {s['act']:8s} {s['fn']:12s} N={s['N']:3d} p={s['p']:2d} anchor {s['anchor']:.3f} wall {s['wall']:.3f} E(anchor) {s['E_anchor']:.1e} Emin {s['E_min']:.1e} theta_M {s['theta_M']:.2f}")


if __name__ == "__main__":
    main()
