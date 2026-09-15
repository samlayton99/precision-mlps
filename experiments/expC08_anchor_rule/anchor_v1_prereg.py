"""Pre-registered test of the lambda anchor rule, version 1 (2026-09-10).

MOVED 2026-09-10 from experiments/expC07_lambda_energy_rule/lambda_rule/ into expC08; only the path constants below changed.
The byte-identical file whose SHA-256 is recorded in PREREGISTRATION_anchor_v1.md is archive/anchor_v1_prereg_as_registered.py.

Rule under test (frozen; from anchor_rule_v1.md): lambda_v1 = sup{lambda on the 600-point log grid [0.03, 1.5] :
B R_{K,r}(lambda, N, omega_bar) <= eps_p}, eps_p = 2^(1-p), with B = sum|b_j| (raw target units), omega_bar = the
|b_j|-weighted mean |omega_j|, theta = 2 omega_bar / N, the first-pair ratio with denominator min{1, H(theta/lambda)} and
the 1/(1 - rho_K) tail factor. Implementation: anchor_rule.lambda_anchor (unchanged since 2026-09-09).

Arms, all fixed before any solve:
  v1     the frozen rule above                                                  (primary)
  const  the target-free constant  H(2 pi / lambda) = eps_p                     (baseline)
  v1n    v1 with B replaced by B / ||f||_inf                                    (declared amplitude fix, secondary)
  max    v1 with omega_bar replaced by the top line frequency, line targets only (diagnostic)

Ten targets never used in expC07..C10 / precision_ladder / anchor_rule (see TARGETS), four activations, two sweeps:
  width:      native fp64, N in {48, 96, 192, 384}     (none of these widths was in the 2026-09-09 sweep)
  precision:  N = 96, p in {11, 16, 24, 32, 40, 48, 53} emulated as in precision_ladder.py, plus NATIVE fp32 (p = "24n")
Halo 32, 40-point lambda grid [0.03, 1.5], relative L2 on 4001 points.

Scoring (fixed before any solve; see PREREGISTRATION_anchor_v1.md written by --predict):
  smoothed curve S = 5-point running median of log error; E_min = min S; floor = median of S over {S <= 5 E_min}
  cell excluded if E_min > 1e-3 (unresolvable) or the arm's lambda is undefined (theta >= pi / no feasible point)
  regret(arm) = S(lambda_arm) / E_min  (log-interpolated on the SAME smoothed curve)
  on_wall(arm) = exact fiber floor at lambda_arm / floor > 2  (noise-free)
  C1 median regret <= 3;  C2 90th percentile regret <= 30;  C3 on-wall fraction <= 0.10   (per activation, per sweep)
  verdict per activation: PASS iff C1-C3 hold in both sweeps
  C4 v1 vs const: per-cell log(regret_v1 / regret_const); per activation over both sweeps, median ratio and a two-sided
     exact sign test on the nonzero differences; "v1 better" iff median ratio < 0.8 and p < 0.05, "worse" iff > 1.25 and
     p < 0.05, else "indistinguishable"
  C5 amplitude: on the cells whose RAW B is outside [0.5, 2] (amp3_mix, amp001_mix, equal_5_7_11, lorentz_off), median regret v1n vs v1
     (amended before --run: the first draft keyed on B/||f||_inf, which is within [0.5, 2] for every target and selects nothing)
Priors written down before the run: tanh, gelu, swish PASS; gaussian fails C3; C4 indistinguishable for r >= 1.
Nothing in this file is to be changed after --run; results go to anchor_v1_prereg_results.md.

Run: uv run --extra dev python experiments/expC08_anchor_rule/anchor_v1_prereg.py --predict
     uv run --extra dev python experiments/expC08_anchor_rule/anchor_v1_prereg.py --run
"""
from __future__ import annotations
import hashlib, json, os, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent; REPO_ROOT = HERE.parents[1]; sys.path.insert(0, str(HERE))
LR = REPO_ROOT / "experiments" / "expC07_lambda_energy_rule" / "lambda_rule"   # run_c09_general.py (fiber projection) lives there
import anchor_rule as ar  # noqa: E402  (frozen rule: lambda_anchor, lambda_const, log_khat, ORDER, apply_act, rnd, smooth)
OUT = REPO_ROOT / "results" / "checkpoint_C_geometry" / "expC08_anchor_rule"
DATA, FIG = OUT / "data", OUT / "figures"

LAM = list(ar.LAM); ACTS = list(ar.ACTS); ORDER = ar.ORDER; HALO = ar.HALO
WIDTHS = [48, 96, 192, 384]; N_PREC = 96; PBITS = [11, 16, 24, 32, 40, 48, 53]
PREC_ARMS = [str(p) for p in PBITS] + ["24n"]        # "24n" = native float32 solve
ARMS = ["v1", "const", "v1n", "max"]
TWO_PI = 2 * np.pi

# ----------------------------------------------------------------------------- the ten fresh targets
# kind: "lines" (exact coefficients), "line" (whole-line target, numerical transform), "periodic" (period given)
TARGETS = {
    "amp3_mix":     (r"$3\sin3\pi x+0.3\sin9\pi x$",                          lambda x: 3 * np.sin(3 * np.pi * x) + 0.3 * np.sin(9 * np.pi * x), "lines", [(3.0, 3), (0.3, 9)]),
    "amp001_mix":   (r"$0.01(\sin2\pi x+\sin4\pi x)$",                        lambda x: 0.01 * (np.sin(2 * np.pi * x) + np.sin(4 * np.pi * x)), "lines", [(0.01, 2), (0.01, 4)]),
    "tail_cos":     (r"$\cos\pi x+0.1\cos7\pi x+0.01\cos13\pi x+0.001\cos19\pi x$", lambda x: np.cos(np.pi * x) + 0.1 * np.cos(7 * np.pi * x) + 0.01 * np.cos(13 * np.pi * x) + 0.001 * np.cos(19 * np.pi * x), "lines", [(1.0, 1), (0.1, 7), (0.01, 13), (0.001, 19)]),
    "equal_5_7_11": (r"$\sin5\pi x+\sin7\pi x+\sin11\pi x$",                  lambda x: np.sin(5 * np.pi * x) + np.sin(7 * np.pi * x) + np.sin(11 * np.pi * x), "lines", [(1.0, 5), (1.0, 7), (1.0, 11)]),
    "runge9":       (r"$1/(1+9x^2)$",                                          lambda x: 1.0 / (1.0 + 9.0 * x ** 2), "line", None),
    "gauss8":       (r"$e^{-8x^2}$",                                           lambda x: np.exp(-8.0 * x ** 2), "line", None),
    "invcos":       (r"$1/(2+\cos3\pi x)$",                                    lambda x: 1.0 / (2.0 + np.cos(3 * np.pi * x)), "periodic", 2.0 / 3.0),
    "lorentz_off":  (r"$1/((x+0.3)^2+0.09)$",                                  lambda x: 1.0 / ((x + 0.3) ** 2 + 0.09), "line", None),
    "sech6":        (r"$\mathrm{sech}(6x)$",                                   lambda x: 1.0 / np.cosh(6.0 * x), "line", None),
    "xgauss":       (r"$x\,e^{-4x^2}$",                                        lambda x: x * np.exp(-4.0 * x ** 2), "line", None),
}
TEX = {k: v[0] for k, v in TARGETS.items()}


def f_of(name): return TARGETS[name][1]


def spectral_inputs(name):
    """B = sum|b_j| (raw units), omega_bar = |b_j|-weighted mean |omega_j|, omega_max (lines only), ||f||_inf on [-1,1]."""
    tex, f, kind, extra = TARGETS[name]
    finf = float(np.max(np.abs(f(np.linspace(-1, 1, 4001)))))
    if kind == "lines":                                   # a sin(k pi x) or a cos(k pi x): two coefficients of magnitude a/2 at +-k pi
        B = float(sum(a for a, _ in extra)); wbar = float(sum(a * k for a, k in extra) * np.pi / B); wmax = float(max(k for _, k in extra) * np.pi)
        return {"B": B, "omega_bar": wbar, "omega_max": wmax, "f_inf": finf, "kind": kind}
    if kind == "periodic":
        P = extra; M = 2 ** 14; x = np.arange(M) * P / M; c = np.abs(np.fft.fft(f(x)) / M); w = np.abs(TWO_PI * np.fft.fftfreq(M, d=P / M))
        return {"B": float(c.sum()), "omega_bar": float((w * c).sum() / c.sum()), "omega_max": float("nan"), "f_inf": finf, "kind": kind}
    L = 256.0; M = 2 ** 21; dx = 2 * L / M; x = -L + np.arange(M) * dx
    F = np.abs(np.fft.fft(f(x)) * dx); w = np.abs(TWO_PI * np.fft.fftfreq(M, d=dx))
    return {"B": float(F.sum() / (2 * L)), "omega_bar": float((w * F).sum() / F.sum()), "omega_max": float("nan"), "f_inf": finf, "kind": kind}


def predict_cell(act, N, p, spec):
    eps = 2.0 ** (1 - p)
    out = {"const": ar.lambda_const(act, eps)}
    out["v1"] = ar.lambda_anchor(act, N, spec["omega_bar"], spec["B"], eps)
    out["v1n"] = ar.lambda_anchor(act, N, spec["omega_bar"], spec["B"] / spec["f_inf"], eps)
    out["max"] = ar.lambda_anchor(act, N, spec["omega_max"], spec["B"], eps) if np.isfinite(spec["omega_max"]) else float("nan")
    out["theta_bar"] = 2 * spec["omega_bar"] / N; out["theta_max"] = 2 * spec["omega_max"] / N if np.isfinite(spec["omega_max"]) else float("nan")
    for k in ("v1", "v1n", "max"):
        out[k + "_status"] = "undefined" if not np.isfinite(out[k]) else ("upper_search_limit" if out[k] >= 1.5 - 1e-12 else "ok")
    return out


# ----------------------------------------------------------------------------- solves
def solve(job):
    from scipy.linalg import lstsq
    act, lam, N, fn, p = job["act"], job["lam"], job["N"], job["fn"], job["p"]; f = f_of(fn)
    h = 2.0 / N; c = -1.0 + np.arange(-HALO, N + HALO + 1) * h; g = lam / h
    x = np.linspace(-1, 1, max(2003, 2 * c.size + 3)); xe = np.linspace(-1, 1, 4001); fe = f(xe)
    if p == "24n":                                        # native float32, as fp32_quick.py
        dt = np.float32; c32 = c.astype(dt); g32 = dt(g); x32 = x.astype(dt)
        A = np.hstack([ar.apply_act(g32 * (x32[:, None] - c32[None, :]), act), np.ones((x.size, 1), dt)]).astype(dt)
        sol = lstsq(A, f(x).astype(dt), lapack_driver="gelsd")[0]
        xe32 = xe.astype(dt)
        fit = (ar.apply_act(g32 * (xe32[:, None] - c32[None, :]), act) @ sol[:-1].astype(dt) + sol[-1]).astype(np.float64)
        return {**job, "rel_l2": float(np.linalg.norm(fit - fe) / np.linalg.norm(fe))}
    p = int(p); eps = 2.0 ** (1 - p)
    A = ar.rnd(np.hstack([ar.apply_act(g * (x[:, None] - c[None, :]), act), np.ones((x.size, 1))]), p)
    b = ar.rnd(f(x), p)
    sol = ar.rnd(lstsq(A, b, cond=eps, lapack_driver="gelsd")[0], p)
    Ae = ar.rnd(ar.apply_act(g * (xe[:, None] - c[None, :]), act), p)
    if p >= 53:
        fit = Ae @ sol[:-1] + sol[-1]
    else:
        fit = ar.rnd(np.full(xe.size, sol[-1]), p)
        for k in range(Ae.shape[1]):
            fit = ar.rnd(fit + ar.rnd(Ae[:, k] * sol[k], p), p)
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


def curve(rows, **key):
    pts = sorted([r for r in rows if all(r[k] == v for k, v in key.items())], key=lambda r: r["lam"])
    return np.array([r["lam"] for r in pts]), np.array([r["rel_l2"] for r in pts])


def interp(l0, lam, y): return float(np.exp(np.interp(np.log(l0), np.log(lam), np.log(np.maximum(y, 1e-300)))))


# ----------------------------------------------------------------------------- pre-registration
def predict():
    DATA.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
    spec = {n: spectral_inputs(n) for n in TARGETS}
    # closed-form checks of the numerical spectrum routine on targets with known inputs (not part of the test)
    chk = {}
    for name, f, kind, extra, want in [("runge25", lambda x: 1 / (1 + 25 * x ** 2), "line", None, (1.0, 5.0)),
                                       ("gauss20", lambda x: np.exp(-20 * x ** 2), "line", None, (1.0, float(np.sqrt(80 / np.pi)))),
                                       ("expsin", lambda x: np.exp(np.sin(3 * np.pi * x)), "periodic", 2 / 3, (float(np.e), 6.349))]:
        TARGETS[name] = ("", f, kind, extra); s = spectral_inputs(name); del TARGETS[name]
        chk[name] = {"B": s["B"], "omega_bar": s["omega_bar"], "want": want}
    preds = {"written": datetime.now(timezone.utc).isoformat(), "eps": "2^(1-p)", "grid": "600-point log grid [0.03, 1.5] (anchor_rule.LAM_GRID_RULE)",
             "spectral_inputs": spec, "spectrum_routine_check": chk, "width": {}, "precision": {}}
    for act in ACTS:
        for n in TARGETS:
            for N in WIDTHS: preds["width"][f"{act}|{n}|{N}"] = predict_cell(act, N, 53, spec[n])
            for p in PBITS: preds["precision"][f"{act}|{n}|{p}"] = predict_cell(act, N_PREC, p, spec[n])
            preds["precision"][f"{act}|{n}|24n"] = predict_cell(act, N_PREC, 24, spec[n])
    pp = DATA / "anchor_v1_prereg_predictions.json"; pp.write_text(json.dumps(preds, indent=1))
    h_pred = hashlib.sha256(pp.read_bytes()).hexdigest(); h_self = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    h_rule = hashlib.sha256((HERE / "anchor_rule.py").read_bytes()).hexdigest()
    n_up = {sw: sum(v["v1_status"] == "upper_search_limit" for v in preds[sw].values()) for sw in ("width", "precision")}
    n_un = {sw: sum(v["v1_status"] == "undefined" for v in preds[sw].values()) for sw in ("width", "precision")}
    doc = f"""# Pre-registration: lambda anchor rule v1 on ten fresh targets

Written {preds['written']} by `anchor_v1_prereg.py --predict`, BEFORE any least-squares solve of this test.

## Hashes
- `anchor_v1_prereg.py` (rule call, targets, arms, sweeps, scoring, criteria): `{h_self}`
- `anchor_rule.py` (the frozen rule implementation `lambda_anchor`, unchanged since 2026-09-09): `{h_rule}`
- `data/anchor_v1_prereg_predictions.json` (every lambda for every arm and cell): `{h_pred}`

## Rule under test
$\\lambda_{{v1}}=\\sup\\{{\\lambda\\in G: B\\,\\mathcal R_{{K,r}}(\\lambda,N,\\bar\\omega)\\le 2^{{1-p}}\\}}$, $G$ the 600-point log grid on $[0.03,1.5]$, $B=\\sum|b_j|$ in raw target units, $\\bar\\omega=\\sum|b_j||\\omega_j|/B$, first-pair ratio with denominator $\\min\\{{1,H(\\theta/\\lambda)\\}}$ and factor $1/(1-\\rho_K)$, exactly as in `anchor_rule_v1.md`.

## Arms
v1 (primary), const ($H(2\\pi/\\lambda)=2^{{1-p}}$, baseline), v1n ($B\\to B/\\|f\\|_\\infty$, declared amplitude fix), max ($\\bar\\omega\\to\\omega_{{\\max}}$, line targets only, diagnostic).

## Targets (none used before) and their rule inputs
| target | kind | B | $\\bar\\omega/\\pi$ | $\\omega_{{\\max}}/\\pi$ | $\\|f\\|_\\infty$ | $B/\\|f\\|_\\infty$ |
|---|---|---:|---:|---:|---:|---:|
""" + "\n".join(f"| {TEX[n]} | {s['kind']} | {s['B']:.4g} | {s['omega_bar']/np.pi:.3f} | {s['omega_max']/np.pi if np.isfinite(s['omega_max']) else float('nan'):.3g} | {s['f_inf']:.4g} | {s['B']/s['f_inf']:.3g} |" for n, s in spec.items()) + f"""

Numerical-spectrum routine checked on known inputs: runge25 B={chk['runge25']['B']:.6f} (1), omega_bar={chk['runge25']['omega_bar']:.5f} (5); gauss20 B={chk['gauss20']['B']:.6f} (1), omega_bar={chk['gauss20']['omega_bar']:.5f} ({chk['gauss20']['want'][1]:.5f}); expsin B={chk['expsin']['B']:.6f} (e), omega_bar={chk['expsin']['omega_bar']:.4f} (6.349).

## Sweeps
Width: native fp64, $N\\in\\{{48,96,192,384\\}}$. Precision: $N=96$, $p\\in\\{{11,16,24,32,40,48,53\\}}$ emulated (inputs, coefficients, partial sums rounded to $p$ bits; SVD cutoff $2^{{1-p}}$; solver internals fp64) plus native float32. Halo 32, 40 $\\lambda$ points on $[0.03,1.5]$, rel $L_2$ on 4001 points. Four activations: tanh ($r=1$), gelu ($r=2$), swish ($r=2$), gaussian $e^{{-x^2}}$ ($r=0$).

Status counts for v1 before the run: upper_search_limit width {n_up['width']}, precision {n_up['precision']}; undefined ($\\theta\\ge\\pi$ or no feasible point) width {n_un['width']}, precision {n_un['precision']}. Upper-limit cells are scored but flagged; they carry no threshold claim.

## Scoring and criteria (fixed now)
- $S$ = 5-point running median of $\\log$ error; $E_{{\\min}}=\\min S$; floor = median of $S$ over $\\{{S\\le5E_{{\\min}}\\}}$.
- Cell excluded if $E_{{\\min}}>10^{{-3}}$ or the arm's $\\lambda$ is undefined.
- regret(arm) $=S(\\lambda_{{\\rm arm}})/E_{{\\min}}$, log-interpolated on the same smoothed curve.
- on_wall(arm): exact fiber floor at $\\lambda_{{\\rm arm}}$ (exact line sum for line targets, expC09 whole-line projection otherwise) divided by the floor $>2$.
- **C1** median regret $\\le3$; **C2** 90th percentile regret $\\le30$; **C3** on-wall fraction $\\le0.10$; each per activation and per sweep. **Verdict per activation: PASS iff C1 to C3 hold in both sweeps.**
- **C4** v1 vs const: per-cell $\\log(\\text{{regret}}_{{v1}}/\\text{{regret}}_{{\\rm const}})$ pooled over both sweeps per activation; median ratio and two-sided exact sign test on nonzero differences. "v1 better" iff median $<0.8$ and $p<0.05$; "worse" iff median $>1.25$ and $p<0.05$; else indistinguishable.
- **C5** amplitude: on cells whose raw $B$ is outside $[0.5,2]$ (four targets: $B=3.3,\ 0.02,\ 3,\ 11.1$), median regret of v1n against v1. (Amended before the run: the first draft keyed on $B/\\|f\\|_\\infty$, which is within $[0.5,2]$ for every target and selected nothing.)

## Priors (stated now)
tanh, gelu, swish: PASS. gaussian: fails C3 (on-wall fraction about 0.15 in the 2026-09-09 mean-frequency run). C4: indistinguishable for $r\\ge1$; if anything v1 better on gaussian. C5: v1n no worse than v1.

No criterion, arm, target, width or precision will be changed after `--run`. Results: `anchor_v1_prereg_results.md`.
"""
    (OUT / "PREREGISTRATION_anchor_v1.md").write_text(doc)
    print(doc)


# ----------------------------------------------------------------------------- run, score, plot
def run():
    preds = json.loads((DATA / "anchor_v1_prereg_predictions.json").read_text()); spec = preds["spectral_inputs"]
    h_pred = hashlib.sha256((DATA / "anchor_v1_prereg_predictions.json").read_bytes()).hexdigest()
    jobs_w = [{"act": a, "lam": l, "N": N, "fn": t, "p": 53} for a in ACTS for t in TARGETS for N in WIDTHS for l in LAM]
    jobs_p = [{"act": a, "lam": l, "N": N_PREC, "fn": t, "p": p} for p in PREC_ARMS for a in ACTS for t in TARGETS for l in LAM]
    rows_w = run_jobs(jobs_w, DATA / "anchor_v1_prereg_rows_width.json")
    rows_p = run_jobs(jobs_p, DATA / "anchor_v1_prereg_rows_precision.json")

    import importlib.util
    sp = importlib.util.spec_from_file_location("c9", LR / "run_c09_general.py"); c9 = importlib.util.module_from_spec(sp); sp.loader.exec_module(c9)
    c9.log_khat = ar.log_khat; c9.KERNEL_ORDER = ORDER
    for n, (tex, f, kind, extra) in TARGETS.items(): c9.TARGETS[n] = (tex, f)
    fib_cache = {}

    def fiber(act, n, N):
        if (act, n, N) not in fib_cache:
            kind, extra = TARGETS[n][2], TARGETS[n][3]
            if kind == "lines":
                lines = [(a / 2, s * k * np.pi) for a, k in extra for s in (+1, -1)]
                r = ORDER[act]; out = []
                for lam in LAM:
                    num = den = 0.0
                    for b, w in lines:
                        th = abs(w) * 2.0 / N
                        if th >= np.pi: num += b * b; den += b * b; continue
                        ms = np.arange(-8, 9); tt = th + TWO_PI * ms
                        lt = 2 * ar.log_khat(act, np.abs(tt) / lam) - 2 * r * np.log(np.abs(tt)); lt -= lt.max(); v = np.exp(lt)
                        num += b * b * v[ms != 0].sum() / v.sum(); den += b * b
                    out.append(np.sqrt(num / den))
                fib_cache[(act, n, N)] = np.array(out)
            else:
                fib_cache[(act, n, N)] = np.asarray(c9.fiber_prediction(act, n, N, LAM, reach="main", halo=HALO)["err"])
        return fib_cache[(act, n, N)]

    cells = []
    for sweep, rows in (("width", rows_w), ("precision", rows_p)):
        for key, pr in preds[sweep].items():
            act, n, third = key.split("|"); N = int(third) if sweep == "width" else N_PREC; p = 53 if sweep == "width" else third
            lam, rel = curve(rows, act=act, N=N, fn=n, p=p); S = ar.smooth(rel)
            emin = float(S.min()); floor = float(np.median(S[S <= 5 * emin])); ff = fiber(act, n, N)
            cell = {"sweep": sweep, "act": act, "fn": n, "N": N, "p": p, "E_min": emin, "floor": floor, "argmin": float(lam[S.argmin()]),
                    "excluded": emin > 1e-3, "theta_bar": pr["theta_bar"], "theta_max": pr["theta_max"], "B_over_finf": spec[n]["B"] / spec[n]["f_inf"]}
            for arm in ARMS:
                la = pr[arm]
                if not (isinstance(la, float) and np.isfinite(la)): cell[arm] = None; continue
                cell[arm] = {"lam": la, "status": pr.get(arm + "_status", "ok"), "regret": interp(la, lam, S) / emin, "fiber_over_floor": interp(la, LAM, ff) / floor,
                             "E": interp(la, lam, S)}
            cells.append(cell)

    # ---- criteria
    from scipy.stats import binomtest
    def pct(v, q): return float(np.percentile(v, q)) if len(v) else float("nan")
    verdict = {}
    for act in ACTS:
        V = {"C4": None, "C5": None}
        for sweep in ("width", "precision"):
            C = [c for c in cells if c["act"] == act and c["sweep"] == sweep and not c["excluded"] and c["v1"] is not None]
            reg = [c["v1"]["regret"] for c in C]; ow = [c["v1"]["fiber_over_floor"] > 2 for c in C]
            V[sweep] = {"n": len(C), "median_regret": pct(reg, 50), "p90_regret": pct(reg, 90), "on_wall_frac": float(np.mean(ow)) if C else float("nan"),
                        "C1": pct(reg, 50) <= 3, "C2": pct(reg, 90) <= 30, "C3": (float(np.mean(ow)) <= 0.10) if C else False}
            for arm in ("const", "v1n", "max"):
                A = [c for c in C if c[arm] is not None]
                V[sweep][arm] = {"n": len(A), "median_regret": pct([c[arm]["regret"] for c in A], 50), "p90_regret": pct([c[arm]["regret"] for c in A], 90),
                                 "on_wall_frac": float(np.mean([c[arm]["fiber_over_floor"] > 2 for c in A])) if A else float("nan")}
        V["PASS"] = all(V[s][k] for s in ("width", "precision") for k in ("C1", "C2", "C3"))
        P = [c for c in cells if c["act"] == act and not c["excluded"] and c["v1"] is not None and c["const"] is not None]
        d = np.array([np.log(c["v1"]["regret"] / c["const"]["regret"]) for c in P]); nz = d[np.abs(d) > 1e-9]
        pval = float(binomtest(int((nz < 0).sum()), int(nz.size), 0.5).pvalue) if nz.size else float("nan"); med = float(np.exp(np.median(d))) if d.size else float("nan")
        V["C4"] = {"n": int(d.size), "median_ratio_v1_over_const": med, "n_v1_better": int((nz < 0).sum()), "n_nonzero": int(nz.size), "sign_test_p": pval,
                   "call": "v1 better" if (med < 0.8 and pval < 0.05) else "v1 worse" if (med > 1.25 and pval < 0.05) else "indistinguishable"}
        Q = [c for c in cells if c["act"] == act and not c["excluded"] and c["v1"] is not None and c["v1n"] is not None and not (0.5 <= spec[c["fn"]]["B"] <= 2)]
        V["C5"] = {"n": len(Q), "median_regret_v1": pct([c["v1"]["regret"] for c in Q], 50), "median_regret_v1n": pct([c["v1n"]["regret"] for c in Q], 50),
                   "median_ratio_v1n_over_v1": float(np.exp(np.median([np.log(c["v1n"]["regret"] / c["v1"]["regret"]) for c in Q]))) if Q else float("nan")}
        verdict[act] = V
    (DATA / "anchor_v1_prereg_scores.json").write_text(json.dumps({"predictions_sha256": h_pred, "cells": cells, "verdict": verdict}, indent=1))

    # ---- figures
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    cmap = plt.get_cmap("viridis")

    def mark(ax, lam, rel, color, pr):
        ax.loglog(lam, rel, "-", color=color, lw=1.2)
        lc = pr["const"]; ax.axvline(lc, color=color, lw=0.9, ls=":", alpha=0.35, zorder=1)
        la = pr["v1"]
        if isinstance(la, float) and np.isfinite(la):
            y = interp(la, lam, rel); ax.axvline(la, color=color, lw=0.9, alpha=0.85); ax.plot(la, y, "o", color=color, ms=6, mec="k", mew=0.6, zorder=6)
        ln = pr["v1n"]
        if isinstance(ln, float) and np.isfinite(ln) and abs(np.log(ln / la)) > 1e-6 if isinstance(la, float) and np.isfinite(la) else False:
            ax.plot(ln, interp(ln, lam, rel), "D", color=color, ms=5, mfc="none", mec=color, mew=1.2, zorder=6)

    def label(n, spec_n): return f"{TEX[n]}\nB={spec_n['B']:.3g}, $\\bar\\omega$={spec_n['omega_bar']/np.pi:.2f}$\\pi$, B/$\\|f\\|_\\infty$={spec_n['B']/spec_n['f_inf']:.2g}\nrel $L_2$"

    cw = {N: cmap(i / (len(WIDTHS) - 1)) for i, N in enumerate(WIDTHS)}
    fig, axes = plt.subplots(len(TARGETS), len(ACTS), figsize=(18, 34), sharex=True, sharey=True)
    for i, n in enumerate(TARGETS):
        for j, act in enumerate(ACTS):
            ax = axes[i, j]
            for N in WIDTHS:
                lam, rel = curve(rows_w, act=act, N=N, fn=n, p=53); mark(ax, lam, rel, cw[N], preds["width"][f"{act}|{n}|{N}"])
            ax.set_xlim(0.03, 1.5); ax.set_ylim(1e-16, 1e1); ax.grid(True, which="both", alpha=0.25)
            if i == 0: ax.set_title(f"{act}  (r = {ORDER[act]})", fontsize=12)
            if j == 0: ax.set_ylabel(label(n, spec[n]), fontsize=8)
            if i == len(TARGETS) - 1: ax.set_xlabel(r"$\lambda$")
    handles = [Line2D([0], [0], color=cw[N], lw=1.6, label=f"N={N}") for N in WIDTHS]
    handles += [Line2D([0], [0], marker="o", color="gray", lw=0, ms=6, mec="k", label=r"v1: $\lambda_{v1}$ (solid vertical), $y$ = measured error there"),
                Line2D([0], [0], marker="D", color="gray", lw=0, ms=5, mfc="none", label=r"v1n ($B/\|f\|_\infty$), where it differs from v1"),
                Line2D([0], [0], color="gray", lw=0.9, ls=":", alpha=0.6, label=r"const: $H(2\pi/\lambda)=\varepsilon$ (dotted, faded)")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=4, frameon=False, fontsize=9)
    fig.suptitle(r"Pre-registered v1 test, width sweep: ten fresh targets, native fp64, $\varepsilon=2^{-52}$, halo 32", y=0.975, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.968]); fig.savefig(FIG / "anchor_v1_prereg_width.png", dpi=120); plt.close(fig)

    cp = {p: cmap(i / (len(PBITS) - 1)) for i, p in enumerate(PBITS)}
    fig, axes = plt.subplots(len(TARGETS), len(ACTS), figsize=(18, 34), sharex=True, sharey=True)
    for i, n in enumerate(TARGETS):
        for j, act in enumerate(ACTS):
            ax = axes[i, j]
            for p in PBITS:
                lam, rel = curve(rows_p, act=act, N=N_PREC, fn=n, p=str(p)); mark(ax, lam, rel, cp[p], preds["precision"][f"{act}|{n}|{p}"])
            lam, rel = curve(rows_p, act=act, N=N_PREC, fn=n, p="24n"); ax.loglog(lam, rel, "--", color=cp[24], lw=1.4)
            la = preds["precision"][f"{act}|{n}|24n"]["v1"]
            if isinstance(la, float) and np.isfinite(la): ax.plot(la, interp(la, lam, rel), "s", color=cp[24], ms=6, mec="k", mew=0.6, zorder=6)
            ax.loglog(LAM, fiber(act, n, N_PREC), "k-.", lw=0.9, zorder=2)
            ax.set_xlim(0.03, 1.5); ax.set_ylim(1e-16, 1e1); ax.grid(True, which="both", alpha=0.25)
            if i == 0: ax.set_title(f"{act}  (r = {ORDER[act]})", fontsize=12)
            if j == 0: ax.set_ylabel(label(n, spec[n]), fontsize=8)
            if i == len(TARGETS) - 1: ax.set_xlabel(r"$\lambda$")
    handles = [Line2D([0], [0], color=cp[p], lw=1.6, label=f"p={p}" + (" (fp64)" if p == 53 else " (emul.)")) for p in PBITS]
    handles += [Line2D([0], [0], color=cp[24], lw=1.4, ls="--", label="native fp32 (square = v1 on it)"),
                Line2D([0], [0], marker="o", color="gray", lw=0, ms=6, mec="k", label=r"v1: $\lambda_{v1}$, $y$ = measured error there"),
                Line2D([0], [0], marker="D", color="gray", lw=0, ms=5, mfc="none", label="v1n where it differs"),
                Line2D([0], [0], color="gray", lw=0.9, ls=":", alpha=0.6, label="const (dotted, faded)"),
                Line2D([0], [0], color="k", lw=0.9, ls="-.", label="fiber floor, exact arithmetic")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=6, frameon=False, fontsize=9)
    fig.suptitle(rf"Pre-registered v1 test, precision sweep: N = {N_PREC}, halo 32, $\varepsilon_p=2^{{1-p}}$; p < 53 EMULATED (solver internals fp64) plus native fp32", y=0.975, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.968]); fig.savefig(FIG / "anchor_v1_prereg_precision.png", dpi=120); plt.close(fig)

    # regret summary: per activation, v1 vs const paired
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.6), sharex=True, sharey=True)
    for ax, act in zip(axes, ACTS):
        for sweep, mk in (("width", "o"), ("precision", "^")):
            C = [c for c in cells if c["act"] == act and c["sweep"] == sweep and not c["excluded"] and c["v1"] is not None]
            ax.scatter([c["const"]["regret"] for c in C], [c["v1"]["regret"] for c in C], marker=mk, s=30, alpha=0.7, edgecolors="k", linewidths=0.4,
                       c=["#d1352b" if c["v1"]["fiber_over_floor"] > 2 else "#2a6fbf" for c in C])
        ax.plot([1e-1, 1e4], [1e-1, 1e4], "k--", lw=0.8); ax.axhline(3, color="gray", lw=0.6, ls=":"); ax.axhline(30, color="gray", lw=0.6, ls=":")
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(0.5, 1e4); ax.set_ylim(0.5, 1e4); ax.grid(alpha=0.3, which="both")
        v = verdict[act]; ax.set_title(f"{act}: {'PASS' if v['PASS'] else 'FAIL'}; C4 {v['C4']['call']}", fontsize=11); ax.set_xlabel("regret of const")
    axes[0].set_ylabel("regret of v1")
    handles = [Line2D([0], [0], marker="o", color="gray", lw=0, label="width sweep"), Line2D([0], [0], marker="^", color="gray", lw=0, label="precision sweep"),
               Line2D([0], [0], marker="o", color="#d1352b", lw=0, label="v1 on the wall (fiber floor > 2x measured floor)"),
               Line2D([0], [0], marker="o", color="#2a6fbf", lw=0, label="v1 on the floor"), Line2D([0], [0], color="gray", ls=":", label="C1 (3) and C2 (30) levels")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=5, frameon=False, fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.9]); fig.savefig(FIG / "anchor_v1_prereg_regret.png", dpi=140); plt.close(fig)
    print("saved figures"); print(json.dumps(verdict, indent=1))


if __name__ == "__main__":
    if "--predict" in sys.argv: predict()
    elif "--run" in sys.argv: run()
    else: print(__doc__)
