"""expD21 analysis: paired effect sizes, in decades, with the seed noise floor stated.

WHAT CHANGED FROM `analysis_rank_deprecated.py` AND WHY
-------------------------------------------------------
The old analysis judged the five variants by mean competition rank (1-5) over 18
cells. That statistic cannot report this experiment's result, for three reasons:

  1. Constant-sum.  Each cell hands out the integers 1..5 exactly once, so the
     five mean ranks are algebraically forced to sum to 15 and average to 3.0.
     The reported "spread 2.67-3.11" is the width of a distribution that was
     pinned at its centre by construction, not a measurement.
  2. It is noise.  Uniform ranks on 1..5 have variance 2, so the standard error
     of a mean over 18 cells is sqrt(2/18) = 0.33.  The observed deviations from
     3.0 are at most 0.95 SE -- indistinguishable from shuffling variant labels
     at random inside each cell.
  3. It pools across a sign flip.  The treatment effect reverses between tanh
     and GELU.  Averaging an antisymmetric effect over equal-sized halves is
     guaranteed to return the null.

This analysis replaces rank with a paired, magnitude-preserving effect size and
states the uncertainty next to every number.

THE UNIT
--------
For cell c = (activation, class, problem), variant v, seed s:

    delta[c,v,s] = log10( err[c,baseline,s] / err[c,v,s] )        [decades]

Positive means the variant beats baseline.  +1 decade = 10x lower error.  The
comparison is PAIRED WITHIN SEED, which is legitimate here because seed controls
only the data realization (grid jitter, 2-D resample, PINN interior draw) and is
independent of variant -- the same seed gives every variant the same data.
Pairing removes both problem difficulty and data-draw luck.

Aggregation is a two-level design, and both levels are reported:
  - within a cell: mean over 3 seeds, 95% CI from t(2) -- this is the seed noise;
  - across cells: mean over the 9 cells of an activation, 95% CI from t(8).

WHAT `final_err` MEANS PER CLASS (this is not uniform, and it matters)
---------------------------------------------------------------------
  interp1d, interp2d : eval relative L2 of the fit.  The quantity of interest.
  pinn_inverse       : relative L2 of the FIELD u, NOT the recovered parameter.

For the inverse problems the field error is a poor proxy for the thing the
experiment is about.  Spearman(field_err, param_rel_err) over the 90 PINN runs
is 0.74 pooled, but 0.38 on burgers and 0.06 on allencahn.  So parameter
recovery is scored separately, in its own section and its own figure, and is
never folded into the interpolation effect sizes.
"""

from __future__ import annotations

import importlib.util
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


R = _load("expD21_run", HERE / "run.py")

OUT_DIR, DATA_DIR, FIG_DIR = R.OUT_DIR, R.DATA_DIR, R.FIG_DIR
VARIANTS, ACTS, SEEDS = R.VARIANTS, R.ACTS, R.SEEDS
CLASSES = list(R.PROBLEMS)
TREATED = [v for v in VARIANTS if v != "baseline"]

SHORT = {"baseline": "baseline", "rms_nocenter": "rms (no ctr)",
         "rms_center": "rms + center", "batchnorm_noaffine": "BN (no affine)",
         "layernorm_noaffine": "LN (no affine)"}
COLORS = {"baseline": "#444444", "rms_nocenter": "#1f77b4",
          "rms_center": "#d62728", "batchnorm_noaffine": "#2ca02c",
          "layernorm_noaffine": "#9467bd"}
CLS_COLOR = {"interp1d": "#0072b2", "interp2d": "#e69f00",
             "pinn_inverse": "#009e73"}
CLS_MARK = {"interp1d": "o", "interp2d": "s", "pinn_inverse": "^"}

T95_2 = float(stats.t.ppf(0.975, 2))    # 4.303, within-cell over 3 seeds
T95_8 = float(stats.t.ppf(0.975, 8))    # 2.306, across the 9 cells of an activation


# ============================ loading ============================

def load_rows():
    rows = []
    for f in sorted(DATA_DIR.glob("*.jsonl")):
        for line in open(f):
            rows.append(json.loads(line))
    return rows


def index(rows):
    """(act, cls, prob, seed) -> variant -> row."""
    idx = defaultdict(dict)
    for r in rows:
        idx[(r["activation"], r["class"], r["problem"], r["seed"])][r["variant"]] = r
    return idx


def deltas(rows, key="final_err"):
    """cell -> variant -> np.array of per-seed paired log10 ratios.

    A seed is used only if BOTH baseline and the variant produced a finite
    positive error for it, so a dropped run shrinks n rather than silently
    biasing the pair.
    """
    idx = index(rows)
    out = defaultdict(lambda: defaultdict(list))
    for (act, cls, prob, seed), byv in idx.items():
        if "baseline" not in byv:
            continue
        eb = byv["baseline"].get(key)
        if eb is None or not np.isfinite(eb) or eb <= 0:
            continue
        for v in VARIANTS:
            if v not in byv:
                continue
            ev = byv[v].get(key)
            if ev is None or not np.isfinite(ev) or ev <= 0:
                continue
            out[(act, cls, prob)][v].append(math.log10(eb / ev))
    return {c: {v: np.array(d, float) for v, d in bv.items()} for c, bv in out.items()}


def cell_stat(d):
    """mean, half-width of the 95% CI, n -- for one cell's per-seed deltas."""
    d = np.asarray(d, float)
    n = d.size
    if n == 0:
        return np.nan, np.nan, 0
    if n == 1:
        return float(d[0]), np.nan, 1
    sem = d.std(ddof=1) / math.sqrt(n)
    return float(d.mean()), float(stats.t.ppf(0.975, n - 1) * sem), n


def cells_of(D, act=None, cls=None):
    ks = sorted(D)
    if act:
        ks = [k for k in ks if k[0] == act]
    if cls:
        ks = [k for k in ks if k[1] == cls]
    return ks


# ============================ text report ============================

def hdr(t):
    print("\n" + "=" * 88)
    print(t)
    print("=" * 88)


def report_noise_floor(D):
    """The yardstick: how big is a paired log-ratio that means nothing?"""
    hdr("1. SEED NOISE FLOOR -- the yardstick every effect below must clear")
    print("Per-seed spread of the paired log-ratio, within a (cell, variant).")
    print("This is pure data-draw noise: it is what a variant with zero effect would show.\n")
    print(f"{'class':14s} {'n cells x var':>13s} {'median |sd|':>12s} {'p90 |sd|':>10s} "
          f"{'max |sd|':>10s} {'median 95% CI':>14s}")
    allsd = []
    for cls in CLASSES + [None]:
        sds, cis = [], []
        for c in cells_of(D, cls=cls):
            for v in TREATED:
                d = D[c].get(v)
                if d is None or d.size < 2:
                    continue
                sds.append(d.std(ddof=1))
                cis.append(T95_2 * d.std(ddof=1) / math.sqrt(d.size))
        if not sds:
            continue
        sds, cis = np.array(sds), np.array(cis)
        if cls is None:
            allsd = sds
        name = cls if cls else "ALL"
        print(f"{name:14s} {len(sds):13d} {np.median(sds):12.3f} "
              f"{np.percentile(sds,90):10.3f} {sds.max():10.3f} "
              f"+/- {np.median(cis):10.3f}")
    print("\nUnits are decades. Read: a cell-level effect smaller than roughly")
    print(f"+/- {np.median([T95_2*s/math.sqrt(3) for s in allsd]):.2f} decades "
          f"({10**np.median([T95_2*s/math.sqrt(3) for s in allsd]):.2f}x) is not "
          "distinguishable from seed noise.")
    return float(np.median(allsd))


def report_headline(D):
    """Effect per activation, the axis on which the effect actually varies."""
    hdr("2. HEADLINE -- mean paired effect per activation, in decades (+ = better than baseline)")
    print("Mean over the 9 cells of an activation; CI is 95% t(8) across cells.")
    print("'better/worse' counts cells whose own 95% seed CI excludes zero.\n")
    out = {}
    for act in ACTS:
        ks = cells_of(D, act=act)
        print(f"-- {act}  ({len(ks)} cells) " + "-" * 52)
        print(f"{'variant':16s} {'mean':>7s} {'95% CI':>16s} {'x-fold':>8s} "
              f"{'median':>8s} {'worst cell':>11s} {'best cell':>10s} {'better':>7s} {'worse':>6s}")
        for v in TREATED:
            per = [cell_stat(D[k].get(v, np.array([])))for k in ks]
            m = np.array([p[0] for p in per], float)
            ok = np.isfinite(m)
            m = m[ok]
            if m.size == 0:
                continue
            ci = T95_8 * m.std(ddof=1) / math.sqrt(m.size)
            nb = sum(1 for p in per if np.isfinite(p[0]) and np.isfinite(p[1]) and p[0] - p[1] > 0)
            nw = sum(1 for p in per if np.isfinite(p[0]) and np.isfinite(p[1]) and p[0] + p[1] < 0)
            print(f"{SHORT[v]:16s} {m.mean():+7.3f} [{m.mean()-ci:+6.3f},{m.mean()+ci:+6.3f}] "
                  f"{10**m.mean():8.2f} {np.median(m):+8.3f} {m.min():+11.3f} {m.max():+10.3f} "
                  f"{nb:3d}/{m.size:<3d} {nw:3d}/{m.size:<3d}")
            out[(act, v)] = (m.mean(), ci, m)
        print()
    return out


def report_by_class(D):
    """The resolving view: the sign of the effect is set by problem class."""
    hdr("2a. EFFECT BY (ACTIVATION x CLASS) -- the sign is structured, not random")
    print("The 9-cell aggregate in section 2 averages over problem classes whose effects")
    print("point in opposite directions. Split by class (3 cells each) and the structure")
    print("appears. 'sign' counts how many of the 3 cells agree with the group mean.\n")
    print(f"{'act':6s} {'class':13s} {'variant':18s} {'mean':>8s} {'x-fold':>8s} "
          f"{'median':>8s} {'95% CI over 3 cells':>22s} {'sign':>5s}")
    for act in ACTS:
        for cls in CLASSES:
            ks = cells_of(D, act=act, cls=cls)
            for v in TREATED:
                m = np.array([cell_stat(D[k].get(v, np.array([])))[0] for k in ks], float)
                m = m[np.isfinite(m)]
                if m.size < 2:
                    continue
                ci = float(stats.t.ppf(0.975, m.size - 1)) * m.std(ddof=1) / math.sqrt(m.size)
                same = int((np.sign(m) == np.sign(m.mean())).sum())
                mark = " <<<" if abs(m.mean()) > ci else ""
                print(f"{act:6s} {cls:13s} {SHORT[v]:18s} {m.mean():+8.3f} "
                      f"{10**m.mean():8.2f} {np.median(m):+8.3f} "
                      f"[{m.mean()-ci:+9.3f},{m.mean()+ci:+9.3f}] {same:d}/{m.size:d}{mark}")
            print()
    print("'<<<' = the 3-cell CI excludes zero. With n=3 that bar is very high, so read")
    print("the sign column and the magnitude together, not the CI alone.")


def report_variance(D):
    """Is the effect small, or is it large with an unstable sign? These differ."""
    hdr("2b. VARIANCE DECOMPOSITION -- small effect, or large effect with random sign?")
    print("An aggregate near zero has two very different causes, and they demand")
    print("different follow-ups:")
    print("  (i)  the transform does nothing        -> |effect| per cell ~ seed noise;")
    print("  (ii) it does a lot, sign varies by cell -> |effect| per cell >> seed noise,")
    print("       but between-cell sd >> |mean|.  Here the mean is the wrong summary.")
    print()
    print(f"{'act':6s} {'variant':16s} {'|mean|':>7s} {'rms|cell|':>10s} {'sd across':>10s} "
          f"{'seed sd':>8s} {'cell/noise':>11s} {'sd/|mean|':>10s}  verdict")
    for act in ACTS:
        for v in TREATED:
            ks = cells_of(D, act=act)
            cm = np.array([cell_stat(D[k].get(v, np.array([])))[0] for k in ks], float)
            cm = cm[np.isfinite(cm)]
            sds = [D[k][v].std(ddof=1) for k in ks
                   if v in D[k] and D[k][v].size > 1]
            noise = float(np.median(sds)) if sds else np.nan
            rms = float(np.sqrt((cm ** 2).mean()))
            sdc = float(cm.std(ddof=1))
            snr = rms / noise if noise > 0 else np.inf
            ratio = sdc / abs(cm.mean()) if cm.mean() != 0 else np.inf
            verdict = ("(ii) large, sign varies (see 2a: by class)" if snr > 3 and ratio > 1.5
                       else "(i) inert" if snr < 3
                       else "consistent")
            print(f"{act:6s} {SHORT[v]:16s} {abs(cm.mean()):7.3f} {rms:10.3f} "
                  f"{sdc:10.3f} {noise:8.3f} {snr:11.1f} {ratio:10.1f}  {verdict}")
    print("\ncell/noise = rms cell effect / median seed sd. sd/|mean| = between-cell sd")
    print("over the size of the aggregate; > 1.5 means the mean is not a summary of anything.")


def report_settled(rows):
    """Is the endpoint the answer, or just where the 2000-step budget landed?"""
    hdr("2c. IS THE VERDICT SETTLED AT 2000 STEPS?")
    print("Change in the paired effect over the last third of training (step ~1333 -> 2000).")
    print("A cell still moving by more than ~0.1 decades is reporting the budget, not the variant.\n")
    idx = index(rows)
    moved = []
    for act in ACTS:
        for cls in CLASSES:
            for prob in R.PROBLEMS[cls]:
                for v in TREATED:
                    dd = []
                    for s in SEEDS:
                        byv = idx.get((act, cls, prob, s), {})
                        if "baseline" not in byv or v not in byv:
                            continue
                        eb, ev = dict(byv["baseline"]["evals"]), dict(byv[v]["evals"])
                        st = sorted(set(eb) & set(ev))
                        if not st:
                            continue
                        mid = min(st, key=lambda t: abs(t - st[-1] * 2 / 3))
                        f = lambda t: math.log10(eb[t] / ev[t])
                        dd.append(f(st[-1]) - f(mid))
                    if dd:
                        moved.append((abs(float(np.median(dd))), act, cls, prob, v))
    moved.sort(reverse=True)
    n_big = sum(1 for m in moved if m[0] > 0.1)
    print(f"{n_big} of {len(moved)} (cell, variant) pairs moved more than 0.1 decades "
          f"in the last third.\nThe ten largest:\n")
    print(f"{'|move|':>7s}  {'cell':34s} {'variant':16s}")
    for m, act, cls, prob, v in moved[:10]:
        print(f"{m:7.3f}  {act+'/'+cls+'/'+prob:34s} {SHORT[v]:16s}")
    print(f"\nmedian move {np.median([m[0] for m in moved]):.3f} decades -- so the "
          "orderings are mostly settled,\nbut the cells listed above are not, and they "
          "include the largest effects in the study.")


def report_percell(D, floor):
    hdr("3. PER-CELL EFFECTS -- where the aggregate comes from")
    print("Each entry is the mean over 3 seeds of the paired log10 ratio, +/- 95% t(2).")
    print("'*' marks a cell whose CI excludes zero (an effect that clears seed noise).\n")
    for act in ACTS:
        print(f"-- {act} " + "-" * 74)
        print(f"{'cell':26s} " + " ".join(f"{SHORT[v]:>17s}" for v in TREATED))
        for k in cells_of(D, act=act):
            line = f"{k[1]+'/'+k[2]:26s} "
            for v in TREATED:
                m, ci, n = cell_stat(D[k].get(v, np.array([])))
                if not np.isfinite(m):
                    line += f"{'--':>18s}"
                    continue
                star = "*" if np.isfinite(ci) and abs(m) > ci else " "
                line += f" {m:+7.3f}+/-{ci:5.3f}{star}"
            print(line)
        print()


def report_centering(D):
    """The mechanistic claim: rms_nocenter -> rms_center is one flag, one code path."""
    hdr("4. THE CENTERING CONTRAST -- rms_nocenter vs rms_center, paired per cell")
    print("Same code path, one flag. The difference isolates centering from scaling.")
    print("Reported as (rms_center effect) - (rms_nocenter effect), in decades.\n")
    print(f"{'activation':12s} {'mean':>8s} {'95% CI':>17s} {'x-fold':>8s} "
          f"{'median':>8s} {'cells + / -':>12s}")
    for act in ACTS:
        ks = cells_of(D, act=act)
        diffs = []
        for k in ks:
            a, b = D[k].get("rms_nocenter"), D[k].get("rms_center")
            if a is None or b is None or a.size == 0 or b.size == 0:
                continue
            n = min(a.size, b.size)
            diffs.append(float(np.mean(b[:n] - a[:n])))
        d = np.array(diffs, float)
        ci = T95_8 * d.std(ddof=1) / math.sqrt(d.size)
        print(f"{act:12s} {d.mean():+8.3f} [{d.mean()-ci:+7.3f},{d.mean()+ci:+7.3f}] "
              f"{10**d.mean():8.2f} {np.median(d):+8.3f} {int((d>0).sum()):5d} / {int((d<0).sum()):<5d}")
    print("\nAlso the frozen-vs-running statistics contrast (rms_center vs BatchNorm),")
    print("which is NOT a clean single variable -- BatchNorm additionally applies an eps floor.\n")
    print(f"{'activation':12s} {'mean':>8s} {'95% CI':>17s}")
    for act in ACTS:
        diffs = []
        for k in cells_of(D, act=act):
            a, b = D[k].get("rms_center"), D[k].get("batchnorm_noaffine")
            if a is None or b is None or a.size == 0 or b.size == 0:
                continue
            n = min(a.size, b.size)
            diffs.append(float(np.mean(b[:n] - a[:n])))
        d = np.array(diffs, float)
        ci = T95_8 * d.std(ddof=1) / math.sqrt(d.size)
        print(f"{act:12s} {d.mean():+8.3f} [{d.mean()-ci:+7.3f},{d.mean()+ci:+7.3f}]")


def report_pinn_param(rows):
    """The inverse problems, scored on the quantity they exist to recover."""
    hdr("5. INVERSE PROBLEMS -- scored on PARAMETER RECOVERY, not field error")
    idx = index(rows)
    keys = sorted({(a, p) for (a, c, p, s) in idx if c == "pinn_inverse"})
    print("rel err = |p_hat - p_true| / |p_true|, median over seeds. "
          "'decades' = -log10(rel err).")
    print("Spearman is between field error and parameter error over that problem's "
          "30 runs --\nwhere it is low, ranking cells by field error does not rank "
          "parameter recovery.\n")
    print(f"{'act':6s} {'problem':13s} {'p_true':>8s} {'variant':16s} "
          f"{'p_hat (med)':>13s} {'rel err':>10s} {'decades':>8s} {'field err':>11s}")
    for act, prob in keys:
        fe_all, pe_all = [], []
        for v in VARIANTS:
            rs = [idx[(act, "pinn_inverse", prob, s)][v]
                  for s in SEEDS if v in idx.get((act, "pinn_inverse", prob, s), {})]
            if not rs:
                continue
            pt = rs[0]["param_true"]
            ph = float(np.median([r["param_final"] for r in rs]))
            fe = float(np.median([r["final_err"] for r in rs]))
            rel = abs(ph - pt) / abs(pt)
            dec = -math.log10(rel) if rel > 0 else 99.0
            print(f"{act:6s} {prob:13s} {pt:8.4g} {SHORT[v]:16s} {ph:13.8g} "
                  f"{rel:10.3e} {dec:8.2f} {fe:11.3e}")
            for r in rs:
                fe_all.append(r["final_err"])
                pe_all.append(abs(r["param_final"] - r["param_true"]) / abs(r["param_true"]))
        if len(fe_all) > 3:
            rho = stats.spearmanr(fe_all, pe_all).statistic
            print(f"{'':6s} {'':13s} {'':8s} {'-> spearman(field, param) = ':>16s} {rho:+.3f}\n")


def report_param_fragility(rows):
    """Two ways a parameter-recovery number can look better than it is."""
    hdr("5b. FRAGILITY OF THE PARAMETER-RECOVERY NUMBERS")
    print("(a) CROSSING. p_hat approaches p_true from below and overshoots. |p_hat - p|")
    print("    then dips through a spurious minimum, so where the 2000-step budget lands")
    print("    relative to the crossing sets the reported accuracy, not the method.")
    print("(b) SEED DISPERSION. With 3 seeds the median IS one seed. If the three")
    print("    endpoints span an order of magnitude, that median is a coin flip.\n")
    idx = index(rows)
    print(f"{'act':6s} {'problem':13s} {'variant':16s} {'seed endpoints (rel err)':>34s} "
          f"{'spread':>8s} {'median':>10s} {'dec':>5s} {'cross':>7s}")
    for act in ACTS:
        for prob in R.PROBLEMS["pinn_inverse"]:
            for v in VARIANTS:
                ends, ncross = [], 0
                for s in SEEDS:
                    r = idx.get((act, "pinn_inverse", prob, s), {}).get(v)
                    if r is None:
                        continue
                    pt = r["param_true"]
                    tr = np.array([p for _, p in r["param_traj"]], float)
                    d = tr - pt
                    ncross += int(((d[:-1] * d[1:]) < 0).any())
                    ends.append(abs(d[-1]) / abs(pt))
                if not ends:
                    continue
                e = np.array(ends)
                mid = float(np.median(e))
                flag = "!!" if (e.max() / max(e.min(), 1e-300) > 5 or ncross) else "  "
                print(f"{act:6s} {prob:13s} {SHORT[v]:16s} "
                      f"{'  '.join(f'{x:9.2e}' for x in e):>34s} "
                      f"{e.max()/max(e.min(),1e-300):8.1f} {mid:10.2e} "
                      f"{-math.log10(mid) if mid>0 else 99:5.2f} {ncross:d}/{len(e)}{flag}")
    print("\n'!!' = endpoints span >5x across seeds, or at least one seed crossed p_true.")
    print("On those rows the reported 'correct decimals' is not a property of the variant.")


def report_disagreement(D, rows):
    """Does the field-error verdict agree with the parameter-recovery verdict?"""
    hdr("6. WHERE THE OLD SCORING DISAGREES WITH PARAMETER RECOVERY")
    Dp = deltas(rows, key="_param_rel")
    print(f"{'act':6s} {'problem':13s} {'variant':16s} {'field (dec)':>12s} "
          f"{'param (dec)':>12s} {'agree?':>8s}")
    for act in ACTS:
        for prob in R.PROBLEMS["pinn_inverse"]:
            k = (act, "pinn_inverse", prob)
            if k not in D or k not in Dp:
                continue
            for v in TREATED:
                mf, _, _ = cell_stat(D[k].get(v, np.array([])))
                mp, _, _ = cell_stat(Dp[k].get(v, np.array([])))
                if not (np.isfinite(mf) and np.isfinite(mp)):
                    continue
                agree = "yes" if (mf > 0) == (mp > 0) else "NO"
                print(f"{act:6s} {prob:13s} {SHORT[v]:16s} {mf:+12.3f} {mp:+12.3f} {agree:>8s}")


# ============================ figures ============================

def _legend_above(ax, ncol, **kw):
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=ncol,
              borderaxespad=0, frameon=False, fontsize=8, **kw)


def fig_forest(D):
    """THE figure. Effect size per variant, per activation, every cell shown."""
    fig, axes = plt.subplots(1, len(ACTS), figsize=(13.5, 4.4), sharex=True,
                             sharey=True)
    lim = 0.0
    for c in D:
        for v in TREATED:
            d = D[c].get(v)
            if d is not None and d.size:
                lim = max(lim, abs(float(d.mean())))
    lim = max(0.6, lim * 1.15)

    handles = {}
    for ai, act in enumerate(ACTS):
        ax = axes[ai]
        ks = cells_of(D, act=act)
        ax.axvline(0, color="black", lw=1.4, zorder=1)
        ax.axvspan(-lim, 0, color="#d62728", alpha=0.045, zorder=0)
        ax.axvspan(0, lim, color="#2ca02c", alpha=0.045, zorder=0)
        for vi, v in enumerate(TREATED):
            y0 = len(TREATED) - 1 - vi
            xs, cs = [], []
            for k in ks:
                m, _, _ = cell_stat(D[k].get(v, np.array([])))
                if np.isfinite(m):
                    xs.append(m)
                    cs.append(k[1])
            xs = np.array(xs)
            jit = np.linspace(-0.20, 0.20, len(xs)) if len(xs) > 1 else [0.0]
            for x, cl, j in zip(xs, cs, jit):
                h = ax.scatter([np.clip(x, -lim * 0.985, lim * 0.985)], [y0 + j], s=44,
                               marker=CLS_MARK[cl], facecolor=CLS_COLOR[cl],
                               edgecolor="white", linewidth=0.6, zorder=3)
                handles.setdefault(cl, h)
            m = xs.mean()
            ci = T95_8 * xs.std(ddof=1) / math.sqrt(xs.size)
            ax.plot([m - ci, m + ci], [y0 - 0.34] * 2, color="black", lw=3.2, zorder=4)
            ax.plot([m], [y0 - 0.34], marker="D", ms=7, color="black", zorder=5)
            ax.text(lim * 0.97, y0 - 0.34, f"mean {m:+.2f} dec  ({10**m:.2f}x)",
                    ha="right", va="center", fontsize=8.5, weight="bold",
                    color="#146c2e" if m > 0 else "#8c1d1d")
        ax.set_yticks(range(len(TREATED)))
        ax.set_yticklabels([SHORT[v] for v in TREATED[::-1]])
        ax.set_ylim(-0.75, len(TREATED) - 0.4)
        ax.set_xlim(-lim, lim)
        ax.set_xlabel(r"paired effect  $\log_{10}$(baseline err / variant err)   [decades]")
        ax.set_title(f"{act}   (9 cells)", fontsize=12, weight="bold", pad=30)
        ax.grid(axis="x", alpha=0.25, lw=0.6)
        sec = ax.secondary_xaxis("top")
        tk = [t for t in [-2, -1, 0, 1, 2] if abs(t) <= lim]
        sec.set_xticks(tk)
        sec.set_xticklabels([f"{10**t:g}x" for t in tk], fontsize=7.5)
    fig.legend(handles.values(), handles.keys(), loc="upper center",
               bbox_to_anchor=(0.5, 1.045), ncol=3, frameon=False, fontsize=9,
               title="one marker = one cell (mean of 3 seeds)", title_fontsize=8.5)
    fig.suptitle("expD21: does a feature normalization beat doing nothing?\n"
                 "black diamond + bar = mean over the 9 cells with 95% CI. "
                 "Every CI crosses zero.",
                 fontsize=12.5, y=1.155)
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    p = FIG_DIR / "expD21_effect_forest.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {p}")


def fig_percell(D):
    """Every cell with its seed error bar, so noise-vs-signal is visible."""
    fig, axes = plt.subplots(len(ACTS), len(CLASSES),
                             figsize=(14.0, 4.4 * len(ACTS)), squeeze=False)
    lim = 0.0
    for c in D:
        for v in TREATED:
            m, ci, _ = cell_stat(D[c].get(v, np.array([])))
            if np.isfinite(m):
                lim = max(lim, abs(m) + (ci if np.isfinite(ci) else 0))
    lim = max(0.6, lim * 1.1)

    for ai, act in enumerate(ACTS):
        for ci_, cls in enumerate(CLASSES):
            ax = axes[ai][ci_]
            probs = R.PROBLEMS[cls]
            ax.axhline(0, color="black", lw=1.4)
            ax.axhspan(0, lim, color="#2ca02c", alpha=0.05)
            ax.axhspan(-lim, 0, color="#d62728", alpha=0.05)
            off = np.linspace(-0.26, 0.26, len(TREATED))
            for vi, v in enumerate(TREATED):
                xs, ys, es = [], [], []
                for pi, prob in enumerate(probs):
                    m, e, _ = cell_stat(D.get((act, cls, prob), {}).get(v, np.array([])))
                    if np.isfinite(m):
                        xs.append(pi + off[vi])
                        ys.append(m)
                        es.append(e if np.isfinite(e) else 0.0)
                ax.errorbar(xs, ys, yerr=es, fmt="o", ms=6, capsize=3, lw=1.4,
                            color=COLORS[v], label=SHORT[v] if (ai == 0 and ci_ == 0) else None)
            ax.set_xticks(range(len(probs)))
            ax.set_xticklabels(probs, fontsize=9)
            ax.set_xlim(-0.55, len(probs) - 0.45)
            ax.set_ylim(-lim, lim)
            ax.grid(axis="y", alpha=0.25, lw=0.6)
            ax.set_title(f"{act} / {cls}", fontsize=11, weight="bold", pad=6)
            if ci_ == 0:
                ax.set_ylabel("paired effect [decades]\n(+ = better than baseline)", fontsize=9)
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 0.965), ncol=4,
               frameon=False, fontsize=10)
    fig.suptitle("expD21: per-cell paired effect with 95% seed CI -- "
                 "an error bar crossing 0 is a cell where the variant did nothing measurable",
                 fontsize=12, y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.945))
    p = FIG_DIR / "expD21_percell.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {p}")


def fig_trajectory(rows):
    """Paired effect vs training step: is the verdict settled, or still moving?"""
    idx = index(rows)
    grid = [(a, c) for a in ACTS for c in CLASSES]
    fig, axes = plt.subplots(len(grid), 3, figsize=(14.0, 2.55 * len(grid)),
                             squeeze=False, sharex=True)
    for ri, (act, cls) in enumerate(grid):
        for ci_, prob in enumerate(R.PROBLEMS[cls]):
            ax = axes[ri][ci_]
            ax.axhline(0, color="black", lw=1.2)
            for v in TREATED:
                per_seed = []
                for s in SEEDS:
                    byv = idx.get((act, cls, prob, s), {})
                    if "baseline" not in byv or v not in byv:
                        continue
                    eb = dict(byv["baseline"]["evals"])
                    ev = dict(byv[v]["evals"])
                    steps = sorted(set(eb) & set(ev))
                    per_seed.append((steps, [math.log10(eb[t] / ev[t])
                                             if eb[t] > 0 and ev[t] > 0 else np.nan
                                             for t in steps]))
                if not per_seed:
                    continue
                steps = per_seed[0][0]
                M = np.array([p[1] for p in per_seed if p[0] == steps], float)
                if M.size == 0:
                    continue
                ax.plot(steps, np.nanmedian(M, 0), color=COLORS[v], lw=1.6,
                        label=SHORT[v] if (ri == 0 and ci_ == 0) else None)
                if M.shape[0] > 1:
                    ax.fill_between(steps, np.nanmin(M, 0), np.nanmax(M, 0),
                                    color=COLORS[v], alpha=0.14, lw=0)
            ax.set_title(f"{act} / {cls} / {prob}", fontsize=10, weight="bold", pad=5)
            ax.grid(alpha=0.25, lw=0.6)
            ax.set_ylim(-2.15, 2.15)
            ax.set_yticks([-2, -1, 0, 1, 2])
            if ci_ == 0:
                ax.set_ylabel("effect [decades]", fontsize=8.5)
            if ri == len(grid) - 1:
                ax.set_xlabel("Adam step")
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 0.975), ncol=4,
               frameon=False, fontsize=10)
    fig.suptitle("expD21: paired effect vs training step "
                 "(median over 3 seeds, band = seed min-max); ALL PANELS SHARE THE Y-AXIS\n"
                 "flat at the right edge = the verdict is settled; "
                 "still moving = the 2000-step budget, not the variant, set the answer",
                 fontsize=12, y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.962))
    p = FIG_DIR / "expD21_trajectory.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {p}")


def fig_pinn_param(rows):
    """The inverse problems on their own metric, with the field error alongside."""
    idx = index(rows)
    probs = R.PROBLEMS["pinn_inverse"]
    fig, axes = plt.subplots(len(ACTS), len(probs),
                             figsize=(14.0, 4.2 * len(ACTS)), squeeze=False,
                             sharex=True)
    for ai, act in enumerate(ACTS):
        for pi, prob in enumerate(probs):
            ax = axes[ai][pi]
            pt = None
            for v in VARIANTS:
                curves = []
                for s in SEEDS:
                    r = idx.get((act, "pinn_inverse", prob, s), {}).get(v)
                    if r is None:
                        continue
                    pt = r["param_true"]
                    st = [t for t, _ in r["param_traj"]]
                    rel = [abs(p - pt) / abs(pt) for _, p in r["param_traj"]]
                    curves.append((st, rel))
                if not curves:
                    continue
                st = curves[0][0]
                M = np.array([c[1] for c in curves if c[0] == st], float)
                if M.size == 0:
                    continue
                ax.plot(st, np.median(M, 0), color=COLORS[v], lw=1.7,
                        label=SHORT[v] if (ai == 0 and pi == 0) else None)
                if M.shape[0] > 1:
                    ax.fill_between(st, M.min(0), M.max(0), color=COLORS[v],
                                    alpha=0.13, lw=0)
            for dec, lab in [(1e-1, "1 decade"), (1e-2, "2 decades"), (1e-3, "3 decades")]:
                ax.axhline(dec, color="gray", ls=":", lw=0.9)
                ax.text(20, dec, lab, fontsize=6.5, color="gray", va="bottom")
            ax.set_yscale("log")
            ax.set_ylim(1e-4, 3e0)
            ax.grid(alpha=0.22, lw=0.6)
            ax.set_title(f"{act} / {prob}  (p_true = {pt:g})", fontsize=11,
                         weight="bold", pad=6)
            if pi == 0:
                ax.set_ylabel(r"$|\hat p - p|/|p|$", fontsize=10)
            if ai == len(ACTS) - 1:
                ax.set_xlabel("Adam step")
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 0.955), ncol=5,
               frameon=False, fontsize=10)
    fig.suptitle("expD21: inverse-problem parameter recovery -- the quantity these cells exist to measure\n"
                 "(the ranking in the original analysis scored field error instead; "
                 "the two disagree on burgers and allencahn)",
                 fontsize=12, y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.935))
    p = FIG_DIR / "expD21_pinn_param.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {p}")


# ============================ main ============================

def main():
    rows = load_rows()
    for r in rows:
        if r["class"] == "pinn_inverse":
            r["_param_rel"] = abs(r["param_final"] - r["param_true"]) / abs(r["param_true"])
    print(f"loaded {len(rows)} runs from {DATA_DIR}")

    D = deltas(rows)
    print(f"{len(D)} cells x {len(TREATED)} treated variants x {len(SEEDS)} seeds")

    floor = report_noise_floor(D)
    report_headline(D)
    report_by_class(D)
    report_variance(D)
    report_settled(rows)
    report_percell(D, floor)
    report_centering(D)
    report_pinn_param(rows)
    report_param_fragility(rows)
    report_disagreement(D, rows)

    hdr("FIGURES")
    fig_forest(D)
    fig_percell(D)
    fig_trajectory(rows)
    fig_pinn_param(rows)


if __name__ == "__main__":
    main()
