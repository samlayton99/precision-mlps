"""expD06 -- derivative-initialized readout under Adam: watching the coefficients move.

Init: QI geometry (uniform centers, gamma = lambda*/h, lambda* = 0.25) in affine
coordinates; readout = the derivative law, scaled by an lstsq-calibrated scalar,
halo nodes = 0. Train EVERYTHING (both layers) with full-batch Adam (fp64), linear
LR warmup (100 steps) then constant 1e-3, 1000 steps.

Scale calibration per (function, width): solve the SVD lstsq readout c on the frozen
geometry, keep interior coefficients c_int (|center| <= 1), build p = f'(centers_int),
find  a = argmin_a || a * c_int - p ||_1  (the lstsq solution is the oracle for ONE
scalar). Interior init = p/a; halo = 0.

Five init variants ("seeds"):
  seed1_calibrated   v = p/a
  seed2_over3x       v = 3 p/a
  seed3_under3x      v = p/(3a)
  seed4_noisy        v = p/a + N(0, (std(p/a)/8)^2) on the WHOLE vector (halo too)
  seed5_random       readout at standard random init (geometry unchanged)

Recorded: train MSE + eval rel-L2 EVERY step; readout snapshots on the log-paced
frame schedule (every 2 steps x20, every 4 x10, every 8 x10, doubling..., 67 frames).
Outputs per seed: gifs/coeffs_N{128,256,512}.gif (6-function subplot per frame) and
one 2x3 loss-curve figure (per-function panels, one line per width).

Run:  uv run python experiments/expD06_derivative_readout_init/run.py   (--smoke)
"""
from __future__ import annotations

import argparse
import io
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import sympy as sp
import torch

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT / "experiments" / "expA06_readout_structure"))
from app import build_phi, geometry, make_fn   # noqa: E402

OUT_DIR = REPO_ROOT / "results" / "checkpoint_D_optimizers" / "expD06_derivative_readout_init"
LAM = 0.25
WIDTHS = [128, 256, 512]
STEPS, WARMUP, LR = 1000, 100, 1e-3
N_TRAIN, N_EVAL = 2003, 4001
TARGETS = {  # the canonical 6-category family
    "sine": "sin(2*pi*x)", "sine_8pi": "sin(8*pi*x)", "runge": "1/(1+25*x**2)",
    "sine_mixture": "sin(2*pi*x)+0.5*sin(6*pi*x)+0.25*sin(14*pi*x)",
    "exp": "exp(x)", "abs_cubed": "Abs(x)**3",
}
SEEDS = ["seed1_calibrated", "seed2_over3x", "seed3_under3x", "seed4_noisy", "seed5_random",
         # magnitude-vs-shape controls (Sam's normalized-Gaussian proposal):
         "seed6_gauss_calibmag",   # Gaussian direction, inner L2 norm = ||p/a|| (right magnitude)
         "seed7_gauss_sqrtN",      # Gaussian direction, inner L2 norm = 1/sqrt(N) (bare rule)
         # 3-column parameter movies (Sam): 4 targets, cols = (w | b | v) vs CURRENT centers
         "seed8_noisy_long",       # (a) ideal geometry, noisy p/a readout, 20k steps: does noise anneal?
         "seed9_zero_readout",     # (b) ideal geometry, readout = 0: does it stay smooth?
         "seed10_standard_dl"]     # (c) standard PyTorch init for BOTH layers (scattered centers)

# the 3-column movie seeds: per-seed step count and the 4-target row set
SEEDS3 = {"seed8_noisy_long": 20000, "seed9_zero_readout": 1000, "seed10_standard_dl": 40000}
TARGETS3 = ["sine", "sine_8pi", "runge", "exp"]
_X = sp.Symbol("x", real=True)   # real=True so Abs(x) differentiates to sign-based form


def frame_steps(total=STEPS):
    """Log-paced schedule: stride 2 x20, 4 x10, then doubling x10 each, + step 0 + final."""
    steps, s = [0], 0
    stride, count = 2, 20
    while s < total:
        for _ in range(count):
            s += stride
            if s >= total:
                break
            steps.append(s)
        stride, count = stride * 2, 10
    steps.append(total)
    return sorted(set(steps))


def fprime(text):
    e = sp.sympify(text, locals={"x": _X})
    return sp.lambdify(_X, sp.diff(e, _X), modules=["numpy"])


def calibrate(fn_text, N):
    """lstsq oracle solve -> interior scale a = argmin_a ||a*c_int - p||_1."""
    from scipy.linalg import lstsq as sp_lstsq
    from scipy.optimize import minimize_scalar
    f_vec = make_fn(fn_text)
    centers, gamma, _ = geometry(N, LAM)
    x = np.linspace(-1, 1, N_TRAIN)
    A = np.hstack([build_phi(x, gamma, centers, "tanh"), np.ones((x.size, 1))])
    sol, *_ = sp_lstsq(A, f_vec(x), lapack_driver="gelsd")
    c = sol[:-1]
    interior = np.abs(centers) <= 1.0
    p = fprime(fn_text)(centers[interior])
    a0 = float(np.dot(c[interior], p) / np.dot(c[interior], c[interior]))  # L2 start
    res = minimize_scalar(lambda a: np.abs(a * c[interior] - p).sum(),
                          bracket=(a0 / 10, a0, a0 * 10) if a0 > 0 else None,
                          method="brent")
    a = float(res.x)
    return centers, gamma, interior, p, a


def init_geometry(seed_name, centers, gamma, rng):
    """(w0, b0) per seed. Default = the QI geometry in affine coordinates; seed10
    uses PyTorch nn.Linear(1, W) defaults (kaiming uniform, fan_in=1 -> U(+-1))."""
    if seed_name == "seed10_standard_dl":
        W = centers.size
        return rng.uniform(-1.0, 1.0, W), rng.uniform(-1.0, 1.0, W)
    return np.full(centers.size, gamma), -gamma * centers


def init_readout(seed_name, centers, interior, p, a, rng):
    v = np.zeros(centers.size)
    if seed_name == "seed9_zero_readout":
        return v
    if seed_name in ("seed5_random", "seed10_standard_dl"):
        bound = 1.0 / math.sqrt(centers.size)          # nn.Linear default (kaiming uniform)
        return rng.uniform(-bound, bound, centers.size)
    if seed_name.startswith("seed6") or seed_name.startswith("seed7"):
        g = rng.standard_normal(interior.sum())
        g /= np.linalg.norm(g)
        target = (np.linalg.norm(p / a) if seed_name.startswith("seed6")
                  else 1.0 / math.sqrt(interior.sum()))
        v[interior] = g * target                       # random direction, chosen magnitude
        return v
    v[interior] = p / a
    if seed_name == "seed2_over3x":
        v *= 3.0
    elif seed_name == "seed3_under3x":
        v /= 3.0
    elif seed_name in ("seed4_noisy", "seed8_noisy_long"):
        v = v + rng.normal(0.0, np.std(v[interior] if interior.any() else v) / 8.0, v.size)
    return v


def run_case(job):
    torch.set_num_threads(job["threads"])
    torch.set_default_dtype(torch.float64)
    fn_name, N, seed_name = job["fn"], job["N"], job["seed"]
    fn_text = TARGETS[fn_name]
    f_vec = make_fn(fn_text)

    steps_total = job.get("steps", STEPS)
    centers, gamma, interior, p, a = calibrate(fn_text, N)
    rng = np.random.default_rng(0)
    w0, b0 = init_geometry(seed_name, centers, gamma, rng)
    v0 = init_readout(seed_name, centers, interior, p, a, rng)

    # affine-coordinate tanh net, everything trainable
    w = torch.tensor(w0).requires_grad_(True)
    b = torch.tensor(b0).requires_grad_(True)
    v = torch.tensor(v0).requires_grad_(True)
    c0 = torch.zeros(1, requires_grad=True)

    xt = torch.tensor(np.linspace(-1, 1, N_TRAIN)).unsqueeze(1)
    yt = torch.tensor(f_vec(np.linspace(-1, 1, N_TRAIN)))
    xe = torch.tensor(np.linspace(-1, 1, N_EVAL)).unsqueeze(1)
    ye = torch.tensor(f_vec(np.linspace(-1, 1, N_EVAL)))
    ye_norm = float(torch.linalg.norm(ye))

    def model(x):
        return torch.tanh(x * w + b) @ v + c0

    opt = torch.optim.Adam([w, b, v, c0], lr=LR)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / WARMUP))          # linear warmup -> constant

    frames_at = set(frame_steps(steps_total))
    snaps, losses, rels = {}, [], []

    def snap(step):
        if step in frames_at:
            # v plus the current w, b so frames can place each coefficient at its
            # CURRENT effective center c_k(t) = -b_k/w_k (the geometry trains too)
            snaps[step] = {"v": v.detach().numpy().copy(),
                           "w": w.detach().numpy().copy(),
                           "b": b.detach().numpy().copy()}

    def ev():
        with torch.no_grad():
            r = model(xe) - ye
            return float(torch.linalg.norm(r)) / ye_norm

    snap(0)
    for step in range(1, steps_total + 1):
        opt.zero_grad(set_to_none=True)
        loss = torch.mean((model(xt) - yt) ** 2)
        loss.backward()
        opt.step()
        sched.step()
        losses.append(float(loss.detach()))
        rels.append(ev())
        snap(step)

    # persist per-frame parameter snapshots for the 3D explorer (app3d.py)
    snap_dir = OUT_DIR / "snaps"
    snap_dir.mkdir(parents=True, exist_ok=True)
    ssteps = sorted(snaps)
    np.savez_compressed(
        snap_dir / f"{seed_name}--{fn_name}--N{N}.npz",
        steps=np.array(ssteps, dtype=np.int64),
        w=np.stack([snaps[s]["w"] for s in ssteps]),
        b=np.stack([snaps[s]["b"] for s in ssteps]),
        v=np.stack([snaps[s]["v"] for s in ssteps]),
        losses=np.array(losses), rels=np.array(rels), a=a)

    return {"fn": fn_name, "N": N, "seed": seed_name, "a": a,
            "centers": centers.tolist(), "interior": interior.tolist(),
            "losses": losses, "rels": rels,
            "snaps": {str(k): {kk: vv.tolist() for kk, vv in s.items()}
                      for k, s in snaps.items()}}


# ----------------------------- rendering -----------------------------

FN_ORDER = list(TARGETS)


def render_gif(rows, seed_name, N, path, fps=8):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    def frame_cv(r, s):
        """(current centers, coefficients) for one frame; legacy rows stored only
        v as a flat list -> fall back to the initial centers."""
        sn = r["snaps"][str(s)]
        if isinstance(sn, dict):
            w_, b_, v_ = np.array(sn["w"]), np.array(sn["b"]), np.array(sn["v"])
            return -b_ / w_, v_
        return np.array(r["centers"]), np.array(sn)

    cases = {r["fn"]: r for r in rows if r["seed"] == seed_name and r["N"] == N}
    steps = sorted(int(s) for s in next(iter(cases.values()))["snaps"])
    lims = {}
    for fn, r in cases.items():
        allv = np.array([frame_cv(r, s)[1] for s in steps])
        lo, hi = np.percentile(allv, 0.5), np.percentile(allv, 99.5)
        pad = 0.15 * max(hi - lo, 1e-9)
        lims[fn] = (lo - pad, hi + pad)

    images = []
    for s in steps:
        fig, axes = plt.subplots(2, 3, figsize=(10.5, 5.6))
        for ax, fn in zip(axes.flat, FN_ORDER):
            r = cases[fn]
            c, vv = frame_cv(r, s)
            order = np.argsort(c)
            ax.axvspan(-1, 1, color="#1f4e8c", alpha=0.05)
            ax.plot(c[order], vv[order], color="#1f4e8c", lw=0.9)
            ax.set_ylim(*lims[fn])
            ax.set_title(fn, fontsize=9)
            ax.tick_params(labelsize=7)
        fig.suptitle(f"{seed_name}   N={N}   step {s}/{STEPS}", fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=80)
        plt.close(fig)
        buf.seek(0)
        images.append(Image.open(buf).convert("P", palette=Image.ADAPTIVE))
    path.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(path, save_all=True, append_images=images[1:],
                   duration=int(1000 / fps), loop=0)
    print(f"  saved {path} ({len(images)} frames, {path.stat().st_size/1e6:.1f} MB)")


def render_gif3(rows, seed_name, N, path, fps=8, scatter=False):
    """3-column parameter movie: rows = TARGETS3, cols = (w_k | b_k | v_k), all plotted
    against the CURRENT effective centers c_k = -b_k/w_k. Fixed window [-2,2], [-1,1]
    shaded, off-screen counter, and per-column theory overlays: w = gamma, b = -gamma*c,
    v = (h/2) f'(c).

    scatter=True: draw each neuron as an unconnected dot (no lines), colored by a FIXED
    per-neuron red->blue gradient assigned from the ORIGINAL (step-0) center ordering --
    so a neuron keeps its color as it moves and you can track where it migrates."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from PIL import Image

    XW = 2.0                                   # x-window half-width
    cases = {r["fn"]: r for r in rows if r["seed"] == seed_name and r["N"] == N}
    fns = [f for f in TARGETS3 if f in cases]
    steps = sorted(int(s) for s in next(iter(cases.values()))["snaps"])
    total = steps[-1]
    h = 2.0 / N
    gamma = LAM / h
    cgrid = np.linspace(-1, 1, 401)
    overlays = {fn: (h / 2.0) * fprime(TARGETS[fn])(cgrid) for fn in fns}

    # fixed per-neuron color: rank each neuron by its ORIGINAL center, map red->blue.
    rbcmap = LinearSegmentedColormap.from_list("redblue", ["red", "blue"])
    ncolors = {}
    for fn in fns:
        sn0 = cases[fn]["snaps"][str(steps[0])]
        w0, b0 = np.array(sn0["w"]), np.array(sn0["b"])
        c0 = -b0 / np.where(np.abs(w0) > 1e-12, w0, 1e-12)
        ranks = np.argsort(np.argsort(c0))     # 0..N-1 rank of each neuron by center
        ncolors[fn] = rbcmap(ranks / max(len(c0) - 1, 1))

    # fixed y-limits per (fn, column) across all frames, overlay included.
    # Scale is set ONLY by neurons whose CURRENT center is in ~[-1,1] (the
    # interior) -- neurons that wander off simply leave the screen.
    lims = {}
    for fn in fns:
        r = cases[fn]
        vals = {"w": [], "b": [], "v": []}
        for s in steps:
            sn = r["snaps"][str(s)]
            w_, b_, v_ = (np.array(sn[k]) for k in ("w", "b", "v"))
            c = -b_ / np.where(np.abs(w_) > 1e-12, w_, 1e-12)
            m = np.abs(c) <= 1.1
            if not m.any():
                m = np.abs(c) <= XW
            for k, arr in (("w", w_), ("b", b_), ("v", v_)):
                vals[k].append(arr[m])
        ref = {"w": np.array([gamma]), "b": np.array([-gamma, gamma]), "v": overlays[fn]}
        for j, k in enumerate(("w", "b", "v")):
            allv = np.concatenate(vals[k])
            lo = min(np.percentile(allv, 0.5), ref[k].min())
            hi = max(np.percentile(allv, 99.5), ref[k].max())
            pad = 0.10 * max(hi - lo, 1e-9)
            lims[(fn, j)] = (lo - pad, hi + pad)

    images = []
    for s in steps:
        fig, axes = plt.subplots(len(fns), 3, figsize=(12.5, 2.5 * len(fns)),
                                 squeeze=False)
        for i, fn in enumerate(fns):
            sn = cases[fn]["snaps"][str(s)]
            w_, b_, v_ = (np.array(sn[k]) for k in ("w", "b", "v"))
            c = -b_ / np.where(np.abs(w_) > 1e-12, w_, 1e-12)
            if scatter:                        # keep neuron order so fixed colors align
                cols = ncolors[fn]
            else:                              # legacy connected line: sort by center
                order = np.argsort(c)
                c, w_, b_, v_ = c[order], w_[order], b_[order], v_[order]
            vis = np.abs(c) <= XW
            for j, (y, ttl) in enumerate([(w_, "first-layer w"), (b_, "bias b"),
                                          (v_, "readout v")]):
                ax = axes[i, j]
                ax.axvspan(-1, 1, color="#1f4e8c", alpha=0.05)
                if j == 0:
                    ax.axhline(gamma, color="#2ca02c", lw=1.0, ls="--", alpha=0.8)
                elif j == 1:
                    ax.plot(cgrid, -gamma * cgrid, color="#2ca02c", lw=1.0, ls="--",
                            alpha=0.8)
                else:
                    ax.plot(cgrid, overlays[fn], color="#2ca02c", lw=1.0, ls="--",
                            alpha=0.8)
                if scatter:
                    ax.scatter(c[vis], y[vis], c=cols[vis], s=9, edgecolors="none",
                               zorder=3)
                else:
                    ax.plot(c[vis], y[vis], color="#1f4e8c", lw=0.8, marker=".", ms=1.5)
                ax.set_xlim(-XW, XW)
                ax.set_ylim(*lims[(fn, j)])
                # plain tick labels: never offset notation ("+6.4e1" header
                # with 0.0000 ticks reads as w ~ 0 when w is really ~ gamma)
                try:
                    ax.ticklabel_format(useOffset=False, style="plain",
                                        axis="y")
                except (AttributeError, ValueError):
                    pass
                ax.tick_params(labelsize=6)
                if i == 0:
                    ax.set_title(ttl + "  (green = theory)", fontsize=8)
                if j == 0:
                    ax.set_ylabel(fn, fontsize=8)
            n_off = int((~vis).sum())
            if n_off:
                axes[i, 2].text(0.98, 0.05, f"{n_off} off-screen",
                                transform=axes[i, 2].transAxes, ha="right",
                                fontsize=6, color="#a33")
        fig.suptitle(f"{seed_name}   N={N}   step {s}/{total}   (x = current centers "
                     r"$-b_k/w_k$)", fontsize=10)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=80)
        plt.close(fig)
        buf.seek(0)
        images.append(Image.open(buf).convert("P", palette=Image.ADAPTIVE))
    path.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(path, save_all=True, append_images=images[1:],
                   duration=int(1000 / fps), loop=0)
    print(f"  saved {path} ({len(images)} frames, {path.stat().st_size/1e6:.1f} MB)")


def render_loss_figure(rows, seed_name, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    colors = {128: "tab:blue", 256: "tab:orange", 512: "tab:green"}
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, fn in zip(axes.flat, FN_ORDER):
        for N in WIDTHS:
            rs = [r for r in rows if r["seed"] == seed_name and r["N"] == N and r["fn"] == fn]
            if not rs:
                continue
            r = rs[0]
            steps = np.arange(1, len(r["losses"]) + 1)
            ax.semilogy(steps, r["losses"], color=colors[N], lw=1.2)
            ax.semilogy(steps, r["rels"], color=colors[N], lw=1.0, ls="--", alpha=0.7)
        ax.set_title(fn, fontsize=10)
        ax.set_xlabel("step"); ax.grid(True, which="both", alpha=0.25)
    axes[0, 0].set_ylabel("train MSE (solid) / eval rel L2 (dashed)")
    handles = [Line2D([0], [0], color=colors[N], lw=1.5, label=f"N={N}") for N in WIDTHS]
    handles += [Line2D([0], [0], color="gray", lw=1.2, label="train MSE"),
                Line2D([0], [0], color="gray", lw=1.0, ls="--", label="eval rel L2")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.0),
               ncol=5, frameon=False)
    fig.suptitle(f"expD06 loss curves -- {seed_name}  (every step recorded)", y=1.04)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


# ----------------------------- main -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seeds", default=None, help="comma list; run only these variants")
    args = ap.parse_args()

    widths, seeds, fns = WIDTHS, SEEDS, list(TARGETS)
    global STEPS
    if args.seeds:
        seeds = args.seeds.split(",")
    if args.smoke:
        widths, seeds = [128], ["seed1_calibrated", "seed5_random"]

    threads = max(1, (os.cpu_count() or 4) // args.workers)
    jobs = [{"fn": f, "N": n, "seed": s, "threads": threads,
             "steps": SEEDS3.get(s, STEPS)}
            for s in seeds for n in widths
            for f in (TARGETS3 if s in SEEDS3 else fns)]
    print(f"expD06 | {len(jobs)} runs | widths={widths} | seeds={seeds} | "
          f"steps per seed: " + ", ".join(f"{s}={SEEDS3.get(s, STEPS)}" for s in seeds))

    rows, t0 = [], time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(run_case, j) for j in jobs]
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result(); rows.append(r)
            print(f"  [{i}/{len(jobs)}] {r['seed']:18s} N={r['N']:<4d} {r['fn']:13s} "
                  f"a={r['a']:.3e} final_mse={r['losses'][-1]:.3e} rel={r['rels'][-1]:.3e}",
                  flush=True)
    print(f"training done in {time.time()-t0:.1f}s")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    slim = [{k: r[k] for k in ("fn", "N", "seed", "a", "losses", "rels")} for r in rows]
    rows_path = OUT_DIR / "expD06_rows.json"
    if rows_path.exists():          # merge with prior seeds (partial reruns via --seeds)
        old = [r for r in json.loads(rows_path.read_text())
               if r["seed"] not in {s["seed"] for s in slim}]
        slim = old + slim
    rows_path.write_text(json.dumps(slim))

    for seed_name in seeds:
        sdir = OUT_DIR / seed_name
        render_loss_figure(rows, seed_name, sdir / f"loss_curves_{seed_name}.png")
        for N in widths:
            if seed_name in SEEDS3:
                render_gif3(rows, seed_name, N, sdir / "gifs" / f"params_N{N}.gif",
                            scatter=(seed_name == "seed10_standard_dl"))
            else:
                render_gif(rows, seed_name, N, sdir / "gifs" / f"coeffs_N{N}.gif")
    print(f"done; outputs under {OUT_DIR}")


if __name__ == "__main__":
    main()
