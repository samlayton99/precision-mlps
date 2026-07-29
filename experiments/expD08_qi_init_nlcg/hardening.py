"""expD08 hardening: which mechanisms are load-bearing, and is the tether needed?

Part A  frozen-phi ablations. The iteration_4 core on the pure quadratic
        [Phi, 1] v = y (readout coordinates only, geometry frozen), one
        mechanism switched off at a time. Clean attribution: no tether, no
        nonquadratic drift.
Part B  network ablations. Full expD08 run_case (all params trainable,
        tether 1e6), one mechanism off at a time, 4 targets x N in {64, 256}.
        Control = the iteration_4 results (same code path, ablate=()).
Part C  tether ablation. The UNMODIFIED iteration_4 optimizer at T=1 and
        T=1e3; control T=1e6 = iteration_4. 6 targets x N in {64, 256}.

Run:  uv run python experiments/expD08_qi_init_nlcg/hardening.py [--part a|b|c|all]
Outputs: results/checkpoint_D_optimizers/expD08_qi_init_nlcg/hardening/*.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run  # noqa: E402  (expD08 run.py: core optimizer + run_case)

HARD_DIR = run.OUT_DIR.parent / "hardening"

# NOTE (iteration_5): the original parts B/D ran against the iteration_4
# core and measured no_trust/no_descent/no_beta_clamp as free-or-harmful;
# those mechanisms are now DELETED from the default core (run_case uses
# train_residual_cg_slim). The archived measurements live in
# hardening/{phi,net_ablate,tether,combo}.json. The lists below only carry
# the mechanisms that still exist in the slim core.
ABLATIONS = ["no_exhaust", "single_proj", "fresh_resid", "no_memory",
             "no_refresh"]
FNS_ABL = ["sine", "sine_8pi", "runge", "sine_mixture"]
NS_ABL = [64, 256]
TETHERS = [1.0, 1e3]
NO_REFRESH = 10 ** 9


def _variant_kwargs(variant):
    if variant == "control":
        return {"ablate": (), "refresh": run.REFRESH}
    if variant == "no_refresh":
        return {"ablate": (), "refresh": NO_REFRESH}
    return {"ablate": (variant,), "refresh": run.REFRESH}


# ------------------------- part A: frozen phi -------------------------

def phi_case(job):
    torch.set_num_threads(job["threads"])
    torch.set_default_dtype(torch.float64)
    fn_name, N, variant = job["fn"], job["N"], job["variant"]
    centers, gamma, _ = run.geometry(N, run.LAM)
    x = np.linspace(-1, 1, run.N_TRAIN)
    A = np.hstack([run.build_phi(x, gamma, centers, "tanh"),
                   np.ones((x.size, 1))])
    y = run.make_fn(run.TARGETS[fn_name])(x)
    A_t, y_t = torch.from_numpy(A), torch.from_numpy(y)
    n = x.size
    y_norm = float(np.linalg.norm(y))

    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    keep = s > run.RCOND * s[0]
    sol = Vt.T @ (np.where(keep, 1.0 / np.where(keep, s, 1.0), 0.0)
                  * (U.T @ y))
    floor = float(np.linalg.norm(A @ sol - y) / y_norm)

    def f_resid(v):
        with torch.no_grad():
            return (A_t.mv(v) - y_t).detach()

    def f_jvp(v, d):
        return A_t.mv(d)

    def f_vjp(v, seed):
        return A_t.t().mv(seed)

    def eval_rel(v):
        with torch.no_grad():
            return float(torch.linalg.norm(A_t.mv(v) - y_t)) / y_norm

    kw = _variant_kwargs(variant)
    t0 = time.time()
    losses, rels, _, _, stats = run.train_residual_cg_slim(
        torch.zeros(A.shape[1]), f_resid, f_jvp, f_vjp, n,
        iters=job["iters"], frames_at=set(), get_wbv=lambda th: {},
        eval_rel=eval_rel, refresh=kw["refresh"], ablate=kw["ablate"])
    return {"part": "phi", "variant": variant, "fn": fn_name, "N": N,
            "best_rel": float(np.nanmin(rels)) if rels else float("nan"),
            "floor": floor, "stats": stats,
            "wall_s": round(time.time() - t0, 1)}


# ---------------------- parts B/C: full network ----------------------

def net_case(job):
    kw = _variant_kwargs(job["variant"])
    r = run.run_case({"fn": job["fn"], "N": job["N"], "iters": job["iters"],
                      "threads": job["threads"], "save_snaps": False,
                      "tether": job.get("tether", run.TETHER),
                      "refresh": kw["refresh"], "ablate": kw["ablate"]})
    return {"part": job["part"], "variant": job["variant"],
            "fn": job["fn"], "N": job["N"],
            "tether": job.get("tether", run.TETHER),
            "best_rel": r["best_rel"], "floor": r["floor"],
            "stats": r["stats"], "wall_s": r["wall_s"]}


# ------------------------------- driver -------------------------------

def load_control():
    """(fn, N) -> best_rel from the iteration_4 run (the ablation control)."""
    rows = json.loads((run.OUT_DIR / "expD08_rows.json").read_text())
    return {(r["fn"], r["N"]): r["best_rel"] for r in rows}


def summarize(results, control):
    by_variant: dict[str, list] = {}
    for r in results:
        by_variant.setdefault(r["variant"], []).append(r)
    print(f"\n{'variant':<15s} {'median orders lost':>19s} {'worst cell':>34s}")
    for v, rs in sorted(by_variant.items()):
        deltas = []
        worst = None
        for r in rs:
            c = control.get((r["fn"], r["N"]))
            if c is None or not np.isfinite(r["best_rel"]):
                d = float("inf")
            else:
                d = np.log10(r["best_rel"] / c)
            deltas.append(d)
            if worst is None or d > worst[0]:
                worst = (d, r)
        med = float(np.median([d for d in deltas if np.isfinite(d)])
                    ) if any(np.isfinite(d) for d in deltas) else float("inf")
        w = worst[1]
        print(f"{v:<15s} {med:>+19.2f} "
              f"{w['fn']}@N{w['N']}: {w['best_rel']:.2e} ({worst[0]:+.1f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="all", choices=["a", "b", "c", "all"])
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--iters", type=int, default=run.ITERS)
    ap.add_argument("--phi-iters", type=int, default=1200)
    args = ap.parse_args()
    threads = max(1, (os.cpu_count() or 4) // args.workers)
    HARD_DIR.mkdir(parents=True, exist_ok=True)

    jobs = []
    if args.part in ("a", "all"):
        for v in ["control"] + ABLATIONS:
            for fn in ["sine", "sine_8pi"]:
                jobs.append({"part": "phi", "variant": v, "fn": fn, "N": 256,
                             "iters": args.phi_iters, "threads": threads})
    if args.part in ("b", "all"):
        for v in ABLATIONS:
            for fn in FNS_ABL:
                for N in NS_ABL:
                    jobs.append({"part": "net_ablate", "variant": v,
                                 "fn": fn, "N": N, "iters": args.iters,
                                 "threads": threads})
    if args.part in ("c", "all"):
        for T in TETHERS:
            for fn in run.TARGETS:
                for N in NS_ABL:
                    jobs.append({"part": "tether", "variant": f"T={T:g}",
                                 "fn": fn, "N": N, "tether": T,
                                 "iters": args.iters, "threads": threads})

    print(f"expD08 hardening | {len(jobs)} runs | part={args.part}")
    results, t0 = [], time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(phi_case if j["part"] == "phi" else net_case, j): j
                for j in jobs}
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            results.append(r)
            s = r["stats"]
            print(f"  [{i}/{len(jobs)}] {r['part']:<10s} "
                  f"{r['variant']:<14s} {r['fn']:13s} N={r['N']:<4d} "
                  f"best_rel={r['best_rel']:.3e} floor={r['floor']:.1e} "
                  f"exh={s.get('exhaust', 0)} rst={s.get('full_reset', 0)} "
                  f"fb={s.get('alpha_fallback', 0)} rev={s.get('revert', 0)} "
                  f"({r['wall_s']}s)", flush=True)
    print(f"all runs done in {time.time() - t0:.1f}s")

    for part in ("phi", "net_ablate", "tether", "combo"):
        rs = [r for r in results if r["part"] == part]
        if rs:
            (HARD_DIR / f"{part}.json").write_text(json.dumps(rs))
            print(f"saved {HARD_DIR / f'{part}.json'} ({len(rs)} rows)")

    # summaries: phi vs its own control; net/tether vs iteration_4
    phi_rs = [r for r in results if r["part"] == "phi"
              and r["variant"] != "control"]
    if phi_rs:
        phi_ctrl = {(r["fn"], r["N"]): r["best_rel"] for r in results
                    if r["part"] == "phi" and r["variant"] == "control"}
        print("\n== part A: frozen-phi ablations (vs phi control) ==")
        summarize(phi_rs, phi_ctrl)
    ctrl = load_control()
    net_rs = [r for r in results if r["part"] == "net_ablate"]
    if net_rs:
        print("\n== part B: network ablations (vs iteration_4) ==")
        summarize(net_rs, ctrl)
    tet_rs = [r for r in results if r["part"] == "tether"]
    if tet_rs:
        print("\n== part C: tether (vs current iteration control) ==")
        summarize(tet_rs, ctrl)


if __name__ == "__main__":
    main()
