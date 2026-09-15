"""expD04 corrected rerun, reduced scope, cheapest probe first.

Every probe writes its CSV and figures into its OWN subfolder under
results/checkpoint_D_optimizers/expD04_varpro/varpro_corrected/:
    01_lsmr        VarPro + LSMR-GN            (QI geometry only)
    02_free_adam   free QI geometry + pure Adam (no lstsq)
    03_fullnet     full-network Adam -> GN      (no lstsq)
    04_varpro_gn   VarPro + Adam -> dense GN, 4-init ladder
    05_second_order LBFGS / SSBroyden + lstsq readout (the 1e-8 wall check)
    06_sine_xavier_seeds  the one random-start cell that reached the floor, seeds 1-3

Corrections carried by the probe modules (see their diffs): truncated-SVD readout with the
Kaufman projector restricted to the retained subspace; no normal-equation fallback; every row
records init / final / best, and the plots draw FINAL (solid) against the init refit (dashed).

Usage:  uv run --extra dev python experiments/expD04_varpro/varpro_corrected/run_quick.py [--only 01,04]
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
D04 = HERE.parent
REPO_ROOT = D04.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
# outputs live under results/ (the old, uncorrected figures are in results/.../expD04_varpro/archive/)
OUT_ROOT = REPO_ROOT / "results" / "checkpoint_D_optimizers" / "expD04_varpro" / "varpro_corrected"


def load(name):
    spec = importlib.util.spec_from_file_location(f"d04_{name}", D04 / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[f"d04_{name}"] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"wrote {path} ({len(rows)} rows)", flush=True)


def stage(msg):
    print(f"\n===== {time.strftime('%H:%M:%S')}  {msg}", flush=True)


# ------------------------------------------------------------------ 01: VarPro + LSMR-GN
def run_lsmr(out):
    m = load("varpro_lsmr_probe")
    rows = []
    for target in ("runge", "sine_mixture"):
        for res in (64, 256):
            W, init, best, final = m.run_case(target, res, 0)
            rows.append({"target": target, "resolution": res, "width": W, "seed": 0,
                         "lsmr_init_eval_rel_l2": init, "lsmr_final_eval_rel_l2": final,
                         "lsmr_best_eval_rel_l2": best})
            print(f"  lsmr {target:>12s} N={res:<4d} init={init:.2e} final={final:.2e}", flush=True)
    write_csv(out / "varpro_lsmr_summary.csv", rows)


# ------------------------------------------------------------------ 02: free QI geometry + pure Adam
def run_free_adam(out):
    m = load("free_qi_adam_probe")
    d05 = m.d05
    cfg = d05.RunConfig(targets=("runge", "sine_mixture"), resolutions=(64, 256), seeds=(0,),
                        families=(m.FAMILY,), steps=5000, eval_interval=250, learning_rate=1e-3,
                        n_train=2003, n_eval=4001, mode="full")
    rows = []
    for target in cfg.targets:
        for res in cfg.resolutions:
            s, _, _ = d05.run_case(d05.Case(target=target, resolution=res, seed=0, family=m.FAMILY), cfg)
            rows.append({"target": target, "resolution": res, "width": s["width"], "seed": 0,
                         "adam_init_eval_rel_l2": s["initial_eval_rel_l2"],
                         "adam_final_eval_rel_l2": s["final_eval_rel_l2"],
                         "adam_best_eval_rel_l2": s["best_eval_rel_l2"]})
            print(f"  adam {target:>12s} N={res:<4d} init={s['initial_eval_rel_l2']:.2e} "
                  f"final={s['final_eval_rel_l2']:.2e} best={s['best_eval_rel_l2']:.2e}", flush=True)
    write_csv(out / "free_qi_adam_summary.csv", rows)


# ------------------------------------------------------------------ 03: full-network Adam -> GN
def run_fullnet(out):
    m = load("adam_gn_fullnet_probe")
    import numpy as np
    rows = []
    for target in ("runge", "sine_mixture"):
        t = m.get_target(target)
        xt = np.linspace(-1, 1, m.N_TRAIN); yt = t.fn_numpy(xt).astype(np.float64)
        xe = np.linspace(-1, 1, m.N_EVAL); ye = t.fn_numpy(xe).astype(np.float64)
        for res in (64, 256):
            geom = m.d05.geometry_for_resolution(res); W = geom.width
            st = m.d05.build_initial_state(m.FAMILY, target, geom, 0)
            p0 = np.concatenate([st.input_weights, st.input_biases,
                                 st.readout_weights, [st.readout_bias]]).astype(np.float64)
            init, best, final = m.adam_then_gn(p0, W, xt, yt, xe, ye, 1000, 30)
            rows.append({"target": target, "resolution": res, "width": W, "seed": 0,
                         "adamgn_init_eval_rel_l2": init, "adamgn_final_eval_rel_l2": final,
                         "adamgn_best_eval_rel_l2": best})
            print(f"  fullnet {target:>12s} N={res:<4d} init={init:.2e} final={final:.2e} best={best:.2e}",
                  flush=True)
    write_csv(out / "adam_gn_fullnet_summary.csv", rows)


# ------------------------------------------------------------------ 04: VarPro + Adam -> dense GN, init ladder
def run_varpro_gn(out):
    m = load("varpro_gn_probe")
    m.RESULTS_DIR = out
    m.SUMMARY_CSV = out / "varpro_gn_summary.csv"
    rows = m.run(("sine", "sine_8pi", "runge", "sine_mixture"), (32, 64, 128, 256), (0,),
                 200, 40, m.DEFAULT_N_TRAIN, m.DEFAULT_N_EVAL, inits=None)
    m.write_summary(rows)
    m.plot(rows)                                   # convergence grid, final vs init refit
    p = load("plot_init_slice")
    p.SUMMARY_CSV = m.SUMMARY_CSV
    p.OUT = out / "init_slice_varpro_gn.png"
    p.main()


# ------------------------------------------------------------------ 05: LBFGS / SSBroyden + lstsq readout
def run_second_order(out):
    m = load("second_order_probe")
    rows = []
    for opt_name in ("lbfgs", "ssbroyden"):
        for target in ("sine", "runge", "sine_8pi"):
            for res in (64, 128):
                fn = m.run_case_ssbroyden if opt_name == "ssbroyden" else m.run_case
                W, init, final, best = fn("lstsq", target, res, 0)
                rows.append({"regime": "lstsq", "optimizer": opt_name, "target": target,
                             "resolution": res, "width": W, "seed": 0,
                             "init_eval_rel_l2": init, "final_eval_rel_l2": final,
                             "best_eval_rel_l2": best})
                print(f"  {opt_name:>9s}+lstsq {target:>8s} N={res:<4d} init={init:.2e} final={final:.2e}",
                      flush=True)
        write_csv(out / f"second_order_{opt_name}_summary.csv", [r for r in rows if r["optimizer"] == opt_name])
    # the 4-way lstsq-regime figure (expD02 Adam line from cube.json + our three arms)
    p = load("plot_second_order")
    p.D04 = out
    p.FIGDIR = out
    for reg in p.REGIMES.values():
        reg["lines"] = [(lab, c, mk, (src[0], out / Path(src[1]).name, src[2], src[3]) if src[0] == "csv" else src)
                        for lab, c, mk, src in reg["lines"]]
    # our varpro_gn rows live in 04_varpro_gn; point the lstsq figure at them
    vg = OUT_ROOT / "04_varpro_gn" / "varpro_gn_summary.csv"
    for reg in p.REGIMES.values():
        reg["lines"] = [(lab, c, mk, (src[0], vg, src[2], src[3]) if src[0] == "csv" and "varpro_gn" in str(src[1]) else src)
                        for lab, c, mk, src in reg["lines"]]
    p.main()


# ------------------------------------------------------------------ 06: sine from Xavier, more seeds
def run_sine_xavier_seeds(out):
    """The 04 ladder had ONE cell where VarPro-GN reached the floor from a random start
    (sine, Xavier, N=256, seed 0: 5e-2 refit -> 6.7e-15). Same cell at seeds 1-3, and N=128."""
    m = load("varpro_gn_probe")
    m.RESULTS_DIR = out
    m.SUMMARY_CSV = out / "varpro_gn_summary.csv"
    rows = m.run(("sine",), (128, 256), (1, 2, 3), 200, 40, m.DEFAULT_N_TRAIN, m.DEFAULT_N_EVAL,
                 inits=("xavier",))
    m.write_summary(rows)


def replot():
    """Regenerate the 04/05 figures from the existing CSVs (no reruns)."""
    m = load("varpro_gn_probe")
    out = OUT_ROOT / "04_varpro_gn"
    m.RESULTS_DIR = out; m.SUMMARY_CSV = out / "varpro_gn_summary.csv"
    m.plot(m.read_summary())
    p = load("plot_init_slice"); p.SUMMARY_CSV = m.SUMMARY_CSV; p.OUT = out / "init_slice_varpro_gn.png"; p.main()
    out5 = OUT_ROOT / "05_second_order"
    p = load("plot_second_order"); p.D04 = out5; p.FIGDIR = out5
    for reg in p.REGIMES.values():
        reg["lines"] = [(lab, c, mk, (src[0], out5 / Path(src[1]).name, src[2], src[3]) if src[0] == "csv" else src)
                        for lab, c, mk, src in reg["lines"]]
        reg["lines"] = [(lab, c, mk, (src[0], m.SUMMARY_CSV, src[2], src[3]) if src[0] == "csv" and "varpro_gn" in str(src[1]) else src)
                        for lab, c, mk, src in reg["lines"]]
    p.main()


STAGES = [("01_lsmr", run_lsmr), ("02_free_adam", run_free_adam), ("03_fullnet", run_fullnet),
          ("04_varpro_gn", run_varpro_gn), ("05_second_order", run_second_order),
          ("06_sine_xavier_seeds", run_sine_xavier_seeds)]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", default=None, help="comma list of stage prefixes, e.g. 01,04")
    ap.add_argument("--plot-only", action="store_true", help="regenerate the 04/05 figures from existing CSVs")
    args = ap.parse_args(argv)
    if args.plot_only:
        replot(); return 0
    keep = None if args.only is None else {s.strip() for s in args.only.split(",")}
    t0 = time.time()
    for name, fn in STAGES:
        if keep is not None and name.split("_")[0] not in keep:
            continue
        out = OUT_ROOT / name
        out.mkdir(parents=True, exist_ok=True)
        stage(f"{name}  -> {out}")
        ts = time.time()
        fn(out)
        print(f"===== {name} done in {time.time() - ts:.0f}s (elapsed {time.time() - t0:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
