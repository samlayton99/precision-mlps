"""Build the consolidated expD02 cube -- one JSON, sliceable any way.

The cube has three axes plus width:
  function (6): sine, sine_8pi, runge, sine_mixture, exp, abs_cubed   (one per category)
  init (4):     xavier, scaled_xavier, const_weight, qi               (first-layer init)
  regime (4):   adam_both, adam_first_lstsq, frozen_init_lstsq, qi_lstsq  (per-row columns)
  width (5):    N in {32, 64, 128, 256, 512}

Each row is one (init, function, N): the four regimes are columns, so any slice
(fix an init -> 4 regime curves; fix a regime -> compare inits; etc.) is a filter.

INITS (first layer; gamma* = lambda*/h, lambda* = 0.25):
  xavier         w,b ~ Xavier (gamma ~ 0.1, the standard start; NOT in the regime)
  scaled_xavier  Xavier then scaled so mean|w| = gamma*; centers = raw -b/w (heavy-tailed coverage)
  const_weight   w = gamma* (constant), centers = Xavier -b/w clipped to the span (covers cleanly), b = -gamma* c
  qi             exact QI geometry (w = gamma*, b = -gamma* c_uniform) + lstsq readout

REGIMES (what is trained vs frozen vs refit; relative L2, fp64 refit):
  adam_both          fully trained by Adam (both layers), best over LRs {1e-3, 3e-3}
  adam_first_lstsq   first layer trained by Adam (best checkpoint), readout re-solved by lstsq
  frozen_init_lstsq  first layer frozen at init (untrained), readout by lstsq
  qi_lstsq           QI uniform first layer (pure fp64), readout by lstsq  (init-independent reference)

Reuse (no recompute):
  scaled_xavier, qi  <- results/.../data_lambda_init.json (schemes scaled_xavier, full_qi), all 6 funcs
  xavier (4 funcs)   <- results/.../data.json (original run.py Xavier baseline): sine, sine_8pi, runge, exp
Compute (new, 8 funcs x 5 widths = 40 cells):
  xavier             -> sine_mixture, abs_cubed (the 2 the original run lacked)
  const_weight       -> all 6 funcs (corrected covering-centers version)

Seed note: the cube seed per function is its index in FUNCTIONS (sine=0 ... abs_cubed=5),
EXCEPT xavier on sine/sine_8pi/runge/exp, which is reused verbatim from the original
run.py (seeds 0,1,2,3). That means xavier-exp carries seed 3 while scaled_xavier/
const_weight-exp carry seed 4. Each row records its own `seed`. qi is deterministic
(no random draw), seed = null.

Output: results/checkpoint_D_optimizers/expD02_adam_geometry/cube.json
Usage:  python experiments/expD02_adam_geometry/build_cube.py
"""

import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_lambda_init as rli

RESULTS_DIR = REPO_ROOT / "results" / "checkpoint_D_optimizers" / "expD02_adam_geometry"
CUBE_PATH = RESULTS_DIR / "cube.json"

FUNCTIONS = ["sine", "sine_8pi", "runge", "sine_mixture", "exp", "abs_cubed"]
WIDTHS = [32, 64, 128, 256, 512]
INITS = ["xavier", "scaled_xavier", "const_weight", "qi"]
REGIMES = ["adam_both", "adam_first_lstsq", "frozen_init_lstsq", "qi_lstsq"]


def load_lambda(scheme_key):
    """Rows for one scheme from data_lambda_init.json, mapped to regime names."""
    d = json.load(open(RESULTS_DIR / "data_lambda_init.json"))
    out = {}
    for r in d["rows"]:
        if r["scheme"] == scheme_key:
            out[(r["fn"], r["N"])] = {
                "W": r["W"], "adam_both": r["trained"],
                "adam_first_lstsq": r["learned_refit"],
                "frozen_init_lstsq": r["init_refit"], "qi_lstsq": r["uniform_refit"]}
    return out


def load_xavier_orig():
    """The original run.py Xavier baseline (data.json), mapped to regime names.
    random_refit (untrained Xavier geometry + lstsq) is the frozen_init_lstsq regime."""
    d = json.load(open(RESULTS_DIR / "data.json"))
    seed = {"sine": 0, "sine_8pi": 1, "runge": 2, "exp": 3}
    out = {}
    for r in d["rows"]:
        out[(r["fn"], r["N"])] = {
            "W": r["W"], "seed": seed[r["fn"]], "adam_both": r["trained"],
            "adam_first_lstsq": r["learned_refit"],
            "frozen_init_lstsq": r["random_refit"], "qi_lstsq": r["uniform_refit"]}
    return out


def main():
    sx = load_lambda("scaled_xavier")
    qi = load_lambda("full_qi")
    xav_orig = load_xavier_orig()

    rows = []
    t0 = time.time()
    for fn in FUNCTIONS:
        for N in WIDTHS:
            # --- xavier (a): reuse the original 4 funcs, compute the 2 new ones ---
            if (fn, N) in xav_orig:
                c = xav_orig[(fn, N)]; seed = c["seed"]
            else:
                c = rli.compute_cell("xavier", N, fn); seed = c["seed"]
                print(f"[{time.time()-t0:5.0f}s] computed xavier       {fn:12s} N={N}")
            rows.append({"init": "xavier", "function": fn, "N": N, "W": c["W"],
                         "seed": seed, **{k: c[k] for k in REGIMES}})

            # --- scaled_xavier (b): reuse ---
            c = sx[(fn, N)]
            rows.append({"init": "scaled_xavier", "function": fn, "N": N, "W": c["W"],
                         "seed": rli.TARGET_SEEDS[fn], **{k: c[k] for k in REGIMES}})

            # --- const_weight (c): compute all (corrected covering-centers) ---
            c = rli.compute_cell("const_weight", N, fn)
            print(f"[{time.time()-t0:5.0f}s] computed const_weight {fn:12s} N={N}")
            rows.append({"init": "const_weight", "function": fn, "N": N, "W": c["W"],
                         "seed": c["seed"], **{k: c[k] for k in REGIMES}})

            # --- qi (d): reuse full_qi; deterministic, no seed ---
            c = qi[(fn, N)]
            rows.append({"init": "qi", "function": fn, "N": N, "W": c["W"],
                         "seed": None, **{k: c[k] for k in REGIMES}})

    out = {
        "config": {
            "functions": FUNCTIONS, "inits": INITS, "regimes": REGIMES, "widths_n": WIDTHS,
            "lambda_star": rli.LAMBDA_STAR, "lrs": rli.LRS, "steps": rli.STEPS,
            "n_train": rli.N_TRAIN, "n_eval": rli.N_EVAL, "train_dtype": "float32",
            "metric": "eval relative L2 (||pred-y||/||y||) on a dense fp64 grid",
            "init_defs": {
                "xavier": "w,b ~ Xavier (gamma ~ 0.1)",
                "scaled_xavier": "Xavier scaled so mean|w|=gamma*; centers = raw -b/w (heavy-tailed)",
                "const_weight": "w=gamma* constant; centers = Xavier -b/w clipped to span; b=-gamma*c",
                "qi": "exact QI geometry (w=gamma*, b=-gamma* c_uniform) + lstsq readout",
            },
            "regime_defs": {
                "adam_both": "fully trained by Adam (both layers), best over LRs",
                "adam_first_lstsq": "first layer Adam (best checkpoint), readout by lstsq",
                "frozen_init_lstsq": "first layer frozen at init, readout by lstsq",
                "qi_lstsq": "QI uniform first layer (fp64), readout by lstsq (init-independent)",
            },
            "provenance": {
                "scaled_xavier": "reused from data_lambda_init.json (scheme scaled_xavier)",
                "qi": "reused from data_lambda_init.json (scheme full_qi)",
                "xavier[sine,sine_8pi,runge,exp]": "reused from data.json (original run.py)",
                "xavier[sine_mixture,abs_cubed]": "computed by build_cube.py",
                "const_weight": "computed by build_cube.py (all 6 functions)",
            },
            "seed_note": "seed = function index, except xavier on the 4 original funcs "
                         "(reused from run.py, seeds 0,1,2,3 -> exp is seed 3 not 4). "
                         "qi is deterministic (seed null).",
        },
        "rows": rows,
    }
    with open(CUBE_PATH, "w") as f:
        json.dump(out, f, indent=0)
    print(f"\nSaved {CUBE_PATH}: {len(rows)} rows "
          f"({len(INITS)} inits x {len(FUNCTIONS)} funcs x {len(WIDTHS)} widths) "
          f"in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
