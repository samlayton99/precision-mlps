"""expD17/width_scaling_gelu -- the width-scaling geometry-motion study, GELU.

One variable changed from `../width_scaling/run.py`: the hidden activation.
This file is a thin driver -- it loads that module and rebinds four globals, so
the builders, the training loop, every figure and the analysis are the SAME
code. Nothing here re-implements plotting.

What must change with the activation, and why:

  1. The bandwidth. expC07's aliasing rule sets lambda* by the activation's
     kernel Fourier tail: tanh 0.250, gelu 0.707 (measured 0.716). Running GELU
     at tanh's 0.25 would confound "GELU behaves differently" with "GELU was
     given the wrong bandwidth" -- expA07 measured a 1e4 break in the readout
     norm law at exactly that mistake. Every class's tuned tanh lambda is
     therefore scaled by LAM_RATIO = 0.707/0.25 = 2.828.

  2. Nothing else. Same widths, budgets, seeds, zero-init readout, snapshot
     schedule, metrics.

GELU is kernel order r=2 (K = gelu''), so the readout law is v_k ~ (h^2/lambda)
f''(c_k) rather than tanh's (h/2) f'(c_k), and gelu'' changes sign, so the QI
construction is NOT strictly valid for it (expA04, expC07). What transfers is
the *geometry* -- uniform centers, one shared bandwidth, a halo -- not a proven
construction. That caveat is the point of running this.

Usage:
    uv run --extra dev python experiments/expD17_geometry_motion/width_scaling_gelu/run.py --probe
    uv run --extra dev python experiments/expD17_geometry_motion/width_scaling_gelu/run.py --driver
    uv run --extra dev python experiments/expD17_geometry_motion/width_scaling_gelu/run.py --plot
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

torch.set_default_dtype(torch.float64)
torch.set_num_threads(3)

LAMBDA_GELU = 0.707          # expC07 aliasing rule (measured optimum 0.716)
LAMBDA_TANH = 0.25
LAM_RATIO = LAMBDA_GELU / LAMBDA_TANH


def _load_by_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


WS = _load_by_path("expD17w_shared",
                   Path(__file__).resolve().parents[1] / "width_scaling" / "run.py")

# --- the four rebindings: activation, bandwidths, output directory ---
WS.P.ACT = "gelu"
WS.LAMBDA_STAR = LAMBDA_GELU
WS.LAMBDA_2D = {k: round(v * LAM_RATIO, 4) for k, v in WS.P.LAMBDA_2D.items()}
WS.LAMBDA_PINN = round(WS.P.LAMBDA_PINN * LAM_RATIO, 4)
WS.OUT_DIR = (REPO_ROOT / "results" / "checkpoint_D_optimizers"
              / "expD17_geometry_motion" / "width_scaling_gelu")
WS.DATA_DIR = WS.OUT_DIR / "data"
WS.FIG_DIR = WS.OUT_DIR / "figures"


def lambda_probe(N=128, target="sine"):
    """Sanity gate before the grid: build the 1-D GELU geometry at tanh's
    lambda and at GELU's own, and report the lstsq floor each reaches. Uses the
    study's own builder and probe, so it measures exactly what the grid will."""
    from src.data.targets import get_target

    print(f"lambda probe -- 1-D {target}, N={N}, GELU, lstsq floor:\n")
    print(f"{'lambda':>8} {'W':>6} {'gamma':>9} {'pre-probe rel L2':>18}")
    out = {}
    for lam in (LAMBDA_TANH, LAMBDA_GELU):
        WS.LAMBDA_STAR = lam
        b = WS.build_interp1d(target, "qi", N)
        floor = b["probe_fn"]()
        out[lam] = floor
        print(f"{lam:8.3f} {b['extra']['W']:6d} {b['extra']['gamma']:9.3f} "
              f"{floor:18.3e}")
    WS.LAMBDA_STAR = LAMBDA_GELU
    best = min(out, key=out.get)
    print(f"\nbest: lambda={best} ({out[best]:.3e}); "
          f"ratio {max(out.values())/min(out.values()):.3g}x")
    # tanh reference at its own optimum, same builder, for context
    WS.P.ACT = "tanh"
    WS.LAMBDA_STAR = LAMBDA_TANH
    ref = WS.build_interp1d(target, "qi", N)["probe_fn"]()
    WS.P.ACT = "gelu"
    WS.LAMBDA_STAR = LAMBDA_GELU
    print(f"tanh reference at lambda=0.25: {ref:.3e}")
    return out


if __name__ == "__main__":
    if "--probe" in sys.argv:
        lambda_probe()
    elif "--plot" in sys.argv:
        WS.plot("drift", "expD17w_gelu_drift_from_init.png",
                "$\\|g_i-g_0\\|/\\|g_0\\|$")
        WS.plot("step_size", "expD17w_gelu_step_size.png",
                "$\\|g_i-g_{i-1}\\|/\\|g_0\\|$")
        WS.plot("drift_abs", "expD17w_gelu_drift_from_init_abs.png",
                "$\\|g_i-g_0\\|$  (absolute)")
        WS.plot("step_abs", "expD17w_gelu_step_size_abs.png",
                "$\\|g_i-g_{i-1}\\|$  (absolute)")
        WS.plot_loss("expD17w_gelu_loss.png")
        WS.hist_gifs("gamma")
        WS.hist_gifs("center")
        WS.gamma_vs_update()
        WS.analyze("expD17w_gelu_ratio_scatter.png")
        WS.dead_analysis()
    elif "--job" in sys.argv:
        cls, wi = sys.argv[sys.argv.index("--job") + 1].split(":")
        WS.run_job(cls, int(wi))
    else:
        WS.driver(script=__file__)      # workers must re-enter THIS driver
