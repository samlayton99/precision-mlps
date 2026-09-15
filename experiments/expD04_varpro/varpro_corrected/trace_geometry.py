"""Did the sine/Xavier cell finish, and what happens with a longer budget?

Runs the same cell (sine, N=256, seed 0, Xavier init) at the short budget used by the ladder
(200 Adam + 40 GN) and at a long one (1000 Adam + 300 GN), tracing train residual, eval error,
median gamma of the in-domain neurons, and the in-domain fraction at every logged step.
One figure: error vs iteration (left) and median gamma vs iteration (right), both budgets.

Usage: uv run --extra dev python experiments/expD04_varpro/varpro_corrected/trace_geometry.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
D04 = HERE.parent
REPO_ROOT = D04.parents[1]
sys.path.insert(0, str(REPO_ROOT))
OUT_DIR = REPO_ROOT / "results" / "checkpoint_D_optimizers" / "expD04_varpro" / "varpro_corrected" / "06_sine_xavier_seeds"
spec = importlib.util.spec_from_file_location("d04_varpro_gn_probe", D04 / "varpro_gn_probe.py")
m = importlib.util.module_from_spec(spec); sys.modules["d04_varpro_gn_probe"] = m; spec.loader.exec_module(m)  # type: ignore

TARGET, N, SEED, INIT = "sine", 256, 0, "xavier"
BUDGETS = {"short (200 Adam + 40 GN)": (200, 40), "long (1000 Adam + 300 GN)": (1000, 300)}


def main():
    xt, yt, xe, ye = m.make_data(TARGET, m.DEFAULT_N_TRAIN, m.DEFAULT_N_EVAL)
    geom = m.d05.geometry_for_resolution(N); W = geom.width
    st = m.d05.build_initial_state(m.GEOM_INITS[INIT], TARGET, geom, SEED)
    theta0 = np.concatenate([st.input_weights, st.input_biases]).astype(np.float64)
    traces, finals = {}, {}
    for label, (warm, gn) in BUDGETS.items():
        tr = []
        init, best, final, theta = m.varpro_adam_then_gn(theta0, W, xt, yt, xe, ye, warm, gn, keep_theta=True, trace=tr)
        gn_rows = [r for r in tr if r["stage"] == "gn"]
        stopped = "cap reached" if gn_rows and gn_rows[-1]["it"] == gn and gn_rows[-1]["accepted"] else "no accepted step"
        finals[label] = dict(init=init, final=final, gn_iters_done=gn_rows[-1]["it"] if gn_rows else 0,
                             stop=stopped, gamma_med_final=gn_rows[-1]["gamma_med"] if gn_rows else None,
                             inside_frac_final=gn_rows[-1]["inside_frac"] if gn_rows else None)
        traces[label] = tr
        print(label, json.dumps(finals[label]))
    (OUT_DIR / "trace_sine_xavier_N256_s0.json").write_text(json.dumps(dict(finals=finals, traces=traces), indent=1))

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    colors = {"short (200 Adam + 40 GN)": "#1f77b4", "long (1000 Adam + 300 GN)": "#d62728"}
    for label, tr in traces.items():
        warm = BUDGETS[label][0]
        x = [r["it"] if r["stage"] == "adam" else warm + r["it"] for r in tr]
        c = colors[label]
        ax[0].semilogy(x, [r["eval_rel"] for r in tr], "-", color=c, lw=1.6, label=f"{label}: eval")
        ax[0].semilogy(x, [r["train_rel"] for r in tr], "--", color=c, lw=1.0, label=f"{label}: train")
        ax[0].axvline(warm, color=c, ls=":", lw=0.9)
        ax[1].semilogy(x, [r["gamma_med"] for r in tr], "-", color=c, lw=1.6, label=label)
        ax[1].axvline(warm, color=c, ls=":", lw=0.9)
    ax[0].axhline(5e-14, color="k", ls=":", lw=1.2, label="fp64 floor")
    ax[0].set_xlabel("iteration (Adam, then GN after the dotted line)"); ax[0].set_ylabel("relative $L_2$")
    ax[0].set_ylim(1e-16, 1); ax[0].grid(alpha=0.3, which="both")
    ax[0].legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2, fontsize=8, borderaxespad=0)
    h = 2.0 / N
    ax[1].axhline(0.30 / h, color="k", ls=":", lw=1.2, label=r"construction $\gamma=0.30/h$")
    ax[1].set_xlabel("iteration"); ax[1].set_ylabel(r"median $\gamma$ of in-domain neurons")
    ax[1].set_ylim(1e-2, 1e2); ax[1].grid(alpha=0.3, which="both")
    ax[1].legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2, fontsize=8, borderaxespad=0)
    fig.suptitle(f"expD04 corrected: {TARGET}, N={N} (W={W}), seed {SEED}, {INIT} start -- does a longer budget change what is found?", y=1.08)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "trace_sine_xavier_N256_s0.png", dpi=150, bbox_inches="tight")
    print("wrote", OUT_DIR / "trace_sine_xavier_N256_s0.png")


if __name__ == "__main__":
    main()
