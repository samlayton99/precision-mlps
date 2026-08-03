"""iteration_2 figures. Two figures, built to be read without a legend hunt.

G1  the scorecard: absolute errors, one panel per (case, target). For each arm
    two bars: what the run reached (light), and what one terminal exact solve
    would give from the geometry it learned (dark = the floor). The black line
    is the same quantity at step 0 ("where you started"). Dark bar below the
    line = the geometry got better. Lower = better everywhere.
G2  the control story: for each case (sine target), the error and floor
    trajectories (top) and the alpha each controller chose with the two
    signals it can see overlaid (bottom).
"""
import importlib.util
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
_s = importlib.util.spec_from_file_location("d14_core2", HERE / "core2.py")
c2 = importlib.util.module_from_spec(_s)
_s.loader.exec_module(c2)

FIGS = c2.FIGS
FIGS.mkdir(parents=True, exist_ok=True)

ARMS = ["adam", "alo", "ladder", "rentry", "lm", "snr"]
COLS = {"adam": "0.45", "alo": "#d62728", "ladder": "#8c564b",
        "rentry": "#1f77b4", "lm": "#ff7f0e", "snr": "#2ca02c"}
LABS = {"adam": "Adam, no solve", "alo": "solve exactly always",
        "ladder": "preset schedule", "rentry": "alpha = residual",
        "lm": "LM ratio", "snr": "alpha = max(Adam SNR, residual)"}
CASES = ["qi", "clustered", "datagap", "rand", "rand_scale"]
TARGETS = ["sine", "sine_8pi", "runge"]

recs = c2.load(c2.RESULTS / "t6_mu_signals.jsonl")
by = {}
for r in recs:
    by.setdefault((r["arm"], r["fn"], r["case"]), []).append(r)


def med(cells, key):
    return np.median([c[key] for c in cells]) if cells else np.nan


# ---- G1: the scorecard ------------------------------------------------------
fig, axes = plt.subplots(len(TARGETS), len(CASES), figsize=(16, 10),
                         sharey=True)
xpos = np.arange(len(ARMS))
for i, fn in enumerate(TARGETS):
    for j, case in enumerate(CASES):
        ax = axes[i, j]
        reached = [med(by.get((a, fn, case), []), "final_rel") for a in ARMS]
        floors = [med(by.get((a, fn, case), []), "floor_final") for a in ARMS]
        f0 = med(by.get(("adam", fn, case), []), "floor0")
        ax.bar(xpos - 0.2, reached, 0.38,
               color=[COLS[a] for a in ARMS], alpha=0.35)
        ax.bar(xpos + 0.2, floors, 0.38, color=[COLS[a] for a in ARMS])
        ax.axhline(f0, color="k", lw=1.2)
        ax.set_yscale("log")
        ax.set_ylim(1e-16, 3e2)
        ax.set_xticks(xpos)
        ax.set_xticklabels(ARMS, rotation=45, fontsize=7)
        if i == 0:
            ax.set_title(case)
        if j == 0:
            ax.set_ylabel(f"{fn}\neval rel $L_2$")
        ax.grid(True, axis="y", alpha=0.3)
fig.legend(handles=[
    plt.Rectangle((0, 0), 1, 1, fc="0.6", alpha=0.35,
                  label="reached during the run (light)"),
    plt.Rectangle((0, 0), 1, 1, fc="0.35",
                  label="after one terminal exact solve (dark) -- the geometry's worth"),
    plt.Line2D([], [], color="k", lw=1.2,
               label="the same at step 0: bars below this line = it learned")],
    loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=3)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(FIGS / "G1_scorecard.png", dpi=150)
plt.close(fig)

# ---- G2: the control story --------------------------------------------------
fig, axes = plt.subplots(2, len(CASES), figsize=(17, 7), sharex=True)
fn = "sine"
for j, case in enumerate(CASES):
    top, bot = axes[0, j], axes[1, j]
    for arm in ARMS:
        cells = by.get((arm, fn, case), [])
        if not cells:
            continue
        h, fh = cells[0]["hist"], cells[0]["fhist"]
        top.plot(fh["it"], fh["floor"], color=COLS[arm], lw=1.5)
        if arm != "adam":
            bot.plot(h["it"], h["alpha"], color=COLS[arm], lw=1.3)
    cells0 = by.get(("adam", fn, case), [])
    if cells0:
        h0 = cells0[0]["hist"]
        bot.plot(h0["it"], h0["rentry"], color="k", ls=":", lw=1.0)
        bot.plot(h0["it"], h0["shat"], color="0.4", ls="--", lw=1.0)
        top.axhline(cells0[0]["floor0"], color="k", ls="--", lw=0.8)
    for ax in (top, bot):
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
    top.set_ylim(1e-16, 3e2)
    bot.set_ylim(1e-16, 3)
    top.set_title(case)
    bot.set_xlabel("iteration")
axes[0, 0].set_ylabel("floor of current geometry")
axes[1, 0].set_ylabel(r"$\alpha$ chosen (colors); signals on Adam (black)")
fig.legend(handles=[plt.Line2D([], [], color=COLS[a], lw=1.5, label=LABS[a])
                    for a in ARMS] +
           [plt.Line2D([], [], color="k", ls=":", lw=1.0,
                       label="residual level $r_{entry}$ (on Adam)"),
            plt.Line2D([], [], color="0.4", ls="--", lw=1.0,
                       label="geometry SNR $\\hat s$ (on Adam)")],
           loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=4)
fig.suptitle(f"target = {fn}: what each controller chose, and what it could see",
             y=0.87, fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.85))
fig.savefig(FIGS / "G2_control_story.png", dpi=150)
plt.close(fig)

print("figures written to", FIGS)
