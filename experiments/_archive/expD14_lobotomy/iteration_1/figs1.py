"""iteration_1 figures. All legends OUTSIDE and above the axes (repo rule)."""
import importlib.util
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
_s = importlib.util.spec_from_file_location("d14_core1", HERE / "core1.py")
c1 = importlib.util.module_from_spec(_s)
_s.loader.exec_module(c1)

RES = c1.RESULTS
FIGS = c1.FIGS
FIGS.mkdir(parents=True, exist_ok=True)

ARMS = ["adam", "full", "ls", "lobo0", "tau1"]
COLS = {"adam": "0.45", "full": "#d62728", "ls": "#ff7f0e",
        "lobo0": "#9467bd", "tau1": "#2ca02c"}
LABS = {"adam": "Adam (no solve)", "full": "exact solve, unthrottled",
        "ls": "exact solve + line search", "lobo0": "iteration-0 arm (both throttles)",
        "tau1": "tau=1 under-solve, unthrottled"}
CASES = ["qi", "clustered", "datagap", "rand", "rand_scale"]
TARGETS = ["sine", "sine_8pi", "runge"]
YLIM = (1e-16, 3e2)

recs = c1.load(RES / "t5_confound.jsonl")
by = {}
for r in recs:
    by.setdefault((r["arm"], r["fn"], r["case"]), []).append(r)


def med_traj(cells, hkey, xkey="it", sub="fhist"):
    """Median trajectory across seeds on the shared grid."""
    if not cells:
        return None, None
    xs = cells[0][sub][xkey]
    ys = np.array([c[sub][hkey] for c in cells])
    return np.array(xs), np.median(ys, axis=0)


# ---- F1: the verdict figure -- floor trajectories --------------------------
fig, axes = plt.subplots(len(CASES), len(TARGETS),
                         figsize=(13, 15), sharex=True, sharey=True)
for i, case in enumerate(CASES):
    for j, fn in enumerate(TARGETS):
        ax = axes[i, j]
        for arm in ARMS:
            cells = by.get((arm, fn, case), [])
            x, y = med_traj(cells, "floor")
            if x is None:
                continue
            for c in cells:
                ax.plot(c["fhist"]["it"], c["fhist"]["floor"],
                        color=COLS[arm], lw=0.5, alpha=0.25)
            ax.plot(x, y, color=COLS[arm], lw=1.6)
        cells0 = by.get(("adam", fn, case), [])
        if cells0:
            ax.axhline(np.median([c["floor0"] for c in cells0]),
                       color="k", ls="--", lw=0.8)
        ax.set_yscale("log")
        ax.set_ylim(*YLIM)
        if i == 0:
            ax.set_title(fn)
        if j == 0:
            ax.set_ylabel(f"{case}\nfloor (eval rel $L_2$)")
        if i == len(CASES) - 1:
            ax.set_xlabel("iteration")
        ax.grid(True, which="major", alpha=0.3)
fig.legend(handles=[plt.Line2D([], [], color=COLS[a], lw=1.6, label=LABS[a])
                    for a in ARMS] +
           [plt.Line2D([], [], color="k", ls="--", lw=0.8,
                       label="floor of the initial geometry")],
           loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=3)
fig.suptitle("The floor of the current geometry (what one terminal exact solve"
             " would give), thin = seeds, thick = median", y=0.955, fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(FIGS / "F1_floor_trajectories.png", dpi=150)
plt.close(fig)

# ---- F2: summary bars -- geometry improvement factor -----------------------
fig, axes = plt.subplots(1, len(CASES), figsize=(15, 4.2), sharey=True)
xpos = np.arange(len(TARGETS))
wbar = 0.16
for i, case in enumerate(CASES):
    ax = axes[i]
    for k, arm in enumerate(ARMS):
        vals = []
        for fn in TARGETS:
            cells = by.get((arm, fn, case), [])
            if cells:
                imp = [c["floor0"] / max(c["floor_final"], 1e-300)
                       for c in cells]
                vals.append(np.median(imp))
            else:
                vals.append(np.nan)
        ax.bar(xpos + (k - 2) * wbar, vals, wbar, color=COLS[arm])
    ax.axhline(1.0, color="k", lw=0.8)
    ax.set_yscale("log")
    ax.set_ylim(1e-3, 1e12)
    ax.set_xticks(xpos)
    ax.set_xticklabels(TARGETS, rotation=20)
    ax.set_title(case)
    if i == 0:
        ax.set_ylabel("floor$_0$ / floor$_{final}$  (>1 = geometry improved)")
    ax.grid(True, axis="y", alpha=0.3)
fig.legend(handles=[plt.Line2D([], [], color=COLS[a], lw=6, label=LABS[a])
                    for a in ARMS],
           loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=3)
fig.tight_layout(rect=(0, 0, 1, 0.82))
fig.savefig(FIGS / "F2_geometry_improvement.png", dpi=150)
plt.close(fig)

# ---- F3: the diagnostics -- travel, SNR, ||v|| (rand + rand_scale, sine) ---
fig, axes = plt.subplots(3, 2, figsize=(11, 10), sharex=True)
for j, case in enumerate(["rand", "rand_scale"]):
    for arm in ARMS:
        cells = by.get((arm, "sine", case), [])
        if not cells:
            continue
        x, y = med_traj(cells, "travel", sub="thist")
        axes[0, j].plot(x, y, color=COLS[arm], lw=1.4)
        x, y = med_traj(cells, "snr", sub="hist")
        axes[1, j].plot(x, y, color=COLS[arm], lw=1.4)
        x, y = med_traj(cells, "vnorm", sub="hist")
        axes[2, j].plot(x, y, color=COLS[arm], lw=1.4)
    axes[0, j].set_title(f"{case} / sine")
    axes[2, j].set_yscale("log")
    axes[2, j].set_xlabel("iteration")
axes[0, 0].set_ylabel("coherent travel $D$ on $(w,b)$")
axes[0, 0].set_ylim(0, 1)
axes[0, 1].set_ylim(0, 1)
axes[1, 0].set_ylabel("corrected Adam SNR $\\hat s$ on $(w,b)$")
axes[1, 0].set_ylim(0, 1)
axes[1, 1].set_ylim(0, 1)
axes[2, 0].set_ylabel("$\\|v\\|$")
fig.legend(handles=[plt.Line2D([], [], color=COLS[a], lw=1.6, label=LABS[a])
                    for a in ARMS],
           loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=3)
fig.tight_layout(rect=(0, 0, 1, 0.9))
fig.savefig(FIGS / "F3_diagnostics.png", dpi=150)
plt.close(fig)

# ---- F4: datagap by region -------------------------------------------------
fig, axes = plt.subplots(2, len(TARGETS), figsize=(13, 7),
                         sharex=True, sharey=True)
for j, fn in enumerate(TARGETS):
    for arm in ARMS:
        cells = by.get((arm, fn, "datagap"), [])
        if not cells:
            continue
        x, yi = med_traj(cells, "floor_in")
        _, yo = med_traj(cells, "floor_out")
        axes[0, j].plot(x, yi, color=COLS[arm], lw=1.4)
        axes[1, j].plot(x, yo, color=COLS[arm], lw=1.4)
    for i in (0, 1):
        axes[i, j].set_yscale("log")
        axes[i, j].set_ylim(*YLIM)
        axes[i, j].grid(True, alpha=0.3)
    axes[0, j].set_title(fn)
    axes[1, j].set_xlabel("iteration")
axes[0, 0].set_ylabel("floor INSIDE the gap")
axes[1, 0].set_ylabel("floor outside")
fig.legend(handles=[plt.Line2D([], [], color=COLS[a], lw=1.6, label=LABS[a])
                    for a in ARMS],
           loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=3)
fig.suptitle("datagap: the global average hides this split", y=0.90,
             fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.88))
fig.savefig(FIGS / "F4_datagap_regions.png", dpi=150)
plt.close(fig)

# ---- F5: reached error vs floor, side by side (the tension figure) ---------
fig, axes = plt.subplots(2, len(CASES), figsize=(16, 7), sharex=True,
                         sharey=True)
fn = "runge"
for i, case in enumerate(CASES):
    for arm in ARMS:
        cells = by.get((arm, fn, case), [])
        if not cells:
            continue
        x, y = med_traj(cells, "rel", xkey="it", sub="hist")
        axes[0, i].plot(x, y, color=COLS[arm], lw=1.2)
        x, y = med_traj(cells, "floor")
        axes[1, i].plot(x, y, color=COLS[arm], lw=1.2)
    for r_ in (0, 1):
        axes[r_, i].set_yscale("log")
        axes[r_, i].set_ylim(*YLIM)
        axes[r_, i].grid(True, alpha=0.3)
    axes[0, i].set_title(case)
    axes[1, i].set_xlabel("iteration")
axes[0, 0].set_ylabel("reached (eval rel $L_2$)")
axes[1, 0].set_ylabel("floor of current geometry")
fig.legend(handles=[plt.Line2D([], [], color=COLS[a], lw=1.6, label=LABS[a])
                    for a in ARMS],
           loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=3)
fig.suptitle(f"target = {fn}: the reached error (top) can diverge while the "
             "geometry (bottom) improves -- score the bottom", y=0.89,
             fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.87))
fig.savefig(FIGS / "F5_reached_vs_floor.png", dpi=150)
plt.close(fig)

print("figures written to", FIGS)
