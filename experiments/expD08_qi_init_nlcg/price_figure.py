"""The price-of-the-floor figure: achieved precision vs memory, one point
per mechanism class from the Phase-0 bake-off. The purple line is the
measured exchange rate (spectral density: ~0.03 decades per stored vector).

Run: uv run python experiments/expD08_qi_init_nlcg/price_figure.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
OUT = (REPO / "results" / "checkpoint_D_optimizers" / "expD08_qi_init_nlcg"
       / "iteration_9" / "b_replacement")

rows = json.loads((OUT / "bakeoff_rows.json").read_text())
PHI = [r["id"] for r in rows if r["id"].startswith("phi")]


def geo(arm):
    vals = [r["best_rel"] for r in rows
            if r["arm"] == arm and r["id"].startswith("phi")]
    return float(np.exp(np.mean(np.log(vals)))), len(vals)


# (arm, m-vector-equivalent memory, family, label)
# fn-space n-vectors converted at n/m ~ 4 (phi N256 r4: n=2076, m=520)
POINTS = [
    ("noB", 6, "precision", "no B (slim minus B)"),
    ("noB_dd3", 12, "precision", "compensated recurrence (full)"),
    ("noB_ddmv", 14, "precision", "ALL double-double (diagnostic)"),
    ("replay_gram_x1", 10, "replay", "replayed-stream projector (best)"),
    ("headk64", 70, "param-head", "frozen head k=64"),
    ("headk128", 134, "param-head", "frozen head k=128"),
    ("headk256", 262, "param-head", "frozen head k=256"),
    ("fhead256", 1030, "fn-space", "fn-space head k=256 (x4 n-vec)"),
    ("hyb64f512", 2118, "hybrid", "hybrid p64 + fn512"),
    ("B", 466, "B", "full B (iteration_7)"),
]
FAM_STYLE = {
    "precision": ("#888888", "o"),
    "replay": ("#17becf", "X"),
    "param-head": ("#7a3fc1", "s"),
    "fn-space": ("#e69b00", "D"),
    "hybrid": ("#8c564b", "^"),
    "B": ("#1f77b4", "*"),
}

OFFSETS = {
    "noB": (8, 10), "noB_dd3": (-30, -18), "noB_ddmv": (10, -4),
    "headk256": (-115, -16), "B": (12, 6), "hyb64f512": (-90, 12),
    "fhead256": (-60, 12),
}
fig, ax = plt.subplots(figsize=(9.5, 6.8))
seen = set()
for arm, mem, fam, lab in POINTS:
    g, ncells = geo(arm)
    col, mk = FAM_STYLE[fam]
    ax.scatter(mem, g, s=140 if mk == "*" else 70, color=col, marker=mk,
               zorder=5, label=fam if fam not in seen else None)
    seen.add(fam)
    suffix = "" if ncells == 6 else f" ({ncells} cells)"
    ax.annotate(lab + suffix, (mem, g), textcoords="offset points",
                xytext=OFFSETS.get(arm, (8, 4)), fontsize=8)

ks = np.linspace(0, 285, 100)
ax.plot(6 + ks, 4.9e-7 * 10 ** (-0.0303 * ks), color="#7a3fc1", lw=1.2,
        ls="--", alpha=0.7, zorder=1,
        label="exchange rate: 0.03 decades / stored vector\n(= spectral density of the problem)")
ax.axhline(1.59e-14, color="k", ls=":", lw=1.2, label="SVD floor (B reference)")
ax.axhline(4.9e-7, color="#888888", ls=":", lw=0.8, alpha=0.6)

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(4, 4000)
ax.set_ylim(1e-15, 1e-2)
ax.set_xlabel("optimizer memory (parameter-vector equivalents, log)")
ax.set_ylabel("geo-mean phi eval rel L2 (6 frozen systems)")
ax.grid(alpha=0.3, which="both")
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=3,
          fontsize=8, borderaxespad=0)
fig.suptitle("Phase 0: the price of the floor -- precision bought vs memory "
             "spent\n(only parameter-space own-process deflation reaches the "
             "floor)", y=0.99)
fig.tight_layout(rect=(0, 0, 1, 0.86))
p = OUT / "price_of_the_floor.png"
fig.savefig(p, dpi=150)
print("saved", p)
