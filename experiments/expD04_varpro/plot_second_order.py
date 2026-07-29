"""Second-order optimizer comparison from the QI geometry init.

Two figures, one per regime, each a 2x3 target grid (x = N, y = eval rel L2, log):
lines are OPTIMIZERS, all starting from the free QI geometry (qi_geom_random_readout):

  second_order_both.png   -- regime: init + optimize ALL params, NO least squares.
      Adam (expD02) / LBFGS / SSBroyden / Gauss-Newton (full net).

  second_order_lstsq.png  -- regime: optimize geometry, readout solved by lstsq.
      Adam+lstsq (expD02) / LBFGS+lstsq / SSBroyden+lstsq / Gauss-Newton (VarPro).

Reference lines: fp64 floor (5e-14, dotted) and sqrt(eps_fp64) (1.49e-8, dashed) --
the precision wall a loss-value line search cannot cross.

expD02 Adam is float32/lambda*=0.25 and hands over the QI readout; our lines are
fp64/lambda=0.30 with the readout NOT handed over. Shared x-axis is N.

Draws whatever optimizer CSVs are present (skips missing), so it can be run
before every sweep has finished. Usage:
    python experiments/expD04_varpro/plot_second_order.py
"""

import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter, ScalarFormatter

REPO_ROOT = Path(__file__).resolve().parents[2]
D02 = REPO_ROOT / "results" / "checkpoint_D_optimizers" / "expD02_adam_geometry"
D04 = REPO_ROOT / "results" / "checkpoint_D_optimizers" / "expD04_varpro"
FIGDIR = D04 / "figures"

FLOOR = 5e-14
SQRT_EPS = 2.220446049250313e-16 ** 0.5   # 1.49e-8

# per regime: list of (label, color, marker, source-spec). source-spec is either
#   ("cube", regime_key)                          -> expD02 Adam, qi init
#   ("csv", path, value_col, row_filter)          -> one of our result files
def cube_src(key):
    return ("cube", key)

def csv_src(name, col, filt):
    return ("csv", D04 / name, col, filt)

REGIMES = {
    "both": {
        "title": "regime: init + optimize all params (NO least squares)",
        "out": "second_order_both.png",
        "lines": [
            ("Adam (expD02, QI init)", "#1f77b4", "o", cube_src("adam_both")),
            ("LBFGS",     "#e377c2", "P", csv_src("second_order_lbfgs_summary.csv",     "best_eval_rel_l2", lambda r: r["regime"] == "both")),
            ("SSBroyden", "#bcbd22", "D", csv_src("second_order_ssbroyden_summary.csv", "best_eval_rel_l2", lambda r: r["regime"] == "both")),
            ("Gauss-Newton (full net)", "#17becf", "v", csv_src("adam_gn_fullnet_summary.csv", "adamgn_best_eval_rel_l2", lambda r: True)),
        ],
    },
    "lstsq": {
        "title": "regime: optimize geometry + least-squares readout",
        "out": "second_order_lstsq.png",
        "lines": [
            ("Adam + lstsq (expD02, QI init)", "#1f77b4", "o", cube_src("adam_first_lstsq")),
            ("LBFGS + lstsq",     "#e377c2", "P", csv_src("second_order_lbfgs_summary.csv",     "best_eval_rel_l2", lambda r: r["regime"] == "lstsq")),
            ("SSBroyden + lstsq", "#bcbd22", "D", csv_src("second_order_ssbroyden_summary.csv", "best_eval_rel_l2", lambda r: r["regime"] == "lstsq")),
            ("Gauss-Newton (VarPro) ← ours", "#d62728", "*", csv_src("varpro_gn_summary.csv", "vpgn_best_eval_rel_l2", lambda r: r["geom_init"] == "qi")),
        ],
    },
}


def median(vals):
    vals = sorted(v for v in vals if v == v)
    return vals[len(vals) // 2] if vals else float("nan")


def load_csv_agg(path, col, filt):
    if not path.exists():
        return None
    agg = {}
    for r in csv.DictReader(open(path)):
        if filt(r):
            agg.setdefault((r["target"], int(r["resolution"])), []).append(float(r[col]))
    return agg


def main():
    if not (D02 / "cube.json").exists():
        print("no cube.json"); sys.exit(1)
    cube = json.load(open(D02 / "cube.json"))
    crows, widths, functions = cube["rows"], cube["config"]["widths_n"], cube["config"]["functions"]

    def cube_val(fn, N, key):
        return next(r[key] for r in crows if r["init"] == "qi" and r["function"] == fn and r["N"] == N)

    FIGDIR.mkdir(parents=True, exist_ok=True)
    for regime, spec in REGIMES.items():
        fig, axes = plt.subplots(2, 3, figsize=(18, 10.5))
        drawn = []
        for ax, fn in zip(axes.ravel(), functions):
            for label, color, marker, source in spec["lines"]:
                if source[0] == "cube":
                    ys = [cube_val(fn, N, source[1]) for N in widths]
                else:
                    agg = load_csv_agg(source[1], source[2], source[3])
                    if agg is None or not any(k[0] == fn for k in agg):
                        continue
                    ys = [median(agg.get((fn, N), [])) for N in widths]
                lw = 2.6 if "ours" in label else 2.0
                ms = 13 if marker == "*" else 7
                ax.loglog(widths, ys, f"-{marker}", color=color, ms=ms, lw=lw, label=label)
                if label not in drawn:
                    drawn.append(label)
            ax.axhline(FLOOR, ls=":", color="0.5", lw=1.1)
            ax.axhline(SQRT_EPS, ls="--", color="0.5", lw=1.1)
            ax.set_ylim(1e-15, 3.0)             # fixed axis (no misleading autoscale)
            ax.set_title(fn, fontsize=12)
            ax.grid(True, alpha=0.3, which="both")
            ax.set_xlabel(r"$N$   (width $W = N + 2\,$halo$+1$)")
            ax.set_ylabel(r"eval relative $L_2$")
            ax.set_xticks(widths)
            ax.xaxis.set_major_formatter(ScalarFormatter())
            ax.xaxis.set_minor_formatter(NullFormatter())

        # legend (optimizers + reference lines)
        handles, labels = axes[0][0].get_legend_handles_labels()
        handles.append(plt.Line2D([0], [0], ls="--", color="0.5", lw=1.1))
        labels.append(r"$\sqrt{\varepsilon_{fp64}}$ (1.5e-8)")
        handles.append(plt.Line2D([0], [0], ls=":", color="0.5", lw=1.1))
        labels.append("fp64 floor (5e-14)")
        fig.tight_layout(rect=[0, 0, 1, 0.9])
        fig.suptitle(
            f"Second-order optimizers vs Adam from the QI geometry -- {spec['title']}\n"
            "all start from free QI geometry (readout not handed over); "
            "expD02 Adam = float32/$\\lambda^*{=}0.25$, ours = fp64/$\\lambda{=}0.30$",
            fontsize=13, y=0.99, va="top")
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.945),
                   ncol=len(labels), fontsize=9.5, frameon=True, borderaxespad=0)
        out = FIGDIR / spec["out"]
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out}  (optimizers drawn: {drawn})")


if __name__ == "__main__":
    main()
