"""expF17 plots: 1a/1b (scaling), 2 (comparators), 3 (polynomial ablation), status.

House rules: legends OUTSIDE the axes (single top figure legend), shared axis
scales across subplots, x = ACTUAL column count (never nominal), geometric mean
across seeds with geometric-standard-deviation bars [g/s_g, g*s_g] (SPEC 13.9).
"""
from __future__ import annotations

import json
import time
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import protocol as pr, store

TASK_TITLE = {
    "convection_c40": "convection c=40", "convection_c80": "convection c=80",
    "reaction": "reaction", "wave": "wave", "burgers": "Burgers",
    "poisson_man": "Poisson (perforated)", "darcy_man": "Darcy (manufactured)",
    "darcy_orig": "Darcy (FNO, ref-capped)",
}
# provenance colour groups (Sam, 2026-09-01): bwler / darcy / dysts / other
FAMILY_COLOR = {"bwler": "#1f77b4", "darcy": "#d62728", "dysts": "#2ca02c",
                "other": "#7f7f7f"}
TASK_FAMILY = {"convection_c40": "bwler", "convection_c80": "bwler",
               "reaction": "bwler", "wave": "bwler", "burgers": "bwler",
               "poisson_man": "bwler", "darcy_man": "darcy", "darcy_orig": "darcy"}

# published best-to-beat (dotted line on the scaling plots): BWLer Table 2 for
# its tasks; trained FNO ~1e-2 for Darcy (the repo's recorded convention, expF08)
SOTA = {"convection_c40": 2.04e-13, "convection_c80": 1.10e-12,
        "reaction": 6.94e-11, "wave": 1.26e-11, "burgers": 4.63e-3,
        "darcy_orig": 1e-2, "darcy_man": None, "poisson_man": None}
METHOD_COLOR = {"qi_radon": "C3", "qi_tensor": "C0", "elm": "C2",
                "spectral": "C1", "bwler": "C8", "rbf_imq": "C4", "rbf_phs": "C5",
                "qi_grid": "C3"}
METHOD_LABEL = {"qi_radon": "QI-Radon", "qi_tensor": "QI-tensor", "elm": "ELM",
                "spectral": "spectral (F/C)", "bwler": "BWLer (explicit)",
                "rbf_imq": "RBF-IMQ", "rbf_phs": "RBF-PHS+p1",
                "qi_grid": "QI (grid)"}
YLIM = (1e-16, 3e0)


def _agg(cells, regime, variant="base", methods=None):
    """-> {(task, method): [(cols, geomean, gsd, n_seeds)]} sorted by cols."""
    g = defaultdict(lambda: defaultdict(list))
    for rec in cells.values():
        if rec["regime"] != regime or rec["variant"] != variant:
            continue
        if methods and rec["method"] not in methods:
            continue
        g[(rec["task"], rec["method"])][rec["cols"]].append(rec["rel_l2"])
    out = {}
    for k, by_cols in g.items():
        rows = []
        for cols, vals in sorted(by_cols.items()):
            gm, sg = pr.geomean_gsd(vals)
            rows.append((cols, gm, sg, len(vals)))
        out[k] = rows
    return out


def _grid_fig(n_tasks=8):
    fig, axes = plt.subplots(2, 4, figsize=(19, 9))
    return fig, axes.ravel()


def _style_axis(ax, task, xlim=(80, 12000)):
    ax.set_title(TASK_TITLE.get(task, task), fontsize=10,
                 color=FAMILY_COLOR.get(TASK_FAMILY.get(task, "other"), "k"))
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_ylim(*YLIM)
    ax.set_xlim(*xlim)
    ax.axhline(1e-13, color="gray", lw=0.8, ls=":")
    sota = SOTA.get(task)
    if sota is not None:
        ax.axhline(sota, color="k", lw=1.2, ls=":")
    ax.grid(True, which="both", alpha=0.2)
    ax.set_xlabel("$W$")


def plot_scaling(cells, regime, path, methods, title):
    data = _agg(cells, regime, methods=methods)
    fig, axes = _grid_fig()
    handles = {}
    for k, task in enumerate(pr.TASK_ORDER):
        ax = axes[k]
        _style_axis(ax, task)
        if k % 4 == 0:
            ax.set_ylabel("rel $L_2$ (eval grid)")
        for method in methods:
            rows = data.get((task, method))
            if not rows:
                continue
            x = [r[0] for r in rows]; y = [r[1] for r in rows]
            lo = [r[1] / r[2] for r in rows]; hi = [r[1] * r[2] for r in rows]
            ln = ax.plot(x, y, "o-", color=METHOD_COLOR[method], ms=4,
                         label=METHOD_LABEL[method])[0]
            ax.fill_between(x, lo, hi, color=METHOD_COLOR[method], alpha=0.15)
            handles[method] = ln
    if handles:
        hs = [handles[m] for m in methods if m in handles]
        ls = [METHOD_LABEL[m] for m in methods if m in handles]
        hs.append(plt.Line2D([], [], color="k", ls=":", lw=1.2))
        ls.append("published best (BWLer/FNO)")
        fig.legend(hs, ls, loc="upper center", bbox_to_anchor=(0.5, 0.995),
                   ncol=len(hs), frameon=False)
    fig.suptitle(title, y=0.935, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_comparators(cells, path_png, path_md):
    """Plot 2: best geomean configuration per method per task (both regimes)."""
    lines = ["# expF17 Plot 2 -- best configuration per method\n",
             "geomean rel L2 over seeds at each method's best (W, config); "
             "oracle / dynamic per cell\n"]
    fig, axes = _grid_fig()
    for k, task in enumerate(pr.TASK_ORDER):
        ax = axes[k]
        best = {}
        for regime in ("oracle", "dynamic"):
            data = _agg(cells, regime)
            for method in pr.ALL_METHODS:
                rows = data.get((task, method))
                if rows:
                    c, gm, sg, n = min(rows, key=lambda r: r[1])
                    best[(method, regime)] = (gm, sg, c)
        if best:
            ms = [m for m in pr.ALL_METHODS if (m, "oracle") in best or (m, "dynamic") in best]
            xs = np.arange(len(ms))
            for off, regime, alpha in ((-0.2, "oracle", 1.0), (0.2, "dynamic", 0.55)):
                vals = [best.get((m, regime), (np.nan,) * 3) for m in ms]
                ax.bar(xs + off, [v[0] for v in vals], width=0.38,
                       color=[METHOD_COLOR[m] for m in ms], alpha=alpha,
                       yerr=None, log=True)
            ax.set_xticks(xs)
            ax.set_xticklabels([METHOD_LABEL[m] for m in ms], rotation=40,
                               ha="right", fontsize=7)
            ax.set_ylim(*YLIM)
            ax.axhline(1e-13, color="gray", lw=0.8, ls=":")
        ax.set_title(TASK_TITLE.get(task, task), fontsize=10,
                     color=FAMILY_COLOR.get(TASK_FAMILY.get(task, ""), "k"))
        ax.grid(True, axis="y", which="both", alpha=0.2)
        lines.append(f"\n## {task}\n")
        lines.append("| method | regime | best W | geomean rel L2 | GSD |")
        lines.append("|---|---|---|---|---|")
        for (m, regime), (gm, sg, c) in sorted(best.items()):
            lines.append(f"| {METHOD_LABEL[m]} | {regime} | {c} | {gm:.2e} | {sg:.2f} |")
    fig.suptitle("Plot 2 -- best configuration per method "
                 "(solid = oracle, faded = dynamic)", y=0.985, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path_png, dpi=140)
    plt.close(fig)

    fig, axes = _grid_fig()
    for k, task in enumerate(DYSTS_ORDER):
        ax = axes[k]
        best = {}
        for regime in ("oracle", "dynamic"):
            data = _agg(cells, regime)
            for method in pr.METHODS_1D:
                rows = data.get((task, method))
                if rows:
                    c, gm, sg, n = min(rows, key=lambda r: r[1])
                    best[(method, regime)] = (gm, sg, c)
        if best:
            ms = [m for m in pr.METHODS_1D
                  if (m, "oracle") in best or (m, "dynamic") in best]
            xs = np.arange(len(ms))
            for off, regime, alpha in ((-0.2, "oracle", 1.0), (0.2, "dynamic", 0.55)):
                vals = [best.get((m, regime), (np.nan,) * 3) for m in ms]
                ax.bar(xs + off, [v[0] for v in vals], width=0.38,
                       color=[METHOD_COLOR[m] for m in ms], alpha=alpha, log=True)
            ax.set_xticks(xs)
            ax.set_xticklabels([METHOD_LABEL[m] for m in ms], rotation=40,
                               ha="right", fontsize=7)
            ax.set_ylim(*YLIM)
            ax.axhline(1e-13, color="gray", lw=0.8, ls=":")
        ax.set_title(TASK_TITLE.get(task, task), fontsize=10,
                     color=FAMILY_COLOR["dysts"])
        ax.grid(True, axis="y", which="both", alpha=0.2)
        lines.append(f"\n## {task}\n")
        lines.append("| method | regime | best W | geomean rel L2 | GSD |")
        lines.append("|---|---|---|---|---|")
        for (m, regime), (gm, sg, c) in sorted(best.items()):
            lines.append(f"| {METHOD_LABEL[m]} | {regime} | {c} | {gm:.2e} | {sg:.2f} |")
    fig.suptitle("Plot 2 (dysts page) -- best configuration per method", y=0.985,
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(str(path_png).replace(".png", "_dysts.png"), dpi=140)
    plt.close(fig)
    path_md.write_text("\n".join(lines) + "\n")


def plot_poly(cells, path):
    """Plot 3: QI arms with (dotted) and without (solid) the degree-3 block,
    horizontal line at the best spectral result. Oracle regime."""
    base = _agg(cells, "oracle", "base", methods=["qi_radon", "qi_tensor"])
    poly = _agg(cells, "oracle", "poly", methods=["qi_radon", "qi_tensor"])
    spec = _agg(cells, "oracle", "base", methods=["spectral"])
    fig, axes = _grid_fig()
    for k, task in enumerate(pr.TASK_ORDER):
        ax = axes[k]
        _style_axis(ax, task)
        if k % 4 == 0:
            ax.set_ylabel("rel $L_2$ (oracle fit)")
        for method, col in (("qi_radon", "red"), ("qi_tensor", "blue")):
            for src, ls in ((base, "-"), (poly, ":")):
                rows = src.get((task, method))
                if rows:
                    ax.plot([r[0] for r in rows], [r[1] for r in rows], ls,
                            color=col, marker="o", ms=3)
        rows = spec.get((task, "spectral"))
        if rows:
            ax.axhline(min(r[1] for r in rows), color="C1", lw=1.1, ls="--")
    fig.legend(handles=[
        plt.Line2D([], [], color="red", ls="-", label="QI-Radon"),
        plt.Line2D([], [], color="blue", ls="-", label="QI-tensor"),
        plt.Line2D([], [], color="k", ls=":", label="with deg-3 poly"),
        plt.Line2D([], [], color="C1", ls="--", label="best spectral")],
        loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=4, frameon=False)
    fig.suptitle("Plot 3 -- the polynomial block, applied to BOTH arms",
                 y=0.935, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(path, dpi=140)
    plt.close(fig)


DYSTS_ORDER = ["dysts_Lorenz", "dysts_Rossler", "dysts_Thomas",
               "dysts_Halvorsen", "dysts_Lorenz96", "dysts_InteriorSquirmer",
               "dysts_DoublePendulum", "dysts_MacArthur"]
for _k in DYSTS_ORDER:
    TASK_FAMILY[_k] = "dysts"
    TASK_TITLE[_k] = _k[len("dysts_"):]


def plot_dysts(cells, regime, path, title):
    methods = pr.METHODS_1D
    data = _agg(cells, regime, methods=methods)
    fig, axes = _grid_fig()
    handles = {}
    for k, task in enumerate(DYSTS_ORDER):
        ax = axes[k]
        _style_axis(ax, task, xlim=(32, 1100))
        if k % 4 == 0:
            ax.set_ylabel("aggregate rel $L_2$ (all components)")
        for method in methods:
            rows = data.get((task, method))
            if not rows:
                continue
            x = [r[0] for r in rows]; y = [r[1] for r in rows]
            ln = ax.plot(x, y, "o-", color=METHOD_COLOR[method], ms=4)[0]
            ax.fill_between(x, [r[1] / r[2] for r in rows],
                            [r[1] * r[2] for r in rows],
                            color=METHOD_COLOR[method], alpha=0.15)
            handles[method] = ln
    if handles:
        fig.legend([handles[m] for m in methods if m in handles],
                   [METHOD_LABEL[m] for m in methods if m in handles],
                   loc="upper center", bbox_to_anchor=(0.5, 0.995),
                   ncol=len(handles), frameon=False)
    fig.suptitle(title, y=0.935, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(path, dpi=140)
    plt.close(fig)


EXT = 1.25  # residual-canvas half-side: shows the halo margin around the box


def _best_oracle_cell(cells, task, method):
    rows = [r for r in cells.values()
            if r["task"] == task and r["method"] == method
            and r["regime"] == "oracle" and r["variant"] == "base"]
    return min(rows, key=lambda r: r["rel_l2"]) if rows else None


def residual_figs(cells, tuning):
    """Plot 4 (Sam, 2026-09-01): the error field at each task's best landed
    width, oracle regime, seed 0 refit. 2-D tasks: log10|u_hat - u*| heatmap on
    a [-EXT, EXT]^2 canvas with the train/score box drawn and everything
    outside it (and inside the poisson holes) zeroed out of the log scale
    (masked grey). 1-D tasks: |error(t)| lines per component. 4a = QI-Radon,
    4b = QI-tensor, 4c = QI(grid) on the dysts systems."""
    import numpy as np
    from . import dicts as fd, solve as sv, store as st

    def heat(ax, task, method):
        cell = _best_oracle_cell(cells, task, method)
        if cell is None:
            ax.set_axis_off()
            return
        from .tasks import TASKS
        t = TASKS[task]
        n_ax, cfg = cell["n_ax"], cell["config"]
        d = fd.build_dictionary(method, n_ax, cfg, fd.dict_rng(method, n_ax, 0),
                                fourier_x=t["periodic_fourier"])
        if task == "darcy_orig":
            a, _ = sv.darcy_orig_fit(t, d, 0)
        else:
            a, _ = sv.oracle_fit(t, d, 0)
        g = np.linspace(-EXT, EXT, 240)
        GX, GY = np.meshgrid(g, g)
        P = np.stack([GX.ravel(), GY.ravel()], axis=1)
        inside = (np.abs(P[:, 0]) <= 1.0) & (np.abs(P[:, 1]) <= 1.0)
        if t["mask"] is not None:
            inside &= t["mask"](P)
        cmap = plt.get_cmap("magma").copy()
        cmap.set_bad("0.85")
        if task == "darcy_orig":
            # truth is grid-bound: show the error on the reference grid itself
            Pe, ue = t["eval_grid"]()
            m = int(round(np.sqrt(len(Pe))))
            u_hat = d.rows(Pe, [((0, 0), 1.0)]) @ a
            Eg = np.log10(np.abs(u_hat - ue) + 1e-18).reshape(m, m)
            gx = Pe[:, 0].reshape(m, m)
            gy = Pe[:, 1].reshape(m, m)
            pc = ax.pcolormesh(gx, gy, Eg, shading="auto", cmap=cmap,
                               vmin=-17, vmax=0)
            ax.set_xlim(-EXT, EXT); ax.set_ylim(-EXT, EXT)
        else:
            E = np.full(len(P), np.nan)
            Pi = P[inside]
            u_hat = d.rows(Pi, [((0, 0), 1.0)]) @ a
            E[inside] = np.log10(np.abs(u_hat - t["exact"](Pi)) + 1e-18)
            M = np.ma.masked_invalid(E.reshape(GX.shape))
            pc = ax.pcolormesh(GX, GY, M, shading="auto", cmap=cmap,
                               vmin=-17, vmax=0)
        ax.add_patch(plt.Rectangle((-1, -1), 2, 2, fill=False, ec="k", lw=1.2))
        ax.set_aspect("equal")
        ax.set_title(f"{TASK_TITLE.get(task, task)}  W={cell['cols']}  "
                     f"rel={cell['rel_l2']:.1e}", fontsize=8,
                     color=FAMILY_COLOR.get(TASK_FAMILY.get(task, "other"), "k"))
        return pc

    for method, fname, label in [("qi_radon", "plot4a_residual_radon.png", "QI-Radon"),
                                 ("qi_tensor", "plot4b_residual_tensor.png", "QI-tensor")]:
        fig, axes = _grid_fig()
        pc = None
        for k, task in enumerate(pr.TASK_ORDER):
            r = heat(axes[k], task, method)
            pc = r or pc
        if pc is not None:
            fig.colorbar(pc, ax=list(axes), shrink=0.7,
                         label=r"$\log_{10}|\hat u - u^*|$ (grey = outside the box)")
        fig.suptitle(f"Plot 4 -- residual fields, {label}, best landed W "
                     "(oracle, seed 0)", y=0.98, fontsize=12)
        fig.savefig(st.RESULTS_DIR / fname, dpi=140)
        plt.close(fig)

    # 4c: dysts, QI(grid): per-component |error(t)| at the best landed W
    from . import dicts1d as f1, solve1d as s1
    from .tasks1d import dysts_task
    fig, axes = _grid_fig()
    for k, key in enumerate(DYSTS_ORDER):
        ax = axes[k]
        cell = _best_oracle_cell(cells, key, "qi_grid")
        if cell is None:
            ax.set_axis_off()
            continue
        t = dysts_task(key[len("dysts_"):])
        d = f1.build_dictionary_1d("qi_grid", cell["C"], cell["config"],
                                   fd.dict_rng("qi_grid", cell["C"], 0))
        A, sigma, _ = s1.oracle_fit_1d(t, d, 0)
        srel = 2.0 * t["ts"] / t["T"] - 1.0
        E = np.abs((d.rows(srel, 0) @ A.T) * sigma[None, :] - t["Yref"]) + 1e-18
        for c in range(E.shape[1]):
            ax.semilogy(srel, E[:, c], lw=0.7)
        ax.axvline(-1.0, color="k", lw=1.0)
        ax.axvline(1.0, color="k", lw=1.0)
        ax.set_xlim(-EXT, EXT)
        ax.set_ylim(1e-17, 1e0)
        ax.grid(True, which="both", alpha=0.2)
        ax.set_title(f"{TASK_TITLE.get(key, key)}  W={cell['cols']}  "
                     f"rel={cell['rel_l2']:.1e}", fontsize=8,
                     color=FAMILY_COLOR["dysts"])
    fig.suptitle("Plot 4c -- |error(t)| per component, QI (grid), best landed W "
                 "(oracle, seed 0)", y=0.98, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(st.RESULTS_DIR / "plot4c_residual_dysts.png", dpi=140)
    plt.close(fig)


def write_status(cells):
    q = pr.build_queue()
    landed = sum(1 for e in q if pr.cell_key(e["task"], e["method"], e["C"],
                 e["seed"], e["regime"], e["variant"]) in cells)
    by = defaultdict(lambda: [0, 0])
    for e in q:
        k = (e["regime"], e["variant"])
        by[k][1] += 1
        if pr.cell_key(e["task"], e["method"], e["C"], e["seed"], e["regime"],
                       e["variant"]) in cells:
            by[k][0] += 1
    lines = [f"# expF17 status  ({time.strftime('%Y-%m-%d %H:%M:%S')})",
             f"\n{landed}/{len(q)} core cells landed\n"]
    for (regime, variant), (a, b) in sorted(by.items()):
        lines.append(f"- {regime}/{variant}: {a}/{b}")
    (store.RESULTS_DIR / "status.md").write_text("\n".join(lines) + "\n")


def replot():
    cells = store.load()
    if not cells:
        write_status(cells)
        return
    d = store.RESULTS_DIR
    plot_scaling(cells, "oracle", d / "plot1a_oracle.png",
                 pr.ALL_METHODS, "Plot 1a -- oracle regime (fit the known solution)")
    plot_scaling(cells, "dynamic", d / "plot1b_dynamic.png",
                 pr.ALL_METHODS, "Plot 1b -- dynamic regime (solve the PDE)")
    plot_comparators(cells, d / "plot2_best.png", d / "plot2_table.md")
    plot_poly(cells, d / "plot3_poly.png")
    plot_dysts(cells, "oracle", d / "plot1a_dysts_oracle.png",
               "Plot 1a (dysts page) -- oracle regime (fit the reference)")
    plot_dysts(cells, "dynamic", d / "plot1b_dysts_dynamic.png",
               "Plot 1b (dysts page) -- dynamic regime (collocation ODE solve)")
    write_status(cells)
