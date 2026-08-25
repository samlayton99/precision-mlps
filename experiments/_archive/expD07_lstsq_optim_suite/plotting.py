"""expD07 figures, regenerated from runs.jsonl after every finished run.

Every panel is in the repo's standard unit: relative L2, on a FIXED log scale
[1e-16, 1] (machine epsilon to O(1)) so all figures are directly comparable
and never rescale as runs land. Lower = better everywhere.

  synthetic problems: train relative L2 = ||A v - y|| / ||y||
  phi problems:       eval relative L2 = ||Phi_ev v - f|| / ||f|| (dense grid)
  (eval L_inf is recorded in runs.jsonl but not plotted.)

Grading is BEST-OVER-TRAJECTORY (min across checkpoints), not the final
iterate: early stopping is free, so an optimizer is judged by the best point
it ever visits. Trajectories are drawn as the running minimum for the same
reason. Profiles already use first-hit semantics.

  ranking.png      -- barh per optimizer: geo-mean relative L2, synthetic |
                      phis, with the lstsq-floor geo-mean as a dashed line.
  profiles.png     -- fraction of problems driven below a rel-L2 tolerance
                      vs grad-eval budget (how-far AND how-fast; the
                      dominating curve wins).
  heatmap.png      -- N=256 problems only, median over row-ratios. Rows
                      ordered best -> worst top -> bottom per panel, with the
                      lstsq (SVD) floor as the top row. Two vertically
                      stacked panels (synthetic / phis), one fixed scale.
  trajectories.png -- median rel-L2 trajectory per class, lstsq floor dashed.

Floors come from the MANIFEST (not the cached run records), so rebuilding the
baseline (e.g. a different rcond) re-renders every figure without redoing a
single optimizer run. All legends sit above the axes (house rule).
"""
from __future__ import annotations

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from optimizers import OPTIMIZERS  # noqa: E402

REL_L2_LIM = (1e-16, 1.0)      # fixed axis: machine epsilon .. O(1)
LOG_LIM = (-16.0, 0.0)         # same, in log10 (heatmap color scale)
TOLS = [1e-6, 1e-12]           # profile tolerances (rel L2)
FLOOR_STYLE = dict(color="k", ls="--", lw=1.4)


def _clamp(vals):
    return np.maximum(np.asarray(vals, dtype=float), REL_L2_LIM[0])


def _geo(vals):
    return float(np.exp(np.mean(np.log(_clamp(vals)))))


def _best(r):
    """Best (min) rel L2 over the trajectory -- early stopping is free."""
    key = "rel_l2" if r["kind"] == "synthetic" else "eval_rel_l2"
    return float(np.min(r["traj"][key]))


def floor_lookup(manifest):
    """problem_id -> baseline rel L2 (synthetic: train sqrt(L*/L0);
    phi: eval rel L2 of the truncated-SVD solve)."""
    floors = {}
    for e in manifest:
        if e["kind"] == "synthetic":
            floors[e["id"]] = float(np.sqrt(e["L_star"] / e["L0"]))
        else:
            floors[e["id"]] = e["floor_eval_rel_l2"]
    return floors


def _floor(r, floors):
    if r["problem_id"] in floors:
        return floors[r["problem_id"]]
    return r["floor_rel_l2"] if r["kind"] == "synthetic" else r["floor_eval_rel_l2"]


def _traj_key(kind):
    return "rel_l2" if kind == "synthetic" else "eval_rel_l2"


def _opt_meta(runs):
    cmap = plt.get_cmap("tab20")
    meta = {s["id"]: {"label": s["label"], "color": cmap(i % 20)}
            for i, s in enumerate(OPTIMIZERS)}
    for r in runs:  # runs from configs since removed from the registry
        if r["opt_id"] not in meta:
            meta[r["opt_id"]] = {"label": r["opt_id"], "color": "0.6"}
    return meta


def _opt_order(runs, meta):
    seen = {r["opt_id"] for r in runs}
    return [oid for oid in meta if oid in seen]


def _classes(runs, kind):
    order = []
    for r in runs:
        if r["kind"] == kind and r["cls"] not in order:
            order.append(r["cls"])
    return order


def fig_ranking(runs, meta, order, floors, out):
    fig, axes = plt.subplots(1, 2, figsize=(13, 0.9 + 0.55 * max(3, len(order))))
    for ax, kind, title in [(axes[0], "synthetic", "synthetic (train)"),
                            (axes[1], "phi", "phis (eval)")]:
        rr = [r for r in runs if r["kind"] == kind]
        vals = [(oid, _geo([_best(r) for r in rr if r["opt_id"] == oid]))
                for oid in order if any(r["opt_id"] == oid for r in rr)]
        if not vals:
            continue
        vals.sort(key=lambda t: t[1])                    # best (smallest) first
        ypos = np.arange(len(vals))
        ax.barh(ypos, [v for _, v in vals],
                color=[meta[o]["color"] for o, _ in vals])
        ax.set_yticks(ypos)
        ax.set_yticklabels([meta[o]["label"] for o, _ in vals])
        ax.invert_yaxis()
        for i, (_, v) in enumerate(vals):
            ax.text(v, i, f" {v:.1e}", va="center")
        floor = _geo([_floor(r, floors) for r in rr])
        ax.axvline(floor, **FLOOR_STYLE)
        ax.text(floor, len(vals) - 0.45, " lstsq floor (geo-mean)", fontsize=8,
                rotation=90, va="bottom", ha="right")
        ax.set_xscale("log")
        ax.set_xlim(*REL_L2_LIM)
        ax.set_xlabel(f"{title}: geo-mean BEST relative $L_2$ over the "
                      "trajectory (lower = better)")
        ax.grid(True, axis="x", which="both", alpha=0.3)
    fig.suptitle("expD07 ranking -- geo-mean best-over-trajectory relative "
                 r"$L_2$, fixed scale $[10^{-16}, 1]$")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out, dpi=140)
    plt.close(fig)


def _first_reach(steps, vals, tol):
    for s, val in zip(steps, vals):
        if val <= tol:
            return s
    return None


def fig_profiles(runs, meta, order, out):
    panels = [(kind, tol) for kind in ("synthetic", "phi") for tol in TOLS]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.4), sharey=True)
    for ax, (kind, tol) in zip(axes.ravel(), panels):
        rr = [r for r in runs if r["kind"] == kind]
        key = _traj_key(kind)
        for oid in order:
            mine = [r for r in rr if r["opt_id"] == oid]
            if not mine:
                continue
            budgets = np.array([s for s in mine[0]["traj"]["step"] if s >= 1])
            firsts = [_first_reach(r["traj"]["step"], r["traj"][key], tol)
                      for r in mine]
            frac = [np.mean([f is not None and f <= b for f in firsts])
                    for b in budgets]
            ax.step(budgets, frac, where="post", color=meta[oid]["color"],
                    label=meta[oid]["label"])
        ax.set_xscale("log")
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("grad-eval budget")
        name = "synthetic (train)" if kind == "synthetic" else "phis (eval)"
        ax.set_title(f"{name}: rel $L_2 \\leq$ {tol:g}", fontsize=11)
        ax.grid(True, alpha=0.3, which="both")
    for ax in axes[:, 0]:
        ax.set_ylabel("fraction of problems solved")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.suptitle("expD07 performance profiles -- relative $L_2$ tolerances",
                 y=0.99, va="top")
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.955),
               ncol=min(len(labels), 5), frameon=True, borderaxespad=0)
    fig.savefig(out, dpi=140)
    plt.close(fig)


def fig_heatmap(runs, meta, order, floors, out, heat_n=256):
    """Two vertically stacked panels (synthetic / phis), one fixed color
    scale: median log10 relative L2 at N=heat_n (median over row-ratios).
    Rows sorted best -> worst top -> bottom PER PANEL, lstsq (SVD) floor on
    top."""
    rr_n = [r for r in runs if r["N"] == heat_n] or runs
    syn_cls = _classes(rr_n, "synthetic")
    phi_cls = _classes(rr_n, "phi")
    nrow = len(order) + 1  # + floor row
    fig, axes = plt.subplots(
        2, 1, figsize=(1.15 * max(len(syn_cls), len(phi_cls), 6) + 3,
                       2 * (1.2 + 0.55 * nrow)))

    def _panel(ax, kind, classes, title):
        rr = [r for r in rr_n if r["kind"] == kind]
        # per-optimizer per-class median (log10), then sort rows best-first
        cell = {}
        for oid in order:
            for cls in classes:
                vals = [_best(r) for r in rr
                        if r["opt_id"] == oid and r["cls"] == cls]
                if vals:
                    cell[(oid, cls)] = np.log10(np.median(_clamp(vals)))
        present = [oid for oid in order
                   if any((oid, c) in cell for c in classes)]
        present.sort(key=lambda oid: np.mean(
            [cell[(oid, c)] for c in classes if (oid, c) in cell]))
        nrow_p = len(present) + 1  # rows actually drawn in THIS panel
        grid = np.full((nrow_p, len(classes)), np.nan)
        for j, cls in enumerate(classes):
            fl = [_floor(r, floors) for r in rr if r["cls"] == cls]
            if fl:
                grid[0, j] = np.log10(np.median(_clamp(fl)))
            for i, oid in enumerate(present):
                grid[i + 1, j] = cell.get((oid, cls), np.nan)
        im = ax.imshow(grid, cmap="viridis_r", vmin=LOG_LIM[0],
                       vmax=LOG_LIM[1], aspect="auto")
        ax.set_xticks(range(len(classes)))
        ax.set_xticklabels(classes, rotation=30, ha="right", fontsize=9)
        ax.set_yticks(range(nrow_p))
        ax.set_yticklabels(["lstsq floor (SVD)"]
                           + [meta[o]["label"] for o in present], fontsize=9)
        ax.axhline(0.5, color="w", lw=2)
        for i in range(nrow_p):
            for j in range(len(classes)):
                if np.isfinite(grid[i, j]):
                    ax.text(j, i, f"{grid[i, j]:.1f}", ha="center",
                            va="center", fontsize=8,
                            color="k" if grid[i, j] < -8 else "w")
        ax.set_title(title, fontsize=11)
        return im

    im = _panel(axes[0], "synthetic", syn_cls,
                f"synthetic: median $\\log_{{10}}$ best train relative $L_2$ "
                f"(N={heat_n})")
    _panel(axes[1], "phi", phi_cls,
           f"phis: median $\\log_{{10}}$ best eval relative $L_2$ (N={heat_n})")
    fig.suptitle("expD07 heatmap -- median $\\log_{10}$ best-over-trajectory "
                 f"relative $L_2$ at N={heat_n} (rows best $\\to$ worst; "
                 "lower = better; fixed scale $-16$..$0$)")
    fig.tight_layout(rect=[0, 0, 0.92, 0.95])
    cax = fig.add_axes([0.94, 0.15, 0.015, 0.7])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label(r"$\log_{10}$ relative $L_2$")
    fig.savefig(out, dpi=140)
    plt.close(fig)


def fig_trajectories(runs, meta, order, floors, out):
    syn_cls = _classes(runs, "synthetic")
    phi_cls = _classes(runs, "phi")
    panels = [("synthetic", c) for c in syn_cls] + [("phi", c) for c in phi_cls]
    if not panels:
        return
    ncols = 4
    nrows = int(np.ceil(len(panels) / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(4.6 * ncols, 3.4 * nrows), squeeze=False)

    def _median_curve(rr, key):
        ref = np.array([s for s in rr[0]["traj"]["step"] if s >= 1], dtype=float)
        curves = []
        for r in rr:
            steps = np.array(r["traj"]["step"], dtype=float)
            # running minimum: grade on the best point visited so far
            vals = np.minimum.accumulate(_clamp(r["traj"][key]))
            keep = steps >= 1
            curves.append(np.interp(ref, steps[keep], vals[keep]))
        return ref, np.median(np.array(curves), axis=0)

    for ax, (kind, cls) in zip(axes.ravel(), panels):
        rr_cls = [r for r in runs if r["kind"] == kind and r["cls"] == cls]
        key = _traj_key(kind)
        for oid in order:
            mine = [r for r in rr_cls if r["opt_id"] == oid]
            if not mine:
                continue
            steps, med = _median_curve(mine, key)
            ax.loglog(steps, med, color=meta[oid]["color"],
                      label=meta[oid]["label"])
        fl = [_floor(r, floors) for r in rr_cls]
        if fl:
            ax.axhline(float(np.median(_clamp(fl))), **FLOOR_STYLE,
                       label="lstsq floor (median)")
        ax.set_ylim(*REL_L2_LIM)
        ax.set_title(f"{kind}: {cls}", fontsize=10)
        ax.set_xlabel("grad evals")
        ax.set_ylabel("best train rel $L_2$" if kind == "synthetic"
                      else "best eval rel $L_2$")
        ax.grid(True, alpha=0.3, which="both")
    for ax in axes.ravel()[len(panels):]:
        ax.set_visible(False)

    handles = {}
    for ax in axes.ravel()[: len(panels)]:
        for h, lab in zip(*ax.get_legend_handles_labels()):
            if lab not in handles:
                handles[lab] = h
    labels = list(handles)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.suptitle("expD07 median best-so-far relative-$L_2$ trajectories by "
                 r"problem class (running min; fixed scale $[10^{-16}, 1]$)",
                 y=0.995, va="top")
    fig.legend([handles[lab] for lab in labels], labels, loc="upper center",
               bbox_to_anchor=(0.5, 0.975), ncol=min(len(labels), 6),
               frameon=True, borderaxespad=0)
    fig.savefig(out, dpi=130)
    plt.close(fig)


def make_all(runs, fig_dir, manifest):
    if not runs:
        return
    fig_dir.mkdir(parents=True, exist_ok=True)
    meta = _opt_meta(runs)
    order = _opt_order(runs, meta)
    floors = floor_lookup(manifest)
    fig_ranking(runs, meta, order, floors, fig_dir / "ranking.png")
    fig_profiles(runs, meta, order, fig_dir / "profiles.png")
    fig_heatmap(runs, meta, order, floors, fig_dir / "heatmap.png")
    fig_trajectories(runs, meta, order, floors, fig_dir / "trajectories.png")
