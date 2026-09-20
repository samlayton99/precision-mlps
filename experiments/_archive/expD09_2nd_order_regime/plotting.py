"""expD09 figures. Every panel is a fixed log axis 1e-16..1e1, no autoscale.

Per method (`<solver_id>.png`): a 3x3 grid, rows = width N, columns = target.
Each panel carries the three curves Sam asked for --

  solid   train rel L2 = ||A a - y|| / ||y||        the objective minimized
  dashed  eval  rel L2 = ||A_ev a - y_ev|| / ||y_ev||   what we care about
  dotted  the truncated-SVD floor (black = eval, grey = train)

Plus two cross-method figures:
  compare_eval.png   every solver's eval curve, same 3x3
  null_drift.png     ||P_null a|| per sweep -- the mechanism behind any
                     train/eval gap (null(A_train) is not null for A_eval)
"""
from __future__ import annotations

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from common import FIG_DIR, TARGETS, WIDTHS, YLIM, load_manifest, load_runs  # noqa: E402
import solvers  # noqa: E402

TOP = YLIM[1]
CMAP = plt.get_cmap("turbo")


def _grid(figsize=(14.5, 10.0)):
    fig, axes = plt.subplots(len(WIDTHS), len(TARGETS), figsize=figsize,
                             sharex=True, sharey=True)
    for i, N in enumerate(WIDTHS):
        for j, fn in enumerate(TARGETS):
            ax = axes[i, j]
            ax.set_yscale("log")
            ax.set_ylim(*YLIM)
            ax.grid(True, which="both", alpha=0.25)
            if i == 0:
                ax.set_title(fn, fontsize=11)
            if i == len(WIDTHS) - 1:
                ax.set_xlabel("sweep  (~one pass over $\\Phi$)")
            if j == 0:
                ax.set_ylabel(f"$N={N}$\nrelative $L_2$", fontsize=10)
    return fig, axes


def _plot_clipped(ax, y, **kw):
    """Draw a curve, pegging anything above the axis top with markers so a
    divergence reads as divergence instead of as a run that stopped."""
    y = np.asarray(y, dtype=float)
    x = np.arange(1, y.size + 1)
    ax.plot(x, np.minimum(np.nan_to_num(y, nan=TOP, posinf=TOP), TOP * 0.98), **kw)
    out = np.nonzero(~(y <= TOP))[0]
    if out.size:
        ax.scatter(out + 1, np.full(out.size, TOP * 0.98), s=4, marker="^",
                   color=kw.get("color", "k"), zorder=5)


def _cell(runs, fn, N, solver=None):
    for r in runs:
        if r["fn"] == fn and r["N"] == N and (solver is None
                                              or r["solver"] == solver):
            return r
    return None


def render_method(runs, spec, path):
    fig, axes = _grid()
    for i, N in enumerate(WIDTHS):
        for j, fn in enumerate(TARGETS):
            ax = axes[i, j]
            r = _cell(runs, fn, N, spec["id"])
            if r is None:
                continue
            ax.axhline(r["floor_train"], color="0.55", ls=(0, (1, 2)), lw=1.1,
                       zorder=1)
            ax.axhline(r["floor_eval"], color="k", ls=":", lw=1.4, zorder=1)
            _plot_clipped(ax, r["train_rel"], color="tab:blue", lw=1.4,
                          ls="-", zorder=3)
            _plot_clipped(ax, r["eval_rel"], color="tab:red", lw=1.3,
                          ls="--", zorder=4)
            ax.text(0.97, 0.94, f"best eval {r['best_eval_rel']:.1e}",
                    transform=ax.transAxes, ha="right", va="top", fontsize=7,
                    color="#333")
    handles = [
        Line2D([0], [0], color="tab:blue", lw=1.6,
               label=r"train rel $L_2$ (the objective)"),
        Line2D([0], [0], color="tab:red", lw=1.4, ls="--",
               label=r"eval rel $L_2$ (4001 oversampled points)"),
        Line2D([0], [0], color="k", ls=":", lw=1.4,
               label="SVD floor -- eval"),
        Line2D([0], [0], color="0.55", ls=(0, (1, 2)), lw=1.2,
               label="SVD floor -- train"),
        Line2D([0], [0], marker="^", ls="", color="0.4", ms=5,
               label="pegged at axis top (diverged)"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.965),
               ncol=5, frameon=False, fontsize=9)
    n_over = spec.get("k")
    cost = ("" if n_over is None else
            f"   [block $k={n_over}$, {solvers.flops_per_sweep(spec):.0f}"
            r"$\times$ a matvec of flops per sweep]")
    fig.suptitle(f"expD09 round 1 -- {spec['label']}{cost}", y=0.995,
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"  saved {path}")


def render_compare(runs, specs, path, key="eval_rel", title=None):
    fig, axes = _grid()
    colors = {s["id"]: CMAP(t) for s, t in
              zip(specs, np.linspace(0.05, 0.95, max(len(specs), 2)))}
    for i, N in enumerate(WIDTHS):
        for j, fn in enumerate(TARGETS):
            ax = axes[i, j]
            drew_floor = False
            for spec in specs:
                r = _cell(runs, fn, N, spec["id"])
                if r is None:
                    continue
                if not drew_floor:
                    ax.axhline(r["floor_eval"], color="k", ls=":", lw=1.4,
                               zorder=1)
                    drew_floor = True
                _plot_clipped(ax, r[key], color=colors[spec["id"]], lw=1.3,
                              zorder=3)
    handles = [Line2D([0], [0], color=colors[s["id"]], lw=1.6, label=s["label"])
               for s in specs]
    handles.append(Line2D([0], [0], color="k", ls=":", lw=1.4,
                          label="SVD floor -- eval"))
    ncol = 3
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.975),
               ncol=ncol, frameon=False, fontsize=9)
    fig.suptitle(title or "expD09 round 1 -- eval relative $L_2$, all methods",
                 y=1.0, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


def render_null_drift(runs, specs, path):
    fig, axes = _grid()
    for ax in axes.flat:
        ax.set_ylim(1e-16, 1e6)
    colors = {s["id"]: CMAP(t) for s, t in
              zip(specs, np.linspace(0.05, 0.95, max(len(specs), 2)))}
    for i, N in enumerate(WIDTHS):
        for j, fn in enumerate(TARGETS):
            ax = axes[i, j]
            for spec in specs:
                r = _cell(runs, fn, N, spec["id"])
                if r is None:
                    continue
                ax.plot(np.arange(1, len(r["null_norm"]) + 1),
                        np.maximum(r["null_norm"], 1e-17),
                        color=colors[spec["id"]], lw=1.3)
            if j == 0:
                ax.set_ylabel(f"$N={N}$\n" r"$\|P_{\rm null}a\|$", fontsize=10)
    handles = [Line2D([0], [0], color=colors[s["id"]], lw=1.6, label=s["label"])
               for s in specs]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.975),
               ncol=3, frameon=False, fontsize=9)
    fig.suptitle("expD09 -- null-space drift: the component of the readout the "
                 "train system cannot see\n(min-norm SVD keeps it at 0; a "
                 "coordinate block does not preserve range($\\Phi^\\top$))",
                 y=1.02, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


def render_spectrum(path, k=64):
    """DIAGNOSTIC, run before any optimizer: where does the conditioning live?

    For each problem, the spectrum of the preconditioned Gram
    P^{-1/2} (A^T A) P^{-1/2} restricted to the numerical row space, for
    P = I, diag(G), block-Jacobi with CONTIGUOUS blocks, block-Jacobi with
    RANDOM blocks. Whichever P collapses the spectrum is the geometry the
    solver should exploit; if contiguous collapses it and random does not,
    the ill-conditioning is local in the column index and block SELECTION
    is the lever.
    """
    from common import PROBLEMS_DIR, build_phi_aug
    manifest = load_manifest()
    fig, axes = _grid(figsize=(14.5, 9.5))
    for ax in axes.flat:
        ax.set_ylim(1e-17, 3e0)
    for ax in axes[-1]:
        ax.set_xlabel("index $i$")
    styles = [("none", "0.25", "-"), ("diag", "tab:green", "-"),
              (f"block-Jacobi contiguous $k={k}$", "tab:blue", "-"),
              (f"block-Jacobi random $k={k}$", "tab:red", "--")]
    rng = np.random.default_rng(0)
    for i, N in enumerate(WIDTHS):
        for j, fn in enumerate(TARGETS):
            ax = axes[i, j]
            entry = next(e for e in manifest if e["fn"] == fn and e["N"] == N)
            z = np.load(PROBLEMS_DIR / entry["path"])
            A = build_phi_aug(z["x_tr"], float(z["gamma"]), z["centers"])
            d = A.shape[1]
            # EVERYTHING stays in the ORIGINAL column coordinates. Building
            # the preconditioner in the singular basis instead would make
            # "diagonal" the exact inverse and the whole diagnostic vacuous.
            G = A.T @ A
            gmax = float(np.max(np.diag(G)))
            for name, color, ls in styles:
                if name == "none":
                    sv = np.linalg.svd(A, compute_uv=False)
                else:
                    P = np.zeros_like(G)
                    if name == "diag":
                        P[np.diag_indices_from(P)] = np.diag(G)
                    else:
                        idx = (np.arange(d) if "contiguous" in name
                               else rng.permutation(d))
                        for t in range(0, d, k):
                            C = idx[t:t + k]
                            P[np.ix_(C, C)] = G[np.ix_(C, C)]
                    w, Q = np.linalg.eigh(P)
                    Mi = Q @ np.diag(np.maximum(w, 1e-14 * gmax) ** -0.5) @ Q.T
                    # Singular values of A M^-1 computed from A DIRECTLY. The
                    # obvious shortcut -- eigenvalues of M^-1 (A^T A) M^-1 --
                    # is wrong here and was wrong in the first version of this
                    # figure: forming A^T A in fp64 floors its small
                    # eigenvalues at eps*||A||^2 ~ 1e-10, against a true
                    # sigma_r^2 ~ 1e-24, so every reported kappa was really
                    # the fp64 noise floor of the Gram.
                    sv = np.linalg.svd(A @ Mi, compute_uv=False)
                ax.semilogy(np.arange(sv.size),
                            np.maximum(sv / sv[0], 1e-18), ls, color=color,
                            lw=1.4)
            ax.axvline(entry["rank"], color="0.6", ls="-.", lw=1.0)
            if j == 0:
                ax.set_ylabel(f"$N={N}$\n" r"$\sigma_i/\sigma_1$ (preconditioned)",
                              fontsize=9)
    handles = [Line2D([0], [0], color=c, ls=ls, lw=1.6, label=f"$P$ = {nm}")
               for nm, c, ls in styles]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.975),
               ncol=4, frameon=False, fontsize=9)
    fig.suptitle("expD09 diagnostic -- spectrum of the PRECONDITIONED system "
                 "$P^{-1/2}\\Phi^\\top\\Phi P^{-1/2}$, in ORIGINAL column "
                 "coordinates\n(flatter = better conditioned; grey dash-dot "
                 "= numerical rank of $[\\Phi,\\mathbf{1}]$)",
                 y=1.03, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


def contraction(r, tail=0.5):
    """Per-sweep contraction ratio of the train objective over the tail.

    BCD on a convex quadratic converges linearly, so the asymptotic ratio
    rho is the whole story: it says how many more sweeps the remaining
    orders cost. Measured on the tail so the fast transient is excluded.
    """
    tr = np.asarray(r["train_rel"], dtype=float)
    i0 = int(len(tr) * (1.0 - tail))
    span = len(tr) - 1 - i0
    if span <= 0 or not np.isfinite(tr[-1]) or tr[i0] <= 0 or tr[-1] <= 0:
        return np.nan, np.nan
    rho = (tr[-1] / tr[i0]) ** (1.0 / span)
    if not (0.0 < rho < 1.0):
        return rho, np.inf
    need = np.log(max(r["floor_train"], 1e-16) / tr[-1]) / np.log(rho)
    return rho, max(need, 0.0)


def render_contraction(runs, specs, path):
    """The extrapolation that makes a 'converging' method legible: at the
    MEASURED per-sweep contraction rate, how many more sweeps until the
    train objective reaches its SVD floor?"""
    fig, axes = _grid(figsize=(14.5, 9.5))
    fams = [("contig", "tab:blue", "-o"), ("rand", "tab:red", "--s")]
    for ax in axes.flat:
        ax.set_ylim(1e0, 1e12)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("block size $k$")
    for i, N in enumerate(WIDTHS):
        for j, fn in enumerate(TARGETS):
            ax = axes[i, j]
            for kind, color, style in fams:
                pts = []
                for spec in specs:
                    if spec.get("blocks") != kind or spec["id"].endswith(
                            ("_damped", "_carried")):
                        continue
                    r = _cell(runs, fn, N, spec["id"])
                    if r is None:
                        continue
                    _, need = contraction(r)
                    pts.append((spec["k"], need))
                if pts:
                    pts.sort()
                    ax.plot([p[0] for p in pts],
                            [min(max(p[1], 1.0), 1e12) for p in pts],
                            style, color=color, lw=1.6, ms=5)
            ax.axhline(1e3, color="0.4", ls=":", lw=1.1)
            if j == 0:
                ax.set_ylabel(f"$N={N}$\nsweeps still needed", fontsize=10)
    handles = [Line2D([0], [0], color=c, lw=1.7, ls=st[0] if st[0] != "-" else "-",
                      marker=st[-1], label=f"{nm} blocks")
               for nm, c, st in fams]
    handles.append(Line2D([0], [0], color="0.4", ls=":", lw=1.1,
                          label="$10^3$ sweeps (a generous budget)"))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.975),
               ncol=3, frameon=False, fontsize=9)
    fig.suptitle("expD09 round 1 -- sweeps still needed to reach the SVD floor "
                 "at the MEASURED per-sweep contraction rate\n"
                 "(linear convergence on a convex quadratic: the tail ratio "
                 r"$\rho$ fixes the remaining cost)", y=1.02, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


def render_all():
    runs = load_runs()
    have = {r["solver"] for r in runs}
    specs = [s for s in solvers.SOLVERS if s["id"] in have]
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for spec in specs:
        render_method(runs, spec, FIG_DIR / f"{spec['id']}.png")
    render_compare(runs, specs, FIG_DIR / "compare_eval.png")
    render_compare(runs, specs, FIG_DIR / "compare_train.png", key="train_rel",
                   title="expD09 round 1 -- TRAIN relative $L_2$, all methods")
    render_null_drift(runs, specs, FIG_DIR / "null_drift.png")
    render_contraction(runs, specs, FIG_DIR / "contraction.png")
    render_spectrum(FIG_DIR / "block_spectrum.png")
