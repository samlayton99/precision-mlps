"""expD09 validation figures.

Every panel uses a fixed log axis (no autoscale). Legends sit OUTSIDE the
axes, above. Ordered quantities (k, noise, batch size) use sequential
colormaps so the trend is readable without consulting the legend.
"""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

import common_val as cv  # noqa: E402

TOP = cv.YLIM[1]


def _grid(nrow, ncol, figsize, ylab, xlab, titles_top, labels_left):
    fig, axes = plt.subplots(nrow, ncol, figsize=figsize, sharex=True,
                             sharey=True, squeeze=False)
    for i in range(nrow):
        for j in range(ncol):
            ax = axes[i, j]
            ax.set_yscale("log")
            ax.set_ylim(*cv.YLIM)
            ax.grid(True, which="both", alpha=0.25)
            if i == 0:
                ax.set_title(titles_top[j], fontsize=11)
            if i == nrow - 1:
                ax.set_xlabel(xlab)
            if j == 0:
                ax.set_ylabel(f"{labels_left[i]}\n{ylab}", fontsize=9)
    return fig, axes


def _x(vals, stride):
    return (np.arange(1, len(vals) + 1)) * stride


def _clip(y):
    y = np.asarray(y, dtype=float)
    return np.minimum(np.nan_to_num(y, nan=TOP, posinf=TOP), TOP * 0.98)


def _finish(fig, path, handles, title, ncol=4, rect_top=0.88, y=1.0):
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, y),
               ncol=ncol, frameon=False, fontsize=9)
    fig.suptitle(title, y=y + 0.045, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, rect_top])
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


# ------------------------------------------------------------------ A ------

def fig_A(rows):
    ks = sorted({r["k"] for r in rows})
    cmap = plt.get_cmap("viridis")
    col = {k: cmap(t) for k, t in zip(ks, np.linspace(0.1, 0.85, len(ks)))}
    fig, axes = _grid(len(cv.WIDTHS), len(cv.TARGETS), (14.5, 10),
                      r"relative $L_2$", "LSQR iteration",
                      cv.TARGETS, [f"$N={n}$" for n in cv.WIDTHS])
    for i, N in enumerate(cv.WIDTHS):
        for j, fn in enumerate(cv.TARGETS):
            ax = axes[i, j]
            drew = False
            for k in ks:
                r = next((r for r in rows if r["fn"] == fn and r["N"] == N
                          and r["k"] == k), None)
                if r is None:
                    continue
                if not drew:
                    ax.axhline(r["floor"], color="k", ls=":", lw=1.5, zorder=1)
                    drew = True
                x = _x(r["eval"], r["stride"])
                ax.plot(x, _clip(r["eval"]), color=col[k], lw=1.7, zorder=3)
            ax.set_xscale("log")
    handles = [Line2D([0], [0], color=col[k], lw=1.8, label=f"$k={k}$")
               for k in ks]
    handles += [Line2D([0], [0], color="k", ls=":", lw=1.7,
                       label="truncated-SVD floor")]
    _finish(fig, cv.FIGS / "A_trajectories_3x3.png", handles,
            "A -- block-QR whitening + LSQR: eval relative $L_2$ vs iteration.\n"
            "Rows = width, columns = target, colour = block size $k$. "
            "(Train residual is not drawn: it lies exactly on the eval curve.)\n"
            "NOTE: fixed iteration budget -- the $k$ ordering here is "
            "budget-limited, see G_uncoupling_corrected.png", ncol=4)


# ------------------------------------------------------------------ B ------

def fig_B(rows):
    fig = plt.figure(figsize=(14.5, 9.5))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.15], hspace=0.42,
                          wspace=0.25)
    Ns = sorted({r["N"] for r in rows})
    cmap = plt.get_cmap("plasma")
    col = {N: cmap(t) for N, t in zip(Ns, np.linspace(0.05, 0.8, len(Ns)))}
    # top: error vs k, one panel per target, one line per width
    for j, fn in enumerate(cv.TARGETS):
        ax = fig.add_subplot(gs[0, j])
        for N in Ns:
            rs = sorted([r for r in rows if r["fn"] == fn and r["N"] == N],
                        key=lambda r: r["k"])
            if not rs:
                continue
            ax.loglog([r["k"] for r in rs], [max(r["best"], 1e-17) for r in rs],
                      "-o", color=col[N], ms=4, lw=1.6)
            ax.axhline(rs[0]["floor"], color=col[N], ls=":", lw=1.0, alpha=0.6)
        ax.set_ylim(*cv.YLIM)
        ax.set_xlabel("block size $k$")
        ax.set_title(fn, fontsize=11)
        ax.grid(True, which="both", alpha=0.25)
        if j == 0:
            ax.set_ylabel(r"best eval rel $L_2$")
    # bottom-left: k* vs d -- the O(d) question
    ax = fig.add_subplot(gs[1, :2])
    marks = {"sine": "o", "sine_8pi": "s", "runge": "^"}
    for fn in cv.TARGETS:
        pts = []
        for N in Ns:
            rs = sorted([r for r in rows if r["fn"] == fn and r["N"] == N],
                        key=lambda r: r["k"])
            if not rs:
                continue
            fl = rs[0]["floor"]
            hit = [r for r in rs if r["best"] <= 10 * fl]
            if hit:
                pts.append((rs[0]["d"], hit[0]["k"]))
        if pts:
            pts.sort()
            ax.loglog([p[0] for p in pts], [p[1] for p in pts],
                      marker=marks[fn], ls="-", lw=1.6, ms=7, label=fn)
    ds = np.array(sorted({r["d"] for r in rows}), dtype=float)
    if ds.size:
        ax.loglog(ds, np.full_like(ds, 64.0), "k--", lw=1.2,
                  label=r"$k^\ast$ constant  $\Rightarrow$ state $=O(d)$")
        ax.loglog(ds, 64.0 * ds / ds[0], color="tab:red", ls="-.", lw=1.2,
                  label=r"$k^\ast\propto d$  $\Rightarrow$ state $=O(d^2)$")
    ax.set_xlabel("$d$  (number of readout parameters)")
    ax.set_ylabel(r"$k^\ast$ = smallest $k$ within $10\times$ of the floor")
    ax.set_ylim(4, 1024)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=3,
              frameon=False, fontsize=8)
    # bottom-right: threshold-FREE version -- ratio to floor at FIXED k
    ax2 = fig.add_subplot(gs[1, 2])
    for fn in cv.TARGETS:
        pts = []
        for N in Ns:
            r = next((r for r in rows if r["fn"] == fn and r["N"] == N
                      and r["k"] == 128), None)
            if r:
                pts.append((r["d"], max(r["best"], 1e-17) / max(r["floor"], 1e-30)))
        if pts:
            pts.sort()
            ax2.loglog([p[0] for p in pts], [p[1] for p in pts],
                       marker=marks[fn], ls="-", lw=1.6, ms=7, label=fn)
    ax2.axhline(1.0, color="k", ls=":", lw=1.4)
    ax2.set_xlabel("$d$")
    ax2.set_ylabel(r"(achieved) / (SVD floor)  at fixed $k=128$")
    ax2.set_ylim(1e-2, 1e4)
    ax2.grid(True, which="both", alpha=0.3)
    ax2.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=3,
               frameon=False, fontsize=8)
    fig.suptitle("B -- is the state coefficient $k$ coupled to $d$?   "
                 "(top: accuracy vs $k$;  bottom: the required $k$ vs problem "
                 "size)", y=0.99, fontsize=12)
    cv.FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(cv.FIGS / "B_k_vs_d.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {cv.FIGS / 'B_k_vs_d.png'}")


# ------------------------------------------------------------------ C ------

def fig_C(rows):
    nzs = sorted({r["noise"] for r in rows})
    cmap = plt.get_cmap("inferno")
    col = {z: cmap(t) for z, t in zip(nzs, np.linspace(0.05, 0.78, len(nzs)))}
    fig, axes = _grid(len(cv.WIDTHS), len(cv.TARGETS), (14.5, 10),
                      r"eval rel $L_2$ vs CLEAN target", "LSQR iteration",
                      cv.TARGETS, [f"$N={n}$" for n in cv.WIDTHS])
    for i, N in enumerate(cv.WIDTHS):
        for j, fn in enumerate(cv.TARGETS):
            ax = axes[i, j]
            for z in nzs:
                r = next((r for r in rows if r["fn"] == fn and r["N"] == N
                          and r["noise"] == z), None)
                if r is None:
                    continue
                x = _x(r["eval"], r["stride"])
                ax.plot(x, _clip(r["eval"]), color=col[z], lw=1.4)
                bi = int(np.argmin(r["eval"]))
                ax.plot(x[bi], max(r["eval"][bi], cv.YLIM[0]), marker="v",
                        ms=5, color=col[z])
            ax.set_xscale("log")
    handles = [Line2D([0], [0], color=col[z], lw=1.8,
                      label=("noiseless" if z == 0 else f"$\\sigma_{{rel}}={z:.0e}$"))
               for z in nzs]
    handles.append(Line2D([0], [0], marker="v", ls="", color="0.35", ms=6,
                          label="best iterate (early stop)"))
    _finish(fig, cv.FIGS / "C_noise_3x3.png", handles,
            "C -- y-noise: eval error against the CLEAN target. The rise after "
            "the marker is the solver fitting noise (semiconvergence).", ncol=4)


def fig_C2(rows):
    fig, ax = plt.subplots(figsize=(9.5, 6.4))
    marks = {"sine": "o", "sine_8pi": "s", "runge": "^"}
    cmap = plt.get_cmap("plasma")
    Ns = sorted({r["N"] for r in rows})
    col = {N: cmap(t) for N, t in zip(Ns, np.linspace(0.05, 0.8, len(Ns)))}
    for fn in cv.TARGETS:
        for N in Ns:
            rs = sorted([r for r in rows if r["fn"] == fn and r["N"] == N
                         and r["noise"] > 0], key=lambda r: r["noise"])
            if not rs:
                continue
            ax.loglog([r["noise"] for r in rs], [r["best"] for r in rs],
                      marker=marks[fn], color=col[N], lw=1.3, ms=5, alpha=0.85)
    nz = np.array([r["noise"] for r in rows if r["noise"] > 0])
    if nz.size:
        g = np.array(sorted(set(nz)))
        ax.loglog(g, g, "k--", lw=1.4, label=r"$\varepsilon_{\rm eval}=\sigma_{rel}$")
        # Correct statistical floor for a rank-r regression: the noise-induced
        # error in the FITTED FUNCTION is sigma * sqrt(r/n), not sigma/sqrt(n)
        # -- r free parameters are estimated, each picking up noise.
        rn = [np.sqrt(cv.make_problem("sine", N)["rank"]
                      / cv.make_problem("sine", N)["n"])
              for N in sorted({r["N"] for r in rows})]
        c = float(np.median(rn))
        ax.loglog(g, c * g, color="tab:green", ls="-.", lw=1.4,
                  label=r"statistical floor $\sigma_{rel}\sqrt{r/n}$")
    ax.set_xlabel(r"relative noise level $\sigma_{rel}=\|\varepsilon\|/\|y\|$")
    ax.set_ylabel(r"best eval rel $L_2$ (vs clean target)")
    ax.set_ylim(1e-17, 1e0)
    ax.grid(True, which="both", alpha=0.3)
    h, l = ax.get_legend_handles_labels()
    h += [Line2D([0], [0], color=col[N], lw=1.6, label=f"$N={N}$") for N in Ns]
    h += [Line2D([0], [0], marker=marks[f], ls="", color="0.35", label=f)
          for f in cv.TARGETS]
    ax.legend(handles=h, loc="lower center", bbox_to_anchor=(0.5, 1.02),
              ncol=4, frameon=False, fontsize=9)
    fig.suptitle("C2 -- achieved accuracy vs noise level, against the two "
                 "reference lines", y=1.10, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(cv.FIGS / "C2_noise_summary.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {cv.FIGS / 'C2_noise_summary.png'}")


# ------------------------------------------------------------------ D ------

def fig_D1(rows):
    fr = sorted({r["frac"] for r in rows}, reverse=True)
    cmap = plt.get_cmap("cividis")
    col = {f: cmap(t) for f, t in zip(fr, np.linspace(0.05, 0.85, len(fr)))}
    fig, axes = _grid(len(cv.WIDTHS), len(cv.TARGETS), (14.5, 10),
                      r"eval rel $L_2$", "LSQR iteration",
                      cv.TARGETS, [f"$N={n}$" for n in cv.WIDTHS])
    for i, N in enumerate(cv.WIDTHS):
        for j, fn in enumerate(cv.TARGETS):
            ax = axes[i, j]
            drew = False
            for f in fr:
                r = next((r for r in rows if r["fn"] == fn and r["N"] == N
                          and r["frac"] == f), None)
                if r is None:
                    continue
                if not drew:
                    ax.axhline(r["floor"], color="k", ls=":", lw=1.5, zorder=1)
                    drew = True
                ax.plot(_x(r["eval"], r["stride"]), _clip(r["eval"]),
                        color=col[f], lw=1.5)
            ax.set_xscale("log")
    handles = [Line2D([0], [0], color=col[f], lw=1.8,
                      label=f"batch $={f:g}\\,n$") for f in fr]
    handles.append(Line2D([0], [0], color="k", ls=":", lw=1.5,
                          label="full-data SVD floor"))
    _finish(fig, cv.FIGS / "D1_batch_trajectories_3x3.png", handles,
            "D1 -- batching: whitening AND solve use the same random row "
            "batch (line colour = batch size)", ncol=5)


def fig_D2(rows):
    """Rows = width, columns = target, one clearly-labelled line per k.

    The previous version encoded width as line TRANSPARENCY with no legend
    entry, which made six lines per panel of which two were explained.
    """
    Ns = sorted({r["N"] for r in rows})
    ks = sorted({r["k"] for r in rows})
    kcol = {64: "tab:blue", 128: "tab:green"}
    fig, axes = plt.subplots(len(Ns), len(cv.TARGETS),
                             figsize=(14.5, 3.4 * len(Ns)),
                             sharex=True, sharey=True, squeeze=False)
    for i, N in enumerate(Ns):
        for j, fn in enumerate(cv.TARGETS):
            ax = axes[i, j]
            for k in ks:
                rs = sorted([r for r in rows if r["fn"] == fn and r["k"] == k
                             and r["N"] == N and r["mult"] > 0],
                            key=lambda r: r["mult"])
                if rs:
                    ax.loglog([r["mult"] for r in rs],
                              [max(r["best"], 1e-17) for r in rs], "-o",
                              color=kcol.get(k, "0.4"), ms=5, lw=1.8)
                full = [r for r in rows if r["fn"] == fn and r["k"] == k
                        and r["N"] == N and r["mult"] < 0]
                if full:
                    ax.axhline(max(full[0]["best"], 1e-17),
                               color=kcol.get(k, "0.4"), ls=":", lw=1.4)
            ax.axvline(8, color="tab:red", ls="--", lw=1.3)
            ax.set_ylim(*cv.YLIM)
            ax.grid(True, which="both", alpha=0.25)
            if i == 0:
                ax.set_title(fn, fontsize=11)
            if i == len(Ns) - 1:
                ax.set_xlabel(r"QR rows $s$, in multiples of $k$")
            if j == 0:
                ax.set_ylabel(f"$N={N}$\nbest eval rel $L_2$", fontsize=9)
    handles = [Line2D([0], [0], color=kcol[k], lw=2.0, label=f"$k={k}$, subset QR")
               for k in ks if k in kcol]
    handles += [Line2D([0], [0], color="0.4", ls=":", lw=1.6,
                       label="same $k$, ALL rows (the target to match)"),
                Line2D([0], [0], color="tab:red", ls="--", lw=1.4,
                       label=r"$s=8k$ (the threshold I originally claimed)")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.0),
               ncol=4, frameon=False, fontsize=9)
    fig.suptitle("D2 -- how many rows the QR needs (the solve always uses all "
                 "rows).\nEach solid curve should meet its own dotted line; "
                 "it does so only near the full row count, so there is NO "
                 "threshold at $8k$.", y=1.06, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    cv.FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(cv.FIGS / "D2_qr_rows.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {cv.FIGS / 'D2_qr_rows.png'}")


def render_all():
    cv.FIGS.mkdir(parents=True, exist_ok=True)
    for name, fn in (("A_traj", fig_A), ("B_scaling", fig_B),
                     ("C_noise", fig_C), ("D1_batch_traj", fig_D1),
                     ("D2_qr_rows", fig_D2)):
        rows = cv.load_rows(name)
        if rows:
            fn(rows)
    rows = cv.load_rows("C_noise")
    if rows:
        fig_C2(rows)
