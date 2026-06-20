"""Render the 2D target functions of exp10 and exp11 as 3D surfaces over the
unit disk, with the disk highlighted and the actual training sample points shown.

Sample points are the exact training set used by both experiments:
    disk_uniform(8000, radius=1.0, seed=0)
Targets and definitions come straight from src/data/targets2d.py, so what is
plotted is what is fit.

Outputs:
    results/exp10_radon_hex2d/targets_3d.png   (6 targets)
    results/exp11_geometry_zoo_2d/targets_3d.png (4 targets)

Run: ~/.venvs/precisionMLPs/bin/python experiments/_viz_targets_3d.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm

# Make src importable
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.data.sampling2d import disk_uniform  # noqa: E402
from src.data.targets2d import get_target_2d  # noqa: E402

RADIUS = 1.0
N_TRAIN = 8000          # matches exp10/exp11 N_TRAIN
TRAIN_SEED = 0          # matches disk_uniform(..., seed=0)
N_SHOW = 1500           # subsample of the 8000 train points actually drawn (legibility)
SURF_SIDE = 220         # surface mesh resolution

PRETTY = {
    "sine2d": r"sine2d  $\sin(\pi x)\cos(\pi y)$",
    "sine2d_hi": r"sine2d_hi  $\sin(4\pi x)\cos(4\pi y)$",
    "gauss_bump": r"gauss_bump  $e^{-(x^2+y^2)}$",
    "runge2d": r"runge2d  $1/(1+25(x^2+y^2))$",
    "mixed2d": r"mixed2d  $e^{\sin\pi x+\cos\pi y}$",
    "planewave": r"planewave  $\tanh(3(0.6x+0.8y))$",
}


def disk_surface(fn):
    """Evaluate fn on a Cartesian mesh over [-R,R]^2, masking outside the disk."""
    g = np.linspace(-RADIUS, RADIUS, SURF_SIDE)
    xx, yy = np.meshgrid(g, g)
    inside = xx ** 2 + yy ** 2 <= RADIUS ** 2
    pts = np.stack([xx.ravel(), yy.ravel()], axis=1)
    zz = fn(pts).reshape(xx.shape)
    zz = np.where(inside, zz, np.nan)        # surface only over the disk
    return xx, yy, zz


def plot_panel(ax, tname):
    fn = get_target_2d(tname).fn_numpy
    xx, yy, zz = disk_surface(fn)

    zmin = np.nanmin(zz)
    zmax = np.nanmax(zz)
    pad = 0.08 * (zmax - zmin + 1e-12)
    floor = zmin - 0.55 * (zmax - zmin + 1e-12)   # base plane below the surface

    # Surface
    ax.plot_surface(
        xx, yy, zz, cmap=cm.viridis, linewidth=0, antialiased=True,
        rstride=2, cstride=2, alpha=0.92, vmin=zmin, vmax=zmax,
    )

    # Disk highlight: filled translucent base disk + bold boundary ring (on floor)
    th = np.linspace(0, 2 * np.pi, 400)
    cx, cy = RADIUS * np.cos(th), RADIUS * np.sin(th)
    ax.plot(cx, cy, floor, color="crimson", lw=2.2, zorder=5)
    # faint filled disk on the base plane
    rr = np.linspace(0, RADIUS, 2)
    ax.plot_trisurf(
        np.r_[0.0, cx], np.r_[0.0, cy], np.full(cx.size + 1, floor),
        color="crimson", alpha=0.06, linewidth=0,
    )

    # Training sample points (the exact set we fit), shown on the surface
    Xs = disk_uniform(N_TRAIN, radius=RADIUS, seed=TRAIN_SEED)
    rng = np.random.default_rng(0)
    idx = rng.choice(N_TRAIN, size=min(N_SHOW, N_TRAIN), replace=False)
    Xs = Xs[idx]
    zs = fn(Xs)
    ax.scatter(
        Xs[:, 0], Xs[:, 1], zs, s=4, c="black", alpha=0.55,
        depthshade=True, edgecolors="none", zorder=6,
    )
    # ...and projected onto the base plane to show the disk sampling pattern
    ax.scatter(
        Xs[:, 0], Xs[:, 1], np.full(Xs.shape[0], floor), s=2,
        c="dimgray", alpha=0.30, edgecolors="none", zorder=4,
    )

    ax.set_title(PRETTY.get(tname, tname), fontsize=10, pad=2)
    ax.set_xlabel("x", fontsize=8, labelpad=-6)
    ax.set_ylabel("y", fontsize=8, labelpad=-6)
    ax.set_zlim(floor, zmax + pad)
    ax.tick_params(labelsize=6, pad=-2)
    ax.set_box_aspect((1, 1, 0.7))
    ax.view_init(elev=28, azim=-60)


def render(targets, ncols, out_path, suptitle):
    nrows = int(np.ceil(len(targets) / ncols))
    fig = plt.figure(figsize=(5.2 * ncols, 4.6 * nrows))
    for i, tname in enumerate(targets):
        ax = fig.add_subplot(nrows, ncols, i + 1, projection="3d")
        plot_panel(ax, tname)
    fig.suptitle(suptitle, fontsize=13, y=0.99)
    # one shared legend note
    fig.text(
        0.5, 0.005,
        f"Surface = target over unit disk (radius {RADIUS:.0f}); crimson ring = disk boundary; "
        f"black dots = training samples on the surface; gray dots = same samples projected "
        f"(disk_uniform, n={N_TRAIN}, seed={TRAIN_SEED}; {N_SHOW} of {N_TRAIN} shown).",
        ha="center", fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print("wrote", out_path)


def main():
    # exp10: 6 targets
    exp10_targets = ["sine2d", "sine2d_hi", "gauss_bump", "runge2d", "mixed2d", "planewave"]
    render(
        exp10_targets, ncols=3,
        out_path=os.path.join(ROOT, "results", "exp10_radon_hex2d", "targets_3d.png"),
        suptitle="exp10 -- 2D targets over the unit disk (with training samples)",
    )
    # exp11: 4 targets
    exp13_targets = ["gauss_bump", "runge2d", "sine2d", "mixed2d"]
    render(
        exp13_targets, ncols=2,
        out_path=os.path.join(ROOT, "results", "exp11_geometry_zoo_2d", "targets_3d.png"),
        suptitle="exp11 -- 2D targets over the unit disk (with training samples)",
    )


if __name__ == "__main__":
    main()
