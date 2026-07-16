"""expG04 figures: fit+residual, band-colored basis contributions, summary."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BAND_COLORS = ["tab:blue", "tab:orange", "tab:red", "tab:green"]
BAND_NAMES = ["sharp", "mid", "soft", "band3"]


def _shade_regions(ax, regions):
    for lo, hi, *_ in regions:
        ax.axvspan(lo, hi, color="0.85", zorder=0)


def _fit_residual_fig(out_dir, n_bands, protocol, target, f, x_test, u_hat, regions):
    u_true = f(x_test)
    fig, (a0, a1) = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    _shade_regions(a0, regions)
    a0.plot(x_test, u_true, "b-", lw=1.5, label="target")
    a0.plot(x_test, u_hat, "r-", lw=1.0, alpha=0.7, label="fit")
    a0.set_ylabel("f, f_hat")
    a0.legend(fontsize=8)
    a0.set_title(f"nb={n_bands} / {protocol} / {target}")
    _shade_regions(a1, regions)
    resid = u_true - u_hat
    a1.plot(x_test, np.sign(resid) * np.log10(np.abs(resid) + 1e-18), "k-", lw=0.8)
    a1.set_ylabel("sign(r)·log10|r|")
    a1.set_xlabel("x")
    fig.tight_layout()
    fig.savefig(Path(out_dir) / f"fit_nb{n_bands}_{protocol}_{target}.png", dpi=140)
    plt.close(fig)


def _basis_band_fig(out_dir, n_bands, protocol, target, f, centers, gamma_vec,
                    band_idx, v, bias, regions, solver):
    x = np.linspace(min(-1.3, centers.min()), max(1.3, centers.max()), 600)
    contrib, b = solver.basis_contributions(x, centers, gamma_vec, v, bias)
    fig, ax = plt.subplots(figsize=(7, 5))
    _shade_regions(ax, regions)
    for k in range(n_bands):
        cols = np.where(band_idx == k)[0]
        for j in cols:
            ax.plot(x, contrib[:, j], color=BAND_COLORS[k], lw=0.4, alpha=0.2,
                    zorder=1 + k)
        # one labeled proxy line per band
        ax.plot([], [], color=BAND_COLORS[k], lw=1.2,
                label=f"{BAND_NAMES[k]} band")
    ax.plot(x, contrib.sum(axis=1) + b, "k-", lw=1.6, label="sum = f_hat", zorder=9)
    ax.plot(x, f(x), "g--", lw=1.0, label="target", zorder=9)
    ax.set_title(f"basis by band — nb={n_bands} / {protocol} / {target}")
    ax.set_xlabel("x")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(Path(out_dir) / f"basis_nb{n_bands}_{protocol}_{target}.png", dpi=140)
    plt.close(fig)


def _summary_fig(out_dir, records):
    fig, ax = plt.subplots(figsize=(7, 5))
    keys = sorted({(r["protocol"], r["target"]) for r in records})
    for protocol, target in keys:
        cells = sorted((r for r in records
                        if r["protocol"] == protocol and r["target"] == target),
                       key=lambda r: r["n_bands"])
        nb = [r["n_bands"] for r in cells]
        held = [r["rel_l2_held"] for r in cells]
        ax.semilogy(nb, held, "o-", label=f"{protocol}/{target}")
    ax.set_xlabel("n_bands")
    ax.set_ylabel("held-out rel L2")
    ax.set_xticks([1, 2, 3])
    ax.set_title("expG04: held-out error vs number of bands")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(Path(out_dir) / "summary_held_vs_nbands.png", dpi=140)
    plt.close(fig)


def make_all_figures(records, out_dir, targets, protocols_mod, cascade_mod,
                     solver, N, lambdas, coarsen):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for rec in records:
        n_bands, protocol, target = rec["n_bands"], rec["protocol"], rec["target"]
        f = targets[target]
        x_train, x_test, regions = protocols_mod.PROTOCOLS[protocol]()
        centers, gamma_vec, band_idx = cascade_mod.cascade_geometry(
            N, lambdas[:n_bands], coarsen)
        v, bias, _ = solver.fit(x_train, f(x_train), centers, gamma_vec)
        u_hat = solver.predict(x_test, centers, gamma_vec, v, bias)
        _fit_residual_fig(out_dir, n_bands, protocol, target, f, x_test, u_hat, regions)
        _basis_band_fig(out_dir, n_bands, protocol, target, f, centers, gamma_vec,
                        band_idx, v, bias, regions, solver)
    if records:
        _summary_fig(out_dir, records)
