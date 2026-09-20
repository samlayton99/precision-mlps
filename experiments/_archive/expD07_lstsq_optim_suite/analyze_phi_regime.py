"""Which synthetic regime does the QI feature system [Phi, 1] sit in?

Two diagnostics per problem (N=256, R=4x ratio), computed from the SVD
A = U diag(s) V^T:

  1. Normalized singular spectrum s_i/s_1 vs index fraction i/m -- the
     conditioning shape (flat head? exponential tail? cliff?).
  2. Truncation curve err(tau) = ||y - P_tau y|| / ||y||, where P_tau
     projects onto left singular directions with s_i >= tau * s_1: the best
     achievable relative L2 using only directions above tau. For plain GD at
     lr = 1/L, direction i converges on the timescale (s_1/s_i)^2 steps, so
     this curve IS the idealized error-vs-time curve of a first-order
     method, reparameterized by tau = s_i/s_1.

Prints, per problem: the tau needed for rel L2 <= 1e-6 / 1e-13 and the
implied GD steps (s_1/tau)^2. Saves figures/phi_regime.png. Pure analysis --
reads the problem artifacts, touches no run cache.

Usage: uv run python experiments/expD07_lstsq_optim_suite/analyze_phi_regime.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from common import FIG_DIR, load_manifest, load_problem  # noqa: E402

SYN_IDS = [
    ("syn_wellcond_iso_N256_R1024", "wellcond_iso", "tab:gray"),
    ("syn_illcond8_iso_N256_R1024", "illcond8_iso", "tab:blue"),
    ("syn_illcond8_head_N256_R1024", "illcond8_head", "tab:cyan"),
    ("syn_illcond8_tail_N256_R1024", "illcond8_tail", "tab:purple"),
    ("syn_phispec_iso_N256_R1024", "phispec_iso", "tab:brown"),
]
PHI_IDS = [
    ("phi_sine_N256_r4", "phi: sine", "tab:green"),
    ("phi_runge_N256_r4", "phi: runge", "tab:orange"),
    ("phi_sine_8pi_N256_r4", "phi: sine_8pi", "tab:red"),
]
TOLS = [1e-6, 1e-13]


def diagnostics(entry):
    prob = load_problem(entry)
    A, y = prob["A"], prob["y"]
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    p = U.T @ y                       # y energy per left singular direction
    # off-range energy via the residual itself (a norm-difference cancels
    # catastrophically and would floor the curve at ~1e-8)
    off_range = float(np.linalg.norm(y - U @ p) ** 2)
    # err(tau_i): residual keeping directions 0..i (s sorted descending)
    tail_energy = np.cumsum((p ** 2)[::-1])[::-1]  # energy in directions >= i
    tail = np.empty_like(tail_energy)
    tail[:-1] = tail_energy[1:]
    tail[-1] = 0.0
    err = np.sqrt(np.maximum(tail + max(off_range, 0.0), 0.0)) / np.linalg.norm(y)
    return s / s[0], err


def main():
    manifest = {e["id"]: e for e in load_manifest()}
    curves = []
    print(f"{'problem':34s} " + "  ".join(
        f"tau(rel_l2<={tol:g})   GD steps" for tol in TOLS))
    for pid, label, color in SYN_IDS + PHI_IDS:
        tau, err = diagnostics(manifest[pid])
        curves.append((pid, label, color, tau, err))
        cells = []
        for tol in TOLS:
            hit = np.where(err <= tol)[0]
            if hit.size:
                t = tau[hit[0]]
                cells.append(f"{t:9.1e}  {1.0 / t**2:9.1e}")
            else:
                cells.append(f"{'never':>9s}  {'--':>9s}")
        print(f"{label:34s} " + "  ".join(cells))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    ax = axes[0]
    for pid, label, color, tau, err in curves:
        ls = "-" if pid.startswith("phi") else "--"
        m = tau.size
        ax.semilogy(np.arange(m) / m, np.maximum(tau, 1e-17), ls, color=color,
                    label=label)
    ax.set_ylim(1e-16, 2)
    ax.set_xlabel("index fraction $i/m$")
    ax.set_ylabel(r"$\sigma_i/\sigma_1$")
    ax.set_title("normalized singular spectrum", fontsize=11)
    ax.grid(True, alpha=0.3, which="both")

    ax = axes[1]
    for pid, label, color, tau, err in curves:
        ls = "-" if pid.startswith("phi") else "--"
        ax.loglog(np.maximum(tau, 1e-17), np.maximum(err, 1e-17), ls,
                  color=color, label=label)
    ax.set_xlim(1e-16, 2)
    ax.set_ylim(1e-16, 2)
    ax.invert_xaxis()  # progress runs left -> right (like time)
    ax.set_xlabel(r"kept directions down to $\sigma/\sigma_1 = \tau$"
                  "\n(GD needs $\\sim (1/\\tau)^2$ steps to get there)")
    ax.set_ylabel("best achievable relative $L_2$")
    ax.set_title("truncation curve: where the solution's energy lives",
                 fontsize=11)
    ax.grid(True, alpha=0.3, which="both")

    handles, labels = axes[1].get_legend_handles_labels()
    fig.tight_layout(rect=[0, 0, 1, 0.86])
    fig.suptitle("expD07 -- where [Phi, 1] sits among the synthetic regimes "
                 "(N=256, R=4x; phi solid, synthetic dashed)", y=0.99, va="top")
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.94),
               ncol=4, fontsize=9, frameon=True, borderaxespad=0)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "phi_regime.png"
    fig.savefig(out, dpi=140)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
