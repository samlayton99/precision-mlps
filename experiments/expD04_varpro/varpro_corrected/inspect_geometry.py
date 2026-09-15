"""What geometry did VarPro + Adam->GN find from a Xavier start on `sine`?

Reruns one cell of the 04/06 ladder (default: sine, N=256, seed 0, Xavier init) with the final
geometry kept, then plots the learned neurons ordered by center c_j = -b_j / a_j:
  left    gamma_j = |a_j| vs c_j, dotted = the construction's gamma = lambda / h (lambda = 0.30 as in
          the probe, and 0.25 for reference), h = 2 / N
  middle  sign-corrected readout v_j vs c_j, dotted = the derivative law v(x) = (h / 2) f'(x)
  right   histogram of the centers, dotted = the uniform-grid count per bin
Writes the figure and a small JSON of summary numbers next to the 06 outputs.

Usage: uv run --extra dev python experiments/expD04_varpro/varpro_corrected/inspect_geometry.py [--N 256] [--seed 0] [--init xavier]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
D04 = HERE.parent
REPO_ROOT = D04.parents[1]
sys.path.insert(0, str(REPO_ROOT))
OUT_DIR = REPO_ROOT / "results" / "checkpoint_D_optimizers" / "expD04_varpro" / "varpro_corrected" / "06_sine_xavier_seeds"

spec = importlib.util.spec_from_file_location("d04_varpro_gn_probe", D04 / "varpro_gn_probe.py")
m = importlib.util.module_from_spec(spec); sys.modules["d04_varpro_gn_probe"] = m; spec.loader.exec_module(m)  # type: ignore


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", default="sine")
    ap.add_argument("--N", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init", default="xavier", choices=list(m.GEOM_INITS))
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--gn", type=int, default=40)
    args = ap.parse_args(argv)

    xt, yt, xe, ye = m.make_data(args.target, m.DEFAULT_N_TRAIN, m.DEFAULT_N_EVAL)
    geom = m.d05.geometry_for_resolution(args.N); W = geom.width
    st = m.d05.build_initial_state(m.GEOM_INITS[args.init], args.target, geom, args.seed)
    theta0 = np.concatenate([st.input_weights, st.input_biases]).astype(np.float64)
    init, best, final, theta = m.varpro_adam_then_gn(theta0, W, xt, yt, xe, ye, args.warmup, args.gn, keep_theta=True)
    print(f"{args.target} N={args.N} W={W} seed={args.seed} init={args.init}: init refit {init:.2e} -> final {final:.2e}")

    # learned geometry, sign-corrected: tanh(a x + b) = sign(a) tanh(|a| x + sign(a) b)
    a, b = theta[:W], theta[W:]
    _, _, _, _, c_hat, _ = m.solve_readout(theta, W, xt, yt)
    v = c_hat[:W]
    sgn = np.sign(a); sgn[sgn == 0] = 1.0
    gamma = np.abs(a); centers = -b / a; v_eff = sgn * v
    order = np.argsort(centers); gamma, centers, v_eff = gamma[order], centers[order], v_eff[order]
    a0, b0 = theta0[:W], theta0[W:]; c0 = -b0 / a0; g0 = np.abs(a0)

    # theory: construction gamma = lambda / h, readout v = (h / 2) f'(x)  (tanh kernel mass M = 2)
    h = 2.0 / args.N
    t = m.get_target(args.target)
    xs = np.linspace(-1.25, 1.25, 2001); eps = 1e-6
    fprime = (t.fn_numpy(xs + eps) - t.fn_numpy(xs - eps)) / (2 * eps)
    v_theory = 0.5 * h * fprime
    g030, g025 = 0.30 / h, 0.25 / h

    inside = (centers >= -1) & (centers <= 1)
    summary = dict(target=args.target, N=args.N, W=W, seed=args.seed, init=args.init,
                   init_refit=init, final=final,
                   centers_inside_frac=float(inside.mean()),
                   centers_inside_count=int(inside.sum()),
                   gamma_median_inside=float(np.median(gamma[inside])),
                   gamma_iqr_inside=[float(np.percentile(gamma[inside], 25)), float(np.percentile(gamma[inside], 75))],
                   gamma_theory_l030=g030, gamma_theory_l025=g025,
                   spacing_inside_median=float(np.median(np.diff(centers[inside]))) if inside.sum() > 2 else None,
                   spacing_uniform=h,
                   local_lambda_median=float(np.median(gamma[inside][1:] * np.diff(centers[inside]))) if inside.sum() > 2 else None,
                   v_abs_max=float(np.abs(v_eff).max()), v_theory_abs_max=float(np.abs(v_theory).max()),
                   init_gamma_median=float(np.median(g0)), init_centers_inside_frac=float(((c0 >= -1) & (c0 <= 1)).mean()))
    print(json.dumps(summary, indent=1))

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
    xlim = (-1.3, 1.3)
    # left: gamma vs center
    ax[0].plot(c0, g0, ".", color="0.65", ms=4, label="init (Xavier)")
    ax[0].plot(centers, gamma, ".", color="#d62728", ms=5, label="learned")
    ax[0].axhline(g030, ls=":", color="k", lw=1.4, label=r"construction $\gamma=0.30/h$")
    ax[0].axhline(g025, ls=":", color="0.4", lw=1.0, label=r"$\gamma=0.25/h$")
    ax[0].set_yscale("log"); ax[0].set_xlim(*xlim); ax[0].set_xlabel("center $c_j=-b_j/a_j$"); ax[0].set_ylabel(r"$\gamma_j=|a_j|$")
    ax[0].axvline(-1, color="0.8", lw=0.8); ax[0].axvline(1, color="0.8", lw=0.8)
    ax[0].legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2, fontsize=8, borderaxespad=0)
    # middle: readout vs center
    ax[1].plot(xs, v_theory, ":", color="k", lw=1.6, label=r"derivative law $v=(h/2)f'(c)$")
    ax[1].plot(centers, v_eff, ".-", color="#d62728", ms=5, lw=0.8, label="learned readout (sign-corrected)")
    ax[1].set_xlim(*xlim); ax[1].set_xlabel("center $c_j$"); ax[1].set_ylabel("$v_j$")
    vmax = 3 * np.abs(v_theory).max(); ax[1].set_ylim(-vmax, vmax)
    ax[1].axvline(-1, color="0.8", lw=0.8); ax[1].axvline(1, color="0.8", lw=0.8)
    ax[1].legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2, fontsize=8, borderaxespad=0)
    # right: center histogram
    bins = np.linspace(-1.3, 1.3, 53); bw = bins[1] - bins[0]
    ax[2].hist(c0, bins=bins, color="0.75", alpha=0.7, label="init (Xavier)")
    ax[2].hist(centers, bins=bins, color="#d62728", alpha=0.7, label="learned")
    ax[2].axhline(bw / h, ls=":", color="k", lw=1.4, label=f"uniform grid ({bw / h:.1f} per bin)")
    ax[2].set_xlim(*xlim); ax[2].set_xlabel("center $c_j$"); ax[2].set_ylabel("count")
    ax[2].axvline(-1, color="0.8", lw=0.8); ax[2].axvline(1, color="0.8", lw=0.8)
    ax[2].legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=3, fontsize=8, borderaxespad=0)
    fig.suptitle(f"expD04 corrected: geometry found by VarPro + Adam→GN from a {args.init} start on {args.target}, "
                 f"N={args.N} (W={W}), seed {args.seed}: init refit {init:.1e} → final {final:.1e}", y=1.06)
    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"geometry_{args.target}_{args.init}_N{args.N}_s{args.seed}"
    fig.savefig(OUT_DIR / f"{stem}.png", dpi=150, bbox_inches="tight")
    (OUT_DIR / f"{stem}.json").write_text(json.dumps(summary, indent=1))
    np.savez(OUT_DIR / f"{stem}.npz", theta0=theta0, theta=theta, v=v)
    print(f"wrote {OUT_DIR / stem}.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
