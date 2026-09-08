"""Numerical check of the fiber (aliasing) theorem behind the lambda rule.

Compares the exact prediction  rel-L2 error = sqrt(Lambda^(r)_lambda(theta0))  for a pure tone at
grid frequency theta0 = omega0 * h against three least-squares fits with tanh-family kernels:

  (a) periodic sech^2 (bump, r=0) network fitting f'      -- the clean torus setting,
  (b) periodic tanh   (step, r=1) network fitting f,
  (c) interval [-1,1] + halo tanh network fitting f        -- the repo's actual setting,

for f = sin(pi x), N = 128, over lambda on the aliasing wall; and against the stored expC03 sweep
(sine = sin 2pi x, sine_8pi = sin 8pi x; N = 64..1024) at lambda = 0.40, 0.50.

Writes results/checkpoint_C_geometry/expC07_lambda_energy_rule/figures/fiber_theory_check.png.
Run:  uv run --extra dev python experiments/expC07_lambda_energy_rule/fiber_check.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results/checkpoint_C_geometry/expC07_lambda_energy_rule/figures"


def khat(w: float) -> float:
    """Normalised sech^2 transform, Khat(w)/Khat(0) = (pi w/2)/sinh(pi w/2)."""
    w = abs(w)
    return 1.0 if w < 1e-12 else (math.pi * w / 2) / math.sinh(math.pi * w / 2)


def alias_amp(lam: float) -> float:
    return khat(2 * math.pi / lam)


def big_lambda(theta0: float, lam: float, r: int, m_max: int = 8) -> float:
    """Exact approximation kernel Lambda^(r)_lambda(theta0) of the fiber theorem."""
    num = den = 0.0
    for m in range(-m_max, m_max + 1):
        t = theta0 + 2 * math.pi * m
        v = (khat(t / lam) / lam) ** 2 / (abs(t) ** (2 * r) if r > 0 else 1.0)
        den += v
        if m != 0:
            num += v
    return num / den


def lstsq_rel_l2(phi_tr, y_tr, phi_ev, y_ev) -> float:
    phi_tr = np.hstack([phi_tr, np.ones((len(phi_tr), 1))])
    phi_ev = np.hstack([phi_ev, np.ones((len(phi_ev), 1))])
    coef, *_ = np.linalg.lstsq(phi_tr, y_tr, rcond=1e-13)
    res = phi_ev @ coef - y_ev
    return float(np.linalg.norm(res) / np.linalg.norm(y_ev))


def main() -> None:
    N = 128
    h = 2.0 / N
    theta0 = math.pi * h  # f = sin(pi x): omega0 = pi
    lams = np.array([0.30, 0.33, 0.36, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70])

    f = lambda x: np.sin(np.pi * x)  # noqa: E731
    fp = lambda x: np.pi * np.cos(np.pi * x)  # noqa: E731

    # periodic problem on the circle of circumference 2 (N centers, period-2 images)
    xs = -1 + 2.0 * np.arange(4096) / 4096
    xe = -1 + 2.0 * (np.arange(8001) + 0.37) / 8001
    c_per = -1 + h * np.arange(N)

    def per_feats(x, gam, kind):
        phi = np.zeros((len(x), N))
        for j in range(-6, 7):
            z = gam * (x[:, None] - c_per[None, :] - 2 * j)
            if kind == "sech2":
                phi += 1 / np.cosh(np.clip(z, -300, 300)) ** 2
            else:  # tanh, anchored so the image sum converges
                phi += np.tanh(z) - np.tanh(gam * (-c_per[None, :] - 2 * j))
        return phi

    # interval + halo problem (repo setting)
    R = 70
    c_int = -1 + h * np.arange(-R, N + R + 1)
    xs_i = np.linspace(-1, 1, 4096)
    xe_i = np.linspace(-1, 1, 8001)

    rows = []
    print(f"N={N}, f=sin(pi x), theta0={theta0:.4f}")
    print(" lam | bump periodic: meas / exact | tanh periodic: meas / exact | tanh interval: meas / exact")
    for lam in lams:
        gam = lam / h
        e_bump = lstsq_rel_l2(per_feats(xs, gam, "sech2"), fp(xs), per_feats(xe, gam, "sech2"), fp(xe))
        e_tanh = lstsq_rel_l2(per_feats(xs, gam, "tanh"), f(xs), per_feats(xe, gam, "tanh"), f(xe))
        e_int = lstsq_rel_l2(
            np.tanh(gam * (xs_i[:, None] - c_int[None, :])), f(xs_i),
            np.tanh(gam * (xe_i[:, None] - c_int[None, :])), f(xe_i),
        )
        p0 = math.sqrt(big_lambda(theta0, lam, 0))
        p1 = math.sqrt(big_lambda(theta0, lam, 1))
        rows.append((lam, e_bump, p0, e_tanh, e_int, p1))
        print(f" {lam:.2f} | {e_bump:.3e} / {p0:.3e} ({e_bump / p0:.3f}) | {e_tanh:.3e} / {p1:.3e} ({e_tanh / p1:.3f})"
              f" | {e_int:.3e} / {p1:.3e} ({e_int / p1:.3f})")
    rows = np.array(rows)

    # stored expC03 sweep
    sweep = REPO / "results/checkpoint_C_geometry/expC03_lambda_basin/full_sweep.json"
    if sweep.exists():
        d = json.load(open(sweep))
        print("\nexpC03 interval sweep vs exact fiber prediction (r=1)")
        for tgt, w0 in [("sine", 2 * math.pi), ("sine_8pi", 8 * math.pi)]:
            for Nn in [64, 128, 256, 1024]:
                r = d[f"{tgt}_N{Nn}"]
                lam_arr = np.array(r["lambdas"])
                err = np.array(r["rel_l2"])
                th = w0 * 2.0 / Nn
                parts = []
                for lp in (0.40, 0.50):
                    i = int(np.argmin(abs(lam_arr - lp)))
                    pred = math.sqrt(big_lambda(th, lam_arr[i], 1))
                    parts.append(f"lam={lam_arr[i]:.2f} meas {err[i]:.2e} pred {pred:.2e} ratio {err[i] / pred:.2f}")
                print(f"  {tgt:9s} N={Nn:5d} theta0={th:.3f}: " + " | ".join(parts))

    # cardinal interpolation (the paper's QI construction, fp64 path) vs Theorem 3's in-band defect |1 - M(theta0)|
    try:
        import sys
        sys.path.insert(0, str(REPO))
        from src.construction import construct_qi, evaluate_qi
        from src.construction.qi_mpmath import default_halo

        def one_minus_m(theta, lam, m_max=6):
            dsum = sum(khat((theta + 2 * math.pi * m) / lam) for m in range(-m_max, m_max + 1))
            return (dsum - khat(theta / lam)) / dsum

        w0 = 2 * math.pi
        th = w0 * h
        fq = lambda x: math.sin(2 * math.pi * x)  # noqa: E731
        fqp = lambda x: 2 * math.pi * math.cos(2 * math.pi * x)  # noqa: E731
        xq = np.linspace(-1, 1, 8001)
        yq = np.sin(2 * math.pi * xq)
        print(f"\nfp64 QI construction (interpolation), f=sin(2pi x), N={N}, theta0={th:.3f}")
        print(" lam | QI meas | pred |1-M(theta0)| | lstsq floor pred sqrt(Lambda^(1))")
        for lam in (0.35, 0.40, 0.45, 0.50, 0.60):
            q = construct_qi(fq, fqp, N, precision="fp64", lambda_star=lam, halo=default_halo(N, lambda_star=lam))
            rel = float(np.linalg.norm(evaluate_qi(q, xq, kahan=True) - yq) / np.linalg.norm(yq))
            print(f" {lam:.2f} | {rel:.3e} | {one_minus_m(th, lam):.3e} | {math.sqrt(big_lambda(th, lam, 1)):.3e}")
    except Exception as exc:  # pragma: no cover - optional check
        print("QI interpolation check skipped:", exc)

    # figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    OUT.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 5))
    ll = np.linspace(0.28, 0.72, 200)
    ax.plot(ll, [math.sqrt(2) * alias_amp(l) for l in ll], "k-", lw=1.2,
            label=r"theory, bump net: $\sqrt{2}\,A(\lambda)$")
    ax.plot(ll, [math.sqrt(big_lambda(theta0, l, 1)) for l in ll], "k--", lw=1.2,
            label=r"theory, tanh net: $\sqrt{\Lambda^{(1)}_\lambda(\theta_0)}\approx\sqrt{2}\,A(\lambda)\,\theta_0/2\pi$")
    ax.plot(rows[:, 0], rows[:, 1], "o", color="tab:red", ms=6, label="measured: periodic sech$^2$ net fitting $f'$")
    ax.plot(rows[:, 0], rows[:, 3], "s", color="tab:blue", ms=6, label="measured: periodic tanh net fitting $f$")
    ax.plot(rows[:, 0], rows[:, 4], "^", color="tab:green", ms=7, mfc="none",
            label="measured: interval + halo tanh net (repo setting)")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\lambda=\gamma h$")
    ax.set_ylabel("eval relative $L_2$ error")
    ax.set_ylim(1e-15, 1e-4)
    ax.set_xlim(0.28, 0.72)
    ax.grid(alpha=0.3)
    ax.set_title(r"Aliasing wall for $f=\sin\pi x$, $N=128$ ($\theta_0=\pi h$): measurement vs the fiber formula",
                 fontsize=10)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.06), ncol=2, fontsize=8, borderaxespad=0, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    out = OUT / "fiber_theory_check.png"
    fig.savefig(out, dpi=150)
    print("saved", out)


if __name__ == "__main__":
    main()
