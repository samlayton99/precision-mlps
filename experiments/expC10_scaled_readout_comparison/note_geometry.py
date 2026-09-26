"""Run the note's geometry and scaling with authorized baseline defaults.

The supplied short note does not define numerical values/rules for epsilon_eff
or its recovery-budget SVD threshold. Sam explicitly authorized baseline defaults
where unspecified. This is a numerical implementation, not a full certificate.
"""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import time

import mpmath as mp
import numpy as np
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("c10_note_base", HERE / "run.py")
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
OUT = base.OUT / "note_geometry"
A_H = 1.0  # Matches the actual PR's core.geometry().


def upper_float(interval):
    """Round an interval's rigorous upper endpoint upward to binary64."""
    with mp.workdps(90):
        upper = mp.mpf(interval._mpi_[1])
        result = float(upper)
        if mp.mpf(result) < upper:
            result = np.nextafter(result, np.inf)
        return result


def interval_envelopes(n, radius, lam, delta):
    """The note's products, with interval arithmetic from inputs to sqrt."""
    iv = mp.iv
    old_dps = iv.dps
    iv.dps = 70
    try:
        h = iv.mpf(2) / n
        ll, dd = iv.mpf(float(lam)), iv.mpf(float(delta))
        pole = iv.pi * h / (2 * ll)
        if not bool((dd - pole).a > 0):
            return None
        width = n + 2 * radius + 1
        alpha = [h / (2 * (dd - pole)) for _ in range(width)]
        m = (radius + 1) // 2
        zeta = iv.exp(-2 * ll)
        products = [iv.mpf(1)]
        for ell in range(1, m + 1):
            products.append(products[-1] * (1 - zeta**ell))
        d_lambda = iv.pi / (2 * ll) + 4 * iv.ln(2) / iv.pi
        for i in range(1, m + 1):
            li = zeta ** (iv.mpf(i * (i + 1) - 1) / 2)
            li /= products[i - 1] * products[m - i]
            for j in range(1, m + 1):
                if j != i:
                    li *= 1 + zeta ** (iv.mpf(j) - iv.mpf(1) / 2)
            correction = h * d_lambda * li / (2 * dd)
            alpha[i - 1] = alpha[i - 1] + correction
            alpha[-i] = alpha[-i] + correction
        alpha = [1 + sum(alpha)] + alpha
        return (np.array([upper_float(a) for a in alpha]),
                np.array([upper_float(iv.sqrt(a)) for a in alpha]))
    finally:
        iv.dps = old_dps


def run():
    root = base.OUT / "data"
    meta = json.loads((root / "metadata.json").read_text())
    cfg = meta["config"]
    data = OUT / "data"
    data.mkdir(parents=True, exist_ok=True)
    source_hashes = {}
    with np.load(root / "samples.npz") as z:
        x, xe, y, ye = [z[k].copy() for k in ["x_train", "x_eval", "y_train", "y_eval"]]
    records = []
    with threadpool_limits(limits=cfg["threads"]):
        for wi, n in enumerate(cfg["interior_resolutions"]):
            source = root / f"N{n}.npz"
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            source_hashes[source.name] = digest
            path = data / source.name
            radius = math.ceil(A_H * math.sqrt(n))
            if path.exists():
                with np.load(path) as z:
                    assert str(z["source_sha256"]) == digest
                    records.append(json.loads(str(z["record_json"])))
                print(f"Reusing note geometry N={n}, R={radius}", flush=True)
                continue
            start = time.perf_counter()
            with np.load(source) as z:
                lambdas = z["lambdas"].copy()
            centers = -1 + 2 / n * np.arange(-radius, n + radius + 1)
            p, nf, nk = len(centers) + 1, len(cfg["targets"]), len(lambdas)
            errors = np.full((nk, nf), np.nan)
            train = np.full_like(errors, np.nan)
            coefficients = np.full((nk, p, nf), np.nan)
            native = np.full_like(coefficients, np.nan)
            scales = np.full_like(coefficients, np.nan)
            alpha = np.full_like(coefficients, np.nan)
            singular = np.full_like(coefficients, np.nan)
            ranks = np.full((nk, nf), -1, dtype=int)
            theorem_conditions = np.zeros((nk, nf), dtype=bool)
            deltas = np.array([cfg["delta_by_target"].get(f, cfg["delta_default"])
                               for f in cfg["targets"]])
            map_discrepancy = 0.0
            for li, lam in enumerate(lambdas):
                gamma = lam * n / 2
                a = np.c_[np.ones(len(x)), np.tanh(gamma * (x[:, None] - centers))]
                e = np.c_[np.ones(len(xe)), np.tanh(gamma * (xe[:, None] - centers))]
                for delta in np.unique(deltas):
                    ids = np.flatnonzero(deltas == delta)
                    enclosed = interval_envelopes(n, radius, lam, delta)
                    if enclosed is None:
                        continue
                    aa, d = enclosed
                    c, v, rank, s = base.scaled_solve(
                        a, y[:, ids], d, cfg["relative_svd_cutoff"])
                    residual = e @ c - ye[:, ids]
                    tr = a @ c - y[:, ids]
                    for k, f in enumerate(ids):
                        errors[li, f] = np.linalg.norm(residual[:, k]) / np.linalg.norm(ye[:, f])
                        train[li, f] = np.linalg.norm(tr[:, k]) / np.linalg.norm(y[:, f])
                        coefficients[li, :, f] = c[:, k]
                        native[li, :, f] = v[:, k]
                        scales[li, :, f] = d
                        alpha[li, :, f] = aa
                        ranks[li, f] = rank
                        singular[li, :, f] = s
                        # Necessary geometric conditions in the older full theorem.
                        # Delta can cover this finite halo for these analytic targets;
                        # this mask is NOT an error/recovery/arithmetic certificate.
                        theorem_conditions[li, f] = (
                            lam <= 1 and gamma * delta >= 4 * np.pi
                            and lam * radius >= 2 * np.log(2)
                            and n >= 16 * max(1, A_H**2))
                    if lam == cfg["standard_lambda"]:
                        discrepancy = np.linalg.norm(e @ c - (e * d) @ v, axis=0)
                        map_discrepancy = max(
                            map_discrepancy,
                            float(np.max(discrepancy / np.linalg.norm(ye[:, ids], axis=0))))
            assert len(centers) == n + 2 * radius + 1
            valid = ranks >= 0
            assert np.isfinite(errors[valid]).all()
            assert np.isnan(errors[~valid]).all()
            assert hashlib.sha256(source.read_bytes()).hexdigest() == digest
            record = dict(N=n, R=radius, W=len(centers), seconds=time.perf_counter() - start,
                          max_mapping_discrepancy_at_025=map_discrepancy)
            records.append(record)
            base.save(path, lambdas=lambdas, centers=centers, halo_per_side=radius,
                      eval_rel_l2=errors, train_rel_l2=train, ranks=ranks,
                      physical_coefficients=coefficients, native_coefficients=native,
                      scales=scales, alpha_upper=alpha, singular_values=singular,
                      geometric_theorem_conditions=theorem_conditions,
                      source_sha256=digest, record_json=json.dumps(record))
            print(f"Completed N={n}, R={radius}, W={len(centers)}; {record['seconds']:.1f}s",
                  flush=True)
    (data / "metadata.json").write_text(json.dumps({
        "scope": "Note geometry and scaling with user-authorized baseline numerical defaults",
        "baseline_config": cfg, "halo_rule": "ceil(a_H*sqrt(N))", "a_H": A_H,
        "halo_source": "Merged PR #2, experiments/expD06_fixed_center_scales/core.py:47",
        "envelope_rounding": "70-decimal interval arithmetic; upward binary64 endpoint conversion",
        "default_authorization": "Sam: use baselines as default if uncertain.",
        "baseline_defaults": {
            "relative_svd_cutoff": cfg["relative_svd_cutoff"],
            "effective_aliasing_budget": cfg["effective_aliasing_budget"],
            "delta_default": cfg["delta_default"], "delta_by_target": cfg["delta_by_target"],
            "bandwidth_bracket": [cfg["lambda_min"], cfg["lambda_max"]],
            "sampling": "Saved 2049 endpoint-inclusive training points and 8191 test midpoints",
            "evaluator": "Baseline float64 NumPy features and BLAS summation"},
        "certificate_limits": [
            "Baseline cutoff is not derived from the newer theorem's unspecified recovery budget.",
            "Ordinary NumPy/BLAS evaluation is not the older theorem's certified evaluator.",
            "Geometric eligibility does not certify sampling or recovery budgets."],
        "source_hashes": source_hashes, "runs": records}, indent=2) + "\n")
    return cfg, meta


def draw(cfg, meta):
    import os
    import tempfile
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "precisionmlps-matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(4, 6, figsize=(23.5, 14), sharex=True, sharey=True)
    fig.subplots_adjust(left=.065, right=.99, bottom=.15, top=.855, wspace=.17, hspace=.30)
    colors = ["#2866ad", "#d47816", "#7c3e9d"]
    summary = []
    for row, n in enumerate(cfg["interior_resolutions"]):
        with np.load(base.OUT / "data" / f"N{n}.npz") as z:
            lambdas, old = z["lambdas"].copy(), z["eval_rel_l2"].copy()
        with np.load(OUT / "data" / f"N{n}.npz") as z:
            np.testing.assert_array_equal(lambdas, z["lambdas"])
            new = z["eval_rel_l2"].copy()
            radius = int(z["halo_per_side"])
            eligible = z["geometric_theorem_conditions"].copy()
        for col, name in enumerate(cfg["targets"]):
            ax = axes[row, col]
            selection = meta["selections"][row][col]
            ax.set(xscale="log", yscale="log", xlim=(cfg["lambda_min"], cfg["lambda_max"]),
                   ylim=(1e-16, 2))
            invalid = np.pi / (n * selection["delta"])
            ax.axvspan(cfg["lambda_min"], max(cfg["lambda_min"], invalid), color=".94", zorder=-5)
            ax.plot(lambdas, old[:, 0, col], color=colors[0], lw=1.4)
            ax.plot(lambdas, old[:, 1, col], color=colors[1], lw=1.3, ls="--")
            ax.plot(lambdas, new[:, col], color=colors[2], lw=1.6)
            ax.axvline(.25, color=colors[0], ls=":", lw=.8, alpha=.7)
            at = int(np.flatnonzero(lambdas == .25)[0])
            record = dict(N=n, R=radius, target=name,
                          raw_R24_at_025=float(old[at, 0, col]),
                          scaled_R24_at_025=float(old[at, 1, col]),
                          scaled_sqrtN_at_025=float(new[at, col]))
            predicted = selection["lambda"]
            if predicted is not None:
                pi = int(np.argmin(abs(lambdas - predicted)))
                ax.axvline(predicted, color=colors[2], lw=.8, ls=":", alpha=.8)
                ax.scatter([predicted], [new[pi, col]], s=35, marker="D", color=colors[2],
                           edgecolors="white", linewidths=.5, zorder=5)
                record.update(lambda_pred=predicted,
                              old_at_prediction=float(old[pi, 1, col]),
                              new_at_prediction=float(new[pi, col]),
                              geometric_conditions_at_prediction=bool(eligible[pi, col]))
            else:
                ax.text(.96, .94, "No admissible\nfrequency-rule prediction",
                        transform=ax.transAxes, ha="right", va="top", fontsize=8.5, color=".35")
            summary.append(record)
            ax.set_yticks([1e-16, 1e-12, 1e-8, 1e-4, 1])
            ax.xaxis.set_major_locator(FixedLocator([.05, .1, .25, .5, 1]))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
            ax.xaxis.set_minor_locator(NullLocator())
            ax.grid(axis="y", alpha=.17, lw=.5)
            if row == 0:
                ax.set_title(base.TITLES[name], pad=32, fontsize=12)
                ax.text(.5, 1.06, base.EQUATIONS[name], transform=ax.transAxes,
                        ha="center", va="bottom", fontsize=9)
            if col == 0:
                ax.set_ylabel(rf"$N={n}$; halo $24\ \to\ {radius}$"
                              + "\nTest relative $L_2$ error", fontsize=11)
            if row == 3:
                ax.set_xlabel(r"Relative bandwidth $\lambda=\gamma h$")
    fig.suptitle(r"Note geometry and scaling: $R=\lceil\sqrt{N}\rceil$ versus the earlier fixed halo",
                 fontsize=20, y=.986)
    fig.text(.525, .947,
             r"Baseline numerical defaults retained where unspecified; same samples, bandwidths, and $\mathrm{rcond}=10^{-14}$",
             ha="center", fontsize=13)
    handles = [
        Line2D([], [], color=colors[0], lw=2, label=r"Original raw solve: $R=24$"),
        Line2D([], [], color=colors[1], lw=2, ls="--", label=r"Earlier note scaling: $R=24$"),
        Line2D([], [], color=colors[2], lw=2, label=r"Note scaling: $R=\lceil\sqrt{N}\rceil$, interval-rounded"),
        Line2D([], [], color=colors[0], ls=":", label=r"Reference $\lambda=0.25$"),
        Line2D([], [], color=colors[2], marker="D", ls="", label=r"Previous frequency-rule $\lambda_{\rm pred}$"),
        Patch(facecolor=".94", label=r"Envelope undefined: $\pi/(N\lambda)\geq\delta$")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.525, .055),
               ncol=3, frameon=False, fontsize=11)
    fig.text(.525, .036,
             r"2,049 training points; 8,191 test midpoints. $a_H=1$ from PR #2. "
             r"Current-$\lambda$ envelopes, including halo and bias.", ha="center", fontsize=10.5)
    fig.text(.525, .019,
             r"Defaults: $\varepsilon_{\rm eff}=2.22\times10^{-16}$; $\delta=0.25$ (Runge: $0.19$); "
             r"bandwidth bracket $[0.03,1.5]$; baseline FP64 evaluator.", ha="center", fontsize=10.5)
    fig.text(.525, .003,
             "Measured numerical fits, including outside sufficient theorem conditions; no full numerical certificate claimed.",
             ha="center", fontsize=10)
    fig.savefig(OUT / "comparison.png", dpi=190, facecolor="white")
    plt.close(fig)
    summary = [{k: (None if isinstance(v, float) and not np.isfinite(v) else v)
                for k, v in row.items()} for row in summary]
    (OUT / "data/selected_points.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(f"Saved {OUT / 'comparison.png'}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    if args.plot_only:
        meta = json.loads((base.OUT / "data/metadata.json").read_text())
        cfg = meta["config"]
    else:
        cfg, meta = run()
    draw(cfg, meta)


if __name__ == "__main__":
    main()
