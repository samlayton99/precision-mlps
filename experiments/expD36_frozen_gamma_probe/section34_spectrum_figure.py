"""Plot finite relative rates and theorem intervals from the saved evaluations."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bounds", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    bandwidths = [1/32, 1/16, 1/8, 1/4]
    ranks = [4, 14, 30]
    rows, sources = [], []
    for bandwidth in bandwidths:
        path = args.bounds / f"lambda{bandwidth:g}_q16_p24.npz"
        meta_path = path.with_suffix(".json")
        meta = json.loads(meta_path.read_text())
        with np.load(path) as data:
            assert abs(data["actual_target_weights"].sum() - 1) < 1e-10
            for rank in ranks:
                index = rank - 1
                lower = float(data["rho_lower"][index])
                actual = float(data["actual_rho"][index])
                upper = float(data["rho_upper"][index])
                assert 0 < lower <= actual <= upper <= 1
                rows.append(dict(bandwidth=bandwidth, gamma=meta["gamma"],
                                 rank=rank, lower=lower, actual=actual, upper=upper,
                                 target_weight=float(data["actual_target_weights"][index])))
        sources.append(dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                            metadata_sha256=hashlib.sha256(meta_path.read_bytes()).hexdigest(),
                            input_sha256=meta["input_sha256"],
                            numerical_status=meta["numerical_status"]))

    plt.rcParams.update({"font.family": "sans-serif", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42,
                         "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8})
    fig, ax = plt.subplots(figsize=(5.5, 2.85), layout="constrained")
    palette = ["#0072B2", "#009E73", "#CC6677"]
    for rank, color in zip(ranks, palette):
        selected = [row for row in rows if row["rank"] == rank]
        lower, actual, upper = [np.array([row[key] for row in selected])
                                for key in ("lower", "actual", "upper")]
        ax.fill_between(bandwidths, lower, upper, color=color, alpha=.16, linewidth=0)
        ax.plot(bandwidths, lower, color=color, lw=.85, ls="--")
        ax.plot(bandwidths, upper, color=color, lw=.85, ls="--")
        ax.plot(bandwidths, actual, color=color, lw=1.6, marker="o", ms=4)
        ax.annotate(f"rank {rank}", (bandwidths[-1], actual[-1]),
                    xytext=(8, 0), textcoords="offset points", va="center", color=color)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlim(.028, .37)
    ax.set_ylim(1e-9, .3)
    ax.set_xticks(bandwidths, ["1/32", "1/16", "1/8", "1/4"])
    ax.set_yticks([1e-8, 1e-6, 1e-4, 1e-2])
    ax.set_xlabel(r"Relative bandwidth $\lambda=\gamma h$ (fixed spacing $h$)")
    ax.set_ylabel(r"Relative eigenvalue $\rho_i=\mu_i/\mu_1$")
    ax.grid(axis="y", which="major", color="#dddddd", lw=.6)
    ax.set_axisbelow(True)
    handles = [Line2D([], [], color="#444444", marker="o", lw=1.6, ms=4,
                      label="Finite dictionary"),
               Line2D([], [], color="#444444", ls="--", lw=.85, label="Theorem interval")]
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=8)
    fig.savefig(args.output / "relative_rates.pdf")
    fig.savefig(args.output / "relative_rates.png", dpi=220)
    plt.close(fig)
    provenance = dict(sources=sources, rows=rows,
                      rank_selection="Illustrative ranks selected after inspecting the saved spectra; "
                                     "ordered rank does not track a fixed eigenvector across bandwidths.",
                      interpolation="Lines connect four evaluated bandwidths as visual guides.")
    (args.output / "relative_rates.json").write_text(json.dumps(provenance, indent=2) + "\n")
    endpoint_rows = [row for row in rows if row["rank"] == 30]
    print(json.dumps(dict(verified_intervals=len(rows),
                         rank30_rate_ratio=endpoint_rows[-1]["actual"]/endpoint_rows[0]["actual"])))


if __name__ == "__main__":
    main()
