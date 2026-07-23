"""expD06 -- Does gradient descent scale geometrically with width?

QI init vs Xavier init, both trained with pure full-batch Adam (NO lstsq refit).
The question: as hidden width grows, does a QI-structured initialization let GD
descend geometrically toward the fp64 floor, while a standard Xavier init
plateaus?  See docs/superpowers/specs/2026-07-21-expD06-gd-width-scaling-design.md.

Three trained lines (pure Adam):
  - standard_xavier_affine      -- plateau baseline
  - exact_qi_oracle             -- GD *preserves* the geometric construction?
  - qi_geom_random_readout      -- QI geometry, random readout; can GD *learn* it?
Plus a dashed construction reference = the oracle family's UNTRAINED
(initial) eval error, which is recorded for free per cell.

All training/init machinery is imported verbatim from expD05 so the numbers are
directly comparable; this driver only reshapes the sweep into a width axis.

Usage:
    uv run --extra dev python experiments/expD06_gd_width_scaling/run.py --mode smoke
    uv run --extra dev python experiments/expD06_gd_width_scaling/run.py --mode full
    uv run --extra dev python experiments/expD06_gd_width_scaling/run.py --plot-only
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Load the expD05 driver as a module (it guards its CLI behind __main__, so the
# import is side-effect-free) and reuse its Case / RunConfig / run_case.
_D05_PATH = REPO_ROOT / "experiments" / "expD05_scale_init_story" / "run.py"
_spec = importlib.util.spec_from_file_location("expd05_run", _D05_PATH)
d05 = importlib.util.module_from_spec(_spec)
assert _spec and _spec.loader
sys.modules["expd05_run"] = d05  # dataclass() needs the module in sys.modules
_spec.loader.exec_module(d05)  # type: ignore[union-attr]

EXPERIMENT = "expD06_gd_width_scaling"
RESULTS_DIR = REPO_ROOT / "results" / "checkpoint_D_optimizers" / EXPERIMENT
FIGURES_DIR = RESULTS_DIR / "figures"
SUMMARY_CSV = RESULTS_DIR / "summary.csv"
REPORT_MD = RESULTS_DIR / "expD06_results.md"

# The three trained families, drawn in this order.
FAMILIES = ("standard_xavier_affine", "exact_qi_oracle", "qi_geom_random_readout")
FAMILY_LABELS = {
    "standard_xavier_affine": "Xavier init + GD",
    "exact_qi_oracle": "QI init (oracle) + GD",
    "qi_geom_random_readout": "QI geom, random readout + GD",
}
FAMILY_COLORS = {
    "standard_xavier_affine": "#d62728",   # red   -- expected plateau
    "exact_qi_oracle": "#2ca02c",          # green -- expected geometric descent
    "qi_geom_random_readout": "#1f77b4",   # blue  -- the open test
}
# Construction reference = untrained oracle initial eval error.
REF_FAMILY = "exact_qi_oracle"
FLOOR = 5e-14

DEFAULT_RESOLUTIONS = (32, 64, 128, 256, 512)
DEFAULT_TARGETS = ("runge", "exp")
DEFAULT_SEEDS = (0, 1, 2)
DEFAULT_STEPS = 20000
DEFAULT_LR = 1.0e-3
DEFAULT_N_TRAIN = 2003
DEFAULT_N_EVAL = 4001
DEFAULT_EVAL_INTERVAL = 250


def build_config(args: argparse.Namespace) -> "d05.RunConfig":
    if args.mode == "smoke":
        return d05.RunConfig(
            targets=("runge",),
            resolutions=(32, 64),
            seeds=(0,),
            families=FAMILIES,
            steps=args.steps or 200,
            eval_interval=args.eval_interval or 20,
            learning_rate=args.learning_rate or DEFAULT_LR,
            n_train=args.n_train or 257,
            n_eval=args.n_eval or 513,
            mode="smoke",
        )
    targets = tuple(args.targets) if args.targets else DEFAULT_TARGETS
    resolutions = tuple(args.resolutions) if args.resolutions else DEFAULT_RESOLUTIONS
    seeds = tuple(args.seeds) if args.seeds else DEFAULT_SEEDS
    return d05.RunConfig(
        targets=targets,
        resolutions=resolutions,
        seeds=seeds,
        families=FAMILIES,
        steps=args.steps or DEFAULT_STEPS,
        eval_interval=args.eval_interval or DEFAULT_EVAL_INTERVAL,
        learning_rate=args.learning_rate or DEFAULT_LR,
        n_train=args.n_train or DEFAULT_N_TRAIN,
        n_eval=args.n_eval or DEFAULT_N_EVAL,
        mode="full",
    )


def enumerate_cases(config: "d05.RunConfig") -> list["d05.Case"]:
    cases: list[d05.Case] = []
    for target in config.targets:
        for resolution in config.resolutions:
            for seed in config.seeds:
                for family in config.families:
                    cases.append(d05.Case(target=target, resolution=resolution, seed=seed, family=family))
    return cases


# Only the columns this experiment needs downstream; run_case computes many more.
KEEP_COLUMNS = (
    "target", "resolution", "width", "halo", "h", "seed", "family",
    "steps", "learning_rate", "n_train", "n_eval",
    "initial_eval_rel_l2", "final_eval_rel_l2", "best_eval_rel_l2",
    "best_step", "final_eval_linf", "final_train_rel_l2",
    "refit_eval_rel_l2", "lambda_final_median",
)


def run_matrix(config: "d05.RunConfig") -> list[dict[str, object]]:
    cases = enumerate_cases(config)
    rows: list[dict[str, object]] = []
    total = len(cases)
    for index, case in enumerate(cases, start=1):
        summary, _traces, _atoms = d05.run_case(case, config)
        row = {key: summary[key] for key in KEEP_COLUMNS if key in summary}
        rows.append(row)
        print(
            f"[{index:3d}/{total}] {case.target:>6s} N={case.resolution:<4d} "
            f"W={summary['width']:<4d} seed={case.seed} {case.family:<24s} "
            f"trained={summary['best_eval_rel_l2']:.2e} "
            f"init={summary['initial_eval_rel_l2']:.2e}",
            flush=True,
        )
    return rows


def write_summary(rows: list[dict[str, object]]) -> None:
    SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        SUMMARY_CSV.write_text("")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with SUMMARY_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_summary() -> list[dict[str, str]]:
    with SUMMARY_CSV.open(newline="") as f:
        return list(csv.DictReader(f))


def _row_key(row: dict) -> tuple:
    return (row["target"], int(row["resolution"]), int(row["seed"]), row["family"])


def merge_and_write(new_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Merge new_rows into an existing summary.csv (new rows replace matching
    target/resolution/seed/family keys). Used by --append to add a target facet
    without re-running the whole matrix."""
    existing: dict[tuple, dict] = {}
    if SUMMARY_CSV.exists() and SUMMARY_CSV.stat().st_size > 0:
        for r in read_summary():
            existing[_row_key(r)] = dict(r)
    for r in new_rows:
        existing[_row_key(r)] = r
    merged = list(existing.values())
    write_summary(merged)
    return merged


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _median_band(values: list[float]) -> tuple[float, float, float]:
    arr = np.array([v for v in values if np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return math.nan, math.nan, math.nan
    return float(np.median(arr)), float(arr.min()), float(arr.max())


def plot(rows: list[dict[str, str]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker

    targets: list[str] = []
    for r in rows:
        if r["target"] not in targets:
            targets.append(r["target"])

    n = len(targets)
    fig, axes = plt.subplots(1, n, figsize=(6.2 * n, 5.0), squeeze=False)
    axes = axes[0]

    for ax, target in zip(axes, targets):
        trows = [r for r in rows if r["target"] == target]
        widths = sorted({int(r["width"]) for r in trows})

        # Trained lines: median best_eval_rel_l2 over seeds, min-max band.
        for family in FAMILIES:
            med, lo, hi = [], [], []
            for w in widths:
                vals = [
                    float(r["best_eval_rel_l2"])
                    for r in trows
                    if int(r["width"]) == w and r["family"] == family
                ]
                m, a, b = _median_band(vals)
                med.append(m)
                lo.append(a)
                hi.append(b)
            color = FAMILY_COLORS[family]
            ax.plot(widths, med, "-o", color=color, label=FAMILY_LABELS[family], zorder=3)
            ax.fill_between(widths, lo, hi, color=color, alpha=0.15, zorder=1)

        # Construction reference: untrained oracle initial eval error.
        ref_med = []
        for w in widths:
            vals = [
                float(r["initial_eval_rel_l2"])
                for r in trows
                if int(r["width"]) == w and r["family"] == REF_FAMILY
            ]
            m, _, _ = _median_band(vals)
            ref_med.append(m)
        ax.plot(
            widths, ref_med, "--", color="0.35", lw=1.8,
            label="QI construction (untrained ref)", zorder=2,
        )

        ax.axhline(FLOOR, color="0.6", ls=":", lw=1.2, zorder=0)
        ax.text(widths[0], FLOOR * 1.4, "fp64 floor", color="0.5", fontsize=8, va="bottom")

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks(widths)
        ax.set_xticklabels([str(w) for w in widths])
        ax.get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.tick_params(axis="x", which="minor", length=0)
        ax.set_xlabel("hidden width $W$")
        ax.set_ylabel("eval relative $L_2$")
        ax.set_title(target)
        ax.grid(True, which="both", ls=":", alpha=0.4)
        ax.legend(
            loc="lower center", bbox_to_anchor=(0.5, 1.06),
            ncol=1, fontsize=8, borderaxespad=0, framealpha=0.9,
        )

    fig.suptitle(
        "expD06: gradient descent width-scaling -- QI init vs Xavier init "
        "(pure Adam, no refit)",
        y=1.02, fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / "error_vs_width.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def parse_int_list(value: str | None) -> tuple[int, ...] | None:
    if not value:
        return None
    return tuple(int(v) for v in value.split(","))


def parse_str_list(value: str | None) -> tuple[str, ...] | None:
    if not value:
        return None
    return tuple(v.strip() for v in value.split(","))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "full"), default="full")
    parser.add_argument("--plot-only", action="store_true", help="re-plot from existing summary.csv")
    parser.add_argument("--append", action="store_true", help="merge into existing summary.csv instead of overwriting")
    parser.add_argument("--targets", type=parse_str_list, default=None)
    parser.add_argument("--resolutions", type=parse_int_list, default=None)
    parser.add_argument("--seeds", type=parse_int_list, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--eval-interval", dest="eval_interval", type=int, default=None)
    parser.add_argument("--learning-rate", dest="learning_rate", type=float, default=None)
    parser.add_argument("--n-train", dest="n_train", type=int, default=None)
    parser.add_argument("--n-eval", dest="n_eval", type=int, default=None)
    args = parser.parse_args(argv)

    if args.plot_only:
        rows = read_summary()
        if not rows:
            print("no summary.csv to plot", file=sys.stderr)
            return 1
        plot(rows)
        return 0

    config = build_config(args)
    print(
        f"expD06 {config.mode}: targets={config.targets} resolutions={config.resolutions} "
        f"seeds={config.seeds} families={len(config.families)} steps={config.steps}",
        flush=True,
    )
    rows = run_matrix(config)
    if args.append:
        all_rows = merge_and_write(rows)
        print(f"wrote {SUMMARY_CSV} ({len(rows)} new, {len(all_rows)} total rows)")
    else:
        write_summary(rows)
        all_rows = rows
        print(f"wrote {SUMMARY_CSV} ({len(rows)} rows)")
    plot([{k: str(v) for k, v in r.items()} for r in all_rows])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
