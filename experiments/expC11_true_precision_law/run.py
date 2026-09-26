"""expC11: the chirp precision law with the model and the least-squares solve both at p bits.

Every operation of the model (geometry, tanh features, readout) and of the solve (reference LAPACK
3.12.1 DGELSS, ported statement by statement) is correctly rounded in a binary format with a p-bit
significand. Spec: experiments/expC11_true_precision_law/SPEC.md. Library: src/precision/.

Run from the repository root:
    uv run --extra dev python experiments/expC11_true_precision_law/run.py --sweep      # p = 8..53
    uv run --extra dev python experiments/expC11_true_precision_law/run.py --anchors    # fp32/fp64 vs reference LAPACK
    uv run --extra dev python experiments/expC11_true_precision_law/run.py --standard   # numpy/scipy pipelines, for comparison
    uv run --extra dev python experiments/expC11_true_precision_law/plot.py
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.precision import pbit, reference_lapack  # noqa: E402

OUT = ROOT / "results/checkpoint_C_geometry/expC11_true_precision_law"
DATA = OUT / "data"
PREDICTIONS = ROOT / "results/checkpoint_C_geometry/expC09_bandwidth_figures/precision_law_W1024/data/config.json"
CONFIG = {
    "target": "chirp", "formula": "sin(8 pi (x + 1)^2)", "width": 1024, "halo_per_side": 24,
    "train_points": 4801, "eval_points": 8001, "precisions": list(range(8, 54)),
    "cutoff_factors": [2, 1, 8, 32],  # DGELSS RCOND = factor * 2^-p; the first (2^(1-p)) is the panel's
    "exponent_range": [pbit.PANEL_EMIN, pbit.PANEL_EMAX],
    "solver": "reference LAPACK 3.12.1 DGELSS, unblocked, ported to p-bit arithmetic",
}
SOURCES = [Path(__file__), Path(__file__).with_name("SPEC.md"), ROOT / "src/precision/pbit.py",
           ROOT / "src/precision/pbit_emul.c", ROOT / "src/precision/pbit_native.c",
           ROOT / "src/precision/pbit_algo.h", ROOT / "src/precision/lapack_gelss.h",
           ROOT / "src/precision/reference_lapack.py", PREDICTIONS]


def target(x):
    return np.sin(8 * np.pi * (x + 1) ** 2)


def lambdas() -> dict[int, float]:
    """The refined-rule bandwidths of expC09 (chosen offline, per p); rechecked against the rule."""
    from experiments.expC09_bandwidth_figures.run import selector
    pred = json.loads(PREDICTIONS.read_text())["predictions"]
    out = {}
    for pr in pred:
        assert pr["width"] == CONFIG["width"] and pr["halo"] == CONFIG["halo_per_side"]
        theta = 2 / pr["N"] * pr["omega_scale"]
        assert abs(selector.log_alias_score("tanh", pr["refined"]["lambda"], theta) - np.log(2.0 ** (1 - pr["p"]))) < 1e-11
        out[pr["p"]] = pr["refined"]["lambda"]
    return out


def geometry():
    W, H = CONFIG["width"], CONFIG["halo_per_side"]
    return W - 2 * H - 1, H


def _tool_version(cmd):
    return subprocess.run(cmd, capture_output=True, text=True).stdout.splitlines()[0]


def manifest() -> dict:
    return {"config": CONFIG,
            "sources": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in SOURCES},
            "lapack_tarball_sha256": reference_lapack.TARBALL_SHA256,
            "cc": _tool_version(["cc", "--version"]), "gfortran": _tool_version(["gfortran", "--version"]),
            "machine": platform.platform()}


def measure(p: int, lam: float) -> tuple[dict, dict]:
    N, H = geometry()
    x = np.linspace(-1, 1, CONFIG["train_points"])
    xe = np.linspace(-1, 1, CONFIG["eval_points"])
    fmt = pbit.panel_format(p)
    t = time.monotonic()
    r = pbit.run(x, target(x), xe, N, H, lam, fmt, rconds=[k * 2.0 ** -p for k in CONFIG["cutoff_factors"]])
    elapsed = time.monotonic() - t
    truth = target(xe)
    errs = [float(np.linalg.norm(f - truth) / np.linalg.norm(truth)) for f in r["fit_all"]]
    for arr in (r["weights_all"], r["fit_all"], r["sigma"]):
        assert np.array_equal(pbit.round_p(arr.ravel(), fmt), arr.ravel())
    row = {"p": p, "format": list(fmt), "width": CONFIG["width"], "N": N, "lambda": lam,
           "lambda_p": r["lambda_p"], "relative_l2": errs[0], "linf": float(np.max(np.abs(r["fit"] - truth))),
           "rank": r["rank"], "cutoff_factors": CONFIG["cutoff_factors"], "relative_l2_by_cutoff": errs,
           "rank_by_cutoff": r["rank_all"], "sigma_max": float(r["sigma"][0]),
           "readout_norm": float(np.linalg.norm(r["weights"])), "events": r["events"], "seconds": elapsed}
    model = {"weights": r["weights_all"], "fit": r["fit_all"], "sigma": r["sigma"]}
    return row, model


def sweep(workers: int):
    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / "models").mkdir(exist_ok=True)
    man = manifest()
    path = DATA / "config.json"
    if path.exists():
        assert json.loads(path.read_text()) == man, "sources or config changed since the saved measurements"
    else:
        path.write_text(json.dumps(man, indent=2) + "\n")
    lam = lambdas()
    rows_path = DATA / "measurements.jsonl"
    rows = [json.loads(s) for s in rows_path.read_text().splitlines()] if rows_path.exists() else []
    done = {r["p"] for r in rows}
    pending = sorted((p for p in CONFIG["precisions"] if p not in done), key=lambda p: p)
    start = time.monotonic()
    with ProcessPoolExecutor(workers) as ex, rows_path.open("a") as fh:
        futures = {ex.submit(measure, p, lam[p]): p for p in pending}
        for fut in as_completed(futures):
            row, model = fut.result()
            np.savez(DATA / "models" / f"p{row['p']}.npz", **model)
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            rows.append(row)
            print(f"p={row['p']:2d}  error={row['relative_l2']:.3e}  error*2^p={row['relative_l2'] * 2 ** row['p']:8.1f}  "
                  f"rank={row['rank']}  ({row['seconds']:.0f}s, elapsed {time.monotonic() - start:.0f}s)", flush=True)
    rows.sort(key=lambda r: r["p"])
    assert [r["p"] for r in rows] == CONFIG["precisions"]
    summary = [{"p": r["p"], "width": r["width"], "rule_lambda": r["lambda"], "rule_error": r["relative_l2"]}
               for r in rows]
    (DATA / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def anchors():
    """Full-size bit-identity: the port in exact fp32 / fp64 against reference LAPACK S/DGELSS on
    the same matrix, and the native builds against the emulator; plus the panel-range runs at
    p = 24 and 53 against the IEEE-range runs."""
    DATA.mkdir(parents=True, exist_ok=True)
    lam = lambdas()
    N, H = geometry()
    x = np.linspace(-1, 1, CONFIG["train_points"])
    xe = np.linspace(-1, 1, CONFIG["eval_points"])
    out = {}
    for name, fmt, native, dtype in [("fp32", pbit.FP32, "f32", np.float32), ("fp64", pbit.FP64, "f64", np.float64)]:
        p = fmt[0]
        r = pbit.run(x, target(x), xe, N, H, lam[p], fmt)
        rn = pbit.run(x, target(x), xe, N, H, lam[p], fmt, backend=native)
        phi = pbit.features(x, N, H, lam[p], fmt)
        A = np.hstack([phi, np.ones((x.size, 1))])
        ref = reference_lapack.gelss(A.astype(dtype), pbit.round_p(target(x), fmt).astype(dtype), 2.0 ** (1 - p), dtype)
        fit_ref = pbit.evaluate(ref["x"], xe, N, H, lam[p], fmt)
        panel = pbit.run(x, target(x), xe, N, H, lam[p], pbit.panel_format(p))
        truth = target(xe)
        err = lambda f: float(np.linalg.norm(f - truth) / np.linalg.norm(truth))
        out[name] = {
            "format": list(fmt), "lambda": lam[p], "reference_info": ref["info"], "reference_rank": ref["rank"],
            "port_rank": r["rank"],
            "port_vs_reference_bit_identical": {"sigma": bool(np.array_equal(r["sigma"], ref["sigma"])),
                                                "weights": bool(np.array_equal(r["weights"], ref["x"])),
                                                "fit": bool(np.array_equal(r["fit"], fit_ref))},
            "native_build_vs_emulator_bit_identical": bool(np.array_equal(rn["weights"], r["weights"])
                                                           and np.array_equal(rn["fit"], r["fit"])
                                                           and np.array_equal(rn["sigma"], r["sigma"])),
            "emulator_events": r["events"], "error_ieee_range": err(r["fit"]), "error_panel_range": err(panel["fit"]),
            "panel_range_bit_identical_to_ieee_range": bool(np.array_equal(panel["weights"], r["weights"])
                                                            and np.array_equal(panel["fit"], r["fit"])),
        }
        print(name, json.dumps(out[name]), flush=True)
    (DATA / "anchors.json").write_text(json.dumps(out, indent=2) + "\n")


def standard():
    """What the everyday libraries give (not p-bit faithful; for comparison only): features with
    numpy tanh in float32/float64, solved by scipy's LAPACK drivers with cond = 2^(1-p), evaluated
    in the same dtype."""
    import scipy.linalg as sl
    lam = lambdas()
    N, H = geometry()
    h = 2.0 / N
    centers = -1.0 + np.arange(-H, N + H + 1) * h
    x = np.linspace(-1, 1, CONFIG["train_points"])
    xe = np.linspace(-1, 1, CONFIG["eval_points"])
    truth = target(xe)
    out = {}
    for name, dtype, p in [("fp32", np.float32, 24), ("fp64", np.float64, 53)]:
        g = dtype(lam[p] / h)
        c = centers.astype(dtype)
        A = np.hstack([np.tanh(g * (x.astype(dtype)[:, None] - c[None, :])), np.ones((x.size, 1), dtype)])
        Ae = np.hstack([np.tanh(g * (xe.astype(dtype)[:, None] - c[None, :])), np.ones((xe.size, 1), dtype)])
        y = target(x).astype(dtype)
        out[name] = {}
        for driver in ("gelsd", "gelss", "gelsy"):
            w = sl.lstsq(A, y, cond=2.0 ** (1 - p), lapack_driver=driver)[0]
            fit = (Ae @ w).astype(np.float64)
            out[name][driver] = float(np.linalg.norm(fit - truth) / np.linalg.norm(truth))
        print(name, out[name], flush=True)
    (DATA / "standard.json").write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--anchors", action="store_true")
    ap.add_argument("--standard", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    if not (args.sweep or args.anchors or args.standard):
        ap.error("choose --sweep, --anchors and/or --standard")
    if args.sweep:
        sweep(args.workers)
    if args.anchors:
        anchors()
    if args.standard:
        standard()
