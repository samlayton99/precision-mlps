"""expG04 sweep driver: band-count ablation x protocols x targets.

Usage (from repo root):
    uv run --extra dev python experiments/expG04_cascade_multiband/run.py [--smoke] [--plot]

Writes results incrementally to
results/checkpoint_G_generalization/expG04_cascade_multiband/data.json
(safe to re-run: finished cells are skipped).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
# Append expG03: it uniquely provides `solver`/`protocols`; appending keeps
# expG04's own `run`/`viz`/`cascade` ahead so they are not shadowed.
sys.path.append(str(REPO_ROOT / "experiments" / "expG03_extrapolation"))

import solver
import protocols as P
import cascade as C

RESULTS_DIR = (REPO_ROOT / "results" / "checkpoint_G_generalization"
               / "expG04_cascade_multiband")

N_DEFAULT = 128
COARSEN = 2
LAMBDAS = [0.25, 0.10, 0.05]
N_BANDS = [1, 2, 3]
TARGETS = {
    "sine": lambda x: np.sin(2 * np.pi * x),
    "runge": lambda x: 1.0 / (1.0 + 25.0 * x**2),
    "exp": lambda x: np.exp(x),
}


def _rel_over(u_hat, u_true, mask):
    if mask.sum() == 0:
        return float("nan")
    return solver.rel_l2(u_hat[mask], u_true[mask])


def evaluate_cell(n_bands, protocol, target, N=N_DEFAULT, coarsen=COARSEN):
    """Fit one (n_bands, protocol, target) cell and return its metric record."""
    t0 = time.time()
    f = TARGETS[target]
    x_train, x_test, regions = P.PROTOCOLS[protocol]()
    centers, gamma_vec, band_idx = C.cascade_geometry(N, LAMBDAS[:n_bands], coarsen)
    v, bias, _ = solver.fit(x_train, f(x_train), centers, gamma_vec)
    u_hat = solver.predict(x_test, centers, gamma_vec, v, bias)
    u_true = f(x_test)
    held = P.in_regions(x_test, regions)
    per_band = {int(k): float(np.linalg.norm(v[band_idx == k]))
                for k in range(n_bands)}
    return dict(
        n_bands=n_bands, protocol=protocol, target=target, N=N, coarsen=coarsen,
        rel_l2_entire=solver.rel_l2(u_hat, u_true),
        rel_l2_unmasked=_rel_over(u_hat, u_true, ~held),
        rel_l2_held=_rel_over(u_hat, u_true, held),
        linf_held=float(np.max(np.abs((u_hat - u_true)[held]))) if held.sum() else float("nan"),
        coeff_norm=float(np.linalg.norm(v)),
        per_band_norm=per_band,
        t_solve=time.time() - t0,
    )


def cell_key(rec):
    return (rec["n_bands"], rec["protocol"], rec["target"])


def load_records():
    path = RESULTS_DIR / "data.json"
    return json.loads(path.read_text()) if path.exists() else []


def save_records(records):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "data.json").write_text(json.dumps(records, indent=1))


def run_sweep(smoke=False):
    nbands = [3] if smoke else N_BANDS
    targets = ["runge"] if smoke else list(TARGETS)
    records = load_records()
    done = {cell_key(r) for r in records}
    for n_bands in nbands:
        for protocol in P.PROTOCOLS:
            for target in targets:
                if (n_bands, protocol, target) in done:
                    continue
                rec = evaluate_cell(n_bands, protocol, target)
                records.append(rec)
                save_records(records)
                print(f"nb={n_bands} {protocol} {target}: "
                      f"held={rec['rel_l2_held']:.2e} "
                      f"unmasked={rec['rel_l2_unmasked']:.2e} "
                      f"||v||={rec['coeff_norm']:.2e} "
                      f"bands={ {k: round(x,3) for k,x in rec['per_band_norm'].items()} }",
                      flush=True)
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()
    if args.plot:
        import viz
        viz.make_all_figures(load_records(), RESULTS_DIR, TARGETS, P, C, solver,
                             N_DEFAULT, LAMBDAS, COARSEN)
        return
    records = run_sweep(smoke=args.smoke)
    import viz
    viz.make_all_figures(records, RESULTS_DIR, TARGETS, P, C, solver,
                         N_DEFAULT, LAMBDAS, COARSEN)


if __name__ == "__main__":
    main()
