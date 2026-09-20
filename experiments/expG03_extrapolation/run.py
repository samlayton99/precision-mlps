"""expG03 sweep driver: 3 hold-out protocols x targets x lambda.

Usage (from repo root):
    uv run --extra dev python experiments/expG03_extrapolation/run.py [--smoke] [--plot]

Writes results incrementally to
results/checkpoint_G_generalization/expG03_extrapolation/data.json
(safe to re-run: finished cells are skipped).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import solver
import protocols as P

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = (REPO_ROOT / "results" / "checkpoint_G_generalization"
               / "expG03_extrapolation")

N_DEFAULT = 128
LAMBDAS = [0.25, 0.10, 0.05]
TARGETS = {
    "sine": lambda x: np.sin(2 * np.pi * x),
    "runge": lambda x: 1.0 / (1.0 + 25.0 * x**2),
    "exp": lambda x: np.exp(x),
}


def _rel_over(u_hat, u_true, mask):
    if mask.sum() == 0:
        return float("nan")
    return solver.rel_l2(u_hat[mask], u_true[mask])


def evaluate_cell(protocol, target, lam, N=N_DEFAULT):
    """Fit one (protocol, target, lambda) cell and return its metric record."""
    t0 = time.time()
    f = TARGETS[target]
    x_train, x_test, regions = P.PROTOCOLS[protocol]()
    centers, gamma_vec = solver.geometry(N, lam)
    v, bias, _ = solver.fit(x_train, f(x_train), centers, gamma_vec)
    u_hat = solver.predict(x_test, centers, gamma_vec, v, bias)
    u_true = f(x_test)
    held = P.in_regions(x_test, regions)
    return dict(
        protocol=protocol, target=target, lam=lam, N=N,
        rel_l2_entire=solver.rel_l2(u_hat, u_true),
        rel_l2_unmasked=_rel_over(u_hat, u_true, ~held),
        rel_l2_held=_rel_over(u_hat, u_true, held),
        linf_held=float(np.max(np.abs((u_hat - u_true)[held]))) if held.sum() else float("nan"),
        coeff_norm=float(np.linalg.norm(v)),
        t_solve=time.time() - t0,
    )


def cell_key(rec):
    return (rec["protocol"], rec["target"], rec["lam"])


def load_records():
    path = RESULTS_DIR / "data.json"
    return json.loads(path.read_text()) if path.exists() else []


def save_records(records):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "data.json").write_text(json.dumps(records, indent=1))


def run_sweep(smoke=False):
    protocols_ = list(P.PROTOCOLS)
    targets = ["runge"] if smoke else list(TARGETS)
    lams = [0.25] if smoke else LAMBDAS
    records = load_records()
    done = {cell_key(r) for r in records}
    for protocol in protocols_:
        for target in targets:
            for lam in lams:
                if (protocol, target, lam) in done:
                    continue
                rec = evaluate_cell(protocol, target, lam)
                records.append(rec)
                save_records(records)
                print(f"{protocol} {target} lam={lam}: "
                      f"held={rec['rel_l2_held']:.2e} "
                      f"unmasked={rec['rel_l2_unmasked']:.2e} "
                      f"||v||={rec['coeff_norm']:.2e}", flush=True)
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()
    if args.plot:
        import viz
        viz.make_all_figures(load_records(), RESULTS_DIR, TARGETS, P, solver, N_DEFAULT)
        return
    records = run_sweep(smoke=args.smoke)
    import viz
    viz.make_all_figures(records, RESULTS_DIR, TARGETS, P, solver, N_DEFAULT)


if __name__ == "__main__":
    main()
