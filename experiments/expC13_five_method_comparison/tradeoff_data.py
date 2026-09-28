"""Data for the neuron-precision trade-off figures (QUILLS and ChebNet only).

Grid: hidden neurons N from 32 to 1024 (four per doubling); precision p = 9, 11, ..., 51 in the p-bit
format (p, -958, 959), and p = 53 in native IEEE binary64 with standard numpy/scipy routines.
Every row stores the validation-grid error (the figures' metric) and the reporting-grid error.

    .venv/bin/python experiments/expC13_five_method_comparison/tradeoff_data.py --quills-pbit   # widths the sweep lacks
    .venv/bin/python experiments/expC13_five_method_comparison/tradeoff_data.py --fp64          # p = 53, both methods
Writes data/tradeoff_rows.jsonl (resumable).
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from fractions import Fraction
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pfloat  # noqa: E402
from scipy.fft import dct  # noqa: E402
from scipy.linalg import lstsq  # noqa: E402

from experiments.expC13_five_method_comparison import common as C  # noqa: E402
from experiments.expC13_five_method_comparison.methods import chebnet, quills  # noqa: E402
from experiments.expC13_five_method_comparison.selection import report_meter, _meter  # noqa: E402

OUT = C.OUT / "data" / "tradeoff_rows.jsonl"
TARGETS = ["exp", "sine", "runge", "chirp"]
N_GRID = [int(round(32 * 2 ** (k / 4))) for k in range(21)]          # 32 .. 1024
P_GRID = list(range(9, 52, 2))                                      # p-bit; p = 53 is native fp64
SWEEP_WIDTHS = [64, 96, 128, 192, 256, 384, 512, 768, 1024]         # already in the sweep (halo 24)
NEW_WIDTHS = [w for w in N_GRID if w not in SWEEP_WIDTHS and w <= 512] + [64]
NUMPY_F = {"exp": np.exp, "sine": lambda x: np.sin(4 * np.pi * x), "runge": lambda x: 1 / (1 + 25 * x * x),
           "chirp": lambda x: np.sin(8 * np.pi * (x + 1) ** 2)}


def halo(W: int) -> int:
    return min(24, W // 4)


def _grids():
    xv = np.array([float(v) for v in C.midpoint_points(C.VALIDATION_POINTS)])
    xr = np.array([float(v) for v in C.linspace_points(C.REPORT_POINTS)])
    return xv, xr


def _errors(target, yv, yr):
    F = pfloat.FP64
    return (_meter(target, "validation").rel_l2(pfloat.array(yv, F)),
            report_meter(target).rel_l2(pfloat.array(yr, F)))


# ---------------------------------------------------------------- QUILLS at p bits (new widths)

def quills_pbit(target, W, p):
    pfloat.set_num_threads(1)
    H = halo(W)
    try:
        lam = quills.rule_lambda(target, p, W, H)
    except ValueError:
        return {"target": target, "method": "quills", "pipeline": "pbit", "p": p, "width": W, "halo": H,
                "status": "rule_inadmissible"}
    net, meta = quills.build(target, p, width=W, halo=H, lam=lam)
    F = C.fmt(p)
    xv = C.inputs(C.midpoint_points(C.VALIDATION_POINTS), F)
    xr = C.inputs(C.linspace_points(C.REPORT_POINTS), F)
    return {"target": target, "method": "quills", "pipeline": "pbit", "p": p, "width": W, "halo": H, "lambda": lam,
            "neurons": W, "val_rel_l2": _meter(target, "validation").rel_l2(net.forward(xv)),
            "rel_l2": report_meter(target).rel_l2(net.forward(xr)), "status": "ok"}


# ---------------------------------------------------------------- native fp64, standard routines

def quills_fp64(target, W):
    H = halo(W)
    try:
        lam = quills.rule_lambda(target, 53, W, H)
    except ValueError:
        return {"target": target, "method": "quills", "pipeline": "fp64", "p": 53, "width": W, "status": "rule_inadmissible"}
    N = W - 2 * H - 1
    h = 2.0 / N
    centers = -1.0 + np.arange(-H, N + H + 1) * h
    gamma = lam / h
    bias = -gamma * centers
    x = np.linspace(-1.0, 1.0, quills.TRAIN_POINTS)
    A = np.column_stack([np.tanh(np.outer(x, np.full(W, gamma)) + bias), np.ones(x.size)])
    w = lstsq(A, NUMPY_F[target](x), cond=2.0 ** -52, lapack_driver="gelsd")[0]
    xv, xr = _grids()
    net = lambda z: np.tanh(np.outer(z, np.full(W, gamma)) + bias) @ w[:W] + w[W]  # noqa: E731
    ev, er = _errors(target, net(xv), net(xr))
    return {"target": target, "method": "quills", "pipeline": "fp64", "p": 53, "width": W, "halo": H, "lambda": lam,
            "neurons": W, "val_rel_l2": ev, "rel_l2": er, "status": "ok"}


def chebyshev_fp64(target, degree, points=4096):
    theta = np.pi * (np.arange(points) + 0.5) / points
    c = dct(NUMPY_F[target](np.cos(theta)), type=2) / points
    c[0] *= 0.5
    return c[:degree + 1]


def chebnet_fp64(target, degree, normalize):
    F = pfloat.FP64
    cheb = pfloat.array(chebyshev_fp64(target, degree), F)
    net = chebnet.build(cheb, degree, normalize)                    # weights formed in binary64
    layers = []
    for L in net.layers:                                            # dense float64 matrices for numpy
        n_in = int(L.idx.max()) + 1 if L.mask.any() else 1
        Wd = np.zeros((L.bias.shape[0], n_in))
        wv = L.w.to_numpy()
        for i, t in zip(*np.nonzero(L.mask)):
            Wd[i, L.idx[i, t]] += wv[i, t]
        layers.append((Wd, L.bias.to_numpy()))

    def forward(z):
        hcur = z[None, :]
        for k, (Wd, b) in enumerate(layers):
            hcur = Wd[:, :hcur.shape[0]] @ hcur + b[:, None]
            if k < len(layers) - 1:
                hcur = np.maximum(hcur, 0.0) ** 2
        return hcur[0]

    xv, xr = _grids()
    ev, er = _errors(target, forward(xv), forward(xr))
    return {"target": target, "method": "chebnet", "pipeline": "fp64", "p": 53, "degree": degree,
            "normalize": normalize, "neurons": net.neurons(), "val_rel_l2": ev, "rel_l2": er, "status": "ok"}


def run(jobs):
    done = set()
    if OUT.exists():
        for s in OUT.read_text().splitlines():
            r = json.loads(s)
            done.add(json.dumps([r["target"], r["method"], r["pipeline"], r["p"], r.get("width"), r.get("degree"),
                                 r.get("normalize")]))
    todo = [j for j in jobs if json.dumps(j[1]) not in done]
    print(f"{len(todo)} jobs", flush=True)
    with ProcessPoolExecutor(9) as ex, OUT.open("a") as fh:
        futs = [ex.submit(fn, *args) for fn, _, args in todo]
        for i, fut in enumerate(as_completed(futs)):
            fh.write(json.dumps(fut.result()) + "\n")
            fh.flush()
            if i % 200 == 0:
                print(f"{i}/{len(todo)}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quills-pbit", action="store_true")
    ap.add_argument("--fp64", action="store_true")
    a = ap.parse_args()
    jobs = []
    if a.quills_pbit:
        jobs += [(quills_pbit, [t, "quills", "pbit", p, W, None, None], (t, W, p))
                 for t in TARGETS for W in sorted(set(NEW_WIDTHS), reverse=True) for p in P_GRID]
    if a.fp64:
        jobs += [(quills_fp64, [t, "quills", "fp64", 53, W, None, None], (t, W)) for t in TARGETS for W in N_GRID]
        jobs += [(chebnet_fp64, [t, "chebnet", "fp64", 53, None, n, norm], (t, n, norm))
                 for t in TARGETS for n in chebnet.NEURON_DEGREES for norm in (False, True)]
    run(jobs)


if __name__ == "__main__":
    main()
