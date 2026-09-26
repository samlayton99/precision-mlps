"""QUILLS: uniform tanh geometry from the bandwidth rule, readout by least squares; all at p bits.

Geometry (as expC11 SPEC, in the shared affine form): N = W - 2H - 1 intervals, h = div(2, Q(N)),
centers c_j = add(-1, mul(Q(j), h)) for j = -H..N+H, gamma = div(Q(lambda), h); hidden slope gamma
and bias neg(mul(gamma, c_j)). Features tanh_p(add(mul(gamma, x), bias_j)) on the M training points
with the oracle's labels; readout from pfloat.lstsq (reference LAPACK DGELSS ported to p bits,
RCOND = 2^(1-p)) on [features, 1]. lambda is the expC09 refined rule's value for (W, H, p, target),
an external hyperparameter rounded once.
"""
from __future__ import annotations

from fractions import Fraction

import numpy as np
import pfloat

from experiments.expC13_five_method_comparison import common as C
from experiments.expC13_five_method_comparison.selection import Selection

WIDTH, HALO, TRAIN_POINTS = 1024, 24, 4801


def rule_lambda(target: str, p: int, width: int = WIDTH, halo: int = HALO) -> float:
    """expC09's refined rule (first alias pair at the target's frequency scale) with e_tol = 2^(1-p);
    the basic rule for exp, which has no oscillation scale. Computed offline in binary64."""
    from experiments.expC09_bandwidth_figures.run import geometry as g09, selector
    from experiments.expC09_bandwidth_figures.additional_targets import frequency_scale
    _, spacing, _ = g09(width, halo)
    omega = None if target == "exp" else frequency_scale(target)
    return float(selector.choose_lambda("tanh", spacing=spacing, e_tol=2.0 ** (1 - p), omega_scale=omega)["lambda"])


def geometry(F: pfloat.Format, width: int, halo: int, lam: float):
    N = width - 2 * halo - 1
    h = C.constant(2, F) / C.constant(N, F)
    j = pfloat.array(np.arange(-halo, N + halo + 1), F)
    centers = C.constant(-1, F) + j * h
    gamma = C.constant(Fraction(lam), F) / h
    slope = pfloat.ones(width, F) * gamma
    bias = -(slope * centers)
    return slope, bias, {"N": N, "halo": halo, "lambda": lam}


def build(target: str, p: int, *, width: int = WIDTH, halo: int = HALO, train_points: int = TRAIN_POINTS,
          lam: float | None = None):
    F = C.fmt(p)
    lam = rule_lambda(target, p, width, halo) if lam is None else lam
    slope, bias, meta = geometry(F, width, halo, lam)
    x = C.inputs(C.linspace_points(train_points), F)
    y = C.sample(target, x)
    shell = C.TanhNet(slope, bias, pfloat.zeros(width, F), pfloat.array(0, F))
    A = pfloat.concatenate([shell.features(x), pfloat.ones((train_points, 1), F)], axis=1)
    w, _, rank, _ = pfloat.lstsq(A, y, rcond=Fraction(2) ** (1 - p))
    net = C.TanhNet(slope, bias, w[:width], w[width])
    meta.update(rank=int(rank), train_points=train_points)
    return net, meta


WIDTHS = [64, 96, 128, 192, 256, 384, 512, 768, 1024]


def select(ctx, widths=WIDTHS) -> Selection:
    """Width W chosen on the validation grid (halo 24, as expC09's width panel); lambda from the rule
    at each W. widths=[1024] is the expC11 configuration (row quills_w1024)."""
    sel = Selection("quills", ctx.target, ctx.p)
    for W in widths:
        try:
            lam = rule_lambda(ctx.target, ctx.p, W, HALO)
        except ValueError:          # the grid frequency h * omega is outside (0, pi): no admissible lambda
            sel.offer({"width": W, "halo": HALO}, None, None, ctx.val_meter, status="rule_inadmissible")
            continue
        net, meta = build(ctx.target, ctx.p, width=W, lam=lam)
        sel.offer({"width": W, "halo": HALO, "lambda": meta["lambda"], "rank": meta["rank"]}, net,
                  net.forward(ctx.val_x), ctx.val_meter)
    return sel
