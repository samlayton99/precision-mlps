"""Costarelli-Spigler sampled sigmoidal series (Annali di Matematica 194 (2015) 289-306, Section 5,
Theorem 5.4; sampled as in the paper's appendix); all at p bits.

q(x) = sum_{k=-K}^{K} A_k psi(w x - k), psi(t) = sigma(t+1) - sigma(t), sigma(t) = (1 + tanh t)/2, with
scale w and truncation K > w (Theorem 5.4 needs K > w max(|a|,|b|) = w on [-1,1]). A_k = F(u_k): F is f
on [-1,1] extended by linear tapers to zero on [1,2] and [-2,-1] (the appendix's extension), and the
sample location is u_k = k/w (Theorem 5.4's coefficients sampled, the appendix's choice) or, in the
labelled "centred" variant, u_k = (k - 1/2)/w, the centre of the bump psi(w x - k), which removes the
half-cell shift behind the O(1/w) rate in the interior.

Samples: the oracle at Q(u_k) for |u_k| <= 1; outside, A_k = mul(y(+-1), sub(2, |u_k|)) with |u_k|
computed as div(Q(|numerator|), Q(denominator)). Summation by parts merges adjacent sigmoids:
q = sum_{j=-K-1}^{K} (A_{j+1} - A_j) sigma(w x - j), A_{+-(K+1)} = 0. As a tanh network: readouts
a_j = mul(sub(A_{j+1}, A_j), 1/2), slope Q(w), biases Q(-j), offset (A_{K+1} - A_{-K-1})/2 = 0 (the
constant parts telescope exactly). Chosen on validation: w, K in {2w (appendix), w + ceil(p ln2 / 2) + 1
(the smallest K whose truncation is below unit roundoff, since psi decays like e^{-2|t|})}, centred.
"""
from __future__ import annotations

from fractions import Fraction
import math

import pfloat

from experiments.expC13_five_method_comparison import common as C
from experiments.expC13_five_method_comparison.selection import Selection

SCALES = [2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 160, 192, 224, 255, 288, 320, 384, 448, 490]


def truncations(w: int, p: int) -> dict:
    return {"2w": 2 * w, "tight": min(2 * w, w + math.ceil(p * math.log(2) / 2) + 1)}


def coefficients(target: str, p: int, w: int, K: int, centred: bool) -> pfloat.PArray:
    """A_k, k = -K..K."""
    F = C.fmt(p)
    num = [2 * k - 1 for k in range(-K, K + 1)] if centred else list(range(-K, K + 1))
    den = 2 * w if centred else w
    inside = [i for i, n in enumerate(num) if abs(n) <= den]
    x = C.inputs([Fraction(num[i], den) for i in inside], F)
    y = C.sample(target, x)
    ends = C.sample(target, pfloat.array([-1, 1], F))
    out = pfloat.zeros(len(num), F)
    out[inside] = y
    outside = [i for i, n in enumerate(num) if abs(n) > den]
    if outside:
        mag = pfloat.array([abs(num[i]) for i in outside], F) / C.constant(den, F)
        taper = C.constant(2, F) - mag
        edge = pfloat.stack([ends[1] if num[i] > 0 else ends[0] for i in outside])
        out[outside] = edge * taper
    return out


def build(target: str, p: int, w: int, K: int, centred: bool) -> C.TanhNet:
    F = C.fmt(p)
    A = coefficients(target, p, w, K, centred)
    zero = pfloat.zeros(1, F)
    Ae = pfloat.concatenate([zero, A, zero])                       # A_{-K-1} .. A_{K+1}
    a = (Ae[1:] - Ae[:-1]) * C.constant(Fraction(1, 2), F)         # j = -K-1..K
    slope = pfloat.ones(2 * K + 2, F) * C.constant(w, F)
    bias = -pfloat.array(list(range(-K - 1, K + 1)), F)
    return C.TanhNet(slope, bias, a, pfloat.array(0, F))


def candidates(p: int, scales=SCALES, faithful_only: bool = False):
    for w in scales:
        for kname, K in truncations(w, p).items():
            for centred in (False, True):
                if faithful_only and (centred or kname != "2w"):
                    continue
                yield {"w": w, "K": K, "truncation": kname, "centred": centred}


def select(ctx, scales=SCALES, faithful_only: bool = False) -> Selection:
    sel = Selection("costarelli", ctx.target, ctx.p)
    seen = set()
    for hp in candidates(ctx.p, scales, faithful_only):
        key = (hp["w"], hp["K"], hp["centred"])
        if key in seen:                     # 'tight' equals '2w' for small w
            continue
        seen.add(key)
        net = build(ctx.target, ctx.p, hp["w"], hp["K"], hp["centred"])
        if net.params() > ctx.budget:
            sel.offer(hp, None, None, ctx.val_meter, status="over_budget")
            continue
        sel.offer(hp, net, net.forward(ctx.val_x), ctx.val_meter)
    return sel
