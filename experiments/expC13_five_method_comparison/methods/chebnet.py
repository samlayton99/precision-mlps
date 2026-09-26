"""ChebNet (Tang, Li, Yu, arXiv:1911.05467 v3, Theorem 3; layer algorithm of Li, Tang, Yu, CiCP 2020,
Theorem 2.2, with T_{2^k} = 2 T_{2^(k-1)}^2 - 1 in place of powers): a deep ReQU network that
represents the degree-n Chebyshev truncation exactly in exact arithmetic; built and run at p bits.

Primitives (Lemma 2): sigma(t) = max(t, 0)^2, beta1 = [1,1,-1,-1]/4, omega1 = [1,-1,1,-1],
gamma1 = [1,-1,-1,1], omega2 = [1,-1]:
  identity  t  = beta1 . sigma(omega1 t + gamma1)        (4 neurons)
  product   ty = beta1 . sigma(omega1 t + gamma1 y)      (4 neurons)
  square    t^2 = sigma(t) + sigma(-t)                    (2 neurons)
Coefficients: the p-bit Chebyshev projection (shared with Mhaskar), zero-padded to 2^(m+1) with
m = floor(log2 n), then the hierarchical split (2.18)/(2.22) at p bits: r_0 = c_0, q_0 = c_M,
r_j = c_j - c_{2M-j}, q_j = 2 c_{M+j}, recursively. The network carries, after hidden layer k, the
affine values P^(k)_i = sum_{l < 2^k} ct[i 2^k + l] That_l(x) and T^(k) = T_{2^k}(x), combining
P^(k)_i = P^(k-1)_{2i} + T^(k-1) P^(k-1)_{2i+1}. Layers are collapsed (A_k = A_k1 A_k0) so the network
is a standard MLP. Nodes with i 2^k > n do not exist and nodes with i 2^k = n are constants carried
in biases (the pruned tree; for n = 2^(m+1)-1 it is the paper's padded construction). n = 1 and
n = 2 use the paper's explicit networks. Option `normalize`: scale ct by the power of two nearest
1/sum|c| and undo it in the output weights (exact; an implementation choice, not in the paper).
"""
from __future__ import annotations

from fractions import Fraction
import math

import numpy as np
import pfloat

from experiments.expC13_five_method_comparison import common as C
from experiments.expC13_five_method_comparison.methods import chebyshev
from experiments.expC13_five_method_comparison.selection import Selection

BETA1 = [0.25, 0.25, -0.25, -0.25]
OMEGA1 = [1, -1, 1, -1]
GAMMA1 = [1, -1, -1, 1]
OMEGA2 = [1, -1]
DEGREES = list(range(1, 17)) + [20, 24, 28, 31, 32, 36, 40, 44, 48, 52, 56, 60, 63, 64, 68, 72, 76, 80,
                                  84, 88, 92, 96, 100, 104, 112, 120, 127]


def hier(c: list) -> list:
    """The hierarchical coefficient split at p bits (c: list of 0-d PArrays, power-of-two length)."""
    if len(c) == 2:
        return list(c)
    M = len(c) // 2
    two = C.constant(2, c[0].fmt)
    r = [c[0]] + [c[j] - c[2 * M - j] for j in range(1, M)]
    q = [c[M]] + [two * c[M + j] for j in range(1, M)]
    return hier(r) + hier(q)


class _Aff:
    """An affine value sum_u w_u h_u + const over the previous layer's neurons (w: dict u -> scalar)."""

    def __init__(self, w=None, const=None):
        self.w = w or {}
        self.const = const

    def is_const(self):
        return not self.w


class _Layer:
    def __init__(self, F):
        self.F = F
        self.rows = []      # list of (list[(input u, weight)], bias)

    def neuron(self, terms, bias) -> int:
        self.rows.append((terms, bias))
        return len(self.rows) - 1


def _combine(F, parts, bias0):
    """Weights and bias of z = sum_k s_k * A_k + bias0 for (sign-or-power-of-two s_k, _Aff A_k): the
    weights are exact products; the bias adds s_k * const_k (exact products) to bias0 in order."""
    terms, bias = [], bias0
    for s, A in parts:
        for u in sorted(A.w):
            terms.append((u, A.w[u] * s))
        if A.const is not None:
            k = A.const * s
            bias = k if bias is None else bias + k
    return terms, bias


def _to_layer(L: _Layer) -> C.ReQULayer:
    F = L.F
    fan = max(1, max(len(t) for t, _ in L.rows))
    n = len(L.rows)
    idx = np.zeros((n, fan), dtype=np.int64)
    mask = np.zeros((n, fan), dtype=bool)
    w = pfloat.zeros((n, fan), F)
    bias = pfloat.zeros(n, F)
    for i, (terms, b) in enumerate(L.rows):
        for t, (u, wt) in enumerate(terms):
            idx[i, t], mask[i, t] = u, True
            w[i, t] = wt
        if b is not None:
            bias[i] = b
    return C.ReQULayer(idx, w, bias, mask)


def build(cheb: pfloat.PArray, n: int, normalize: bool) -> C.ReQUNet:
    F = cheb.fmt
    K = lambda v: C.constant(v, F)  # noqa: E731  exact small constants
    c = [cheb[j] for j in range(n + 1)]
    scale_out = None
    if normalize:
        total = pfloat.sum(pfloat.absolute(cheb[:n + 1]))
        tot = float(total.to_numpy(rounding=True))
        if tot > 0 and math.isfinite(tot):
            e = -round(math.log2(tot))
            two_e = C.constant(Fraction(2) ** e, F)
            c = [cj * two_e for cj in c]                                     # exact power-of-two scaling
            scale_out = e
    x_in = _Aff({0: K(1)})                                       # the network input as an affine value
    if n <= 2:
        L1 = _Layer(F)
        idn = [L1.neuron(*_combine(F, [(K(OMEGA1[r]), x_in)], K(GAMMA1[r]))) for r in range(4)]
        sq = [L1.neuron(*_combine(F, [(K(OMEGA2[r]), x_in)], None)) for r in range(2)] if n == 2 else []
        out = _Layer(F)
        terms = [(u, c[1] * K(BETA1[r])) for r, u in enumerate(idn)]
        bias = c[0]
        if n == 2:
            terms += [(u, K(2) * c[2]) for u in sq]
            bias = c[0] - c[2]
        layers = [L1, out]
        out.neuron(terms, bias)
        return _finish(layers, scale_out, F)
    m = int(math.floor(math.log2(n)))
    size = 2 ** (m + 1)
    padded = c + [K(0)] * (size - (n + 1))
    ct = hier(padded)
    # ---- layer 1: identity block on x and (if m >= 1) the square block starting the T chain
    L1 = _Layer(F)
    X = [L1.neuron(*_combine(F, [(K(OMEGA1[r]), x_in)], K(GAMMA1[r]))) for r in range(4)]
    S = [L1.neuron(*_combine(F, [(K(OMEGA2[r]), x_in)], None)) for r in range(2)]
    xval = _Aff({u: K(BETA1[r]) for r, u in enumerate(X)})
    T = _Aff({u: K(2) for u in S}, K(-1))                       # T_2 = 2 x^2 - 1
    P = {}
    for i in range(size // 2):
        if 2 * i > n:
            continue
        if 2 * i == n:
            P[i] = _Aff({}, ct[2 * i])
        else:
            P[i] = _Aff({u: ct[2 * i + 1] * w for u, w in xval.w.items()}, ct[2 * i])
    layers = [L1]
    for k in range(2, m + 2):
        L = _Layer(F)
        newP = {}
        for i in range(size // 2 ** k):
            if i * 2 ** k > n:
                continue
            low, high = P.get(2 * i), P.get(2 * i + 1)
            w, const = {}, None
            if low.is_const():
                const = low.const
            else:
                ids = [L.neuron(*_combine(F, [(K(OMEGA1[r]), low)], K(GAMMA1[r]))) for r in range(4)]
                w.update({u: K(BETA1[r]) for r, u in enumerate(ids)})
            if high is not None:
                prods = [L.neuron(*_combine(F, [(K(OMEGA1[r]), high), (K(GAMMA1[r]), T)], None)) for r in range(4)]
                w.update({u: K(BETA1[r]) for r, u in enumerate(prods)})
            newP[i] = _Aff(w, const)
        if k <= m:
            sq = [L.neuron(*_combine(F, [(K(OMEGA2[r]), T)], None)) for r in range(2)]
            T = _Aff({u: K(2) for u in sq}, K(-1))
        P = newP
        layers.append(L)
    out = _Layer(F)
    top = P[0]
    out.neuron(*_combine(F, [(K(1), top)], None))
    layers.append(out)
    return _finish(layers, scale_out, F)


def _finish(layers, scale_out, F):
    net = C.ReQUNet([_to_layer(L) for L in layers])
    if scale_out:
        L = net.layers[-1]
        f = C.constant(Fraction(2) ** (-scale_out), F)
        L.w[...] = L.w * f                                               # exact: undo the power-of-two scaling
        L.bias[...] = L.bias * f
    return net


NEURON_BUDGET = 1024
NEURON_DEGREES = DEGREES + [136, 144, 160, 176, 192, 208, 224, 240, 250, 255]


def select(ctx, degrees=DEGREES, faithful_only: bool = False, neuron_budget: bool = False) -> Selection:
    """neuron_budget: the sensitivity arm, at most 1024 hidden neurons instead of 3073 parameters."""
    sel = Selection("chebnet", ctx.target, ctx.p)
    if neuron_budget:
        sel.limit_key, sel.limit, degrees = "neurons", NEURON_BUDGET, NEURON_DEGREES
    cheb = chebyshev.projection(ctx.target, ctx.p, max(degrees))
    for n in degrees:
        for normalize in ((False,) if faithful_only else (False, True)):
            net = build(cheb, n, normalize)
            hp = {"degree": n, "normalize": normalize}
            if (net.neurons() > NEURON_BUDGET) if neuron_budget else (net.params() > ctx.budget):
                sel.offer(hp, None, None, ctx.val_meter, status="over_budget")
                continue
            sel.offer(hp, net, net.forward(ctx.val_x), ctx.val_meter)
    return sel
