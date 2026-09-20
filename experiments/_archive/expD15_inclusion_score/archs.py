"""expD15 -- does the rule survive real architectures? ResNet and convolution.

Everything tested so far has been one hidden layer plus extras. Two questions
the method has never faced:

  RESNET   5 residual blocks, h_{l+1} = h_l + W2_l tanh(W1_l h_l + b1_l) + b2_l,
           output v.h_5 + c0. This is the interesting case, because the affine
           block is NOT UNIQUE. The output is affine in (v, c0). It is ALSO
           affine in (W2_4, b2_4, c0), the last block's projection, because h_4
           does not depend on them and the map h_5 -> out is linear. But the
           UNION is not affine: out contains v . W2_4 z, a product of two
           unknowns, so the two blocks are mutually exclusive. The method has to
           pick one, and whichever it picks must actually be affine.

  CONV     Weight sharing: one parameter acts at many spatial positions. Two
           variants, to separate "finds the head" from "finds a conv kernel":
             conv_head  conv-tanh-conv-tanh-linear head. Affine = head only.
             conv_lin   conv-tanh-conv-sum. The SECOND CONV has no activation
                        after it, so its kernel is affine. Nothing else is.

SCORING
  For these, enumerating the true maximal blocks by hand is error-prone (see
  the resnet's non-uniqueness), so the returned set is scored by the exact
  second-difference identity instead: S is affine iff
      f(theta+s) - 2 f(theta) + f(theta-s) = 0
  identically for s supported on S. That is ground truth, not a guess, and it
  is reported alongside which named tensors the set actually contains.
"""
from __future__ import annotations

import numpy as np


class Blocks:
    """A model as a list of named parameter tensors plus a forward function."""

    def __init__(self, shapes, forward):
        self.names = [n for n, _ in shapes]
        self.shapes = dict(shapes)
        self.slices, o = {}, 0
        for n, sh in shapes:
            sz = int(np.prod(sh))
            self.slices[n] = slice(o, o + sz)
            o += sz
        self.m = o
        self._fwd = forward

    def labels(self):
        out = np.empty(self.m, dtype=object)
        for n in self.names:
            out[self.slices[n]] = n
        return out.astype(str)

    def unpack(self, th):
        return {n: th[self.slices[n]].reshape(self.shapes[n]) for n in self.names}

    def f(self, X, th):
        return self._fwd(X, self.unpack(th))

    def jac(self, X, th, eps=1e-6):
        n = self.f(X, th).shape[0]
        J = np.empty((n, self.m))
        for i in range(self.m):
            tp = th.copy(); tp[i] += eps
            tm = th.copy(); tm[i] -= eps
            J[:, i] = (self.f(X, tp) - self.f(X, tm)) / (2 * eps)
        return J


def make_resnet(win=4, nblocks=5, seed=0):
    rng = np.random.default_rng(seed)
    shapes = [("Win", (win, 1)), ("bin", (win,))]
    for l in range(nblocks):
        shapes += [(f"W1_{l}", (win, win)), (f"b1_{l}", (win,)),
                   (f"W2_{l}", (win, win)), (f"b2_{l}", (win,))]
    shapes += [("v", (win,)), ("c0", (1,))]

    def fwd(X, p):
        h = X @ p["Win"].T + p["bin"]
        for l in range(nblocks):
            h = h + np.tanh(h @ p[f"W1_{l}"].T + p[f"b1_{l}"]) @ p[f"W2_{l}"].T \
                + p[f"b2_{l}"]
        return h @ p["v"] + p["c0"][0]

    net = Blocks(shapes, fwd)
    th = np.concatenate([rng.standard_normal(int(np.prod(sh))) *
                         (0.4 if n.startswith("W") else 0.2)
                         for n, sh in shapes])
    return net, th, nblocks


def _conv1d(x, K, b):
    """x: (n, cin, L) ; K: (cout, cin, k) -> (n, cout, L-k+1)."""
    n, cin, L = x.shape
    cout, _, k = K.shape
    out = np.zeros((n, cout, L - k + 1))
    for t in range(L - k + 1):
        out[:, :, t] = np.tensordot(x[:, :, t:t + k], K, axes=([1, 2], [1, 2])) + b
    return out


def make_conv(variant="head", L=12, c1=3, c2=2, k=3, seed=0):
    rng = np.random.default_rng(seed)
    Lo1, Lo2 = L - k + 1, L - 2 * k + 2
    shapes = [("K1", (c1, 1, k)), ("b1", (c1,)), ("K2", (c2, c1, k)), ("b2", (c2,))]
    if variant == "head":
        shapes += [("v", (c2 * Lo2,)), ("c0", (1,))]
    else:
        shapes += [("c0", (1,))]

    def fwd(X, p):
        h = np.tanh(_conv1d(X, p["K1"], p["b1"]))
        h = _conv1d(h, p["K2"], p["b2"])
        if variant == "head":
            h = np.tanh(h).reshape(X.shape[0], -1)
            return h @ p["v"] + p["c0"][0]
        return h.sum(axis=(1, 2)) + p["c0"][0]

    net = Blocks(shapes, fwd)
    th = np.concatenate([rng.standard_normal(int(np.prod(sh))) * 0.4
                         for _, sh in shapes])
    return net, th


def is_affine(net, X, th, idx, reps=3, eps=1e-2, seed=0):
    """Exact second-difference test: returns the worst relative curvature over
    `reps` random probes supported on idx. ~1e-16 means affine, O(1e-3) not."""
    if len(idx) == 0:
        return 0.0
    rng = np.random.default_rng(seed)
    f0 = net.f(X, th)
    sc = np.maximum(np.abs(th), 1e-2)
    worst = 0.0
    for _ in range(reps):
        s = np.zeros(net.m)
        z = rng.integers(0, 2, len(idx)).astype(float) * 2 - 1
        s[idx] = eps * sc[idx] * z
        fp, fm = net.f(X, th + s), net.f(X, th - s)
        lin = np.linalg.norm(fp - fm)
        worst = max(worst, np.linalg.norm(fp - 2 * f0 + fm) / max(lin, 1e-300))
    return float(worst)


def discover(net, X, th, keep=0.97, rounds=600, tol=1e-6, delta=1e-2, seed=1):
    """The shipped rule: prune whatever's own Jacobian column moves under a
    fixed-size probe supported on the current set."""
    m = net.m
    L = np.arange(m)
    rr = np.random.default_rng(seed)
    J0 = net.jac(X, th)
    n0 = np.maximum(np.linalg.norm(J0, axis=0), 1e-300)
    sc = np.maximum(np.abs(th), 1e-2)
    for it in range(rounds):
        z = rr.integers(0, 2, len(L)).astype(float) * 2 - 1
        s = np.zeros(m)
        s[L] = delta * sc[L] * z
        mv = np.linalg.norm(net.jac(X, th + s)[:, L] - J0[:, L], axis=0) / n0[L]
        if mv.max() <= tol:
            break
        kk = max(1, int(keep * len(L)))
        if (mv <= tol).sum() > kk:
            kk = int((mv <= tol).sum())
        L = L[np.argsort(mv)[:kk]]
        if len(L) <= 1:
            break
    return np.sort(L), it + 1
