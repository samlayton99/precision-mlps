"""expD15 -- a small transformer, and the membership inventory for every
architecture tested.

WHY A TRANSFORMER IS THE INTERESTING CASE
  It has several candidate affine blocks and most of them are mutually
  exclusive, because attention and the MLP are full of PRODUCTS of two weight
  matrices:

      attention output  =  A(Q,K) . X . W_V . W_O          (bilinear in V, O)
      output head       =  v . (residual stream)           (bilinear with
                                                            anything feeding it
                                                            linearly)

  So "which parameters are affine" has no single answer, and layer norm may
  destroy the candidates upstream of it entirely. Rather than predict, every
  named tensor and combination is measured with the exact second-difference
  identity, and then discovery is run and compared against that measurement.

MODEL
  pre-norm block, single head:
      h <- h + Attn(LN1(h))
      h <- h + W2 tanh(W1 LN2(h))
  then mean-pool over positions, then a linear head v.h + c0.
"""
from __future__ import annotations

import numpy as np

from archs import Blocks


def _ln(x, g, b, eps=1e-5):
    mu = x.mean(-1, keepdims=True)
    sd = np.sqrt(x.var(-1, keepdims=True) + eps)
    return (x - mu) / sd * g + b


def _softmax(z):
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def make_transformer(dmod=4, dff=6, nblocks=2, T=5, seed=0):
    shapes = []
    for l in range(nblocks):
        shapes += [(f"ln1g_{l}", (dmod,)), (f"ln1b_{l}", (dmod,)),
                   (f"Wq_{l}", (dmod, dmod)), (f"Wk_{l}", (dmod, dmod)),
                   (f"Wv_{l}", (dmod, dmod)), (f"Wo_{l}", (dmod, dmod)),
                   (f"ln2g_{l}", (dmod,)), (f"ln2b_{l}", (dmod,)),
                   (f"W1_{l}", (dff, dmod)), (f"W2_{l}", (dmod, dff))]
    shapes += [("v", (dmod,)), ("c0", (1,))]

    def fwd(X, p):
        h = X                                        # (n, T, dmod)
        for l in range(nblocks):
            z = _ln(h, p[f"ln1g_{l}"], p[f"ln1b_{l}"])
            q = z @ p[f"Wq_{l}"].T
            k = z @ p[f"Wk_{l}"].T
            v = z @ p[f"Wv_{l}"].T
            a = _softmax(q @ k.transpose(0, 2, 1) / np.sqrt(dmod))
            h = h + (a @ v) @ p[f"Wo_{l}"].T
            z = _ln(h, p[f"ln2g_{l}"], p[f"ln2b_{l}"])
            h = h + np.tanh(z @ p[f"W1_{l}"].T) @ p[f"W2_{l}"].T
        return h.mean(1) @ p["v"] + p["c0"][0]

    net = Blocks(shapes, fwd)
    rng = np.random.default_rng(seed)
    parts = []
    for n, sh in shapes:
        if n.startswith("ln") and "g_" in n:
            parts.append(np.ones(int(np.prod(sh))))
        elif n.startswith("ln"):
            parts.append(np.zeros(int(np.prod(sh))))
        else:
            parts.append(rng.standard_normal(int(np.prod(sh))) * 0.4)
    return net, np.concatenate(parts), T, dmod
