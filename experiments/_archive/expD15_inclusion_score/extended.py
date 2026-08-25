"""expD15 -- does the fixed point find linear blocks that are NOT the readout,
and can its cost be amortised across optimizer steps with a running belief?

TWO QUESTIONS
  1. The whole point of discovering L rather than hard-coding it is that the
     linear block need not be the last layer. Test architectures where other
     parameters are exactly linear and check they are admitted:
       skip   f = tanh(xA'+b) v + c.x + c0      -- c is a FIRST-layer-shaped
                                                   parameter, exactly linear
       basis  f = tanh(xA'+b) v + u.psi(x) + c0 -- u weights FIXED features,
                                                   exactly linear, disjoint
                                                   from the readout
       depth2 f = tanh(tanh(xA'+b) W2' + b2) v + c0  -- only v, c0 are linear;
                                                   W2, b2 are NOT, even though
                                                   they are closer to the output
     The last one is the guard against "discovers a convention": a naive rule
     that admits whatever is near the output would wrongly take W2, b2.

  2. Sam: use the previous iterate as a prior and update it. Discovery is then
     one cheap round per step instead of a full sweep per solve event, and the
     belief sharpens as evidence accumulates.

BAYESIAN UPDATE
  Per parameter, a Beta(a_i, b_i) belief that it is linear. Each round produces
  a binary observation -- did column i move? -- and the update is conjugate:
      moved     -> b_i += 1
      not moved -> a_i += 1
  with a decay toward the prior so the belief can track a geometry that is
  still changing (staleness is a refresh policy, not a blocker). Membership is
  the posterior mean above a threshold. Zero extra passes over the round it
  already runs; state is 2 floats per parameter, i.e. Adam's class.
"""
from __future__ import annotations

import numpy as np


class NetX:
    """f(x) = tanh(x A' + b) v + c.x + u.psi(x) + c0, optional depth-2 trunk.

    Parameter order: A | b | v | c | u | c0  (or A|b|W2|b2|v|c0 when depth2).
    Exactly-linear parameters: v, c, u, c0 -- and, in the depth-2 model, only
    v and c0."""

    def __init__(self, A, b, v, c=None, U=None, c0=0.0, psi=None,
                 W2=None, b2=None):
        self.W, self.dim = A.shape
        self.psi = psi
        self.depth2 = W2 is not None
        parts, names = [A.ravel(), b], [("A", A.size), ("b", b.size)]
        if self.depth2:
            parts += [W2.ravel(), b2]
            names += [("W2", W2.size), ("b2", b2.size)]
        parts.append(v)
        names.append(("v", v.size))
        self.h = v.size
        if c is not None:
            parts.append(c)
            names.append(("c", c.size))
        if U is not None:
            parts.append(U)
            names.append(("u", U.size))
        parts.append(np.array([c0]))
        names.append(("c0", 1))
        self.theta = np.concatenate(parts)
        self.names = names
        self.slices, o = {}, 0
        for nm, sz in names:
            self.slices[nm] = slice(o, o + sz)
            o += sz
        self.m = o

    def labels(self):
        out = np.empty(self.m, dtype=object)
        for nm, _ in self.names:
            out[self.slices[nm]] = nm
        return out.astype(str)

    def linear_truth(self):
        """Ground truth: which parameters the model is exactly affine in."""
        lin = {"v", "c", "u", "c0"}
        lab = self.labels()
        return np.isin(lab, list(lin))

    def f(self, X, th=None):
        th = self.theta if th is None else th
        s = self.slices
        A = th[s["A"]].reshape(self.W, self.dim)
        z = np.tanh(X @ A.T + th[s["b"]])
        if self.depth2:
            W2 = th[s["W2"]].reshape(self.h, self.W)
            z = np.tanh(z @ W2.T + th[s["b2"]])
        out = z @ th[s["v"]] + th[s["c0"]][0]
        if "c" in s:
            out = out + X @ th[s["c"]]
        if "u" in s:
            out = out + self.psi(X) @ th[s["u"]]
        return out

    def jac(self, X, th=None, eps=1e-6):
        """Full Jacobian by complex-step-free central differences on the few
        BLOCKS (not per parameter) -- exact analytic forms exist but this keeps
        the extended architectures short. Only used to build test problems."""
        th = self.theta if th is None else th
        n = X.shape[0]
        J = np.empty((n, self.m))
        for i in range(self.m):
            tp = th.copy(); tp[i] += eps
            tm = th.copy(); tm[i] -= eps
            J[:, i] = (self.f(X, tp) - self.f(X, tm)) / (2 * eps)
        return J


class Belief:
    """Beta posterior per parameter that it is linear, updated once per round.

    State: 2 floats per parameter (Adam's class). Zero extra passes -- it reads
    the observation the round already produced."""

    def __init__(self, m, prior=1.0, decay=0.98):
        self.a = np.full(m, prior)
        self.b = np.full(m, prior)
        self.prior, self.decay = prior, decay

    def update(self, moved):
        d, p = self.decay, self.prior
        self.a = d * self.a + (1 - d) * p + (~moved).astype(float)
        self.b = d * self.b + (1 - d) * p + moved.astype(float)

    def mean(self):
        return self.a / (self.a + self.b)

    def select(self, thresh=0.9):
        return np.nonzero(self.mean() >= thresh)[0]
