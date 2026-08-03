"""expD15 -- amortised discovery: a running belief, one probe per step.

THE PROBLEM WITH A SWEEP
  The fixed-point sweep is exact and blind, but its cost is
  rounds = log(linear fraction)/log(keep) -- ~1100 passes for a 1e4 head in a
  1e9-parameter model. That is a per-solve-event cost and it is too much.

  It also has a structural recall failure. In a depth-2 net J_v depends on
  EVERY geometry parameter, so while any of them is still in L and moving, the
  readout's own columns move too and the quota evicts them. Measured: 56%
  recall on depth2 against 100% on the shallow architectures.

THE FIX, WHICH IS THE SAME FOR BOTH (Sam's suggestion)
  Keep a Beta(a_i, b_i) belief per parameter that it is linear and update it
  from ONE probe per optimizer step. Two consequences:

  - Cost amortises. One probe is 1 pass; the belief sharpens over the steps the
    optimizer was going to take anyway, instead of a sweep per solve event.
  - Recall recovers. Early rounds are noisy because L still contains movers,
    but that is evidence, not a verdict. Once geometry has been voted out, later
    probes disturb a quiet L, the readout stops moving, and the belief flips
    back. A sweep cannot do this -- eviction there is permanent.

  State is 2 floats per parameter: Adam's class.

EXPLORATION
  The probe set is the believed-linear set plus a small random sample of the
  rest, so a parameter wrongly voted out can be re-tested. The sample is kept
  small because its members are the disturbance that makes genuinely linear
  columns move.

DECAY
  Evidence decays toward the prior, so the belief tracks a geometry that is
  still changing rather than fossilising. Staleness becomes a rate, not a re-run.
"""
from __future__ import annotations

import numpy as np


class Amortised:
    def __init__(self, m, tol=1e-5, delta=1e-2, decay=0.995, prior=1.0,
                 thresh=0.5, explore=0.05, seed=0):
        self.m, self.tol, self.delta = m, tol, delta
        self.decay, self.prior, self.thresh = decay, prior, thresh
        self.explore = explore
        self.a = np.full(m, prior)
        self.b = np.full(m, prior)
        self.rng = np.random.default_rng(seed)
        self.passes = 0

    def mean(self):
        return self.a / (self.a + self.b)

    def selected(self):
        return np.nonzero(self.mean() >= self.thresh)[0]

    def probe_set(self):
        sel = self.selected()
        rest = np.setdiff1d(np.arange(self.m), sel)
        n_ex = int(np.ceil(self.explore * max(len(rest), 1)))
        if len(rest) and n_ex:
            ex = self.rng.choice(rest, min(n_ex, len(rest)), replace=False)
            return np.union1d(sel, ex), sel
        return sel, sel

    def step(self, net, X, th, J0=None, n0=None):
        """One probe: perturb inside the probe set, see whose columns moved,
        update the belief. 1 forward-Jacobian pass (2 in the matrix-free form
        via Hutchinson, which preserves exact zeros with zero variance)."""
        if J0 is None:
            J0 = net.jac(X, th)
        if n0 is None:
            n0 = np.maximum(np.linalg.norm(J0, axis=0), 1e-300)
        L, _ = self.probe_set()
        if len(L) == 0:
            L = np.arange(self.m)
        sc = np.maximum(np.abs(th), 1e-2)
        z = self.rng.integers(0, 2, len(L)).astype(float) * 2 - 1
        s = np.zeros(self.m)
        s[L] = self.delta * sc[L] * z
        mv = np.linalg.norm(net.jac(X, th + s)[:, L] - J0[:, L], axis=0) / n0[L]
        self.passes += 1

        moved = mv > self.tol
        d, p = self.decay, self.prior
        self.a = d * self.a + (1 - d) * p
        self.b = d * self.b + (1 - d) * p
        self.a[L[~moved]] += 1.0
        self.b[L[moved]] += 1.0
        return mv, L
