"""expD15 -- propose-and-verify: an EXACT set-level linearity test.

WHY A RANKING CANNOT WIN
  Measured: the best per-parameter ranking (L_self, 64-68% readout purity in
  2-D) does not reach the top five of the acid test in any 2-D cell. The solve
  is all-or-nothing -- iteration 11 section 10, "solving 90% of a least-squares
  system is not 90% as good". So the target is not a good ordering, it is a set
  with NO wrong members. A score needs a separating GAP, and every estimator
  here has a noise floor that destroys one.

THE EXACT TEST
  For any s supported on a set S, f is affine on S if and only if

      D(S) = f(theta+s) - 2 f(theta) + f(theta-s) = 0

  identically -- not approximately, not up to an estimator's variance, only up
  to floating-point rounding. That is a real gap: linear sets read ~1e-16
  relative, nonlinear ones read O(1).

  Cost: 3 forward passes per test (2 with f(theta) cached), INDEPENDENT of |S|
  and of P. Memory O(P). It is a group test -- it reports whether S contains
  any nonlinear member, not which -- so the ranking's only job is to order the
  candidates well enough that offenders are rare, and exactness comes from the
  verifier.

SEARCH
  Walk the ranked list in chunks. Accept a chunk if the set still verifies;
  otherwise bisect the chunk to isolate and drop its offenders. Passes are
  counted and reported, because "feasible" has to mean a number.
"""
from __future__ import annotations

import numpy as np


class Verifier:
    """Exact second-difference linearity test, with a pass counter."""

    def __init__(self, env, th, eps=1e-3, reps=2, tol=1e-9, seed=0):
        self.env, self.th = env, th
        self.net = env["net"]
        self.X = env["Xtr"]
        self.eps, self.reps, self.tol = eps, reps, tol
        self.rng = np.random.default_rng(seed)
        self.scale = np.maximum(np.abs(th), 1e-3)
        self.f0 = self.net.f(self.X, th)
        self.passes = 1
        self.tests = 0

    def rel_curv(self, idx):
        """max over reps of ||f(+s) - 2f(0) + f(-s)|| / ||f(+s) - f(-s)||.

        The denominator makes it scale-free: it is the linear part of the same
        probe, so the ratio is 'how much of this step's effect is not affine'."""
        if len(idx) == 0:
            return 0.0
        worst = 0.0
        for _ in range(self.reps):
            z = self.rng.integers(0, 2, len(idx)).astype(float) * 2 - 1
            s = np.zeros(self.th.size)
            s[idx] = self.eps * self.scale[idx] * z
            fp = self.net.f(self.X, self.th + s)
            fm = self.net.f(self.X, self.th - s)
            self.passes += 2
            lin = np.linalg.norm(fp - fm)
            curv = np.linalg.norm(fp - 2 * self.f0 + fm)
            worst = max(worst, curv / max(lin, 1e-300))
        self.tests += 1
        return float(worst)

    def ok(self, idx):
        return self.rel_curv(idx) <= self.tol


def grow_verified(env, th, order, chunk=16, tol=1e-9, max_tests=400, **kw):
    """Walk the ranked order in chunks, keeping the set exactly affine.

    Returns (selected_idx, stats). A chunk that breaks the test is bisected so
    its clean members are still recovered -- without that, one bad parameter
    early in the ranking would cap the set at its position and destroy recall."""
    V = Verifier(env, th, tol=tol, **kw)
    sel = []

    def admit(cand, depth=0):
        if len(cand) == 0 or V.tests >= max_tests:
            return
        if V.ok(np.array(sel + list(cand), dtype=int)):
            sel.extend(cand)
            return
        if len(cand) == 1:
            return                      # lone offender: drop it
        h = len(cand) // 2
        admit(cand[:h], depth + 1)
        admit(cand[h:], depth + 1)

    for i in range(0, len(order), chunk):
        if V.tests >= max_tests:
            break
        admit(list(order[i:i + chunk]))
    return (np.sort(np.array(sel, dtype=int)),
            dict(passes=V.passes, tests=V.tests, size=len(sel)))


def prefix_verified(env, th, order, tol=1e-12, eps=1e-1, reps=1, **kw):
    """Longest ranked PREFIX that is exactly affine, by binary search.

    This is the feasible form. Chunked growth costs O(P) tests because it must
    visit every parameter; binary search on the prefix length costs
    ceil(log2 P) tests -- 9 for P=289, 30 for P=1e9 -- i.e. ~60 forward passes
    at ANY width, which is the gate the whole exercise is held to.

    Capping at the first offender is free here, and that is a measured fact
    rather than a hope: dropping 20% of the readout moves the solve from
    8.127e-03 to 8.100e-03 (nothing), while ADDING 2% geometry moves it to
    4.4e-02 and 5% to 7.1e+01. Purity is everything, recall is free -- so the
    right search is the most conservative one that keeps the set exactly clean.
    """
    V = Verifier(env, th, eps=eps, reps=reps, tol=tol, **kw)
    lo, hi = 0, len(order)
    if V.ok(np.array(order[:hi], dtype=int)):
        lo = hi
    else:
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if V.ok(np.array(order[:mid], dtype=int)):
                lo = mid
            else:
                hi = mid
    return (np.sort(np.array(order[:lo], dtype=int)),
            dict(passes=V.passes, tests=V.tests, size=lo))
