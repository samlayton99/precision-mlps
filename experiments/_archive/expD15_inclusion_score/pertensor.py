"""expD15 -- the shipped rule: per-tensor sampling with verification.

  visit tensors in descending total value
  sample a few parameters, test them against the current L
  if clean, tentatively admit the WHOLE tensor and verify it once
  keep only if the verification passes
  stop when the remaining tensors' value falls below eps_val * captured

Two probes per tensor, no cap, cost O(#tensors) rather than O(P). The only
structure used is that parameters arrive in named tensors, which every
framework already provides; nothing about the architecture is assumed, and a
wrong generalisation from the samples is caught by the verification rather than
trusted.
"""
from __future__ import annotations

import numpy as np


def _probe(net, X, th, J0, n0, idx, delta, rr):
    m = th.size
    L = np.sort(np.asarray(idx))
    z = rr.integers(0, 2, len(L)).astype(float) * 2 - 1
    s = np.zeros(m)
    s[L] = delta * np.maximum(np.abs(th[L]), 1e-2) * z
    mv = np.linalg.norm(net.jac(X, th + s)[:, L] - J0[:, L], axis=0) / n0[L]
    return L, mv


def discover_pertensor(net, X, y, th, groups, tol=1e-6, delta=1e-2, seed=1,
                       nsamp=2, eps_val=1e-3, sparse=False):
    """groups: dict name -> index array.

    sparse=False admits whole tensors. sparse=True falls back, on a tensor that
    fails verification, to keeping only the members that stayed clean in the
    full-tensor probe -- which is still safe (a non-mover under a probe of the
    whole tensor is affine w.r.t. L plus itself) and recovers partial tensors
    such as the bilinear-conflict cases.
    """
    m = th.size
    rr = np.random.default_rng(seed)
    J0 = net.jac(X, th)
    n0 = np.maximum(np.linalg.norm(J0, axis=0), 1e-300)
    r = net.f(X, th) - y
    val = (J0.T @ r) ** 2 / np.maximum((J0 * J0).sum(0), 1e-300)
    names = sorted(groups, key=lambda n: -float(val[groups[n]].sum()))
    tval = {n: float(val[groups[n]].sum()) for n in names}
    inL = np.zeros(m, bool)
    captured = 0.0
    passes = 1
    for pos_i, n in enumerate(names):
        rem = sum(tval[q] for q in names[pos_i:])
        if captured > 0 and rem < eps_val * captured:
            break
        idx = groups[n]
        S = rr.choice(idx, min(nsamp, len(idx)), replace=False)
        L, mv = _probe(net, X, th, J0, n0, np.concatenate([np.nonzero(inL)[0], S]),
                       delta, rr)
        passes += 1
        pos = {p: j for j, p in enumerate(L)}
        if not all(mv[pos[c]] <= tol for c in S):
            continue
        L2, mv2 = _probe(net, X, th, J0, n0,
                         np.concatenate([np.nonzero(inL)[0], idx]), delta, rr)
        passes += 1
        pos2 = {p: j for j, p in enumerate(L2)}
        keep = idx if mv2.max() <= tol else \
            (np.array([c for c in idx if mv2[pos2[c]] <= tol], dtype=int)
             if sparse else np.array([], dtype=int))
        if len(keep):
            inL[keep] = True
            captured += float(val[keep].sum())
    return np.nonzero(inL)[0], passes


def groups_from_labels(lab):
    return {n: np.nonzero(lab == n)[0] for n in sorted(set(lab.tolist()))}
