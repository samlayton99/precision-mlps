"""expD15 -- the depth hypothesis, and end-to-end ratios.

HYPOTHESIS UNDER TEST
  Coverage of the affine set degrades with the DEPTH between that set and the
  input. Mechanism: J_L depends on every upstream parameter, so a candidate is
  admitted only when the probe batch happens to contain no upstream weight, and
  that probability falls as the upstream grows.

THREE FAMILIES, each at depths 1..5, affine-set size held constant so coverage
is comparable across depths:

  mlp     h <- tanh(W_l h + b_l), out = v.h + c0            affine {v, c0}
  resnet  h <- h + W2_l tanh(W1_l h + b1_l) + b2_l          affine {v,c0} OR
          out = v.h + c0                                    {W2_last,b2_last,c0}
  skip    mlp trunk plus a direct c.x term                  affine {v, c, c0}

THE FIX THIS SUGGESTED, AND WHY IT SHOULD WORK
  With batch size 1 the admission test is EXACT and depth cannot matter: L is
  clean by construction, so perturbing L + {c} moves only affine parameters,
  and a candidate whose column depends solely on upstream weights (which are
  not in L) does not move. It is admitted regardless of how deep it sits.
  Batch size 1 costs one round per candidate, so the batch is made ADAPTIVE in
  the direction opposite to my first attempt: it DOUBLES after a round where
  every candidate was admitted, and halves after a round where none were.
  Clean regions clear fast; contested regions fall back to exact single tests.
"""
from __future__ import annotations

import numpy as np

from archs import Blocks


def make_family(kind, depth, w=4, seed=0):
    rng = np.random.default_rng(seed)
    shapes = [("Win", (w, 1)), ("bin", (w,))]
    if kind == "resnet":
        for l in range(depth):
            shapes += [(f"W1_{l}", (w, w)), (f"b1_{l}", (w,)),
                       (f"W2_{l}", (w, w)), (f"b2_{l}", (w,))]
    else:
        for l in range(depth):
            shapes += [(f"W_{l}", (w, w)), (f"b_{l}", (w,))]
    if kind == "skip":
        shapes += [("c", (1,))]
    shapes += [("v", (w,)), ("c0", (1,))]

    def fwd(X, p):
        h = np.tanh(X @ p["Win"].T + p["bin"])
        if kind == "resnet":
            for l in range(depth):
                h = h + np.tanh(h @ p[f"W1_{l}"].T + p[f"b1_{l}"]) @ p[f"W2_{l}"].T \
                    + p[f"b2_{l}"]
        else:
            for l in range(depth):
                h = np.tanh(h @ p[f"W_{l}"].T + p[f"b_{l}"])
        out = h @ p["v"] + p["c0"][0]
        if kind == "skip":
            out = out + X[:, 0] * p["c"][0]
        return out

    net = Blocks(shapes, fwd)
    th = np.concatenate([
        (np.ones if n.endswith("g") else rng.standard_normal)(int(np.prod(sh)))
        * (0.5 if n[0] == "W" else 0.3) for n, sh in shapes])
    return net, th


def best_affine(net, kind, depth):
    """The largest affine set available, by construction of the family."""
    lab = net.labels()
    base = ["v", "c0"] + (["c"] if kind == "skip" else [])
    cand = [base]
    if kind == "resnet":
        cand.append([f"W2_{depth-1}", f"b2_{depth-1}", "c0"])
    best, bn = None, -1
    for names in cand:
        idx = np.concatenate([np.nonzero(lab == n)[0] for n in names])
        if len(idx) > bn:
            best, bn = np.sort(idx), len(idx)
    return best


def value_grow(net, X, y, th, rounds=400, tol=1e-6, delta=1e-2, seed=1,
               b0=1, adaptive=True, batch=None):
    """Grow from empty in value order. `adaptive` doubles the batch after a
    fully-successful round and halves it after a fully-failed one."""
    m = net.m
    rr = np.random.default_rng(seed)
    J0 = net.jac(X, th)
    n0 = np.maximum(np.linalg.norm(J0, axis=0), 1e-300)
    r = net.f(X, th) - y
    val = (J0.T @ r) ** 2 / np.maximum((J0 * J0).sum(0), 1e-300)
    order = list(np.argsort(-val))
    sc = np.maximum(np.abs(th), 1e-2)
    inL = np.zeros(m, bool)
    tried = np.zeros(m, bool)
    k = b0 if adaptive else (batch or 2)
    passes = 1
    for it in range(rounds):
        rest = [c for c in order if not inL[c] and not tried[c]]
        if not rest:
            tried[:] = False
            rest = [c for c in order if not inL[c]]
            if not rest:
                break
        C = np.array(rest[:max(1, k)])
        L = np.sort(np.concatenate([np.nonzero(inL)[0], C]))
        z = rr.integers(0, 2, len(L)).astype(float) * 2 - 1
        s = np.zeros(m)
        s[L] = delta * sc[L] * z
        mv = np.linalg.norm(net.jac(X, th + s)[:, L] - J0[:, L], axis=0) / n0[L]
        passes += 1
        pos = {p: i for i, p in enumerate(L)}
        good = [c for c in C if mv[pos[c]] <= tol]
        for c in C:
            if mv[pos[c]] <= tol:
                inL[c] = True
            else:
                tried[c] = True
        if adaptive:
            if len(good) == len(C):
                k = min(2 * k, m)
            elif len(good) == 0:
                k = max(1, k // 2)
    return np.nonzero(inL)[0], passes


def solve_and_eval(net, Xtr, ytr, Xev, yev, th, idx, alpha=1e-8):
    """Damped least-squares solve on `idx`, applied, then eval relative L2."""
    if len(idx) == 0:
        return float(np.linalg.norm(net.f(Xev, th) - yev) / np.linalg.norm(yev))
    J = net.jac(Xtr, th)[:, idx]
    r = net.f(Xtr, th) - ytr
    U, s, Vt = np.linalg.svd(J, full_matrices=False)
    mu = (alpha * s[0]) ** 2
    d = Vt.T @ ((s / (s ** 2 + mu)) * (U.T @ (-r)))
    th2 = th.copy()
    th2[idx] += d
    return float(np.linalg.norm(net.f(Xev, th2) - yev) / np.linalg.norm(yev))


def grow_v2(net, X, y, th, tol=1e-6, delta=1e-2, seed=1, eps_val=1e-3,
            max_rounds=100000):
    """Bottom-up growth with a PRINCIPLED stopping rule and no round cap.

    Candidates are visited once, in descending order of
        val_i = (r . J_i)^2 / ||J_i||^2,
    the residual reduction from solving i alone. Two things follow.

    STOPPING. Walk down the sorted list and stop when the total remaining value
    falls below eps_val times the value already captured: what is left cannot
    change the answer. One scalar comparison per round, no scan of the
    parameter vector, and it terminates on the quantity we actually care about.
    Certifying true maximality instead would need a batch-1 pass over every
    remaining parameter, which is O(P) and unusable at scale.

    BATCH. Doubles after a round in which every candidate was admitted, halves
    after one in which none was. At batch 1 the test is exact -- L is clean, so
    perturbing L+{c} moves only affine parameters -- so contested regions get
    the exact test while clean regions clear at exponentially growing batch.
    """
    m = net.m
    rr = np.random.default_rng(seed)
    J0 = net.jac(X, th)
    n0 = np.maximum(np.linalg.norm(J0, axis=0), 1e-300)
    r = net.f(X, th) - y
    val = (J0.T @ r) ** 2 / np.maximum((J0 * J0).sum(0), 1e-300)
    order = np.argsort(-val)
    sc = np.maximum(np.abs(th), 1e-2)
    inL = np.zeros(m, bool)
    captured = 0.0
    i, k, passes = 0, 1, 1
    reason = "exhausted"
    while i < m and passes < max_rounds:
        remaining = float(val[order[i:]].sum())
        if captured > 0 and remaining < eps_val * captured:
            reason = "value-exhausted"
            break
        C = order[i:i + max(1, k)]
        L = np.sort(np.concatenate([np.nonzero(inL)[0], C]))
        z = rr.integers(0, 2, len(L)).astype(float) * 2 - 1
        s = np.zeros(m)
        s[L] = delta * sc[L] * z
        mv = np.linalg.norm(net.jac(X, th + s)[:, L] - J0[:, L], axis=0) / n0[L]
        passes += 1
        pos = {p: j for j, p in enumerate(L)}
        good = [c for c in C if mv[pos[c]] <= tol]
        for c in good:
            inL[c] = True
            captured += val[c]
        i += len(C)
        if len(good) == len(C):
            k = min(2 * k, m)
        elif len(good) == 0:
            k = max(1, k // 2)
    return np.nonzero(inL)[0], passes, reason
