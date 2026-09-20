"""expD18 -- the minimal assembled QI optimizer, and the lstsq probes.

The QI optimizer here is the deployable composition distilled from the expD14
iteration-4 latch (experiments/_archive/expD14_lobotomy/iteration_4/core4.py),
rebuilt minimal per STEP2_SOLVER_SPEC for the tabular setting:

  - Adam on ALL parameters (manual implementation, identical code path to the
    plain-Adam control arm so the only difference is the solve machinery);
  - every `cadence` steps, a damped correction solve of the readout block on
    the fit rows:  d = argmin ||A d + r||^2 + mu ||d||^2  with the damping
    floor  alpha = clip(||r||/||y_fit||, ...)  and  mu = (alpha * sigma_1)^2
    (never damp finer than the residual you walked in with -- on noisy real
    data r_entry is noise-dominated, which is exactly the regime the floor
    rule was built for);
  - after the first solve, Adam's geometry step gets the exact-line-search
    cap  t* = -(r . J p) / ||J p||^2  clipped to [0, 1] (one closed-form JVP
    on the batch; the measured fix for the post-solve sawtooth);
  - one VALIDATED terminal finisher: damping chosen on a held-out tenth of
    the train rows, with "keep the in-run readout" as a candidate (the raw
    min-norm refit was worse than the in-run state in 26/132 expD14 cells).

L is hard-set to the readout (fc2.weight, fc2.bias). Discovery is bypassed:
on a 1-hidden-layer MLP the readout's Jacobian columns [Phi, 1] depend only
on fc1, so they are bitwise-invariant under perturbations of L -- expD15's
exact-zero rule would return exactly this set.

Kill-list compliance: no control decision compares loss values (alpha comes
from residual norms; the finisher validates on held-out residual norms, as
iteration_4 did); no trust-region clip on the solve step.
"""
from __future__ import annotations

import math
import time

import numpy as np
import torch

ALPHA_MIN, ALPHA_MAX = 1e-15, 1.0
RCOND = 1e-13
ALPHAS_VAL = (0.0, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1e-1)


# ----------------------------- linear algebra -----------------------------

def _design(model, x):
    """A = [tanh(fc1(x)), 1] on given rows, detached numpy fp64."""
    with torch.no_grad():
        Z = torch.tanh(model.fc1(x))
        A = torch.cat([Z, torch.ones(Z.shape[0], 1, dtype=Z.dtype)], dim=1)
    return A.numpy()


def _readout_vec(model):
    w = model.fc2.weight.detach().numpy().ravel()
    b = model.fc2.bias.detach().numpy().ravel()
    return np.concatenate([w, b])


def _set_readout(model, v):
    with torch.no_grad():
        model.fc2.weight.copy_(torch.tensor(v[:-1], dtype=torch.float64).reshape(1, -1))
        model.fc2.bias.fill_(float(v[-1]))


def damped_correction(A_fit, y_fit, v_cur, alpha):
    """d = argmin ||A d + r||^2 + mu ||d||^2, mu = (alpha * s1)^2. Returns v_cur + d."""
    r = A_fit @ v_cur - y_fit
    U, s, Vt = np.linalg.svd(A_fit, full_matrices=False)
    mu = (alpha * s[0]) ** 2
    inv = s / (s ** 2 + mu)
    d = Vt.T @ (inv * (U.T @ (-r)))
    return v_cur + d


def validated_solve(A_fit, y_fit, A_ho, y_ho, v_keep=None):
    """Full damped solve of y with the damping chosen on held-out rows.
    v_keep (the in-run readout) competes as a candidate. Returns (v, tag)."""
    U, s, Vt = np.linalg.svd(A_fit, full_matrices=False)
    keep = s > RCOND * s[0]
    utr = U.T @ y_fit
    best_v, best_ho, best_tag = None, np.inf, None
    for a in ALPHAS_VAL:
        if a == 0.0:
            inv = np.where(keep, 1.0 / np.where(keep, s, 1.0), 0.0)
        else:
            mu = (a * s[0]) ** 2
            inv = s / (s ** 2 + mu)
        v = Vt.T @ (inv * utr)
        ho = float(np.linalg.norm(A_ho @ v - y_ho))
        if ho < best_ho:
            best_v, best_ho, best_tag = v, ho, f"alpha={a:g}"
    if v_keep is not None:
        ho = float(np.linalg.norm(A_ho @ v_keep - y_ho))
        if ho < best_ho:
            best_v, best_ho, best_tag = v_keep, ho, "keep"
    return best_v, best_tag


def probe(model, x_fit, y_fit, x_ho, y_ho, x_te, y_te):
    """The lstsq probe: validated readout solve on the CURRENT geometry
    (solved on fit rows, damping validated on held-out rows), scored on the
    test set. Does not modify the model. Returns test rel L2."""
    A_fit = _design(model, x_fit)
    A_ho = _design(model, x_ho)
    v, tag = validated_solve(A_fit, y_fit.numpy(), A_ho, y_ho.numpy())
    A_te = _design(model, x_te)
    err = A_te @ v - y_te.numpy()
    return float(np.linalg.norm(err) / np.linalg.norm(y_te.numpy())), tag


# ----------------------------- training loop -----------------------------

def rel_l2(model, x, y):
    with torch.no_grad():
        pred = model(x).squeeze(-1)
        return float(torch.linalg.norm(pred - y) / torch.linalg.norm(y))


def train_arm(model, kind, x_fit, y_fit, x_ho, y_ho, x_te, y_te, *,
              steps=4000, batch=128, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
              cadence=200, eval_every=25, seed=0, probe_every=250):
    """One training run. kind in {"adam", "qiopt"}. Returns history dict.

    Both arms share the manual-Adam code path; qiopt adds the solve events,
    the line-search cap on the geometry step, and the validated finisher.

    probe_every: every that many steps (plus step 0 and the final step), run
    a NON-INVASIVE terminal-solve probe on both arms: the exact validated
    finisher procedure (damping chosen on held-out rows, the current in-run
    readout competing as a candidate), computed on copies and scored on test.
    The live model is never modified by a probe.
    """
    g = torch.Generator().manual_seed(seed)
    params = [model.fc1.weight, model.fc1.bias, model.fc2.weight, model.fc2.bias]
    is_geo = [True, True, False, False]
    mbuf = [torch.zeros_like(p) for p in params]
    vbuf = [torch.zeros_like(p) for p in params]
    n = x_fit.shape[0]
    y_fit_norm = float(torch.linalg.norm(y_fit))
    b1, b2 = betas
    t_adam = 0
    passes = 0
    solved_once = False
    solve_log = []
    hist = {"step": [], "test_rel": [], "train_rel": [], "passes": []}
    probes = {"step": [], "test_rel": [], "tag": []}
    y_fit_np, y_ho_np, y_te_np = y_fit.numpy(), y_ho.numpy(), y_te.numpy()
    y_te_norm = float(np.linalg.norm(y_te_np))

    def log_eval(step):
        hist["step"].append(step)
        hist["test_rel"].append(rel_l2(model, x_te, y_te))
        hist["train_rel"].append(rel_l2(model, x_fit, y_fit))
        hist["passes"].append(passes)

    def probe_solve(step):
        """Non-invasive: what would the validated terminal finisher give NOW?
        Computed on copies; the live model is untouched, training continues."""
        A_fit = _design(model, x_fit)
        A_ho = _design(model, x_ho)
        v, tag = validated_solve(A_fit, y_fit_np, A_ho, y_ho_np,
                                 v_keep=_readout_vec(model))
        A_te = _design(model, x_te)
        rel = float(np.linalg.norm(A_te @ v - y_te_np) / y_te_norm)
        probes["step"].append(step)
        probes["test_rel"].append(rel)
        probes["tag"].append(tag)

    log_eval(0)
    if probe_every:
        probe_solve(0)
    perm = torch.randperm(n, generator=g)
    ptr = 0
    for step in range(1, steps + 1):
        if ptr + batch > n:
            perm = torch.randperm(n, generator=g)
            ptr = 0
        idx = perm[ptr:ptr + batch]
        ptr += batch
        xb, yb = x_fit[idx], y_fit[idx]

        model.zero_grad(set_to_none=True)
        pred = model(xb).squeeze(-1)
        loss = torch.mean((pred - yb) ** 2)
        loss.backward()
        passes += 2
        t_adam += 1

        upds = []
        with torch.no_grad():
            for p, mb, vb in zip(params, mbuf, vbuf):
                gr = p.grad
                mb.mul_(b1).add_(gr, alpha=1 - b1)
                vb.mul_(b2).addcmul_(gr, gr, value=1 - b2)
                mh = mb / (1 - b1 ** t_adam)
                vh = vb / (1 - b2 ** t_adam)
                upds.append(-lr * mh / (vh.sqrt() + eps))

        scale = 1.0
        if kind == "qiopt" and solved_once:
            # exact-line-search cap on the geometry step: one closed-form JVP
            with torch.no_grad():
                z = model.fc1(xb)
                sech2 = 1.0 - torch.tanh(z) ** 2
                dz = xb @ upds[0].T + upds[1]
                dpred = (sech2 * dz) @ model.fc2.weight.T
                dpred = dpred.squeeze(-1)
                r = model(xb).squeeze(-1) - yb
                denom = float(torch.dot(dpred, dpred))
                if denom > 0:
                    tstar = float(-torch.dot(r, dpred) / denom)
                    scale = min(max(tstar, 0.0), 1.0)
            passes += 1

        with torch.no_grad():
            for p, u, geo in zip(params, upds, is_geo):
                p.add_(u, alpha=scale if geo else 1.0)

        if kind == "qiopt" and step % cadence == 0:
            A_fit = _design(model, x_fit)
            v_cur = _readout_vec(model)
            r_entry = float(np.linalg.norm(A_fit @ v_cur - y_fit.numpy())) / y_fit_norm
            alpha = float(np.clip(r_entry, ALPHA_MIN, ALPHA_MAX))
            v_new = damped_correction(A_fit, y_fit.numpy(), v_cur, alpha)
            _set_readout(model, v_new)
            solved_once = True
            solve_log.append({"step": step, "alpha": alpha})
            passes += 1  # the design-matrix build; the SVD is offline work

        if step % eval_every == 0 or step == steps:
            log_eval(step)

        if probe_every and (step % probe_every == 0 or step == steps):
            probe_solve(step)

    result = {"hist": hist, "passes": passes, "solve_log": solve_log,
              "probes": probes,
              "own_final_prefinisher": hist["test_rel"][-1]}

    if kind == "qiopt":
        # validated terminal finisher, in-run readout competes
        A_fit = _design(model, x_fit)
        A_ho = _design(model, x_ho)
        v, tag = validated_solve(A_fit, y_fit.numpy(), A_ho, y_ho.numpy(),
                                 v_keep=_readout_vec(model))
        _set_readout(model, v)
        result["finisher_tag"] = tag
    result["own_final"] = rel_l2(model, x_te, y_te)
    return result
