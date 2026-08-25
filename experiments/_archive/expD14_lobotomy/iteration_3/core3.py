"""expD14 iteration 3 -- the Adam handoff: cool the geometry step as it is
ready, so r_entry can follow the floor down and the run banks.

Frozen from iteration_2: the r_entry-damped direct solve on the Method-C
discovered L, stock Adam moments, the floor trajectory as the score. The ONLY
thing that varies is how the geometry step's length is cooled:

  none    no cooling (iteration_2's `rentry` arm; baseline rows come from t6)
  ls      exact-line-search cap: lenA = min(||stepA||, t*), with
          t* = -(r . J pA)/||J pA||^2 -- one JVP, linear in the residual, so
          it vanishes as the problem is solved. Kills the ||v||*eta sawtooth.
  snr     scale the geometry step by shat_geo, the corrected Adam SNR:
          cools when learning is DONE, even if the error is still high
          (the clustered stall), and reads 1 at startup so early learning
          is untouched.
  ls_snr  both, composed multiplicatively: the cap handles the residual
          scale, the gate handles persistence.
  cos     cosine decay of the geometry step to 0 over the run -- the
          classical no-signal schedule; expected to fail on one init or the
          other, like the preset mu ladder did.
  perp    scale the geometry step by ||r_perp||/||r||, the fraction of the
          residual the current features CANNOT express (r_perp = r - U U^T r,
          free from the SVD the solve already computes). This is the only
          candidate that separates the inits at step 1: ~1e-13 on qi (all of
          the start error is expressible, geometry should not move on it),
          ~0.4 on rand (most of the error needs new geometry). It encodes
          "never move the geometry on error the solve is about to remove."

The two coupled predictions this tests: on `qi` the cooled arms should hold
the 1e-14 floor AND let the reached error follow alpha to ~1e-15 (recovering
iteration_0's result without hand throttles); on `rand` cooling must NOT
stun the early geometry learning (cos is the arm expected to break this).
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]

_s1 = importlib.util.spec_from_file_location(
    "d14_core1", HERE.parent / "iteration_1" / "core1.py")
core1 = importlib.util.module_from_spec(_s1)
_s1.loader.exec_module(core1)

RESULTS = REPO / "results" / "checkpoint_D_optimizers" / "expD14_lobotomy" / "iteration_3"
FIGS = RESULTS / "figures"

build_case = core1.build_case
discover_pertensor = core1.discover_pertensor
coherent_travel = core1.coherent_travel
snr_corrected = core1.snr_corrected
append, load = core1.append, core1.load

RCOND = 1e-15
COL_FLOOR = 1e-8
ALPHA_MIN, ALPHA_MAX = 1e-15, 1.0


def train_handoff(env, iters, *, cool, lr=1e-3, betas=(0.9, 0.999),
                  eps_adam=1e-8, refresh=200, record=5, floor_every=50,
                  travel_window=100, seed=0, verbose=False):
    """iteration_2's rentry trainer plus one cooling rule on the A step.
    cool in {"none", "ls", "snr", "ls_snr", "cos"}."""
    b1, b2 = betas
    theta = env["theta0"].clone()
    m, W = env["m"], env["W"]
    y_norm = env["y_norm"]
    geo = torch.zeros(m, dtype=torch.bool)
    geo[:2 * W] = True

    ma = torch.zeros(m, dtype=theta.dtype)
    va = torch.zeros(m, dtype=theta.dtype)
    t_adam = 0
    passes = 0

    members, ev = discover_pertensor(env, theta, seed=seed)
    passes += ev
    idx = torch.nonzero(members, as_tuple=False).flatten()
    n_member_changes = 0

    sigma1 = 1.0
    pw = torch.randn(int(idx.numel()), dtype=theta.dtype)
    trav_vec = torch.zeros(m, dtype=theta.dtype)
    trav_len = 0.0

    hist = {k: [] for k in ("it", "rel", "rel_in", "rel_out", "alpha",
                            "shat", "coolf", "B", "vnorm", "passes")}
    fhist = {k: [] for k in ("it", "floor", "floor_in", "floor_out")}
    thist = {k: [] for k in ("it", "travel")}

    def masked_ops(th):
        def av(z):
            dv = torch.zeros(m, dtype=th.dtype)
            dv[idx] = z
            return env["f_jvp"](th, dv)

        def atu(u):
            return env["f_vjp"](th, u)[idx]
        return av, atu

    def log_floor(it):
        fl, fi, fo = env["geometry_floor_split"](theta)
        fhist["it"].append(it)
        fhist["floor"].append(fl)
        fhist["floor_in"].append(fi)
        fhist["floor_out"].append(fo)

    log_floor(0)
    for it in range(1, iters + 1):
        r, g = env["f_residgrad"](theta)
        passes += 2
        r_entry = float(torch.linalg.norm(r)) / y_norm
        alpha = float(np.clip(r_entry, ALPHA_MIN, ALPHA_MAX))

        t_adam += 1
        ma = b1 * ma + (1.0 - b1) * g
        va = b2 * va + (1.0 - b2) * g * g
        mh = ma / (1.0 - b1 ** t_adam)
        vh = va / (1.0 - b2 ** t_adam)
        adam_full = -lr * mh / (vh.sqrt() + eps_adam)
        shat = snr_corrected(mh, vh, geo, b1)

        # ---- L: the r_entry-damped direct solve (first: `perp` needs U) ----
        dmu = None
        fperp = 1.0
        if idx.numel():
            av, atu = masked_ops(theta)
            if it == 1 or (it - 1) % refresh == 0:
                sigma1, pw = core1.power_sigma1(av, atu, pw, iters=3)
                passes += 6
            mu = (alpha * sigma1) ** 2
            J = env["f_jac_cols"](theta, idx)
            cn = torch.linalg.norm(J, dim=0)
            cmax = float(cn.max()) if cn.numel() else 0.0
            live = cn > COL_FLOOR * cmax if cmax > 0 else cn > 0
            dmu = torch.zeros(int(idx.numel()), dtype=theta.dtype)
            if bool(live.any()):
                Jl = J[:, live]
                U, s, Vt = torch.linalg.svd(Jl, full_matrices=False)
                inv = torch.where(s > RCOND * s[0], s / (s ** 2 + mu),
                                  torch.zeros_like(s))
                dmu[live] = Vt.T @ (inv * (U.T @ (-r)))
                utr = U.T @ r
                rperp2 = float(r.dot(r)) - float(utr.dot(utr))
                fperp = float(np.sqrt(max(rperp2, 0.0)) /
                              max(float(torch.linalg.norm(r)), 1e-300))
            passes += int(idx.numel())

        # ---- A: Adam with the cooling rule ----
        stepA = adam_full.clone()
        if idx.numel():
            stepA[idx] = 0.0
        nA = float(torch.linalg.norm(stepA))
        lenA = nA
        if nA > 0 and cool in ("ls", "ls_snr"):
            pA = stepA / nA
            jA = env["f_jvp"](theta, pA)
            passes += 1
            den = float(jA.dot(jA))
            tstar = (-float(r.dot(jA)) / den) if den > 0 else 0.0
            lenA = min(nA, max(tstar, 0.0))
        if cool in ("snr", "ls_snr"):
            lenA = lenA * shat
        if cool == "cos":
            lenA = lenA * 0.5 * (1.0 + np.cos(np.pi * it / iters))
        if cool == "perp":
            lenA = lenA * fperp
        coolf = lenA / nA if nA > 0 else 1.0
        dth = (lenA / nA) * stepA if nA > 0 else stepA
        if dmu is not None:
            dth = dth.clone()
            dth[idx] += dmu

        th_new = theta + dth
        if not np.isfinite(float(torch.linalg.norm(th_new))):
            ma.zero_()
            va.zero_()
            continue
        theta = th_new

        trav_vec += dth * geo
        trav_len += float(torch.linalg.norm(dth[geo]))
        if it % travel_window == 0:
            thist["it"].append(it)
            thist["travel"].append(coherent_travel(trav_vec[geo], trav_len))
            trav_vec.zero_()
            trav_len = 0.0

        if it % refresh == 0:
            new, ev = discover_pertensor(env, theta, seed=seed + it)
            passes += ev
            if not torch.equal(new, members):
                n_member_changes += 1
                members = new
                idx = torch.nonzero(members, as_tuple=False).flatten()
                pw = torch.randn(int(idx.numel()), dtype=theta.dtype) \
                    if idx.numel() else None

        if it % floor_every == 0 or it == iters:
            log_floor(it)
        if it % record == 0 or it == iters:
            rel, rin, rout = env["eval_rel_split"](theta)
            hist["it"].append(it)
            hist["rel"].append(rel)
            hist["rel_in"].append(rin)
            hist["rel_out"].append(rout)
            hist["alpha"].append(float(alpha))
            hist["shat"].append(float(shat))
            hist["coolf"].append(float(coolf))
            hist["B"].append(int(idx.numel()))
            hist["vnorm"].append(float(torch.linalg.norm(theta[env["idx_readout"]])))
            hist["passes"].append(int(passes))
        if verbose and it % max(1, iters // 10) == 0:
            print(f"    it {it:5d}  rel {hist['rel'][-1]:.3e}  "
                  f"cool {coolf:.2e}  floor {fhist['floor'][-1]:.3e}")

    fl, fi, fo = env["geometry_floor_split"](theta)
    return dict(hist=hist, fhist=fhist, thist=thist, theta=theta,
                best_rel=float(np.min(hist["rel"])), final_rel=hist["rel"][-1],
                floor_final=fl, floor_final_in=fi, floor_final_out=fo,
                passes=passes, n_member_changes=n_member_changes)


ARMS = {
    "ls":     dict(cool="ls"),
    "snr":    dict(cool="snr"),
    "ls_snr": dict(cool="ls_snr"),
    "cos":    dict(cool="cos"),
    "perp":   dict(cool="perp"),
}
