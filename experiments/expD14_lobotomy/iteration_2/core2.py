"""expD14 iteration 2 -- one question: which measured signal sets mu.

Everything else is frozen at the iteration_1 harness: Adam fully stock on the
geometry (no throttles -- iteration_1 cleared them of blame), damped direct
solve on the discovered L (the reference instrument, so alpha is meaningful
over its whole range), Method C discovery on a 200-step cadence, the floor
trajectory as the score. The ONLY thing that varies across arms is the
controller mapping measured signals to the damping alpha = sqrt(mu)/sigma_1.

The controllers:

  fixed    alpha pinned (the alpha=1e-15 arm is iteration_1's stun anchor)
  ladder   preset log-linear descent 1 -> 1e-15 over the run; a schedule with
           no signal, so it cannot know which init it was handed
  rentry   alpha_t = ||r||/||y|| at step entry -- the damping floor (spec V.6)
           applied as an equality: never solve past the error you walked in
           with; self-clocking on any init
  lm       the classical Levenberg-Marquardt ratio, but on the COMPOSITE step.
           On the L step alone the ratio is identically 1 (f is exactly linear
           in L, the local model is exact), so it would always say "solve
           exactly" -- the stunning arm. Measured on solve + Adam motion
           together it reads the geometry's nonlinearity. Multiplicative
           update, floored at r_entry, with the loss-noise guard lesson 1
           requires (loss differences are blind below relative ~1e-12).
  snr      alpha_t = max(shat_geo, r_entry): stay soft while Adam's geometry
           gradient is persistent, harden as it cools. Sam's stated target
           behavior encoded directly. shat's single-sample startup reads 1,
           so the run begins soft with no special-casing.

All controllers clip alpha to [1e-15, 1].
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

RESULTS = REPO / "results" / "checkpoint_D_optimizers" / "expD14_lobotomy" / "iteration_2"
FIGS = RESULTS / "figures"

build_case = core1.build_case
discover_pertensor = core1.discover_pertensor
coherent_travel = core1.coherent_travel
snr_corrected = core1.snr_corrected
append, load = core1.append, core1.load

RCOND = 1e-15
COL_FLOOR = 1e-8
ALPHA_MIN, ALPHA_MAX = 1e-15, 1.0
LM_GUARD = 1e-12          # relative predicted improvement below which the
                          # ratio is loss-rounding noise and must not act


def train_mu(env, iters, *, alpha_mode, lr=1e-3, betas=(0.9, 0.999),
             eps_adam=1e-8, refresh=200, record=5, floor_every=50,
             travel_window=100, seed=0, verbose=False):
    """iteration_1's trainer stripped to the mu question. Adam untouched on
    A; damped truncated-SVD solve on L with alpha from the controller.
    alpha_mode in {"adam", "fixed:<v>", "ladder", "rentry", "lm", "snr"}."""
    b1, b2 = betas
    theta = env["theta0"].clone()
    m, W = env["m"], env["W"]
    y_norm = env["y_norm"]
    n_tr = env["n_train"]
    geo = torch.zeros(m, dtype=torch.bool)
    geo[:2 * W] = True

    ma = torch.zeros(m, dtype=theta.dtype)
    va = torch.zeros(m, dtype=theta.dtype)
    t_adam = 0
    passes = 0

    solve_on = alpha_mode != "adam"
    if solve_on:
        members, ev = discover_pertensor(env, theta, seed=seed)
        passes += ev
    else:
        members = torch.zeros(m, dtype=torch.bool)
    idx = torch.nonzero(members, as_tuple=False).flatten()
    n_member_changes = 0

    sigma1 = 1.0
    pw = torch.randn(int(idx.numel()), dtype=theta.dtype) if idx.numel() else None
    alpha_lm = 1.0                    # lm controller state
    prev = None                       # (||r||^2, predicted ||r_new||^2)
    trav_vec = torch.zeros(m, dtype=theta.dtype)
    trav_len = 0.0

    hist = {k: [] for k in ("it", "rel", "rel_in", "rel_out", "alpha",
                            "rentry", "shat", "rho_lm", "B", "vnorm",
                            "snr", "passes")}
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
    alpha = ALPHA_MAX
    rho_lm = float("nan")
    for it in range(1, iters + 1):
        r, g = env["f_residgrad"](theta)
        passes += 2
        rn2 = float(r.dot(r))
        r_entry = float(np.sqrt(rn2)) / y_norm

        # ---- close the LM loop from the previous step's prediction ----
        if alpha_mode == "lm" and prev is not None:
            rn2_prev, pred2_prev = prev
            I = rn2_prev - pred2_prev            # predicted decrease
            G = rn2_prev - rn2                   # realized decrease
            if I > LM_GUARD * rn2_prev:          # noise guard: else freeze
                rho_lm = G / I
                if rho_lm > 0.75:
                    alpha_lm = alpha_lm / 2.0
                elif rho_lm < 0.25:
                    alpha_lm = alpha_lm * 4.0
                alpha_lm = float(np.clip(alpha_lm, ALPHA_MIN, ALPHA_MAX))
        prev = None

        t_adam += 1
        ma = b1 * ma + (1.0 - b1) * g
        va = b2 * va + (1.0 - b2) * g * g
        mh = ma / (1.0 - b1 ** t_adam)
        vh = va / (1.0 - b2 ** t_adam)
        adam_full = -lr * mh / (vh.sqrt() + eps_adam)
        shat = snr_corrected(mh, vh, geo, b1)

        # ---- the controller ----
        if alpha_mode.startswith("fixed:"):
            alpha = float(alpha_mode.split(":")[1])
        elif alpha_mode == "ladder":
            alpha = 10.0 ** (-15.0 * it / iters)
        elif alpha_mode == "rentry":
            alpha = r_entry
        elif alpha_mode == "lm":
            alpha = max(alpha_lm, r_entry)
        elif alpha_mode == "snr":
            alpha = max(shat, r_entry)
        elif alpha_mode != "adam":
            raise ValueError(alpha_mode)
        alpha = float(np.clip(alpha, ALPHA_MIN, ALPHA_MAX))

        # ---- A: stock Adam off L; L: damped direct solve ----
        stepA = adam_full.clone()
        if idx.numel():
            stepA[idx] = 0.0
        dth = stepA.clone()
        if solve_on and idx.numel():
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
            passes += int(idx.numel())
            dth[idx] += dmu

        # ---- LM prediction for the composite step, one JVP ----
        if alpha_mode == "lm":
            jd = env["f_jvp"](theta, dth)
            passes += 1
            pred = r + jd
            prev = (rn2, float(pred.dot(pred)))

        th_new = theta + dth
        if not np.isfinite(float(torch.linalg.norm(th_new))):
            ma.zero_()
            va.zero_()
            prev = None
            continue
        theta = th_new

        trav_vec += dth * geo
        trav_len += float(torch.linalg.norm(dth[geo]))
        if it % travel_window == 0:
            thist["it"].append(it)
            thist["travel"].append(coherent_travel(trav_vec[geo], trav_len))
            trav_vec.zero_()
            trav_len = 0.0

        if solve_on and it % refresh == 0:
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
            hist["rentry"].append(float(r_entry))
            hist["shat"].append(float(shat))
            hist["rho_lm"].append(float(rho_lm))
            hist["B"].append(int(idx.numel()))
            hist["vnorm"].append(float(torch.linalg.norm(theta[env["idx_readout"]])))
            hist["snr"].append(float(shat))
            hist["passes"].append(int(passes))
        if verbose and it % max(1, iters // 10) == 0:
            print(f"    it {it:5d}  rel {hist['rel'][-1]:.3e}  "
                  f"alpha {alpha:.2e}  floor {fhist['floor'][-1]:.3e}")

    fl, fi, fo = env["geometry_floor_split"](theta)
    return dict(hist=hist, fhist=fhist, thist=thist, theta=theta,
                best_rel=float(np.min(hist["rel"])), final_rel=hist["rel"][-1],
                floor_final=fl, floor_final_in=fi, floor_final_out=fo,
                passes=passes, n_member_changes=n_member_changes)


ARMS = {
    "adam":   dict(alpha_mode="adam"),
    "alo":    dict(alpha_mode="fixed:1e-15"),
    "ladder": dict(alpha_mode="ladder"),
    "rentry": dict(alpha_mode="rentry"),
    "lm":     dict(alpha_mode="lm"),
    "snr":    dict(alpha_mode="snr"),
}
