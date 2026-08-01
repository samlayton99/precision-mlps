"""expD14 iteration 0 -- the lobotomy: one step-energy budget, split between
Adam and a damped least-squares solve.

THE IDEA (Sam, 2026-07-31)
  Every iteration updates every parameter. A measured certificate splits the
  parameter vector into L (the block the model is linear in) and A (the rest).
  There is ONE energy budget per step: E = ||stock Adam step||, MEASURED this
  step over all m coordinates. A fraction rho of it goes to L, (1-rho) to A.
  Adam's moments are updated for ALL parameters at every step regardless of who
  spends the energy, so a parameter moving A <-> L never sees a cold optimizer.

    Delta_A = (1 - rho) * s_A,              s_A = Adam's own step on A
    Delta_L = min(||d_mu||, rho * E) * pL,  pL = unit(d_mu)

  A's share is Adam's actual step scaled down, never a renormalized one: at
  rho = 0 with L empty this is stock Adam to the last bit (T1/P5), which is the
  reliability floor the whole program is graded against.

  d_mu is the damped Gauss-Newton step on the L block,
        d_mu = argmin_d  ||J_L d + r||^2 + mu ||d||^2 ,
  and mu is the whole story. Sam's framing: multiply the GN update by mu to put
  it back in RAW (gradient) coordinates, then scale it to the size the energy
  allocation says. That is exactly what happens here, because

        mu -> inf :   mu * d_mu -> J_L^T (-r)   (the raw gradient direction)
        mu -> 0   :   d_mu      -> the exact least-squares step

  so the L update morphs continuously from "Adam-scale gradient descent on L"
  to "the exact solve", and the trust-region clip -- never the damping -- sets
  its length. The clip stops binding on its own once ||d_mu|| shrinks below the
  budget, which is how the terminal solve lands without a mode switch.

THE TWO MEASURED KNOBS (no schedule, no hand-set constant)

  mu, from drift.  alpha = sqrt(mu)/sigma_1(J_L) is exactly the inverse
  effective condition number (expD12 phase 0), so alpha is "the relative scale
  below which I do not trust my own features". Measure it: the A step moves the
  predictions by J*Delta_A, so
        alpha_t = EMA( ||J Delta_A|| / ||y|| ),  clipped to [1e-15, 1],
        mu = (alpha * sigma_1)^2 .
  QI init has a zero readout, so every geometry gradient is zero, so the A step
  is zero, so alpha floors instantly and the solve is undamped from step 1.
  Random init moves the geometry hard, alpha ~ 1, and the L block is damped
  down to a gradient step until the geometry quiets. That is the prediction.

  rho, from predicted decrease.  For a unit direction p, the exact-line-search
  decrease of ||r||^2 is (r.Jp)^2/||Jp||^2 -- one JVP each. Allocate
        rho = gain_L / (gain_A + gain_L) .
  Free of constants, O(d) state, 2 extra passes per step.

COST per step: 1 fused fwd+bwd (Adam) + 2*tau passes (LSQR) + 2 passes (gains).
STATE: theta, Adam (m,v), membership mask, sigma_1 power vector. All O(d).
No block preconditioner anywhere -- scalar (Adam-diagonal) only, by decision.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import importlib.util

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
D08 = REPO / "experiments" / "expD08_qi_init_nlcg"
for p in (str(REPO), str(D08)):
    if p not in sys.path:
        sys.path.insert(0, p)

# load expD08's run.py by explicit path -- "run" is a common module name and a
# plain `import run` resolved to a different file inside spawned workers.
_spec = importlib.util.spec_from_file_location("d08_run", D08 / "run.py")
d08run = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(d08run)

RESULTS = REPO / "results" / "checkpoint_D_optimizers" / "expD14_lobotomy" / "iteration_0"
FIGS = RESULTS / "figures"

N_TRAIN, N_EVAL = 2003, 4001
LAM = 0.25
RCOND = 1e-15
EPS_MACH = 2.220446049250313e-16

Q_ENTER, Q_EXIT = 1e-8, 1e-7      # iteration-11 certificate thresholds
COL_FLOOR = 1e-8                  # observability: drop J columns below this x max
ALPHA_MIN, ALPHA_MAX = 1e-15, 1.0


# =========================================================== the problem ====

def build_case(fn_name, N, init="qi", seed=0):
    """1-D tanh MLP, theta = (w | b | v | c0), m = 3W+1.

    init="qi"    QI geometry in affine coordinates, readout zeroed (the
                 expD06/seed9 protocol iteration 11 used).
    init="rand"  PyTorch's stock nn.Linear init: w,b ~ U(-1,1) (fan_in 1),
                 v ~ U(-1/sqrt(W), 1/sqrt(W)), c0 = 0.
    """
    f_vec = d08run.make_fn(d08run.TARGETS[fn_name])
    centers, gamma, _halo = d08run.geometry(N, LAM)
    W = centers.size
    m = 3 * W + 1

    if init == "qi":
        theta0 = torch.tensor(np.concatenate([
            np.full(W, gamma), -gamma * centers, np.zeros(W), [0.0]]))
    elif init == "rand":
        rng = np.random.default_rng(1234 + seed)
        theta0 = torch.tensor(np.concatenate([
            rng.uniform(-1, 1, W), rng.uniform(-1, 1, W),
            rng.uniform(-1, 1, W) / np.sqrt(W), [0.0]]))
    else:
        raise ValueError(init)

    x_tr = np.linspace(-1, 1, N_TRAIN)
    x_ev = np.linspace(-1, 1, N_EVAL)
    xt = torch.tensor(x_tr).unsqueeze(1)
    yt = torch.tensor(f_vec(x_tr))
    xe = torch.tensor(x_ev).unsqueeze(1)
    ye = torch.tensor(f_vec(x_ev))
    ye_norm = float(torch.linalg.norm(ye))

    def model(x, th):
        return torch.tanh(x * th[:W] + th[W:2 * W]) @ th[2 * W:3 * W] + th[-1]

    def f_pred(th):
        return model(xt, th)

    def f_resid(th):
        with torch.no_grad():
            return (f_pred(th) - yt).detach()

    def f_jvp(th, dvec):
        _, jd = torch.func.jvp(f_pred, (th,), (dvec,))
        return jd.detach()

    def f_vjp(th, seed_v):
        thr = th.detach().requires_grad_(True)
        pred = f_pred(thr)
        (g,) = torch.autograd.grad(pred, thr, grad_outputs=seed_v)
        return g.detach()

    def f_residgrad(th):
        thr = th.detach().requires_grad_(True)
        pred = model(xt, thr)
        r = (pred - yt).detach()
        (g,) = torch.autograd.grad(pred, thr, grad_outputs=(2.0 / N_TRAIN) * r)
        return r, g.detach()

    def eval_rel(th):
        with torch.no_grad():
            return float(torch.linalg.norm(model(xe, th) - ye)) / ye_norm

    def probe_fn(th):
        return d08run.probe_self_curvature(f_pred, th)

    # the frozen-geometry floor of the INITIAL geometry (what a direct solve
    # on theta0's features buys, i.e. the best a readout-only method can do)
    def _floor(th):
        t = th.detach().numpy()
        z = np.tanh(x_tr[:, None] * t[:W][None, :] + t[W:2 * W][None, :])
        A = np.hstack([z, np.ones((x_tr.size, 1))])
        U, s, Vt = np.linalg.svd(A, full_matrices=False)
        keep = s > RCOND * s[0]
        sol = Vt.T @ (np.where(keep, 1.0 / np.where(keep, s, 1.0), 0.0)
                      * (U.T @ f_vec(x_tr)))
        ze = np.tanh(x_ev[:, None] * t[:W][None, :] + t[W:2 * W][None, :])
        A_ev = np.hstack([ze, np.ones((x_ev.size, 1))])
        return float(np.linalg.norm(A_ev @ sol - f_vec(x_ev))
                     / np.linalg.norm(f_vec(x_ev)))

    idx_readout = torch.arange(2 * W, 3 * W + 1)      # oracle L = (v, c0)
    # geometry_floor(theta) is the diagnostic that separates the two jobs:
    # "did the readout get solved" (error vs this number) from "did the
    # geometry get learned" (this number vs its value at theta0).

    def f_jac_cols(th, cols):
        """J[:, cols], analytically, in one pass. Arithmetically identical to
        looping JVPs over the basis vectors, and used ONLY by the reference
        exact-solve arm (the shipped lsolver='lsqr' path is matrix-free and
        never calls this). It exists because a per-column JVP loop costs |L|
        passes PER STEP, which is unaffordable at 3000 steps."""
        with torch.no_grad():
            z = xt * th[:W] + th[W:2 * W]
            t = torch.tanh(z)
            sech2 = 1.0 - t * t
            v = th[2 * W:3 * W]
            J = torch.cat([sech2 * v * xt, sech2 * v, t,
                           torch.ones(xt.shape[0], 1, dtype=th.dtype)], 1)
            return J[:, cols]

    return dict(theta0=theta0, W=W, m=m, centers=centers, gamma=gamma,
                fn=fn_name, N=N, init=init, floor0=_floor(theta0),
                geometry_floor=_floor,
                f_pred=f_pred, f_resid=f_resid, f_jvp=f_jvp, f_vjp=f_vjp,
                f_residgrad=f_residgrad, eval_rel=eval_rel, probe_fn=probe_fn,
                idx_readout=idx_readout, f_jac_cols=f_jac_cols,
                y_norm=float(torch.linalg.norm(yt)),
                x_tr=x_tr, y_tr=f_vec(x_tr))


# ============================================================== the solver ==

def lsqr_damped_op(av, atu, rhs, mu, d, maxiter, atol=1e-15, btol=1e-15):
    """LSQR on [J; sqrt(mu) I] x ~= [rhs; 0], matrix-free, O(d) state.

    av(z) -> J z  (length n);  atu(u) -> J^T u  (length d).  Damping is applied
    implicitly (never by stacking an identity).  Fong-Saunders stopping with
    eps-scaled tolerances (expD10 T7).  Returns (x, iters, atr0, atr).
    """
    n = rhs.numel()
    sq = float(np.sqrt(mu)) if mu > 0 else 0.0
    damp = sq > 0.0

    def AV(z):
        jz = av(z)
        return torch.cat([jz, sq * z]) if damp else jz

    def ATU(u):
        return atu(u[:n]) + sq * u[n:] if damp else atu(u)

    u = torch.cat([rhs, torch.zeros(d, dtype=rhs.dtype)]) if damp else rhs.clone()
    beta = float(torch.linalg.norm(u))
    bnorm = beta
    if beta == 0.0:
        return torch.zeros(d, dtype=rhs.dtype), 0, 0.0, 0.0
    u = u / beta
    v = ATU(u)
    alpha = float(torch.linalg.norm(v))
    if alpha == 0.0:
        return torch.zeros(d, dtype=rhs.dtype), 0, 0.0, 0.0
    v = v / alpha

    w = v.clone()
    x = torch.zeros(d, dtype=rhs.dtype)
    phibar, rhobar = beta, alpha
    anorm2 = alpha ** 2
    atr0 = beta * alpha
    atr = atr0
    it = 0
    for it in range(1, maxiter + 1):
        un = AV(v) - alpha * u
        beta = float(torch.linalg.norm(un))
        if beta <= 1e-14 * np.sqrt(anorm2):
            break
        u = un / beta
        vn = ATU(u) - beta * v
        alpha = float(torch.linalg.norm(vn))
        anorm2 += alpha ** 2 + beta ** 2
        if alpha <= 1e-14 * np.sqrt(anorm2):
            break
        v = vn / alpha
        rho = float(np.hypot(rhobar, beta))
        cs, sn = rhobar / rho, beta / rho
        theta_ = sn * alpha
        rhobar = -cs * alpha
        phi = cs * phibar
        phibar = sn * phibar
        x = x + (phi / rho) * w
        w = v - (theta_ / rho) * w
        atr = abs(phibar * alpha * cs)
        anorm = np.sqrt(anorm2)
        if atr <= atol * anorm * phibar or phibar <= btol * bnorm:
            break
    return x, it, atr0, atr


def power_sigma1(av, atu, z, iters=3):
    """sigma_1(J) by power iteration on J^T J.  z is O(d) carried state."""
    s = 0.0
    for _ in range(iters):
        nz = float(torch.linalg.norm(z))
        if nz == 0.0:
            return 1.0, z
        z = z / nz
        jz = av(z)
        s = float(torch.linalg.norm(jz))
        z = atu(jz)
    return max(s, 1e-300), z


# ========================================================= the certificate ==

def certify(probe_fn, theta, f_pred, info, y_norm, members):
    """iteration-11 certificate: linear AND informed, with hysteresis and the
    absolute rounding guard.  Returns (members, qhat)."""
    q = probe_fn(theta)
    with torch.no_grad():
        p_scale = max(float(torch.linalg.norm(f_pred(theta))), y_norm)
    q_noise = 4.0 * EPS_MACH / d08run.PROBE_EPS ** 2 * p_scale
    q = torch.where(q < 100.0 * q_noise, torch.zeros_like(q), q)
    qm = float(q.max())
    qhat = q / qm if qm > 0 else torch.zeros_like(q)
    informed = info > 0
    stay = members & (qhat < Q_EXIT) & informed
    enter = (~members) & (qhat < Q_ENTER) & informed
    return (stay | enter), qhat


# ============================================================== the trainer ==

def train_lobo(env, iters, *, lr=1e-3, betas=(0.9, 0.999), eps_adam=1e-8,
               lset="certificate", refresh=200, tau=3, lsolver="lsqr",
               alpha_mode="drift", alpha_fixed=None, alpha_ema=0.9,
               rho_mode="gain", rho_fixed=None, clip=True, astep="adam",
               reset_moments=False, adam_on_L=False, record=1, verbose=False):
    """The lobotomized optimizer.  lset="none" reduces it to stock Adam."""
    b1, b2 = betas
    theta = env["theta0"].clone()
    m = env["m"]
    f_jvp, f_vjp = env["f_jvp"], env["f_vjp"]
    y_norm = env["y_norm"]

    ma = torch.zeros(m, dtype=theta.dtype)
    va = torch.zeros(m, dtype=theta.dtype)
    t_adam = 0

    if lset == "none":
        members = torch.zeros(m, dtype=torch.bool)
    elif lset == "readout":
        members = torch.zeros(m, dtype=torch.bool)
        members[env["idx_readout"]] = True
    elif lset == "certificate":
        r0 = env["f_resid"](theta)
        info0 = f_vjp(theta, (2.0 / N_TRAIN) * r0).abs()
        members, _ = certify(env["probe_fn"], theta, env["f_pred"], info0,
                             y_norm, torch.zeros(m, dtype=torch.bool))
    else:
        raise ValueError(lset)
    idx = torch.nonzero(members, as_tuple=False).flatten()
    r_ref = env["f_resid"](theta)
    backoff, windows, n_probes = 1, 0, 1 if lset == "certificate" else 0

    pw = torch.randn(int(idx.numel()), dtype=theta.dtype) if idx.numel() else None
    sigma1 = 1.0
    alpha = ALPHA_MAX
    a_use = ALPHA_MAX
    a_ema = 0.0
    rho_prev = 0.0
    passes = 0
    E_scale = np.sqrt(m)

    hist = {k: [] for k in ("it", "loss", "rel", "alpha", "mu", "rho", "B",
                            "dnorm", "clipped", "gainA", "gainL", "lsqr_it",
                            "sigma1", "vnorm", "passes", "drift", "lenA",
                            "n_drop")}

    def masked_ops(th):
        def av(z):
            dv = torch.zeros(m, dtype=th.dtype)
            dv[idx] = z
            return f_jvp(th, dv)

        def atu(u):
            return f_vjp(th, u)[idx]
        return av, atu

    for it in range(1, iters + 1):
        r, g = env["f_residgrad"](theta)
        passes += 2

        t_adam += 1
        ma = b1 * ma + (1.0 - b1) * g
        va = b2 * va + (1.0 - b2) * g * g
        mh = ma / (1.0 - b1 ** t_adam)
        vh = va / (1.0 - b2 ** t_adam)
        adam_full = -lr * mh / (vh.sqrt() + eps_adam)

        # ---- the A direction (Adam on the non-certified coordinates) ----
        stepA = adam_full.clone()
        if not adam_on_L and idx.numel():
            stepA[idx] = 0.0
        nA = float(torch.linalg.norm(stepA))
        pA = stepA / nA if nA > 0 else stepA

        # ---- the L direction (damped GN on the certified block) ----
        dmu = None
        mu = 0.0
        lsqr_it = 0
        n_drop = 0
        if idx.numel():
            av, atu = masked_ops(theta)
            if it == 1 or (it - 1) % refresh == 0:
                sigma1, pw = power_sigma1(av, atu, pw, iters=3)
                passes += 6
            if alpha_mode == "fixed":
                alpha = float(alpha_fixed)
            a_use = alpha
            if alpha_mode == "drift_gain":
                # Never resolve L below the residual level A still owns.
                # gainA/(gainA+gainL) = 1-rho and gains are quadratic in the
                # residual, so sqrt(1-rho) is that level in residual units --
                # the same units T1/P1 measured alpha in.  Uses the PREVIOUS
                # step's rho (a control signal, not an equation to close).
                a_use = max(alpha, float(np.sqrt(max(1.0 - rho_prev, 0.0))))
                a_use = float(np.clip(a_use, ALPHA_MIN, ALPHA_MAX))
            mu = (a_use * sigma1) ** 2
            if lsolver == "lsqr":
                dmu, lsqr_it, _, _ = lsqr_damped_op(av, atu, -r, mu,
                                                    int(idx.numel()), tau)
                passes += 2 * max(lsqr_it, 1)
            elif lsolver == "direct":
                J = env["f_jac_cols"](theta, idx)
                # observability filter (iteration 11, 6.4): a certified but
                # data-blind coordinate contributes a ~null column, and the
                # min-norm solve amplifies its eps-level content by 1/sigma
                # into a huge bogus correction. Measured gap on this family:
                # live columns O(1), blind columns <= 1e-14.
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
                n_drop = int((~live).sum())
                passes += int(idx.numel())
            else:
                raise ValueError(lsolver)

        # ---- energy allocation from measured predicted decrease ----
        gainA = gainL = 0.0
        tstarA = float("inf")
        rho = float(rho_fixed) if rho_mode == "fixed" else 0.0
        if idx.numel() and dmu is not None:
            nL = float(torch.linalg.norm(dmu))
            pL_sub = dmu / nL if nL > 0 else dmu
            pL = torch.zeros(m, dtype=theta.dtype)
            pL[idx] = pL_sub
        else:
            nL, pL = 0.0, None

        if rho_mode not in ("gain", "fixed"):
            raise ValueError(rho_mode)
        # jA is needed by the allocation AND by the line search; jL only by the
        # allocation. One JVP each, so the whole allocation costs 2 passes.
        if nA > 0 and (rho_mode == "gain" or astep == "linesearch"):
            jA = f_jvp(theta, pA)
            passes += 1
            den = float(jA.dot(jA))
            gainA = (float(r.dot(jA)) ** 2 / den) if den > 0 else 0.0
            # the exact-line-search length along Adam's own direction. It is
            # LINEAR in the residual, so unlike Adam's scale-free step it
            # vanishes as the problem is solved -- which is what kills the
            # ||v||*eta re-injection (iteration 11, 9.1-9.3).
            tstarA = (-float(r.dot(jA)) / den) if den > 0 else 0.0
        if rho_mode == "gain":
            if pL is not None and nL > 0:
                jLv = f_jvp(theta, pL)
                passes += 1
                den = float(jLv.dot(jLv))
                gainL = (float(r.dot(jLv)) ** 2 / den) if den > 0 else 0.0
            tot = gainA + gainL
            rho = (gainL / tot) if tot > 0 else 0.0
        rho_prev = rho

        # ---- the split step ----
        # E is the energy a stock Adam step would have spent, measured now.
        # A keeps its OWN step scaled by (1-rho) (so rho=0 with L empty is
        # bit-for-bit Adam); L gets rho*E as a trust radius.
        E = float(torch.linalg.norm(adam_full))
        dth = torch.zeros(m, dtype=theta.dtype)
        lenA = 0.0
        if nA > 0:
            lenA = nA if astep == "adam" else min(nA, max(tstarA, 0.0))
            dth = dth + (1.0 - rho) * lenA * pA
        clipped = 0
        if pL is not None and nL > 0:
            budget = rho * E
            take = min(nL, budget) if clip else nL
            clipped = int(nL > budget) if clip else 0
            dth = dth + take * pL

        # ---- drift gauge: how far the A step moved the predictions ----
        if alpha_mode in ("drift", "drift_gain"):
            stepA_full = (1.0 - rho) * lenA * pA if nA > 0 else None
            if stepA_full is not None:
                jd = f_jvp(theta, stepA_full)
                passes += 1
                a_now = float(torch.linalg.norm(jd)) / y_norm
            else:
                a_now = 0.0
            # bias-corrected EMA (Adam's own trick): without it the gauge
            # spends ~1/(1-beta) steps decaying from its initial value instead
            # of reading the measurement, which on QI init is the whole run.
            a_ema = alpha_ema * a_ema + (1.0 - alpha_ema) * a_now
            alpha = a_ema / (1.0 - alpha_ema ** t_adam)
            alpha = float(np.clip(alpha, ALPHA_MIN, ALPHA_MAX))

        th_new = theta + dth
        if not np.isfinite(float(torch.linalg.norm(th_new))):
            ma.zero_()
            va.zero_()
            continue
        if reset_moments and idx.numel():
            ma[idx] = 0.0
            va[idx] = 0.0
        theta = th_new

        # ---- re-certify on a cadence, gated by the drift gauge ----
        # iteration 11's rule: re-probe at a refresh boundary only when the
        # residual actually moved, and back off exponentially while membership
        # is stable. The dense probe costs 2(2m+1) forwards, so this is what
        # makes the certificate affordable at all.
        if lset == "certificate" and it % refresh == 0:
            with torch.no_grad():
                r_now = env["f_resid"](theta)
            drift_w = float(torch.linalg.norm(r_now - r_ref))
            r_ref = r_now
            windows += 1
            if drift_w > 10.0 * EPS_MACH * y_norm and windows >= backoff:
                new, _ = certify(env["probe_fn"], theta, env["f_pred"], vh,
                                 y_norm, members)
                n_probes += 1
                windows = 0
                changed = not torch.equal(new, members)
                backoff = 1 if changed else min(backoff * 2, 8)
                if changed:
                    members = new
                    idx = torch.nonzero(members, as_tuple=False).flatten()
                    pw = torch.randn(int(idx.numel()), dtype=theta.dtype) \
                        if idx.numel() else None

        if it % record == 0 or it == iters:
            with torch.no_grad():
                rr = env["f_resid"](theta)
            hist["it"].append(it)
            hist["loss"].append(float(rr.dot(rr)) / N_TRAIN)
            hist["rel"].append(env["eval_rel"](theta))
            hist["alpha"].append(float(a_use))
            hist["mu"].append(float(mu))
            hist["rho"].append(float(rho))
            hist["B"].append(int(idx.numel()))
            hist["dnorm"].append(float(nL))
            hist["clipped"].append(int(clipped))
            hist["gainA"].append(float(gainA))
            hist["gainL"].append(float(gainL))
            hist["lsqr_it"].append(int(lsqr_it))
            hist["sigma1"].append(float(sigma1))
            hist["vnorm"].append(float(torch.linalg.norm(theta[env["idx_readout"]])))
            hist["drift"].append(float(alpha))
            hist["lenA"].append(float(lenA))
            hist["n_drop"].append(int(n_drop))
            hist["passes"].append(int(passes))
        if verbose and it % max(1, iters // 10) == 0:
            print(f"    it {it:5d}  rel {hist['rel'][-1]:.3e}  "
                  f"alpha {alpha:.2e}  rho {rho:.3f}  B {int(idx.numel())}")

    return dict(hist=hist, theta=theta, best_rel=float(np.min(hist["rel"])),
                final_rel=hist["rel"][-1], passes=passes,
                floor_final=env["geometry_floor"](theta),
                n_probes=n_probes, B_final=int(idx.numel()))


# ==================================================================== io ====

def append(path, rec):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")


def load(path):
    p = Path(path)
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
