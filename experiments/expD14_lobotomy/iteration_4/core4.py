"""expD14 iteration 4 -- the held-out handoff, and the assembled optimizer.

iteration_3 found the right KIND of cooling signal (expressibility: never
move the geometry on error the solve can fix) with one measured defect:
computed on the training rows it cannot tell "features right" from
"features overfit the sample", which froze stock-random learning.

THE FIX: measure the same quantity on a held-out split of the training set.
10% of the (kept) train points are excluded from every gradient and solve;
at each step the exact (undamped, truncated) solve direction d* on the fit
rows is evaluated on the held-out rows:

    f_gen = || r_ho + J_ho d* || / || r_ho ||   in [0, 1]

and Adam's geometry step is scaled by it. On paper it reads: ~0.4 on stock
random (features overfit, geometry must learn -> free), ~1e-14 on qi
(solve generalizes -> frozen), small on datagap (the hold-out comes from
the same gappy distribution -> frozen, correctly), large on clustered
(middle points inexpressible -> free, correctly).

THE ASSEMBLED OPTIMIZER (everything a free measurement):
    what to solve   Method C per-tensor discovery      (expD15)
    how hard        alpha = r_entry, the damping floor  (iteration_2)
    the handoff     geometry step *= f_gen              (this iteration)
No hand throttles, no schedules, no hyperparameters beyond Adam's own.

The falsification, stated before running: f_gen must hold qi at machine
epsilon (iteration_3's perp result) AND recover the uncooled arm's
7.5e-7 geometry on rand/sine_8pi. Failing either kills it.
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
d08run = core1.d08run

RESULTS = REPO / "results" / "checkpoint_D_optimizers" / "expD14_lobotomy" / "iteration_4"
FIGS = RESULTS / "figures"

coherent_travel = core1.coherent_travel
snr_corrected = core1.snr_corrected
discover_pertensor = core1.discover_pertensor
append, load = core1.append, core1.load

N_TRAIN, N_EVAL = 2003, 4001
LAM = 0.25
RCOND = 1e-15
GAP_RADIUS, GAP_KEEP = 0.4, 0.04
COL_FLOOR = 1e-8
ALPHA_MIN, ALPHA_MAX = 1e-15, 1.0
HOLDOUT_EVERY = 10          # every 10th kept point is held out of fit
S_REF = 1e-2                # the gate: learn at full speed above this


def build_case(fn_name, N, case="qi", seed=0):
    """core1's five cases, with a fit/holdout split of the (kept) train
    points. Gradients, solves and r_entry use the fit rows only; the
    holdout rows exist solely for the f_gen signal. For datagap the split
    is taken AFTER the gap masking, so the holdout is drawn from the same
    gappy distribution -- the load-bearing subtlety."""
    base = core1.build_case(fn_name, N, case=case, seed=seed)
    W, m = base["W"], base["m"]
    x_all, y_all = base["x_tr"], base["y_tr"]
    ho = np.zeros(x_all.size, bool)
    ho[HOLDOUT_EVERY // 2::HOLDOUT_EVERY] = True
    x_fit, y_fit = x_all[~ho], y_all[~ho]
    x_ho, y_ho = x_all[ho], y_all[ho]

    xt = torch.tensor(x_fit).unsqueeze(1)
    yt = torch.tensor(y_fit)
    xh = torch.tensor(x_ho).unsqueeze(1)
    yh = torch.tensor(y_ho)

    def model(x, th):
        return torch.tanh(x * th[:W] + th[W:2 * W]) @ th[2 * W:3 * W] + th[-1]

    def f_pred(th):
        return model(xt, th)

    def f_resid(th):
        with torch.no_grad():
            return (f_pred(th) - yt).detach()

    def f_resid_ho(th):
        with torch.no_grad():
            return (model(xh, th) - yh).detach()

    def f_jvp(th, dvec):
        _, jd = torch.func.jvp(f_pred, (th,), (dvec,))
        return jd.detach()

    def f_vjp(th, seed_v):
        thr = th.detach().requires_grad_(True)
        (g,) = torch.autograd.grad(f_pred(thr), thr, grad_outputs=seed_v)
        return g.detach()

    def f_residgrad(th):
        thr = th.detach().requires_grad_(True)
        pred = model(xt, thr)
        r = (pred - yt).detach()
        (g,) = torch.autograd.grad(pred, thr,
                                   grad_outputs=(2.0 / x_fit.size) * r)
        return r, g.detach()

    def _jac(x, th):
        with torch.no_grad():
            z = x * th[:W] + th[W:2 * W]
            t = torch.tanh(z)
            sech2 = 1.0 - t * t
            v = th[2 * W:3 * W]
            return torch.cat([sech2 * v * x, sech2 * v, t,
                              torch.ones(x.shape[0], 1, dtype=th.dtype)], 1)

    def f_jac_full(th):
        return _jac(xt, th)

    def f_jac_cols(th, cols):
        return _jac(xt, th)[:, cols]

    def f_jac_cols_ho(th, cols):
        return _jac(xh, th)[:, cols]

    # floor: fit on the FIT rows (what the optimizer can see), eval as core1.
    # VALIDATED: the damping of the terminal solve is chosen on the held-out
    # rows. The raw truncated min-norm refit is worse than the in-run state
    # in 26 of 132 t8 cells (it overfits the fit rows on datagap, and a
    # single solve is cancellation-limited at ~1e-14 where in-run iterative
    # refinement reaches 1e-16), so the finisher must validate its damping.
    ALPHAS_VAL = (0.0, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2)

    def _floor_split(th):
        t = th.detach().numpy()
        z = np.tanh(x_fit[:, None] * t[:W][None, :] + t[W:2 * W][None, :])
        A = np.hstack([z, np.ones((x_fit.size, 1))])
        U, s, Vt = np.linalg.svd(A, full_matrices=False)
        keep = s > RCOND * s[0]
        utr = U.T @ y_fit
        zh = np.tanh(x_ho[:, None] * t[:W][None, :] + t[W:2 * W][None, :])
        A_ho = np.hstack([zh, np.ones((x_ho.size, 1))])
        best_sol, best_ho = None, np.inf
        for a_ in ALPHAS_VAL:
            if a_ == 0.0:
                inv = np.where(keep, 1.0 / np.where(keep, s, 1.0), 0.0)
            else:
                mu_ = (a_ * s[0]) ** 2
                inv = s / (s ** 2 + mu_)
            sol = Vt.T @ (inv * utr)
            ho_err = float(np.linalg.norm(A_ho @ sol - y_ho))
            if ho_err < best_ho:
                best_ho, best_sol = ho_err, sol
        x_ev = np.linspace(-1.0, 1.0, N_EVAL)
        y_ev = d08run.make_fn(d08run.TARGETS[fn_name])(x_ev)
        ze = np.tanh(x_ev[:, None] * t[:W][None, :] + t[W:2 * W][None, :])
        err = np.hstack([ze, np.ones((x_ev.size, 1))]) @ best_sol - y_ev
        ev_in = np.abs(x_ev) < GAP_RADIUS
        fl = float(np.linalg.norm(err) / np.linalg.norm(y_ev))
        fi = float(np.linalg.norm(err[ev_in]) / np.linalg.norm(y_ev[ev_in]))
        fo = float(np.linalg.norm(err[~ev_in]) / np.linalg.norm(y_ev[~ev_in]))
        return fl, fi, fo

    fl0 = _floor_split(base["theta0"])
    env = dict(base)
    env.update(theta0=base["theta0"], n_train=int(x_fit.size),
               floor0=fl0[0], floor0_in=fl0[1], floor0_out=fl0[2],
               geometry_floor=lambda th: _floor_split(th)[0],
               geometry_floor_split=_floor_split,
               f_pred=f_pred, f_resid=f_resid, f_resid_ho=f_resid_ho,
               f_jvp=f_jvp, f_vjp=f_vjp, f_residgrad=f_residgrad,
               f_jac_full=f_jac_full, f_jac_cols=f_jac_cols,
               f_jac_cols_ho=f_jac_cols_ho,
               y_norm=float(torch.linalg.norm(yt)),
               y_norm_ho=float(torch.linalg.norm(yh)))
    return env


def train_v4(env, iters, *, cool="fgen", lr=1e-3, betas=(0.9, 0.999),
             eps_adam=1e-8, refresh=200, record=5, floor_every=50,
             travel_window=100, snap_every=0, seed=0, verbose=False):
    """The assembled optimizer. cool in {"fgen", "none"}; "none" is the
    uncooled iteration_2 arm rerun under the fit/holdout harness so the
    comparison is apples-to-apples. snap_every > 0 records theta snapshots
    (for the parameter-movement gifs)."""
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
                            "fgen", "shat", "coolf", "B", "vnorm", "passes")}
    fhist = {k: [] for k in ("it", "floor", "floor_in", "floor_out")}
    thist = {k: [] for k in ("it", "travel")}
    snaps = {}

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
    if snap_every:
        snaps["0"] = theta.numpy().copy().tolist()
    s_prev = 1.0
    mode_latch = None
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

        # ---- L: the damped solve, plus d* for the signal ----
        # The SVD does not depend on mu, so f_gen is computed FIRST; the
        # fgen2 rule then composes the damping as alpha = r_entry * f_gen:
        # solve down to the level of error that actually generalizes, not
        # the raw residual. (fgen alone deadlocks on rand/sine: the signal
        # says "features fine, freeze geometry" while alpha = r_entry says
        # "error high, do not solve" -- and nothing ever moves.)
        dmu = None
        fgen = 1.0
        if idx.numel():
            av, atu = masked_ops(theta)
            if it == 1 or (it - 1) % refresh == 0:
                sigma1, pw = core1.power_sigma1(av, atu, pw, iters=3)
                passes += 6
            J = env["f_jac_cols"](theta, idx)
            cn = torch.linalg.norm(J, dim=0)
            cmax = float(cn.max()) if cn.numel() else 0.0
            live = cn > COL_FLOOR * cmax if cmax > 0 else cn > 0
            dmu = torch.zeros(int(idx.numel()), dtype=theta.dtype)
            if bool(live.any()):
                Jl = J[:, live]
                U, s, Vt = torch.linalg.svd(Jl, full_matrices=False)
                keep = s > RCOND * s[0]
                utr = U.T @ (-r)
                if cool in ("fgen", "fgen2", "fabs") or \
                        (cool in ("fabsc", "gate", "latch") and
                         (it == 1 or (it - 1) % refresh == 0)):
                    # exact (undamped, truncated) direction on the fit rows,
                    # evaluated on the held-out rows
                    inv_e = torch.where(keep, 1.0 / torch.where(keep, s,
                                        torch.ones_like(s)),
                                        torch.zeros_like(s))
                    dstar = Vt.T @ (inv_e * utr)
                    r_ho = env["f_resid_ho"](theta)
                    J_ho = env["f_jac_cols_ho"](theta, idx)[:, live]
                    passes += 1
                    r_ho_new = r_ho + J_ho @ dstar
                    nho = float(torch.linalg.norm(r_ho))
                    if cool in ("fabs", "fabsc", "gate", "latch"):
                        # the ONE signal: the absolute held-out error the
                        # exact solve would leave. Noise reads ~1e-14 here,
                        # where the relative version reads 1.0 and releases
                        # Adam onto a finished geometry.
                        fgen = float(np.clip(
                            float(torch.linalg.norm(r_ho_new))
                            / env["y_norm_ho"], 0.0, 1.0))
                        s_prev = fgen
                    else:
                        fgen = float(np.clip(
                            float(torch.linalg.norm(r_ho_new))
                            / max(nho, 1e-300), 0.0, 1.0))
                if cool == "fgen2":
                    alpha = float(np.clip(r_entry * fgen,
                                          ALPHA_MIN, ALPHA_MAX))
                elif cool == "fabs":
                    alpha = float(np.clip(fgen, ALPHA_MIN, ALPHA_MAX))
                elif cool == "fabsc":
                    # cadence form: the held signal drives both dials
                    fgen = s_prev
                    alpha = float(np.clip(s_prev, ALPHA_MIN, ALPHA_MAX))
                elif cool == "gate":
                    # the gate: full-speed learning while the held-out
                    # inexpressible error exceeds S_REF; freeze below it.
                    # Cooling PROPORTIONAL to s decelerates learning in
                    # proportion to remaining error (harmonic, not
                    # geometric, convergence) -- measured killing rand.
                    fgen = s_prev
                    alpha = float(np.clip(s_prev, ALPHA_MIN, ALPHA_MAX))
                elif cool == "latch":
                    # THE MODE BIT, measured once at init: s0 separates the
                    # regimes by five orders in every case measured.
                    #   learn mode (s0 >= S_REF): alpha = r_entry, no
                    #     cooling -- t8's `none` arm, the measured winner on
                    #     stock-random. A deep (alpha = s) solve re-stuns
                    #     learning even uncooled (latch probe: 1.5e-3 vs
                    #     6.7e-7), so learn mode keeps the solve soft too.
                    #   preserve mode (s0 < S_REF): the gate.
                    if mode_latch is None:
                        mode_latch = "preserve" if s_prev < S_REF else "learn"
                    fgen = s_prev
                    if mode_latch == "learn":
                        alpha = float(np.clip(r_entry, ALPHA_MIN, ALPHA_MAX))
                    else:
                        alpha = float(np.clip(s_prev, ALPHA_MIN, ALPHA_MAX))
                elif cool == "fabsd":
                    # the DEPLOYABLE form: alpha from the PREVIOUS step's
                    # signal (one-step lag), and the signal itself from the
                    # damped direction the O(d) solver actually produces --
                    # no exact solve, no SVD required in deployment.
                    alpha = float(np.clip(s_prev, ALPHA_MIN, ALPHA_MAX))
                mu = (alpha * sigma1) ** 2
                inv_d = torch.where(keep, s / (s ** 2 + mu),
                                    torch.zeros_like(s))
                dmu[live] = Vt.T @ (inv_d * utr)
                if cool == "fabsd":
                    r_ho = env["f_resid_ho"](theta)
                    J_ho = env["f_jac_cols_ho"](theta, idx)[:, live]
                    passes += 1
                    r_ho_new = r_ho + J_ho @ dmu[live]
                    fgen = float(np.clip(
                        float(torch.linalg.norm(r_ho_new))
                        / env["y_norm_ho"], 0.0, 1.0))
                    s_prev = fgen
            passes += int(idx.numel())

        # ---- A: Adam cooled by f_gen ----
        stepA = adam_full.clone()
        if idx.numel():
            stepA[idx] = 0.0
        if cool == "gate":
            coolf = float(min(1.0, fgen / S_REF))
        elif cool == "latch":
            coolf = 1.0 if mode_latch == "learn" \
                else float(min(1.0, fgen / S_REF))
        elif cool in ("fgen", "fgen2", "fabs", "fabsc", "fabsd"):
            coolf = fgen
        else:
            coolf = 1.0
        dth = coolf * stepA
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

        if snap_every and it % snap_every == 0:
            snaps[str(it)] = theta.numpy().copy().tolist()
        if it % floor_every == 0 or it == iters:
            log_floor(it)
        if it % record == 0 or it == iters:
            rel, rin, rout = env["eval_rel_split"](theta)
            hist["it"].append(it)
            hist["rel"].append(rel)
            hist["rel_in"].append(rin)
            hist["rel_out"].append(rout)
            hist["alpha"].append(float(alpha))
            hist["fgen"].append(float(fgen))
            hist["shat"].append(float(shat))
            hist["coolf"].append(float(coolf))
            hist["B"].append(int(idx.numel()))
            hist["vnorm"].append(float(torch.linalg.norm(theta[env["idx_readout"]])))
            hist["passes"].append(int(passes))
        if verbose and it % max(1, iters // 10) == 0:
            print(f"    it {it:5d}  rel {hist['rel'][-1]:.3e}  "
                  f"fgen {fgen:.2e}  floor {fhist['floor'][-1]:.3e}")

    fl, fi, fo = env["geometry_floor_split"](theta)
    return dict(hist=hist, fhist=fhist, thist=thist, snaps=snaps, theta=theta,
                best_rel=float(np.min(hist["rel"])), final_rel=hist["rel"][-1],
                floor_final=fl, floor_final_in=fi, floor_final_out=fo,
                passes=passes, n_member_changes=n_member_changes)


ARMS = {
    "fgen":  dict(cool="fgen"),
    "fgen2": dict(cool="fgen2"),
    "fabs":  dict(cool="fabs"),
    "fabsc": dict(cool="fabsc"),
    "fabsd": dict(cool="fabsd"),
    "gate":  dict(cool="gate"),
    "latch": dict(cool="latch"),
    "none":  dict(cool="none"),
}
