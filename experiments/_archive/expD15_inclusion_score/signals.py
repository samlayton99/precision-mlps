"""expD15 -- the candidate inclusion signals.

Every signal maps (env, theta) -> score vector of length m, HIGHER = more fit to
be in L. Each carries a cost tag and a `feasible` flag; anything whose cost
scales with the parameter count P is marked infeasible and exists only as a
measurement instrument, never as a shippable candidate.

COST LEGEND
  free      computable from state the optimizer already holds, or from the
            solve it already takes. No extra passes.
  probes    a FIXED number k of random JVP/VJP pairs, independent of P.
  ref       reference instrument: dense SVD or O(P) passes. NOT shippable.

WHAT THE FIRST DIAGNOSTIC ALREADY KILLED (recorded so it is not re-proposed)
  Signals built from raw Jacobian COLUMN magnitudes -- diag(J^T J), effective
  sample count, and the resolution diagonal itself -- are theoretically blind
  to a data gap when the activation saturates. tanh(g(x-c)) tends to +-1 away
  from c, so a column's norm is dominated by its saturated tail, not by the
  data near its own center: measured column norms varied only 7.47..9.04 across
  the whole domain INCLUDING a region with 96% of its data removed, and the
  exact resolution diagonal gave a middle/outer ratio of 0.97 (i.e. nothing).
  The localisation lives in the DIFFERENCE between neighbouring columns, not in
  any column: the same resolution measured in the first-difference basis gives
  0.69 vs 0.89. They are kept in the farm as measured negatives.
"""
from __future__ import annotations

import numpy as np

from core15 import RCOND

EPS = 1e-300


def _norm(s):
    """Rank-preserving squash to [0,1] so scores are comparable across signals."""
    s = np.asarray(s, float)
    good = np.isfinite(s)
    if not good.any():
        return np.zeros_like(s)
    lo, hi = np.min(s[good]), np.max(s[good])
    out = np.zeros_like(s)
    out[good] = (s[good] - lo) / (hi - lo) if hi > lo else 1.0
    return out


def _gram_probes(J, k=8, seed=0):
    """One JVP+VJP pair per probe gives unbiased samples of BOTH
         diag(J^T J)_i        from  z .* (J^T J z)
         sum_j (J^T J)_ij^2   from  ((J^T J) z)_i^2
    so the Gram diagonal and its row norms cost the same k probes, at any P."""
    rng = np.random.default_rng(seed)
    m = J.shape[1]
    dsum = np.zeros(m)
    rsum = np.zeros(m)
    for _ in range(k):
        z = rng.integers(0, 2, m).astype(float) * 2 - 1
        Az = J.T @ (J @ z)
        dsum += z * Az
        rsum += Az * Az
    return dsum / k, rsum / k


def _damped_solve(J, r, mu, rcond=RCOND):
    U, s, Vt = np.linalg.svd(J, full_matrices=False)
    inv = np.where(s > rcond * s[0], s / (s ** 2 + mu), 0.0)
    return Vt.T @ (inv * (U.T @ (-r)))


# =============================================================== signals =====

def make_signals(env, th, k_probes=8, alpha_lo=1e-10, alpha_hi=1e-2):
    """Compute every candidate at once (they share J and the probes)."""
    net = env["net"]
    J = net.jac(env["Xtr"], th)
    r = net.f(env["Xtr"], th) - env["ytr"]
    n, m = J.shape
    s1 = np.linalg.svd(J, compute_uv=False)[0]
    mu_lo, mu_hi = (alpha_lo * s1) ** 2, (alpha_hi * s1) ** 2
    dg, rn = _gram_probes(J, k_probes)
    dg = np.maximum(dg, 0.0)
    out = {}

    # ---- family S: readouts of the solve we already take -------------------
    d_lo = _damped_solve(J, r, mu_lo)
    d_hi = _damped_solve(J, r, mu_hi)
    # S1 small solved magnitude = well determined. A parameter the solver has
    # to give an absurd value to is one the data does not pin down. Measured:
    # on the data-gap case the blown-up weights sit on the gap centers at 23x
    # the outer mean.
    out["S_mag"] = dict(v=_norm(-np.log10(np.abs(d_lo) + 1e-30)),
                        cost="free", feasible=True,
                        why="small solved magnitude = determined (Tikhonov/L-curve)")
    # S2 the causal version, and scale-free: release the damping and see if the
    # answer moves. A determined parameter is insensitive to mu; an unresolved
    # one explodes. The docs' recurring lesson is that a causal test beats a
    # passive statistic, and this is the causal test for identifiability.
    sens = np.log10((np.abs(d_lo) + 1e-30) / (np.abs(d_hi) + 1e-30))
    out["S_sens"] = dict(v=_norm(-sens), cost="free", feasible=True,
                         why="stability of the solved value as mu -> 0")
    # S3 does solving it actually buy residual? predicted decrease per param.
    out["P_gain"] = dict(v=_norm(np.log10((J.T @ r) ** 2 / (dg + EPS) + 1e-30)),
                         cost="free", feasible=True,
                         why="exact-line-search decrease from this parameter alone")

    # ---- family R: resolution / identifiability ----------------------------
    out["R_diag"] = dict(v=_norm(dg / (dg + mu_lo)), cost="probes", feasible=True,
                         why="Gram-diagonal (Jacobi) resolution -- predicted BLIND")
    dd = dg ** 2 / (rn + EPS)
    out["R_domin"] = dict(v=_norm(dd), cost="probes", feasible=True,
                          why="diagonal dominance = conditional/marginal info")
    from core15 import resolution_exact
    out["R_exact"] = dict(v=_norm(resolution_exact(J, mu_lo)), cost="ref",
                          feasible=False, why="exact Backus-Gilbert resolution")

    # ---- family L: linearity ----------------------------------------------
    # Hutchinson nonlinearity: diag(Hess L) - (2/n) diag(J^T J) is exactly zero
    # for a parameter the model is affine in.
    nl = nonlinearity_diag(env, th, k_probes)
    out["L_hutch"] = dict(v=_norm(-np.log10(nl + 1e-30)), cost="probes",
                          feasible=True, why="Hessian minus Gauss-Newton diagonal")

    # ---- family K: blind kernels (Sam's nth-derivative-is-a-bump rule) -----
    from core15 import blind_parity
    par, cov, crowd, r_used = blind_parity(J)
    out["K_parity"] = dict(v=_norm(np.log10(par + 1e-30)), cost="O(nm) elementwise",
                           feasible=True,
                           why="local Nyquist: data points per competing unknown")
    out["K_crowd"] = dict(v=_norm(-np.log10(crowd + 1e-30)),
                          cost="O(nm) elementwise", feasible=True,
                          why="few competitors for the same territory")
    out["C_lin_x_par"] = dict(v=_norm(_norm(-np.log10(nl + 1e-30)) *
                                      _norm(np.log10(par + 1e-30))),
                              cost="probes+O(nm)", feasible=True,
                              why="linear AND locally resolved -- both halves")

    # ---- family L2: SELF-dependence of the Jacobian column ------------------
    # The cleanest characterisation of linearity, and the one every earlier
    # attempt missed: the model is affine in theta_i  <=>  J_i does not depend
    # on theta_i. Not "the second derivative is small" -- that quantity is
    # proportional to v_k and so vanishes for a good geometry (measured, twice).
    # The RELATIVE log-derivative
    #
    #     L_i = d log||J_i|| / d log theta_i     (isolated by Hutchinson)
    #
    # is exactly 0 for the readout (J_v = act(.) has no v in it) and O(1) for
    # geometry (J_b = v*act'(.), so the v cancels in the ratio). Scale-free by
    # construction, so it does NOT degenerate as ||v|| -> 0.
    #
    # T_stale measured TOTAL column motion under the solve step, which conflates
    # self-dependence with cross-dependence; the z_i correlation below isolates
    # the diagonal term, which is the one linearity is actually about.
    # Cost: 2k Jacobian evaluations, independent of P.
    scale_th = np.maximum(np.abs(th), 1e-3)
    acc = np.zeros(m)
    eps_s = 1e-5
    rngL = np.random.default_rng(11)
    Jn2 = np.maximum((J * J).sum(0), 1e-300)
    for _ in range(k_probes):
        zz = rngL.integers(0, 2, m).astype(float) * 2 - 1
        st = eps_s * scale_th * zz
        dJ = (net.jac(env["Xtr"], th + st) - net.jac(env["Xtr"], th - st)) / (2 * eps_s)
        acc += zz * ((dJ * J).sum(0) / Jn2)
    self_dep = np.abs(acc / k_probes)
    out["L_self"] = dict(v=_norm(-np.log10(self_dep + 1e-30)), cost="2k passes",
                         feasible=True,
                         why="d log||J_i|| / d log theta_i -- exactly 0 iff affine")

    # ---- family D: does the DATA determine it? (jackknife) -----------------
    # Perturb the DATA, not the model. Solve on one half of the rows and on the
    # other half; a parameter the data pins down gets the same answer twice, an
    # unresolved one does not. This is the classical stability-of-estimator
    # test, and it is Passage 2 section 3's independent-minibatch coherence
    # statistic C_l applied to solves instead of gradients. No linearity
    # assumption, so it cannot degenerate with ||v||; free in a batched
    # optimizer, which already sees independent row samples every step.
    rngD = np.random.default_rng(5)
    perm = rngD.permutation(n)
    h1, h2 = perm[:n // 2], perm[n // 2:]
    dA = _damped_solve(J[h1], r[h1], mu_lo)
    dB = _damped_solve(J[h2], r[h2], mu_lo)
    denom = np.abs(dA) + np.abs(dB) + np.median(np.abs(dA) + np.abs(dB)) + 1e-300
    disagree = np.abs(dA - dB) / denom
    out["D_jack"] = dict(v=_norm(-disagree), cost="2 half-solves", feasible=True,
                         why="two halves of the data agree on this parameter")
    out["C_self_x_jack"] = dict(
        v=np.minimum(_norm(-np.log10(self_dep + 1e-30)), _norm(-disagree)),
        cost="2k passes+2 solves", feasible=True,
        why="affine AND determined by the data -- both halves, as a conjunction")

    # ---- family U: the unification -- is the solved value REPRODUCIBLE? -----
    # Every criterion tried so far is the same statement in disguise:
    #   nonlinear in theta_i          -> re-solving after the step disagrees
    #   data does not resolve theta_i -> re-solving disagrees
    #   step breaks the linearisation -> re-solving disagrees
    # So ask it once, causally: solve, take the step, solve again, and see
    # which parameters the SECOND solve stops asking about.
    #
    #     U_i = |d2_i| / (|d1_i| + median|d1|)
    #
    # A parameter that was solved correctly is done: the residual is orthogonal
    # to its column and d2_i ~ 0. A parameter that was solved wrongly -- for any
    # of the three reasons -- gets asked for again.
    #
    # This is Passage 3's coherence statistic ("what fraction of the proposed
    # movement is coherent across independent samples") applied to the solve
    # rather than the gradient, and it is the docs' recurring lesson that a
    # causal test beats a passive statistic.
    #
    # Cost: one extra Jacobian and one extra solve per solve EVENT, both of
    # which the optimizer already performs on its normal cadence. Blind.
    # The probe step must be one we would actually TAKE. d_lo is the full
    # parameter set solved at near-zero damping -- a wild Gauss-Newton
    # overshoot on a nonlinear model, after which the second solve asks for
    # everything again and the test inverts (measured: 0% purity). Use the
    # damped step instead: small enough to trust, which is the whole premise
    # of asking whether it was reproducible.
    th1 = th + d_hi
    J1 = net.jac(env["Xtr"], th1)
    r1 = net.f(env["Xtr"], th1) - env["ytr"]
    d2 = _damped_solve(J1, r1, mu_hi)
    scl = np.median(np.abs(d_hi)) + 1e-300
    repro = np.abs(d2 - d_hi) / (np.abs(d_hi) + scl)
    out["U_repro"] = dict(v=_norm(-np.log10(repro + 1e-30)), cost="1 pass+1 solve",
                          feasible=True,
                          why="the second solve stops asking about it")
    # function-space variant: coherence of the PREDICTED CHANGE rather than of
    # the parameter values, which is reparameterisation-invariant (Passage 3 S6)
    u1 = J * d_hi[None, :]
    u2 = J1 * d2[None, :]
    n1 = np.linalg.norm(u1, axis=0)
    n2 = np.linalg.norm(u2, axis=0)
    out["U_func"] = dict(v=_norm(-np.log10(n2 / (n1 + np.median(n1) + 1e-300) + 1e-30)),
                         cost="1 pass+1 solve", feasible=True,
                         why="same test, measured in function space")

    # ---- family T: does the solve's own step invalidate my linearisation? ---
    # The trust-region question, asked per parameter and answered causally.
    # Take the step the solver actually proposes, recompute the Jacobian, and
    # measure how much each column MOVED, relative to itself:
    #
    #     T_i = || J_i(theta + d) - J_i(theta) ||  /  || J_i(theta) ||
    #
    # T_i = 0 means the linear model this parameter lives in is untouched by
    # the step, so solving it was exact. T_i large means the step the solver
    # took has already invalidated the model it used to choose that step.
    #
    # Why this succeeds where the linearity signals fail: it is a RATIO, so it
    # does not vanish as ||v|| -> 0. The per-coordinate second derivative is
    # proportional to v_k, so a good geometry (small readout, no blow-up) makes
    # every parameter look linear and the signal degenerates -- measured. The
    # relative column motion has no such scaling.
    #
    # Cost: ONE extra Jacobian evaluation per solve event. Blind: it assumes
    # nothing about layers or activations, only that we can recompute J.
    th_step = th + d_lo
    J2 = net.jac(env["Xtr"], th_step)
    col0 = np.linalg.norm(J, axis=0)
    stale = np.linalg.norm(J2 - J, axis=0) / np.maximum(col0, 1e-300)
    out["T_stale"] = dict(v=_norm(-np.log10(stale + 1e-30)), cost="1 pass",
                          feasible=True,
                          why="relative Jacobian-column motion under the proposed step")
    sc_stale = _norm(-np.log10(stale + 1e-30))
    sc_par = _norm(np.log10(par + 1e-30))
    out["C_stale_x_par"] = dict(v=_norm(sc_stale * sc_par),
                                cost="1 pass+O(nm)", feasible=True,
                                why="product of the two halves (wrong operator)")
    # The two halves are a CONJUNCTION, not a product: a parameter must be both
    # still-linear-after-the-step AND locally resolved, and failing either one
    # must veto it. A product lets a very high score on one half buy off a
    # failure on the other, which is exactly the observed failure -- the product
    # tracks staleness on `clustered`, where staleness is the half that is
    # wrong. min() is the conjunction.
    out["C_and"] = dict(v=np.minimum(sc_stale, sc_par), cost="1 pass+O(nm)",
                        feasible=True,
                        why="AND of the two halves: min, not product")
    # Absolute, theory-set thresholds instead of a top-k percentile: parity >= 1
    # is the local Nyquist condition (at least one data point per competing
    # unknown), and staleness below the median relative column motion is the
    # linearisation-still-valid condition. No knob chosen by looking at eval.
    admit = (par >= 1.0) & (stale <= np.median(stale))
    out["C_nyquist"] = dict(v=admit.astype(float) + 1e-6 * sc_stale,
                            cost="1 pass+O(nm)", feasible=True,
                            why="absolute rule: parity>=1 AND linearisation intact")

    # ---- family C: combinations -------------------------------------------
    out["C_lin_x_sens"] = dict(v=_norm(out["L_hutch"]["v"] * out["S_sens"]["v"]),
                               cost="free+probes", feasible=True,
                               why="linear AND stable under damping release")
    out["C_lin_x_mag"] = dict(v=_norm(out["L_hutch"]["v"] * out["S_mag"]["v"]),
                              cost="free+probes", feasible=True,
                              why="linear AND small solved magnitude")

    # ---- baselines ---------------------------------------------------------
    rng = np.random.default_rng(7)
    out["B_random"] = dict(v=rng.random(m), cost="free", feasible=True,
                           why="random ranking -- the floor any signal must beat")
    lab = net.labels()
    out["B_oracle_v"] = dict(v=(lab == "v").astype(float) + 1e-9 * rng.random(m),
                             cost="ref", feasible=False,
                             why="hard-coded readout -- the answer expD14 assumed")
    return out, J, r


def nonlinearity_diag(env, th, k=8, seed=1, rel_eps=1e-5):
    """diag of the NONLINEAR part of the Hessian, per parameter:

        N = Hess(L) - (2/n) J^T J = (2/n) sum_k r_k grad^2 f_k

    which is EXACTLY zero for any parameter the model is affine in.

    Two bugs are fixed here, and the second was the real one.

    (1) Absolute finite-difference steps are wrong: geometry sits at ~gamma
        (order 3-40), readout at ~0.03, so one eps is a 1e-6 relative nudge for
        one group and 1e-2 for the other. Steps are scaled per coordinate.

    (2) Estimating diag(Hess) and diag(J^T J) with INDEPENDENT probes and then
        subtracting is fatal. Both estimates carry Hutchinson variance
        proportional to their own off-diagonal mass, and that mass is largest
        precisely for the readout -- so the difference of two noisy estimates
        made the exactly-linear block look like the MOST nonlinear one
        (measured: readout 1.25e+01 vs geometry 6.3e-09, i.e. the signal was
        inverted). Using COMMON RANDOM NUMBERS -- the same z for both terms --
        cancels the J^T J contribution probe by probe, leaving only the term we
        actually want. Same cost, correct answer.
    """
    net, X, y = env["net"], env["Xtr"], env["ytr"]
    n = X.shape[0]
    rng = np.random.default_rng(seed)
    m = th.size
    scale = np.maximum(np.abs(th), 1e-3)
    acc = np.zeros(m)
    J = net.jac(X, th)

    def grad(t):
        Jt = net.jac(X, t)
        return (2.0 / n) * (Jt.T @ (net.f(X, t) - y))

    for _ in range(k):
        z = rng.integers(0, 2, m).astype(float) * 2 - 1
        step = rel_eps * scale * z
        hv = (grad(th + step) - grad(th - step)) / (2 * rel_eps)   # Hess @ (scale*z)
        gn = (2.0 / n) * (J.T @ (J @ (scale * z))) / 1.0           # GN @ (scale*z)
        acc += (z / np.maximum(scale, 1e-300)) * (hv - gn)
    return np.abs(acc / k)
