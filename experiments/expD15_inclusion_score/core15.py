"""expD15 -- the INCLUSION SCORE: which parameters belong in the least-squares set L?

THE QUESTION (Sam, 2026-08-02)
  An optimizer that injects a least-squares solve must decide, per parameter,
  whether that parameter belongs in the solved set L. "Linearity" was the first
  guess and it is provably not sufficient: a readout weight sitting over a hole
  in the data is exactly linear and must NOT be solved, because the data does
  not determine it. So the score has (at least) two halves:

      fitness_i  =  is a solve EXACT here      (linearity)
               AND does the data DETERMINE it  (resolution)

THE THEORY FOR THE SECOND HALF
  The classical object is the model resolution matrix (Backus-Gilbert 1968),

      R(mu) = (J^T J + mu I)^{-1} (J^T J),      gamma_i = R_ii in [0,1],

  read as "what fraction of parameter i is set by the data rather than by the
  regularizer". gamma_i = 1 is fully resolved; gamma_i = 0 means the parameter
  lives in the damped null space. Its trace is MacKay's "number of well-
  determined parameters" from the evidence framework. In seismic tomography --
  whose forward operator is the Radon transform, i.e. LITERALLY our 2-D ridge
  geometry -- this is the standard tool for deciding which cells to invert for,
  and low resolution is exactly the ray-coverage-vs-cell-size mismatch that Sam
  described as data density vs center density.

  The unification that matters: R(mu) is the resolution OF THE DAMPED SOLVE WE
  ALREADY TAKE. The inclusion score is not a separate mechanism -- it is a
  readout of the solver, with mu as its knob. That gives a testable hierarchy,
  cheapest first:

      gamma_i ~ d_i/(d_i + mu)          Gram diagonal only          (free)
      gamma_i ~ corrected by off-diagonal mass                      (cheap)
      gamma_i  = exact diag of R(mu)                                (reference)

FEASIBILITY GATE (Sam, binding)
  A signal is inadmissible if its cost scales with the parameter count P. The
  dead predecessor (expD14's curvature probe) cost 4P forward passes. Every
  shipped candidate here costs either ZERO (state the optimizer already keeps)
  or a FIXED number of random probes, independent of P; memory is O(P).

  The enabling identity: for Rademacher z, one JVP+VJP pair gives unbiased
  samples of BOTH
      diag(J^T J)_i          from  z .* J^T(J z)
      sum_j (J^T J)_ij^2     from  ((J^T J) z)_i^2
  so the Gram diagonal and its row norms -- hence the collinearity correction --
  come from the same probes. k=8 probes is ~32 passes at ANY width.

  Reference quantities (exact resolution, exact linearity) are computed here by
  dense SVD. They are MEASUREMENT INSTRUMENTS, never candidates, and are
  labelled `feasible=False` so they cannot be reported as shippable.

THE MODEL, dimension-agnostic
      f(x) = sum_k v_k tanh(a_k . x + b_k) + c0,     a_k in R^dim
      theta = (A | b | v | c0),  m = W*(dim+2) + 1
  dim=1 is the 1-D problem; dim=2 is the ridge/Radon problem. One code path.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

RESULTS = REPO / "results" / "checkpoint_D_optimizers" / "expD15_inclusion_score"
FIGS = RESULTS / "figures"
RCOND = 1e-15


# ======================================================== the model ==========

class Net:
    """theta = (A.ravel() | b | v | c0). Analytic forward and Jacobian."""

    def __init__(self, A, b, v, c0=0.0):
        self.W, self.dim = A.shape
        self.theta = np.concatenate([A.ravel(), b, v, [c0]])

    @property
    def m(self):
        return self.W * (self.dim + 2) + 1

    def unpack(self, th=None):
        th = self.theta if th is None else th
        W, d = self.W, self.dim
        return (th[:W * d].reshape(W, d), th[W * d:W * d + W],
                th[W * d + W:W * d + 2 * W], th[-1])

    def f(self, X, th=None):
        A, b, v, c0 = self.unpack(th)
        return np.tanh(X @ A.T + b) @ v + c0

    def jac(self, X, th=None):
        """Full Jacobian d f / d theta, shape (n, m). Analytic, one pass.

        columns are ordered (A_11..A_Wd | b_1..b_W | v_1..v_W | c0), matching
        `theta`, so a selected index set can mix geometry and readout freely --
        which it must, since the method is allowed to put first-layer weights
        in L (Sam: "if the method selects b and w in the first layer, you would
        have to do the Jacobian matrix that also includes these")."""
        A, b, v, c0 = self.unpack(th)
        n, W, d = X.shape[0], self.W, self.dim
        t = np.tanh(X @ A.T + b)                    # (n, W)
        s = (1.0 - t * t) * v                       # (n, W)  = v_k sech^2
        JA = (s[:, :, None] * X[:, None, :]).reshape(n, W * d)
        return np.hstack([JA, s, t, np.ones((n, 1))])

    def labels(self):
        """Per-parameter group tag, for scoring and plots."""
        W, d = self.W, self.dim
        return np.array(["A"] * (W * d) + ["b"] * W + ["v"] * W + ["c0"])

    def unit_of(self):
        """Which hidden unit each parameter belongs to (-1 for c0)."""
        W, d = self.W, self.dim
        return np.concatenate([np.repeat(np.arange(W), d), np.arange(W),
                               np.arange(W), [-1]])


# ======================================================== the four cases =====

def _target_1d(X):
    return np.sin(4.0 * X[:, 0])


def _target_2d(X):
    r2 = (X ** 2).sum(1)
    return np.exp(-3.0 * r2) + 0.3 * np.sin(3.0 * X[:, 0]) * np.cos(3.0 * X[:, 1])


def build_case(case, dim=1, N=24, lam=0.30, seed=0, gap_radius=0.4,
               gap_keep=0.04, row_mult=6):
    """The four cases, in either dimension.

      qi         correct uniform geometry, data everywhere      -> all of v in L
      clustered  centers bunched at the edges, sparse middle;    (geometry defect)
                 data everywhere
      datagap    correct uniform geometry, data almost absent    (data defect)
                 from the middle
      random     random init, data everywhere

    `clustered` and `datagap` are the discriminating pair: in one the CENTERS
    are wrong, in the other the centers are right and the DATA is wrong. Any
    score that only measures linearity must give them identical answers.
    """
    rng = np.random.default_rng(1000 + seed)

    if dim == 1:
        # Centers must extend BEYOND the domain by the standard halo, or the
        # "qi" case is not actually a good QI geometry and the whole ladder is
        # anchored to a weak reference (CLAUDE.md warns about exactly this).
        from src.construction.qi_mpmath import default_halo
        edge = 1.0
        h = 2.0 * edge / N
        gamma = lam / h
        halo = int(default_halo(N, lambda_star=lam))
        cen = (-edge + h * np.arange(-halo, N + halo + 1))[:, None]
        n_units_1d = cen.shape[0]
        Xtr = np.linspace(-edge, edge, row_mult * n_units_1d)[:, None]
        Xev = np.linspace(-edge, edge, 2001)[:, None]
        target = _target_1d
    else:
        # 2-D units are RIDGES, not bumps: tanh(gamma*(d_k . x - o_k)) is a line
        # at signed distance o_k along unit normal d_k. Using a tensor center
        # grid here (an earlier bug) gives every unit the SAME direction, i.e.
        # a stack of parallel lines -- degenerate. Use the Radon (direction,
        # offset) tensor grid, which is expE01's geometry and the one the
        # tomography resolution theory is written for.
        edge = 1.0
        n_dir = max(2, int(round(np.sqrt(N / 2))))
        n_off = max(2, int(round(N / n_dir)))
        ang = np.pi * np.arange(n_dir) / n_dir
        dirs = np.stack([np.cos(ang), np.sin(ang)], 1)
        offs = np.linspace(-edge, edge, n_off + 2)[1:-1]
        h = offs[1] - offs[0] if n_off > 1 else 2.0 * edge
        gamma = lam / h
        D = np.repeat(dirs, n_off, axis=0)
        O = np.tile(offs, n_dir)
        cen = D * O[:, None]                       # foot of each ridge line
        rng2 = np.random.default_rng(7)
        ns = int(np.sqrt(row_mult * n_dir * n_off * 4 / np.pi)) + 6
        gx, gy = np.meshgrid(np.linspace(-edge, edge, ns),
                             np.linspace(-edge, edge, ns))
        Xtr = np.stack([gx.ravel(), gy.ravel()], 1)
        Xtr = Xtr[np.linalg.norm(Xtr, axis=1) <= edge]
        gx, gy = np.meshgrid(np.linspace(-edge, edge, 55),
                             np.linspace(-edge, edge, 55))
        Xev = np.stack([gx.ravel(), gy.ravel()], 1)
        Xev = Xev[np.linalg.norm(Xev, axis=1) <= edge]
        target = _target_2d
        ridge_dirs, ridge_offs = D, O

    if case == "clustered":
        # push units away from the middle (1-D: centers; 2-D: ridge offsets),
        # r -> r^0.45, keeping the count but emptying the centre. The GEOMETRY
        # is wrong here; the data is untouched.
        if dim == 1:
            r = np.abs(cen)
            cen = np.sign(cen) * edge * (np.maximum(r, 1e-12) / edge) ** 0.45
        else:
            sgn = np.sign(ridge_offs) + (ridge_offs == 0)
            ridge_offs = sgn * edge * (np.abs(ridge_offs) / edge) ** 0.45
            cen = ridge_dirs * ridge_offs[:, None]

    W0 = cen.shape[0]
    if case == "random":
        A = rng.uniform(-1, 1, (W0, dim))
        b = rng.uniform(-1, 1, W0)
    elif dim == 1:
        A = np.full((W0, 1), gamma)
        b = -(A * cen).sum(1)
    else:
        A = gamma * ridge_dirs
        b = -gamma * ridge_offs

    if case == "datagap":
        # keep only a small fraction of the samples inside the middle
        rtr = np.linalg.norm(Xtr, axis=1)
        inside = rtr < gap_radius
        keep = ~inside
        idx_in = np.nonzero(inside)[0]
        n_keep = max(1, int(round(gap_keep * idx_in.size)))
        keep[rng.choice(idx_in, n_keep, replace=False)] = True
        Xtr = Xtr[keep]

    W = cen.shape[0]
    # Start the readout SMALL-RANDOM, not zero. Zero is degenerate (every
    # geometry Jacobian column is proportional to v_k, so all of them vanish --
    # the trap expD14's probe fell into), and a fully-solved readout is also
    # degenerate for the opposite reason: the residual is already orthogonal to
    # the readout span, so a second solve does nothing and every signal scores
    # identically. Small-random is the realistic mid-training state and leaves
    # the solve real work to do.
    # A PARTIALLY TRAINED readout: half of the exact solve plus noise. Neither
    # extreme is usable. v=0 (and v tiny) is degenerate because every geometry
    # second derivative is proportional to v_k, so the nonlinearity that the
    # linearity signals must measure is itself ~0 and drowns in differencing
    # noise -- measured, and it is the same trap expD14's probe fell into. A
    # fully-solved readout is degenerate the other way: the residual is already
    # orthogonal to the readout span, so the acid test has nothing to solve and
    # every signal ties. Half-solved is the realistic mid-training state and is
    # degenerate in neither direction.
    net = Net(A, b, np.zeros(W), 0.0)
    _J = net.jac(Xtr)[:, W * dim + W: W * dim + 2 * W + 1]
    _r = net.f(Xtr) - target(Xtr)
    _U, _s, _Vt = np.linalg.svd(_J, full_matrices=False)
    # DAMPED, not truncated: on the clustered geometry the exact min-norm
    # readout is enormous, and half of it is a start state with error ~1e7,
    # which makes that whole case meaningless. Damping at 1e-6*sigma_1 keeps
    # every case's start in a comparable regime.
    _mu = (1e-6 * _s[0]) ** 2
    _inv = _s / (_s ** 2 + _mu)
    _d = _Vt.T @ (_inv * (_U.T @ (-_r)))
    v0 = 0.5 * _d[:W] + 0.05 * np.abs(_d[:W]).mean() * rng.standard_normal(W)
    c00 = 0.5 * _d[W]
    # RESCALE so the start explains a fixed fraction of the target in EVERY
    # case and dimension. Without this the QI cells start with a readout of
    # order 1e-9 (the damped solve on a well-conditioned geometry is tiny), the
    # geometry Jacobian columns -- which carry a factor v_k -- are then ~1e-9
    # against the readout's ~30, and the geometry is NUMERICALLY DEAD. Every
    # signal then "separates" the readout trivially, and the 1-D purities that
    # produced (94-99%) measure the start state rather than the signal. In 2-D,
    # where the columns happened to be comparable, the same signals scored
    # 47-78% -- the honest number.
    _probe = Net(A, b, v0, c00)
    _pred = _probe.f(Xtr)
    _ny = np.linalg.norm(target(Xtr))
    _np_ = np.linalg.norm(_pred)
    if _np_ > 0:
        _k = 0.5 * _ny / _np_
        v0, c00 = v0 * _k, c00 * _k
    net = Net(A, b, v0, c00)
    ytr, yev = target(Xtr), target(Xev)
    return dict(case=case, dim=dim, N=N, net=net, cen=cen, gamma=gamma,
                Xtr=Xtr, ytr=ytr, Xev=Xev, yev=yev,
                ny=float(np.linalg.norm(yev)), edge=edge,
                gap_radius=gap_radius, label=f"{case}-{dim}d-N{N}")


def warm_readout(env, th=None):
    """Put the network in a realistic mid-training state: readout solved once on
    the current geometry. Scores measured at an all-zero readout are degenerate
    (every geometry Jacobian column is proportional to v_k and therefore zero),
    which is the trap expD14's probe fell into."""
    net = env["net"]
    th = net.theta.copy() if th is None else th.copy()
    J = net.jac(env["Xtr"], th)
    W, d = net.W, net.dim
    sl = slice(W * d + W, W * d + 2 * W + 1)
    r = net.f(env["Xtr"], th) - env["ytr"]
    U, s, Vt = np.linalg.svd(J[:, sl], full_matrices=False)
    inv = np.where(s > RCOND * s[0], 1.0 / np.where(s > RCOND * s[0], s, 1), 0)
    th[sl] += Vt.T @ (inv * (U.T @ (-r)))
    return th


# ================================================= exact reference quantities =

def resolution_exact(J, mu):
    """diag of R(mu) = (J^T J + mu I)^{-1} (J^T J), via SVD. REFERENCE ONLY."""
    _, s, Vt = np.linalg.svd(J, full_matrices=False)
    return ((Vt ** 2) * (s ** 2 / (s ** 2 + mu))[:, None]).sum(0)


def nonlinearity_exact(env, th, eps=1e-4):
    """||d^2 f / d theta_i^2|| per parameter, by central differences on the
    Jacobian column (exact for this model up to O(eps^2)). REFERENCE ONLY --
    this is the O(P)-forward-passes object the whole exercise exists to avoid."""
    net, X = env["net"], env["Xtr"]
    out = np.zeros(th.size)
    J0 = net.jac(X, th)
    for i in range(th.size):
        tp = th.copy(); tp[i] += eps
        tm = th.copy(); tm[i] -= eps
        out[i] = np.linalg.norm((net.jac(X, tp)[:, i] - net.jac(X, tm)[:, i])
                                / (2 * eps))
    return out


# ============================================== the acid test: solve a subset =

def solve_subset(env, th, idx, mu=0.0, rcond=RCOND):
    """Damped least-squares solve on the SELECTED parameters only, using the
    true Jacobian columns for whatever was selected (geometry included), then
    apply and evaluate. This is the operational test of an inclusion score:
    a score is good if the subset it picks, when actually solved, pays."""
    net = env["net"]
    if len(idx) == 0:
        return eval_rel(env, th), th
    J = net.jac(env["Xtr"], th)[:, idx]
    r = net.f(env["Xtr"], th) - env["ytr"]
    U, s, Vt = np.linalg.svd(J, full_matrices=False)
    keep = s > rcond * s[0]
    inv = np.where(keep, s / (s ** 2 + mu), 0.0)
    d = Vt.T @ (inv * (U.T @ (-r)))
    th2 = th.copy()
    th2[idx] += d
    return eval_rel(env, th2), th2


def eval_rel(env, th):
    return float(np.linalg.norm(env["net"].f(env["Xev"], th) - env["yev"])
                 / env["ny"])


def train_rel(env, th):
    return float(np.linalg.norm(env["net"].f(env["Xtr"], th) - env["ytr"])
                 / max(np.linalg.norm(env["ytr"]), 1e-300))


# ==================================================================== io =====

def append(path, rec):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")


def load(path):
    p = Path(path)
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


# ============================== data / center parity, and the region split ===
# Sam: "be careful to separate L2 between sections with good data-center parity
# and matching, and also sections with bad parity" -- a single global eval L2
# averages the two regimes together and hides the entire effect.
#
# Parity is defined WITHOUT reference to the activation. The only structural
# fact used is the one Sam allows: the derivative is eventually a KERNEL, so
# each hidden unit has a localised response whose shape we can read off the
# Jacobian rather than assume. Unit k's presence at a point x is taken to be
# |d f / d b_k|(x) normalised to unit mass -- a measured kernel, not tanh.

def unit_kernels(env, th, X, ref=None):
    """K[i,k] = response of unit k at point X[i], each unit's kernel normalised
    to unit mass over a FIXED reference grid. Measured from the Jacobian's own
    bias columns -- no activation is assumed, only Sam's allowance that the
    derivative is eventually a kernel."""
    net = env["net"]
    W, d = net.W, net.dim
    ref = env["Xev"] if ref is None else ref
    mass = np.abs(net.jac(ref, th)[:, W * d:W * d + W]).sum(0)
    K = np.abs(net.jac(X, th)[:, W * d:W * d + W])
    return K / np.maximum(mass, 1e-300)


def parity_field(env, th, X):
    """Local samples-per-unit -- the Nyquist number of the geometry/data pair.

      data_mass_k   = how many TRAIN points fall under unit k's kernel
      parity(x)     = kernel-weighted average of data_mass over units at x

    parity > 1 means the data can resolve the units sitting there; < 1 means it
    cannot. (Earlier version normalised each kernel over the same set it then
    measured mass on, which forced data_mass_k = const and made the field
    exactly constant -- a plain bug, caught by the field reading 6.00 at every
    single eval point.)"""
    Kx = unit_kernels(env, th, X)
    Kt = unit_kernels(env, th, env["Xtr"])
    data_mass = Kt.sum(0)
    w = Kx.sum(1)
    return (Kx @ data_mass) / np.maximum(w, 1e-300)


def region_split(env, th, q=0.25):
    """Split EVAL points into good- and bad-parity quarters by the parity field."""
    pr = parity_field(env, th, env["Xev"])
    lo, hi = np.quantile(pr, q), np.quantile(pr, 1 - q)
    return pr >= hi, pr <= lo, pr


def eval_rel_split(env, th, good, bad):
    """eval rel L2 overall / on good-parity points / on bad-parity points."""
    net = env["net"]
    e = net.f(env["Xev"], th) - env["yev"]
    y = env["yev"]

    def rel(mask):
        d = np.linalg.norm(y[mask])
        return float(np.linalg.norm(e[mask]) / d) if d > 0 else float("nan")
    return rel(np.ones(len(y), bool)), rel(good), rel(bad)


# ===================== blind kernels: Sam's rule, stated correctly ==========
# "the activation will have some nth order derivative that is a bump" -- gelu,
# sigmoid, tanh, swish. NOT necessarily the first. So the kernel of a parameter
# is not its Jacobian column and not any hand-picked derivative: it is the
# r-th INPUT derivative of its Jacobian column, for the smallest r that
# localises. r is chosen by measurement, per parameter, so nothing about the
# activation is assumed.
#
# This repairs an earlier wrong conclusion. Judged on raw column magnitude the
# readout looked structurally blind to local data coverage (tanh saturates, so
# the column norm is dominated by its tail). But d/dx of that column IS a bump,
# so coverage is measurable for readout parameters too -- the blindness was in
# the statistic, not the parameter.

def _participation(c):
    """(sum c^2)^2 / sum c^4 -- the effective number of points carrying the
    mass of c. Small = localised, ~n = spread over everything."""
    c2 = c * c
    return (c2.sum() ** 2) / max((c2 * c2).sum(), 1e-300)


def blind_kernels(J, order_max=3):
    """For each column of J, differentiate along the (sorted, gridded) input
    axis until it localises; return the localised kernels and the order used.

    Cost is O(n*m) elementwise work on a Jacobian we already hold, and needs a
    neighbour ordering of the rows -- free on a grid, an O(n) neighbour graph
    otherwise. It does NOT scale with P beyond the Jacobian itself."""
    best = np.abs(J).copy()
    best_pr = np.array([_participation(J[:, i]) for i in range(J.shape[1])])
    best_r = np.zeros(J.shape[1], dtype=int)
    cur = J.copy()
    for r in range(1, order_max + 1):
        cur = np.diff(cur, axis=0)
        pad = np.vstack([cur, np.zeros((r, cur.shape[1]))])
        pr = np.array([_participation(pad[:, i]) for i in range(pad.shape[1])])
        take = pr < best_pr
        best[:, take] = np.abs(pad[:, take])
        best_pr[take] = pr[take]
        best_r[take] = r
    return best, best_pr, best_r


def blind_parity(J, order_max=3):
    """Per-parameter local Nyquist number, blind.

      coverage_i = effective number of data points under parameter i's kernel
      crowd_i    = number of other kernels sharing that territory
      parity_i   = coverage_i / crowd_i        (samples per competing unknown)

    parity >= 1 means the data locally out-numbers the unknowns competing for
    it, which is exactly the condition for the local solve to be determined."""
    K, pr, r = blind_kernels(J, order_max)
    Kn = K / np.maximum(np.linalg.norm(K, axis=0, keepdims=True), 1e-300)
    overlap = Kn.T @ Kn                       # kernel-space cosine overlap
    crowd = np.abs(overlap).sum(1)
    return pr / np.maximum(crowd, 1e-300), pr, crowd, r
