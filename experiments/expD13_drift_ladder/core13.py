"""expD13 -- the mu ladder on a DRIFTING Phi (expD12 redone faithfully).

expD12 solved a frozen Phi. That is not what training looks like. Here Phi moves:
Adam is finding the geometry, so early on Phi is far from the geometry it will
settle into and it takes large random jumps; as training proceeds the jumps
shrink and Phi converges.

THE SCHEDULE (a reverse noise schedule, tied to the damping)
  At ladder level with damping alpha = sqrt(mu)/sigma_1, the geometry is
        g_k = g_true + (nmult * alpha_k) * z_k ,   z_k ~ N(0, I) fresh each level
  so the distance from the true geometry shrinks WITH the damping, the jump
  between consecutive levels is O(alpha_{k-1}), and by the last level the
  perturbation is ~1e-10..1e-12. The terminal solve then runs on the true Phi.

  nmult sweeps the MATCHING between damping and geometry noise:
      nmult = 0     static Phi (the expD12 control)
      nmult = 0.1   damping coarser than the noise
      nmult = 1     matched
      nmult = 10    damping finer than the noise (over-solving a wrong Phi)

WHAT MOVES IS THE GEOMETRY, NOT THE MATRIX
  Perturbing Phi entrywise would be a different (and easier) experiment. Here
  the generating parameters move and BOTH the train and eval matrices are
  rebuilt from them, so the measured error is the error the model would
  actually have at that point in training.
      QI-1D   centers move by eta*h, gamma by (1+eta)
      QI-2D   offsets move by eta*scale, directions rotate by eta
      twin    the right singular subspace V rotates by eta

Warm start is used throughout: each level begins from the previous level's w.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
E01 = REPO / "experiments" / "expE01_geometry_zoo_2d"
for p in (str(REPO), str(E01)):
    if p not in sys.path:
        sys.path.insert(0, p)

RESULTS = REPO / "results" / "checkpoint_D_optimizers" / "expD13_drift_ladder"
FIGS = RESULTS / "figures"
RCOND = 1e-15


# ====================================================== drifting problems ===

class DriftProblem:
    """Holds a TRUE geometry and can rebuild (A, y, A_ev, y_ev) at any
    perturbation level eta. eta=0 gives the true system."""

    family = "base"

    def build(self, eta, seed):
        raise NotImplementedError

    def at(self, eta, seed=0):
        A, y, A_ev, y_ev = self.build(eta, seed)
        return dict(A=A, y=y, A_ev=A_ev, y_ev=y_ev,
                    ny=float(np.linalg.norm(y_ev)), d=A.shape[1], n=A.shape[0])


class QI1D(DriftProblem):
    family = "qi1d"

    def __init__(self, N, lam=0.30, row_mult=6, noise_rel=0.0):
        from src.construction.qi_mpmath import default_halo
        self.N, self.lam, self.row_mult, self.noise_rel = N, lam, row_mult, noise_rel
        self.h = 2.0 / N
        self.gamma = lam / self.h
        halo = int(default_halo(N, lambda_star=lam))
        self.cen = -1.0 + self.h * np.arange(-halo, N + halo + 1)
        self.label = f"QI-1D  N={N}"
        self.xt = np.linspace(-1, 1, row_mult * (self.cen.size + 1))
        self.xe = np.linspace(-1, 1, 4001)
        self.f = lambda z: np.sin(4.0 * z)

    def build(self, eta, seed):
        rng = np.random.default_rng(1000 + seed)
        cen = self.cen + eta * self.h * rng.standard_normal(self.cen.size)
        gam = self.gamma * (1.0 + eta * rng.standard_normal())
        ph = lambda x: np.hstack([np.tanh(gam * (x[:, None] - cen[None, :])),
                                  np.ones((x.size, 1))])
        y = self.f(self.xt)
        if self.noise_rel > 0:
            e = rng.standard_normal(self.xt.size)
            y = y + e * self.noise_rel * np.linalg.norm(y) / np.linalg.norm(e)
        return ph(self.xt), y, ph(self.xe), self.f(self.xe)


class QI2D(DriftProblem):
    family = "qi2d"

    def __init__(self, N, geom="radon_tensor", lam=0.12, row_mult=6, noise_rel=0.0):
        from geometries import GEOMETRIES
        from src.data.sampling2d import disk_grid, disk_uniform
        from src.data.targets2d import get_target_2d
        radius = 2.5 if geom.startswith("radon") else 1.6
        self.dirs, self.offs = GEOMETRIES[geom](N, radius, seed=0)
        self.gamma = lam / (2.8 / np.sqrt(N))
        self.noise_rel = noise_rel
        d = len(self.offs) + 1
        self.Xt = disk_uniform(row_mult * d, radius=1.0, seed=0)
        self.Xe = disk_grid(120, radius=1.0)
        self.f = get_target_2d("gauss_bump").fn_numpy
        self.scale = float(np.std(self.offs))
        self.label = f"QI-2D  N={N}"

    def build(self, eta, seed):
        rng = np.random.default_rng(2000 + seed)
        offs = self.offs + eta * self.scale * rng.standard_normal(self.offs.size)
        dirs = self.dirs + eta * rng.standard_normal(self.dirs.shape)
        dirs = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
        gam = self.gamma * (1.0 + eta * rng.standard_normal())
        ph = lambda P: np.hstack([np.tanh(gam * (P @ dirs.T - offs[None, :])),
                                  np.ones((len(P), 1))])
        y = self.f(self.Xt)
        if self.noise_rel > 0:
            e = rng.standard_normal(len(self.Xt))
            y = y + e * self.noise_rel * np.linalg.norm(y) / np.linalg.norm(e)
        return ph(self.Xt), y, ph(self.Xe), self.f(self.Xe)


class Twin(DriftProblem):
    """Random matrix; the 'geometry' that drifts is the right singular
    subspace V, which is exactly what a moving feature map does."""
    family = "twin"

    def __init__(self, d, spectrum=None, rank_frac=0.6, row_mult=4, n_ev=1500,
                 noise_rel=0.0, label=None, seed=0):
        rng = np.random.default_rng(9000 + d + seed)
        self.sig = (np.asarray(spectrum, float) if spectrum is not None
                    else np.logspace(0, -14, int(rank_frac * d)))[:d]
        r = self.sig.size
        self.ntr = row_mult * d
        self.U = np.linalg.qr(rng.standard_normal((self.ntr, r)))[0]
        self.Ue = np.linalg.qr(rng.standard_normal((n_ev, r)))[0]
        self.V = np.linalg.qr(rng.standard_normal((d, r)))[0]
        self.coef = self.sig * rng.standard_normal(r)
        self.d, self.noise_rel = d, noise_rel
        self.label = label or f"random  d={d}"

    def build(self, eta, seed):
        rng = np.random.default_rng(3000 + seed)
        V = self.V + eta * rng.standard_normal(self.V.shape) / np.sqrt(self.d)
        V = np.linalg.qr(V)[0]
        A = (self.U * self.sig) @ V.T
        A_ev = (self.Ue * self.sig) @ V.T
        y = self.U @ self.coef
        y_ev = self.Ue @ self.coef
        sc = np.sqrt(self.ntr) / np.linalg.norm(y)
        y, y_ev = y * sc, y_ev * sc
        if self.noise_rel > 0:
            e = rng.standard_normal(self.ntr)
            y = y + e * self.noise_rel * np.linalg.norm(y) / np.linalg.norm(e)
        return A, y, A_ev, y_ev


# =============================================================== helpers ====

def eval_err(p, w):
    return float(np.linalg.norm(p["A_ev"] @ w - p["y_ev"]) / p["ny"])


def spectrum(p):
    U, s, Vt = np.linalg.svd(p["A"], full_matrices=False)
    return U, s, Vt, int((s > RCOND * s[0]).sum())


def damped_exact(U, s, Vt, y, mu):
    return Vt.T @ ((s / (s ** 2 + mu)) * (U.T @ y))


def pool_floor(p):
    U, s, Vt, r = spectrum(p)
    kp = s > RCOND * s[0]
    a = (Vt.T * np.where(kp, 1 / np.where(kp, s, 1), 0)) @ (U.T @ p["y"])
    return eval_err(p, a), r, float(s[0])


def floor_1batch(p, b, seed=0, reps=2):
    A, y, n = p["A"], p["y"], p["n"]
    out = []
    for rep in range(reps):
        rng = np.random.default_rng(500 + 17 * rep + seed)
        S = rng.choice(n, min(b, n), replace=False)
        U, s, Vt = np.linalg.svd(A[S], full_matrices=False)
        kp = s > RCOND * s[0]
        a = (Vt.T * np.where(kp, 1 / np.where(kp, s, 1), 0)) @ (U.T @ y[S])
        out.append(eval_err(p, a))
    return float(np.exp(np.mean(np.log(np.maximum(out, 1e-300)))))


# ============================================================== solver ======
# reuse the verified expD12 solver (LSQR with damping, sliding/full reorth,
# breakdown guard at 1e-14, vectorised CGS2)
sys.path.insert(0, str(REPO / "experiments" / "expD12_mu_ladder"))
from core12 import lsqr_damped, probe_sched          # noqa: E402


def append(path, rec):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")


def load(path):
    if not Path(path).exists():
        return []
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
