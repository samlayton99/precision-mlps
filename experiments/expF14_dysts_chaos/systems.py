"""expF14 -- the five dysts chaotic ODE systems, with analytic RHS and Jacobian.

Canonical parameters, initial conditions, dominant period and estimated maximal
Lyapunov exponent are read from `dysts` (Gilpin, NeurIPS 2021 D&B).  The
right-hand sides are re-implemented here in vectorised numpy because the
Gauss-Newton solve needs (a) evaluation at n collocation points at once and
(b) an analytic Jacobian dF/du at those points.

Both re-implementations are verified:
  * `verify_rhs`      -- our F(U) against dysts' own rhs, elementwise;
  * `verify_jacobian` -- our J(U) against the complex-step derivative
                         Im F(U + i h e_c)/h, which is exact to rounding
                         (no subtractive cancellation, unlike finite differences).

Everything is fp64 except the complex-step probe.
"""

from __future__ import annotations

import numpy as np

SYSTEM_ORDER = ["Lorenz", "Rossler", "Thomas", "Halvorsen", "Lorenz96"]


# ---------------------------------------------------------------------------
# right-hand sides:  F(U) with U of shape (n, d) -> (n, d)
# Jacobians:         J(U) with U of shape (n, d) -> (n, d, d), J[:, c, k] = dF_c/du_k
# Both are written so they also accept complex input (for the complex-step test).
# ---------------------------------------------------------------------------

def _lorenz_rhs(U, p):
    x, y, z = U[:, 0], U[:, 1], U[:, 2]
    return np.stack([p["sigma"] * (y - x),
                     p["rho"] * x - x * z - y,
                     x * y - p["beta"] * z], axis=1)


def _lorenz_jac(U, p):
    x, y, z = U[:, 0], U[:, 1], U[:, 2]
    n = len(U)
    J = np.zeros((n, 3, 3))
    J[:, 0, 0] = -p["sigma"]; J[:, 0, 1] = p["sigma"]
    J[:, 1, 0] = p["rho"] - z; J[:, 1, 1] = -1.0; J[:, 1, 2] = -x
    J[:, 2, 0] = y; J[:, 2, 1] = x; J[:, 2, 2] = -p["beta"]
    return J


def _rossler_rhs(U, p):
    x, y, z = U[:, 0], U[:, 1], U[:, 2]
    return np.stack([-y - z,
                     x + p["a"] * y,
                     p["b"] + z * x - p["c"] * z], axis=1)


def _rossler_jac(U, p):
    x, y, z = U[:, 0], U[:, 1], U[:, 2]
    n = len(U)
    J = np.zeros((n, 3, 3))
    J[:, 0, 1] = -1.0; J[:, 0, 2] = -1.0
    J[:, 1, 0] = 1.0; J[:, 1, 1] = p["a"]
    J[:, 2, 0] = z; J[:, 2, 2] = x - p["c"]
    return J


def _thomas_rhs(U, p):
    x, y, z = U[:, 0], U[:, 1], U[:, 2]
    a, b = p["a"], p["b"]
    return np.stack([-a * x + b * np.sin(y),
                     -a * y + b * np.sin(z),
                     -a * z + b * np.sin(x)], axis=1)


def _thomas_jac(U, p):
    x, y, z = U[:, 0], U[:, 1], U[:, 2]
    a, b = p["a"], p["b"]
    n = len(U)
    J = np.zeros((n, 3, 3))
    J[:, 0, 0] = -a; J[:, 0, 1] = b * np.cos(y)
    J[:, 1, 1] = -a; J[:, 1, 2] = b * np.cos(z)
    J[:, 2, 2] = -a; J[:, 2, 0] = b * np.cos(x)
    return J


def _halvorsen_rhs(U, p):
    x, y, z = U[:, 0], U[:, 1], U[:, 2]
    a, b = p["a"], p["b"]
    return np.stack([-a * x - b * y - b * z - y ** 2,
                     -a * y - b * z - b * x - z ** 2,
                     -a * z - b * x - b * y - x ** 2], axis=1)


def _halvorsen_jac(U, p):
    x, y, z = U[:, 0], U[:, 1], U[:, 2]
    a, b = p["a"], p["b"]
    n = len(U)
    J = np.zeros((n, 3, 3))
    J[:, 0, 0] = -a; J[:, 0, 1] = -b - 2 * y; J[:, 0, 2] = -b
    J[:, 1, 0] = -b; J[:, 1, 1] = -a; J[:, 1, 2] = -b - 2 * z
    J[:, 2, 0] = -b - 2 * x; J[:, 2, 1] = -b; J[:, 2, 2] = -a
    return J


def _l96_rhs(U, p):
    """Cyclic Lorenz-96: xdot_i = (x_{i+1} - x_{i-2}) x_{i-1} - x_i + F."""
    Xp1 = np.roll(U, -1, axis=1)
    Xm1 = np.roll(U, 1, axis=1)
    Xm2 = np.roll(U, 2, axis=1)
    return (Xp1 - Xm2) * Xm1 - U + p["f"]


def _l96_jac(U, p):
    n, d = U.shape
    J = np.zeros((n, d, d))
    idx = np.arange(d)
    ip1, im1, im2 = (idx + 1) % d, (idx - 1) % d, (idx - 2) % d
    # accumulate (indices can collide for very small d)
    np.add.at(J, (slice(None), idx, ip1), U[:, im1])
    np.add.at(J, (slice(None), idx, im2), -U[:, im1])
    np.add.at(J, (slice(None), idx, im1), U[:, ip1] - U[:, im2])
    np.add.at(J, (slice(None), idx, idx), -1.0)
    return J


_IMPL = {
    "Lorenz": (_lorenz_rhs, _lorenz_jac),
    "Rossler": (_rossler_rhs, _rossler_jac),
    "Thomas": (_thomas_rhs, _thomas_jac),
    "Halvorsen": (_halvorsen_rhs, _halvorsen_jac),
    "Lorenz96": (_l96_rhs, _l96_jac),
}


# ---------------------------------------------------------------------------
# system objects
# ---------------------------------------------------------------------------

class System:
    def __init__(self, name):
        import dysts.flows as flows
        m = getattr(flows, name)()
        self.name = name
        self.params = {k: float(v) for k, v in m.params.items()}
        self.ic = np.atleast_1d(np.asarray(m.ic, dtype=np.float64)).copy()
        self.d = int(self.ic.size)
        self.period = float(m.period)
        self.lyapunov = float(m.maximum_lyapunov_estimated)
        self._dysts = m
        self._rhs, self._jac = _IMPL[name]

    def F(self, U):
        """(n, d) -> (n, d)."""
        return self._rhs(np.atleast_2d(U), self.params)

    def J(self, U):
        """(n, d) -> (n, d, d), J[:, c, k] = dF_c/du_k."""
        return self._jac(np.atleast_2d(U), self.params)

    def horizon(self, lyap_times):
        """Window length T such that lambda_max * T = lyap_times."""
        return float(lyap_times) / self.lyapunov

    def __repr__(self):
        return (f"<{self.name} d={self.d} lambda={self.lyapunov:.4f} "
                f"period={self.period:.4f}>")


def load_all():
    return {n: System(n) for n in SYSTEM_ORDER}


# ---------------------------------------------------------------------------
# verification
# ---------------------------------------------------------------------------

def verify_rhs(sys, n=64, seed=0):
    """Max |our F - dysts rhs| over random on-scale states, and its ulp scale."""
    rng = np.random.default_rng(seed)
    U = sys.ic[None, :] + 5.0 * rng.standard_normal((n, sys.d))
    ours = sys.F(U)
    theirs = np.empty_like(ours)
    for i in range(n):
        theirs[i] = np.asarray(sys._dysts.rhs(U[i], 0.0), dtype=np.float64).ravel()
    absd = np.max(np.abs(ours - theirs))
    scale = np.max(np.abs(theirs))
    return absd, absd / scale


def verify_jacobian(sys, n=64, seed=0, h=1e-30):
    """Analytic Jacobian vs the complex-step derivative (exact to rounding)."""
    rng = np.random.default_rng(seed)
    U = sys.ic[None, :] + 5.0 * rng.standard_normal((n, sys.d))
    Ja = sys.J(U)
    Jc = np.empty_like(Ja)
    for k in range(sys.d):
        Uc = U.astype(np.complex128)
        Uc[:, k] += 1j * h
        Jc[:, :, k] = np.imag(sys._rhs(Uc, sys.params)) / h
    absd = np.max(np.abs(Ja - Jc))
    scale = max(np.max(np.abs(Jc)), 1e-300)
    return absd, absd / scale


def verify_all(verbose=True):
    """Returns dict name -> (rhs_rel, jac_rel). Raises if anything is off."""
    out = {}
    for name in SYSTEM_ORDER:
        s = System(name)
        r_abs, r_rel = verify_rhs(s)
        j_abs, j_rel = verify_jacobian(s)
        if verbose:
            print(f"  {name:10s} d={s.d}  rhs vs dysts: {r_rel:.2e} (rel)   "
                  f"jac vs complex-step: {j_rel:.2e} (rel)")
        assert r_rel < 1e-14, f"{name}: rhs mismatch {r_rel:.3e}"
        assert j_rel < 1e-13, f"{name}: jacobian mismatch {j_rel:.3e}"
        out[name] = (r_rel, j_rel)
    return out


if __name__ == "__main__":
    print("verifying RHS and Jacobian for the five dysts systems")
    verify_all()
    print("\nsystem metadata:")
    for n in SYSTEM_ORDER:
        s = System(n)
        for lt in (3,):
            print(f"  {s!r}  T(lambdaT={lt})={s.horizon(lt):.3f} "
                  f"= {s.horizon(lt)/s.period:.2f} periods  ic={np.round(s.ic,4).tolist()}")
