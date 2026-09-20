"""2D tensor-product Hermite quasi-interpolant on the tanh basis (fp64 / mpmath).

This is the repo's FIRST true 2D QI construction. Every prior 2D result
(expE01_geometry_zoo_2d) reconstructs f by placing ridge neurons and solving a
dense least-squares readout. Here we instead reconstruct f on [-1,1]^2 by an
EXPLICIT tensor-product convolution -- no dense solve, no conditioning wall --
reusing the repo's own well-conditioned 1D QI machinery
(``src.construction.qi_mpmath``).

The identity
------------
For f on [-1,1]^2, the tensor Hermite / fundamental-theorem-of-calculus identity
is

    f(x,y) = f(x,-1) + f(-1,y) - f(-1,-1) + \\int_{-1}^{x}\\int_{-1}^{y} d2_xy f .

We realize each piece with the tanh-QI basis:

    R(x,y) = Qx(x) + Qy(y) - f(-1,-1) + M(x,y)

- ``Qx`` is the repo's 1D QI reconstruction of the bottom-edge slice
  x |-> f(x,-1) (via ``construct_qi`` + ``evaluate_qi``). ``Qy`` is the 1D QI of
  the left-edge slice y |-> f(-1,y). These separable marginals inherit the 1D
  path's fp64/mpmath precision and c_j cache for free.
- ``f(-1,-1)`` is the corner constant, subtracted once because Qx and Qy each
  already carry it (Qx(-1) = f(-1,-1), Qy(-1) = f(-1,-1)).
- ``M(x,y) = \\sum_n \\sum_m A_{nm} S_n(x) S_m(y)`` is the genuinely-2D cross
  term. The boundary-anchored basis

      S_n(u) = tanh(gamma*(u - x_n)) - tanh(gamma*(-1 - x_n))    (so S_n(-1) = 0)

  makes the mixed term vanish on both edges (M(x,-1) = M(-1,y) = 0), so it does
  not disturb the marginals. Since d/du S_n(u) = gamma*sech^2(gamma*(u - x_n)) is
  exactly the kernel the cardinal filter c_j inverts, the mixed coefficients are
  a DOUBLE convolution of the cross-derivative samples g_{jk} = d2_xy f(x_j, x_k)
  with the SAME c_j along both axes:

      A = conv_axis(conv_axis(g).T).T .

  Grid/centers/halo/gamma match the 1D ``construct_qi`` exactly
  (h = 2/N, gamma = lambda_star/h, x_n = -1 + n*h over n in [-halo, N+halo];
  extended sample grid j in [-(halo+Kc), N+halo+Kc]).

- An optional degree-3 polynomial tail fits {x^a y^b : a+b <= 3} to the residual
  f - R_base on a small well-conditioned interior sample, making low-degree
  polynomials (x+y, x*y, ...) exact.

All computation is float64. The mpmath path builds the cardinal filter c_j at
arbitrary precision (then stores fp64 coefficients) so the construction reaches
machine epsilon, exactly as in the repo's 1D path. Reuses:
``construct_qi``, ``evaluate_qi``, ``default_halo``,
``_build_toeplitz_c_f64``, ``_build_toeplitz_c_mpmath``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Literal, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.construction.qi_mpmath import (  # noqa: E402
    construct_qi,
    evaluate_qi,
    default_halo,
    _build_toeplitz_c_f64,
    _build_toeplitz_c_mpmath,
)

# 80-bit extended precision for the mpmath cross-term convolution. The cardinal
# filter c_j reaches |c_0| ~ 300 with alternating signs, so a plain-fp64 double
# convolution of d2_xy f loses ~4-5 digits (floor ~1e-10). Carrying the c_j and
# the double-conv accumulation in longdouble (initialized from the full-precision
# mpmath c_j, not from a re-cast fp64 value) removes that loss and lets the tanh
# reconstruction reach machine eps -- the same LD strategy as the continuous-mlps
# tensor-QI reference. Only the OFFLINE coefficient build uses LD; A is stored as
# fp64 and every reported reconstruction is evaluated in fp64.
_LD = np.longdouble


class TensorQI2D:
    """Tensor-product Hermite QI reconstruction of f on [-1,1]^2.

    Args:
        N:            Per-axis number of grid intervals (h = 2/N).
        precision:    "fp64" (lambda_star=0.30, fast) or "mpmath"
                      (lambda_star=0.25, machine eps). Matches ``construct_qi``.
        lambda_star:  Override the dimensionless bandwidth gamma*h. If None, uses
                      the precision default (0.30 fp64 / 0.25 mpmath).
        Kc:           Toeplitz stencil half-width (default 160, matches the 1D
                      path and continuous-mlps).
        halo:         Ghost nodes each side. If None, ``default_halo(N, ...)``.
        mp_dps:       mpmath decimal places for the mpmath path.
        poly_tail:    If True (default), fit a degree-3 monomial tail to the
                      residual so low-degree polynomials are exact.
    """

    def __init__(
        self,
        N: int,
        *,
        precision: Literal["fp64", "mpmath"] = "fp64",
        lambda_star: Optional[float] = None,
        Kc: int = 160,
        halo: Optional[int] = None,
        mp_dps: int = 30,
        poly_tail: bool = True,
    ):
        if lambda_star is None:
            lambda_star = 0.25 if precision == "mpmath" else 0.30
        if halo is None:
            halo = default_halo(N, lambda_star=lambda_star)

        self.N = int(N)
        self.precision = precision
        self.lambda_star = float(lambda_star)
        self.Kc = int(Kc)
        self.halo = int(halo)
        self.mp_dps = int(mp_dps)
        self.poly_tail = bool(poly_tail)

        self.h = 2.0 / self.N
        self.gamma = self.lambda_star / self.h

        # ---- cardinal filter c_j (target-independent), same builders as 1D ----
        # mpmath path carries c_j in longdouble (from full-precision mpmath, not a
        # re-cast fp64) so the cross-term double-conv reaches machine eps despite
        # |c_j| ~ 300 with alternating signs. fp64 path stays plain fp64 (fast).
        if precision == "mpmath":
            import mpmath as mp  # noqa: F401
            c_mp, _ = _build_toeplitz_c_mpmath(
                h=self.h, gamma=self.gamma, Kc=self.Kc, mp_dps=self.mp_dps
            )
            self._conv_dtype = _LD
            self.c = np.array([_LD(mp.nstr(v, self.mp_dps)) for v in c_mp], dtype=_LD)
        else:
            c_list, _ = _build_toeplitz_c_f64(h=self.h, gamma=self.gamma, Kc=self.Kc)
            self._conv_dtype = np.float64
            self.c = np.asarray(c_list, dtype=np.float64)

        # ---- centers (n in [-halo, N+halo]) and extended sample grid j ----
        self.n_idx = np.arange(-self.halo, self.N + self.halo + 1)
        self.xn = -1.0 + self.n_idx * self.h                     # centers, (Nc,)
        self.jmin = -(self.halo + self.Kc)
        self.j_idx = np.arange(self.jmin, self.N + self.halo + self.Kc + 1)
        self.xj = -1.0 + self.j_idx * self.h                     # samples, (Nj,)

        # populated by .fit(...)
        self.qi_x = None
        self.qi_y = None
        self.f00 = None
        self.A = None
        self.tail_c = None

    # ------------------------------------------------------------------ #
    # cross-term convolution
    # ------------------------------------------------------------------ #
    def _conv_axis0(self, Y: np.ndarray) -> np.ndarray:
        """a_{n,...} = sum_k c_k Y_{n-k,...}  along axis 0.

        Y is sampled on ``self.j_idx``; output indexed by ``self.n_idx``. Same
        correlation the 1D path applies to the derivative samples, generalized to
        broadcast over trailing axes.
        """
        Y = np.asarray(Y, dtype=self._conv_dtype)
        out = np.zeros((len(self.n_idx),) + Y.shape[1:], dtype=self._conv_dtype)
        for kk in range(len(self.c)):
            k = kk - self.Kc                        # stencil index k in [-Kc, Kc]
            idx = (self.n_idx - k) - self.jmin      # positions into j_idx
            out += self.c[kk] * Y[idx]
        return out

    # ------------------------------------------------------------------ #
    # fit
    # ------------------------------------------------------------------ #
    def fit(
        self,
        f: Callable[[np.ndarray], np.ndarray],
        fx: Callable[[np.ndarray], np.ndarray],
        fy: Callable[[np.ndarray], np.ndarray],
        fxy: Callable[[np.ndarray], np.ndarray],
    ) -> "TensorQI2D":
        """Fit from tensor-Hermite data.

        Args:
            f:   f(x,y)          callable on stacked (P,2) points -> (P,).
            fx:  d_x f(x,y).
            fy:  d_y f(x,y).
            fxy: d2_xy f(x,y).
        """
        # ---- corner constant ----
        self.f00 = float(f(np.array([[-1.0, -1.0]]))[0])

        # ---- separable marginals via the repo's 1D QI ----
        # Qx: 1D QI of x |-> f(x,-1); its x-derivative slice is d_x f(x,-1).
        self.qi_x = construct_qi(
            target_fn=lambda x: float(f(np.array([[x, -1.0]]))[0]),
            target_deriv=lambda x: float(fx(np.array([[x, -1.0]]))[0]),
            N=self.N,
            precision=self.precision,
            lambda_star=self.lambda_star,
            Kc=self.Kc,
            halo=self.halo,
            mp_dps=self.mp_dps,
        )
        # Qy: 1D QI of y |-> f(-1,y); its y-derivative slice is d_y f(-1,y).
        self.qi_y = construct_qi(
            target_fn=lambda y: float(f(np.array([[-1.0, y]]))[0]),
            target_deriv=lambda y: float(fy(np.array([[-1.0, y]]))[0]),
            N=self.N,
            precision=self.precision,
            lambda_star=self.lambda_star,
            Kc=self.Kc,
            halo=self.halo,
            mp_dps=self.mp_dps,
        )

        # ---- mixed coefficients: double conv of d2_xy f on the extended grid ----
        xj = self.xj
        Xg, Yg = np.meshgrid(xj, xj, indexing="ij")
        FXY = fxy(np.stack([Xg.ravel(), Yg.ravel()], 1)).reshape(len(xj), len(xj))
        FXY = np.asarray(FXY, dtype=self._conv_dtype)
        # conv along axis 0, transpose, conv along axis 0 again, transpose back.
        # Kept in self._conv_dtype (longdouble on the mpmath path) so the O(300)
        # cardinal-filter cancellation does not cap precision at ~1e-10.
        self.A = self._conv_axis0(self._conv_axis0(FXY).T).T          # (Nc, Nc)

        # ---- optional degree-3 polynomial tail on the residual ----
        self.tail_c = None
        if self.poly_tail:
            rng = np.random.default_rng(0)
            Ptr = rng.uniform(-0.97, 0.97, (2000, 2))
            resid = f(Ptr) - self._base(Ptr)
            B = self._monos(Ptr)
            self.tail_c, *_ = np.linalg.lstsq(B, resid, rcond=1e-14)
        return self

    # ------------------------------------------------------------------ #
    # evaluation
    # ------------------------------------------------------------------ #
    def _S(self, u: np.ndarray) -> np.ndarray:
        """Boundary-anchored basis S_n(u) = tanh(g(u-x_n)) - tanh(g(-1-x_n)).

        Shape (P, Nc); S_n(-1) = 0.
        """
        u = np.asarray(u, dtype=np.float64)
        return (
            np.tanh(self.gamma * (u[:, None] - self.xn[None, :]))
            - np.tanh(self.gamma * (-1.0 - self.xn[None, :]))
        )

    @staticmethod
    def _monos(P: np.ndarray) -> np.ndarray:
        x, y = P[:, 0], P[:, 1]
        mono = [(0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2),
                (3, 0), (2, 1), (1, 2), (0, 3)]
        return np.stack([x ** a * y ** b for (a, b) in mono], 1)

    def _base(self, P: np.ndarray) -> np.ndarray:
        """R_base = Qx(x) + Qy(y) - f00 + M(x,y) (no polynomial tail)."""
        P = np.asarray(P, dtype=np.float64)
        x, y = P[:, 0], P[:, 1]
        qx = evaluate_qi(self.qi_x, x)
        qy = evaluate_qi(self.qi_y, y)
        # Evaluate the cross term in the convolution dtype (longdouble on the
        # mpmath path) to match the machine-eps coefficients, then cast to fp64.
        Sx = self._S(x).astype(self._conv_dtype)
        Sy = self._S(y).astype(self._conv_dtype)
        mixed = np.einsum("pn,nm,pm->p", Sx, self.A, Sy).astype(np.float64)
        return qx + qy - self.f00 + mixed

    def __call__(self, P: np.ndarray) -> np.ndarray:
        out = self._base(P)
        if self.tail_c is not None:
            out = out + self._monos(np.asarray(P, dtype=np.float64)) @ self.tail_c
        return out
