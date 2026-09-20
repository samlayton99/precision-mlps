"""Machine-eps verification of the 2D tensor-product Hermite QI (constructions/).

For four targets on [-1,1]^2 with analytic tensor-Hermite data (f, d_x f, d_y f,
d2_xy f), reconstruct with ``TensorQI2D`` at both precisions and check the error on
a dense interior grid:

    - mpmath rel-L2 < 1e-11 on all four (should actually reach ~1e-13..1e-14).
    - fp64   rel-L2 < 1e-7 on all four.

Plus two structural checks that answer "is the target a product of x and y?":
    (a) x+y  -> the mixed coefficient tensor ||A||_inf ~= 0  (pure separable sum).
    (b) x*y  -> ||A||_inf is NOT negligible (the cross term IS a product basis).
        For x*y the cross-derivative is the constant 1, so ||A||_inf sits at the
        h^2-scale of the double convolution (~4e-4 at N=48): small but many orders
        above the exact-zero seen for x+y.

Run:  cd /scr/cdeng/precision-mlps && python -m pytest constructions/test_tensor_qi_2d.py -v -s
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tensor_qi_2d import TensorQI2D  # noqa: E402


# ---------------------------------------------------------------------------
# Targets on [-1,1]^2 with analytic (f, fx, fy, fxy). Callables on (P,2) -> (P,).
# ---------------------------------------------------------------------------
def _target_x_plus_y():
    return (
        lambda P: P[:, 0] + P[:, 1],
        lambda P: np.ones(len(P)),
        lambda P: np.ones(len(P)),
        lambda P: np.zeros(len(P)),
    )


def _target_x_times_y():
    return (
        lambda P: P[:, 0] * P[:, 1],
        lambda P: P[:, 1],
        lambda P: P[:, 0],
        lambda P: np.ones(len(P)),
    )


def _target_product_sines():
    tp = 2.0 * np.pi
    return (
        lambda P: np.sin(tp * P[:, 0]) * np.sin(tp * P[:, 1]),
        lambda P: tp * np.cos(tp * P[:, 0]) * np.sin(tp * P[:, 1]),
        lambda P: tp * np.sin(tp * P[:, 0]) * np.cos(tp * P[:, 1]),
        lambda P: tp * tp * np.cos(tp * P[:, 0]) * np.cos(tp * P[:, 1]),
    )


def _target_radial_bump():
    # f = exp(-3(x^2+y^2)); d_x f = -6x f; d2_xy f = 36 x y f.
    e = lambda P: np.exp(-3.0 * (P[:, 0] ** 2 + P[:, 1] ** 2))
    return (
        lambda P: e(P),
        lambda P: -6.0 * P[:, 0] * e(P),
        lambda P: -6.0 * P[:, 1] * e(P),
        lambda P: 36.0 * P[:, 0] * P[:, 1] * e(P),
    )


TARGETS = {
    "x+y": _target_x_plus_y(),
    "x*y": _target_x_times_y(),
    "product_sines": _target_product_sines(),
    "radial_bump": _target_radial_bump(),
}

N_TEST = 48

# Dense interior eval grid over [-0.9, 0.9]^2.
_xs = np.linspace(-0.9, 0.9, 60)
_X, _Y = np.meshgrid(_xs, _xs)
EVAL_P = np.stack([_X.ravel(), _Y.ravel()], 1)


def _errors(qi, f):
    pred = qi(EVAL_P)
    tgt = f(EVAL_P)
    rel_l2 = float(np.linalg.norm(pred - tgt) / np.linalg.norm(tgt))
    linf = float(np.abs(pred - tgt).max())
    return rel_l2, linf


@pytest.mark.parametrize("name", list(TARGETS))
def test_accuracy_both_precisions(name):
    f, fx, fy, fxy = TARGETS[name]

    qi64 = TensorQI2D(N_TEST, precision="fp64").fit(f, fx, fy, fxy)
    rel64, linf64 = _errors(qi64, f)

    qimp = TensorQI2D(N_TEST, precision="mpmath").fit(f, fx, fy, fxy)
    relmp, linfmp = _errors(qimp, f)

    print(
        f"\n[{name:14s}] fp64   rel-L2={rel64:.3e}  L_inf={linf64:.3e}"
        f"\n[{name:14s}] mpmath rel-L2={relmp:.3e}  L_inf={linfmp:.3e}"
    )

    assert rel64 < 1e-7, (
        f"{name}: fp64 rel-L2 {rel64:.3e} not < 1e-7 (L_inf {linf64:.3e})"
    )
    assert relmp < 1e-11, (
        f"{name}: mpmath rel-L2 {relmp:.3e} not < 1e-11 (L_inf {linfmp:.3e})"
    )


def test_structural_A_norm():
    """x+y is a pure separable sum (A ~= 0); x*y needs the product cross term."""
    f_sum, fx_sum, fy_sum, fxy_sum = TARGETS["x+y"]
    qi_sum = TensorQI2D(N_TEST, precision="fp64").fit(f_sum, fx_sum, fy_sum, fxy_sum)
    A_sum = float(np.abs(qi_sum.A).max())

    f_prod, fx_prod, fy_prod, fxy_prod = TARGETS["x*y"]
    qi_prod = TensorQI2D(N_TEST, precision="fp64").fit(f_prod, fx_prod, fy_prod, fxy_prod)
    A_prod = float(np.abs(qi_prod.A).max())

    print(
        f"\n[structural] ||A||_inf  x+y = {A_sum:.3e}  (expect ~0)"
        f"\n[structural] ||A||_inf  x*y = {A_prod:.3e}  (expect NOT ~0)"
    )

    assert A_sum < 1e-9, f"x+y should have A~=0, got ||A||_inf={A_sum:.3e}"
    assert A_prod > 1e-4, f"x*y should have A!=0, got ||A||_inf={A_prod:.3e}"
