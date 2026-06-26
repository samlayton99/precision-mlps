"""Helpers for expC02: frequency-ladder targets and argmin selection.

Pure numpy; imported by run.py and exercised by tests/test_expC02_lambda_vs_frequency.py.
"""

from __future__ import annotations

import numpy as np


def sin_target(k: int):
    """Return (fn, deriv) for sin(k*pi*x) and its analytic derivative."""
    kpi = k * np.pi
    return (lambda x: np.sin(kpi * np.asarray(x)),
            lambda x: kpi * np.cos(kpi * np.asarray(x)))


def best_lambda(lambdas, errors) -> tuple[float, float]:
    """Return (lambda, error) at the minimum finite error; (nan, nan) if none."""
    lambdas = np.asarray(lambdas, dtype=np.float64)
    errors = np.asarray(errors, dtype=np.float64)
    mask = np.isfinite(errors)
    if not mask.any():
        return float("nan"), float("nan")
    i = int(np.argmin(np.where(mask, errors, np.inf)))
    return float(lambdas[i]), float(errors[i])
