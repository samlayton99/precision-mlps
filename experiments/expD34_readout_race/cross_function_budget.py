"""Fork-only movement budgets for the frozen full effective-map model.

These are reduced-model bounds evaluated in floating point, not certified
enclosures of nonlinear GD. No singular-value cutoff is used.
"""

import numpy as np


def spectral_budget(p0, T, e0, eta, horizons, thresholds=(1., 3.2, 16.)):
    """Evaluate the nonoscillating frozen-map model and its movement budgets.

    ``p0`` has D34 order (a, b, c, d), length P=3W+1; ``T`` is (P,M)
    and ``e0`` is (M,). The model updates e <- (I-eta*T.T*T)e and
    p <- p-eta*T*e, using the old residual in both updates.

    Returned shapes, with H horizons and K thresholds:
      displacement (H,P), displacement_norm (H,),
      slope_travel_upper (H,W), initial_occupancy (K,),
      simultaneous_occupancy_upper (H,K), ever_occupancy_upper (H,K).
    Occupancies are neuron counts, including neurons acquired initially.
    Simultaneous occupancy uses the monotone full-parameter displacement
    norm; ever occupancy uses coordinatewise absolute-travel bounds.

    Applicability requires eta*sigma**2 <= 1 for every singular value.
    Outside that regime, movement arrays are NaN and count bounds are -1;
    ``applicable`` and ``reason`` explain this without clipping eigenvalues.
    Exact null modes contribute no motion; tiny positive modes are retained.
    """
    p0, T, e0 = (np.asarray(x, dtype=np.float64) for x in (p0, T, e0))
    raw_horizons = np.asarray(horizons)
    thresholds = np.asarray(thresholds, dtype=np.float64)
    if p0.ndim != 1 or len(p0) < 4 or (len(p0)-1) % 3:
        raise ValueError("p0 must have flat D34 order (a,b,c,d), length 3W+1")
    if T.ndim != 2 or e0.ndim != 1 or T.shape != (len(p0), len(e0)):
        raise ValueError("T must have shape (len(p0), len(e0))")
    if not all(np.isfinite(x).all() for x in (p0, T, e0)):
        raise ValueError("parameters, map, and residual must be finite")
    if not np.isfinite(eta) or eta <= 0:
        raise ValueError("eta must be finite and positive")
    if (raw_horizons.ndim != 1 or not np.isfinite(raw_horizons).all()
            or np.any(raw_horizons < 0)
            or np.any(raw_horizons != np.floor(raw_horizons))
            or np.any(raw_horizons >= 2**63)):
        raise ValueError("horizons must be nonnegative int64 update counts")
    if (thresholds.ndim != 1 or not np.isfinite(thresholds).all()
            or np.any(thresholds <= 0)):
        raise ValueError("thresholds must be finite and positive")
    horizons = raw_horizons.astype(np.int64)
    width = (len(p0)-1)//3
    a0 = np.abs(p0[:width])
    U, sigma, Vt = np.linalg.svd(T, full_matrices=False)
    with np.errstate(over="ignore"):
        eta_lambda = eta*sigma*sigma
    applicable = bool(np.all(np.isfinite(eta_lambda)) and np.all(eta_lambda <= 1))
    result = {
        "model": "frozen_full_effective_map_only",
        "horizons": horizons,
        "thresholds": thresholds,
        "singular_values": sigma,
        "eta_lambda": eta_lambda,
        "applicable": applicable,
        "reason": "" if applicable else "nonoscillating condition eta*sigma**2 <= 1 fails",
        "displacement": np.full((len(horizons), len(p0)), np.nan),
        "displacement_norm": np.full(len(horizons), np.nan),
        "slope_travel_upper": np.full((len(horizons), width), np.nan),
        "initial_occupancy": np.sum(a0[:, None] >= thresholds, axis=0),
        "simultaneous_occupancy_upper": np.full((len(horizons), len(thresholds)), -1, dtype=int),
        "ever_occupancy_upper": np.full((len(horizons), len(thresholds)), -1, dtype=int),
    }
    if not applicable:
        return result

    # Phi = eta * sum_n (1-eta*lambda)**n. expm1/log1p preserve the
    # finite-time response when eta*lambda is too small for 1-x to resolve.
    phi = np.zeros((len(horizons), len(sigma)))
    interior = (eta_lambda > 0) & (eta_lambda < 1)
    for i, n in enumerate(horizons):
        if n == 0:
            continue
        phi[i, eta_lambda == 0] = eta*float(n)
        phi[i, eta_lambda == 1] = eta
        x = eta_lambda[interior]
        phi[i, interior] = eta*(-np.expm1(float(n)*np.log1p(-x))/x)
    amplitudes = phi*(sigma*(Vt @ e0))[None, :]
    displacement = -amplitudes @ U.T
    travel = np.abs(amplitudes) @ np.abs(U[:width, :]).T
    norms = np.linalg.norm(amplitudes, axis=1)
    result.update(displacement=displacement, displacement_norm=norms,
                  slope_travel_upper=travel)
    for k, threshold in enumerate(thresholds):
        distances = np.sqrt(np.cumsum(np.sort(np.maximum(threshold-a0, 0))**2))
        result["simultaneous_occupancy_upper"][:, k] = np.searchsorted(
            distances, norms, side="right")
        result["ever_occupancy_upper"][:, k] = np.sum(
            a0[None, :]+travel >= threshold, axis=1)
    return result
