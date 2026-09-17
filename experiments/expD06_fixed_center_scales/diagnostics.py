"""Detached FP64 diagnostics. No training state or readout is modified here."""

from __future__ import annotations

import numpy as np
from scipy.linalg import svd


def midpoint_grid(count):
    return -1 + 2 * (np.arange(count, dtype=np.float64) + 0.5) / count


def features(x, centers, gamma):
    return np.column_stack((np.ones(len(x)), np.tanh((x[:, None] - centers) * gamma)))


def prediction(x, centers, c, gamma, chunk_size=4096):
    return np.concatenate([features(x[i:i + chunk_size], centers, gamma) @ c
                           for i in range(0, len(x), chunk_size)])


def band_residuals(r):
    """Orthonormal DFT bands, including DC and both signs of every frequency."""
    count = len(r)
    index = np.minimum(np.arange(count), count - np.arange(count))
    spectrum = np.fft.fft(r, norm="ortho")
    bounds = [(0, 1)]
    lo = 1
    while lo <= count // 2:
        bounds.append((lo, min(2 * lo, count // 2 + 1)))
        lo *= 2
    residuals = np.stack([np.fft.ifft(spectrum * ((index >= lo) & (index < hi)),
                                     norm="ortho").real for lo, hi in bounds])
    return np.asarray(bounds), residuals, spectrum


def gradient_bands(r, a, j, projected_j):
    bounds, rb, spectrum = band_residuals(r)
    total = rb @ j
    parallel = rb @ projected_j
    return {"band_bounds": bounds, "residual_fft": spectrum,
            "band_energy": np.sum(rb**2, axis=1), "band_gradient_lambda": total,
            "band_gradient_parallel": parallel, "band_gradient_perpendicular": total - parallel,
            "band_gradient_readout": rb @ a}


def checkpoint_arrays(x, y, centers, h, d_reference, c, gamma, masks, cutoff=1e-12,
                      delta_c=None, delta_lambda=None):
    """Use one scaled SVD for refitting and every retained-span diagnostic."""
    count = len(x)
    root_m = np.sqrt(count)
    a = features(x, centers, gamma) / root_m
    normalized_y = y / root_m
    r = a @ c - normalized_y
    u, s, vh = svd(a * d_reference, full_matrices=False, lapack_driver="gesdd")
    keep = s > cutoff * s[0]
    retained = u[:, keep]
    projected_r = retained @ (retained.T @ r)
    a_star = vh[keep].T @ ((retained.T @ normalized_y) / s[keep])
    c_star = d_reference * a_star
    refit_r = a @ c_star - normalized_y
    distances = x[:, None] - centers
    e = np.exp(-2 * np.abs(distances * gamma))
    j = c[1:] * distances * (4 * e / (1 + e)**2) / (h * root_m)
    projected_j = retained @ (retained.T @ j)
    perpendicular_j = j - projected_j
    output = gradient_bands(r, a, j, projected_j)
    output.update({
        "x_train": x, "target_train": y,
        "prediction_train": (r + normalized_y) * root_m,
        "residual_train": r * root_m,
        "residual_parallel": projected_r * root_m,
        "residual_perpendicular": (r - projected_r) * root_m,
        "residual_refit": refit_r * root_m,
        "readout_refit": c_star, "readout_correction_prediction": a @ (c_star - c) * root_m,
        "singular_values": s, "retained_rank": np.array(keep.sum()), "cutoff": np.array(cutoff),
        "gradient_lambda": j.T @ r, "gradient_readout": a.T @ r,
        "gradient_parallel": projected_j.T @ r,
        "gradient_perpendicular": perpendicular_j.T @ r,
        "tangent_column_norm": np.linalg.norm(j, axis=0),
        "projected_tangent_column_norm": np.linalg.norm(perpendicular_j, axis=0),
        "fft_angular_frequency": 2 * np.pi * np.fft.fftfreq(count, d=x[1] - x[0]),
        "parallel_fft": np.fft.fft(projected_r, norm="ortho"),
        "perpendicular_fft": np.fft.fft(r - projected_r, norm="ortho"),
        "refit_fft": np.fft.fft(refit_r, norm="ortho"),
        "tapered_residual_fft": np.fft.fft(np.hanning(count) * r, norm="ortho"),
    })
    all_masks = {"all": np.ones_like(gamma, dtype=bool), **masks}
    for region, mask in all_masks.items():
        v = np.where(mask, np.sign(gamma), 0.0)
        norm = np.linalg.norm(v)
        if norm:
            v /= norm
        t = j @ v
        pt = projected_j @ v
        zt = t - pt
        rp = r - projected_r
        raw_norm, projected_norm = np.linalg.norm(t), np.linalg.norm(zt)
        alignment_denominator = np.linalg.norm(rp) * projected_norm
        output[f"force_{region}"] = np.array([
            -projected_r @ pt, -rp @ zt, raw_norm, projected_norm,
            projected_norm / raw_norm if raw_norm else np.nan,
            -rp @ zt / alignment_denominator if alignment_denominator else np.nan])
    # Unit-RMS sine/cosine probes have unit norm in normalized residual coordinates.
    # Retain response norms and collective signed responses; full matrices can be rebuilt.
    for label, tangent in (("raw", j), ("perpendicular", perpendicular_j)):
        transform = np.fft.rfft(tangent, axis=0, norm="ortho")
        multiplier = np.full(transform.shape[0], np.sqrt(2))
        multiplier[0] = 1.0
        if count % 2 == 0:
            multiplier[-1] = 1.0
        for region, mask in all_masks.items():
            output[f"probe_{label}_{region}_cos"] = multiplier * np.linalg.norm(transform[:, mask].real, axis=1)
            output[f"probe_{label}_{region}_sin"] = multiplier * np.linalg.norm(transform[:, mask].imag, axis=1)
    if delta_c is not None and delta_lambda is not None:
        output["next_delta_readout"] = delta_c
        output["next_delta_lambda"] = delta_lambda
        output["band_predicted_descent_geometry"] = -output["band_gradient_lambda"] @ delta_lambda
        output["band_predicted_descent_readout"] = -output["band_gradient_readout"] @ delta_c
        changed_a = features(x, centers, gamma + delta_lambda / h) / root_m
        readout_change = a @ delta_c
        geometry_change = (changed_a - a) @ c
        interaction = (changed_a - a) @ delta_c
        measured = changed_a @ (c + delta_c) - a @ c
        output.update({"prediction_change_readout": root_m * readout_change,
                       "prediction_change_geometry": root_m * geometry_change,
                       "prediction_change_interaction": root_m * interaction,
                       "prediction_change_measured": root_m * measured,
                       "prediction_change_linearized": root_m * (readout_change + j @ delta_lambda)})
    return output


def errors(pred, y):
    residual = pred - y
    return {"rms": float(np.sqrt(np.mean(residual**2))),
            "rel_l2": float(np.linalg.norm(residual) / np.linalg.norm(y)),
            "linf": float(np.max(np.abs(residual)))}
