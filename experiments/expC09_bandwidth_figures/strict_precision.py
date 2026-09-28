"""Correctly rounded p-bit inference; FP64 readout recovery is offline."""
from __future__ import annotations

import ctypes
import hashlib
from pathlib import Path
import subprocess
import sys

import gmpy2
import numpy as np
from scipy.linalg import lstsq

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.expC09_bandwidth_figures.run import geometry, round_bits, target_values

OUT = ROOT / "results/checkpoint_C_geometry/expC09_bandwidth_figures/precision_law_W1024_strict"
MODEL = ("MPFR round-to-nearest ties-to-even at p significand bits for inputs, centers, gamma, "
         "readout weights, bias, subtraction, multiplication, tanh, and sequential readout addition; "
         "no FMA; unrestricted exponent; FP64 offline SVD and error measurement")
_LIB = None


def library():
    global _LIB
    if _LIB is not None:
        return _LIB
    source = Path(__file__).with_name("strict_arithmetic.c")
    package = Path(gmpy2.__file__).parent
    libs = package.parent / "gmpy2.libs"
    candidates = list(libs.glob("libmpfr*.dylib")) + list(libs.glob("libmpfr*.so*"))
    if len(candidates) != 1:
        raise RuntimeError("Expected the pinned gmpy2 wheel's bundled MPFR library")
    mpfr_lib = candidates[0]
    fingerprint = hashlib.sha256(source.read_bytes() + gmpy2.mpfr_version().encode()).hexdigest()[:16]
    cache = OUT / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    compiled = cache / f"strict_arithmetic_{fingerprint}.so"
    if not compiled.exists():
        subprocess.run(["cc", "-O3", "-shared", "-fPIC", "-I", str(package), str(source),
                        str(mpfr_lib), "-o", str(compiled)], check=True)
        if sys.platform == "darwin":
            install_name = subprocess.check_output(["otool", "-D", str(mpfr_lib)], text=True).splitlines()[1].strip()
            subprocess.run(["install_name_tool", "-change", install_name, str(mpfr_lib), str(compiled)], check=True)
    lib = ctypes.CDLL(str(compiled))
    array = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
    lib.strict_features.argtypes = [ctypes.c_size_t, ctypes.c_size_t, array, array,
                                   ctypes.c_double, ctypes.c_uint, array]
    lib.strict_readout.argtypes = [ctypes.c_size_t, ctypes.c_size_t, array, array,
                                  ctypes.c_double, ctypes.c_uint, array]
    lib.strict_features.restype = lib.strict_readout.restype = ctypes.c_int
    _LIB = lib
    return lib


def features(x, centers, gamma, p):
    assert 2 <= p <= 53
    x, centers = np.ascontiguousarray(x, dtype=np.float64), np.ascontiguousarray(centers, dtype=np.float64)
    out = np.empty((len(x), len(centers)))
    if library().strict_features(len(x), len(centers), x, centers, gamma, p, out):
        raise MemoryError("MPFR feature workspace")
    return out


def readout(phi, weights, bias, p):
    phi, weights = np.ascontiguousarray(phi, dtype=np.float64), np.ascontiguousarray(weights, dtype=np.float64)
    assert phi.ndim == 2 and phi.shape[1] == len(weights) and 2 <= p <= 53
    out = np.empty(phi.shape[0])
    if library().strict_readout(*phi.shape, phi, weights, bias, p, out):
        raise MemoryError("MPFR readout workspace")
    return out


def measure(width, halo, lam, p, train_points, eval_points, target="chirp", *, save_model=None):
    n, h, centers = geometry(width, halo)
    centers = round_bits(centers, p)
    gamma = float(round_bits(lam/h, p))
    x = np.linspace(-1, 1, train_points)
    phi = features(x, centers, gamma, p)
    a = np.column_stack((phi, np.ones(x.size)))
    y = round_bits(target_values(x, target), p)
    weights, _, rank, singular = lstsq(a, y, cond=2.**(1-p), lapack_driver="gelsd")
    weights = round_bits(weights, p)
    del a, phi
    xe = np.linspace(-1, 1, eval_points)
    phi = features(xe, centers, gamma, p)
    fit = readout(phi, weights[:-1], weights[-1], p)
    truth = target_values(xe, target)
    residual = fit-truth
    assert np.array_equal(round_bits(weights, p), weights)
    assert np.array_equal(round_bits(centers, p), centers)
    assert np.array_equal(round_bits(phi, p), phi)
    assert np.array_equal(round_bits(fit, p), fit)
    if save_model is not None:
        np.savez(save_model, centers=centers, gamma=gamma, weights=weights, p=p,
                 evaluation_inputs=round_bits(xe, p), output=fit)
    return {"target": target, "width": width, "N": n, "halo": halo, "lambda": float(lam),
            "p": p, "gamma": gamma, "rank": int(rank),
            "unique_stored_centers": len(np.unique(centers)),
            "relative_l2": float(np.linalg.norm(residual)/np.linalg.norm(truth)),
            "linf": float(np.max(np.abs(residual))), "readout_norm": float(np.linalg.norm(weights)),
            "sigma_max": float(singular[0]), "train_points": train_points, "eval_points": eval_points,
            "precision_model": MODEL}
