"""Strict p-bit construction/inference. FP64 arrays only store p-bit values."""
from __future__ import annotations

import ctypes
import hashlib
from pathlib import Path
import subprocess
import sys

import gmpy2 as g
import numpy as np

from experiments.expC09_bandwidth_figures.strict_precision import readout
from experiments.expC12_mhaskar_comparison.construction import TanhNetwork, round_bits

OUT = Path(__file__).resolve().parents[2] / "results/checkpoint_C_geometry/expC12_mhaskar_comparison/strict"
_LIB = None


def library():
    global _LIB
    if _LIB is not None:
        return _LIB
    source = Path(__file__).with_name("pbit_kernels.c")
    package = Path(g.__file__).parent
    libs = package.parent / "gmpy2.libs"
    mpfr = next(iter(list(libs.glob("libmpfr*.dylib"))+list(libs.glob("libmpfr*.so*"))))
    cache = OUT / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    fingerprint = hashlib.sha256(source.read_bytes()+g.mpfr_version().encode()).hexdigest()[:16]
    compiled = cache / f"pbit_{fingerprint}.so"
    if not compiled.exists():
        subprocess.run(["cc", "-O3", "-shared", "-fPIC", "-I", str(package), str(source),
                        str(mpfr), "-o", str(compiled)], check=True)
        if sys.platform == "darwin":
            install_name = subprocess.check_output(["otool", "-D", str(mpfr)], text=True).splitlines()[1].strip()
            subprocess.run(["install_name_tool", "-change", install_name, str(mpfr), str(compiled)], check=True)
    lib = ctypes.CDLL(str(compiled))
    array = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
    lib.affine_features.argtypes = [ctypes.c_size_t, ctypes.c_size_t, array, array, array, ctypes.c_uint, array]
    lib.polynomial_data.argtypes = [ctypes.c_size_t, ctypes.c_uint, array, array, ctypes.c_double,
                                    ctypes.c_uint, array, array, array]
    lib.stencil_network.argtypes = [ctypes.c_uint, array, array, ctypes.c_double, ctypes.c_uint, array, array]
    for name in ("affine_features", "polynomial_data", "stencil_network"):
        getattr(lib,name).restype = ctypes.c_int
    _LIB = lib
    return lib


def features(x, slope, bias, p):
    x, slope, bias = [np.ascontiguousarray(a, dtype=np.float64) for a in (x, slope, bias)]
    out = np.empty((len(x),len(slope)), dtype=np.float64)
    if library().affine_features(len(x),len(slope),x,slope,bias,p,out):
        raise MemoryError("MPFR affine features")
    return out


def evaluate(model, x, p):
    return readout(features(x,model.slope,model.bias,p), model.readout,model.offset,p)


def polynomial_data(nodes, labels, degree, p):
    with g.context(precision=p, round=g.RoundToNearest):
        bias = float(g.log(g.mpfr(2))/g.mpfr(2))
    c, poly, t = np.empty(degree+1), np.empty((degree+1,degree+1)), np.empty(degree+1)
    nodes, labels = [np.ascontiguousarray(a,dtype=np.float64) for a in (nodes,labels)]
    if library().polynomial_data(len(nodes),degree,nodes,labels,bias,p,c,poly,t):
        raise MemoryError("MPFR polynomial construction")
    return c, poly, t, bias


def construct(poly, taylor, degree, step, bias, p):
    slope, weights = np.empty(2*degree+1), np.empty(2*degree+1)
    a = np.ascontiguousarray(poly[:degree+1],dtype=np.float64)
    t = np.ascontiguousarray(taylor[:degree+1],dtype=np.float64)
    status = library().stencil_network(degree,a,t,step,p,slope,weights)
    if status:
        raise FloatingPointError(f"p-bit Mhaskar construction status {status}")
    return TanhNetwork(slope,np.full(slope.size,float(round_bits(bias,p))),weights,np.float64(0.))


def quill_geometry(width, halo, p):
    """Compute centers, spacing, predicted lambda, slope and biases at p bits."""
    n = width-2*halo-1
    with g.context(precision=p, round=g.RoundToNearest):
        q = g.mpfr
        one, two, pi = q(1), q(2), g.const_pi()
        spacing = two/q(n)
        omega = q(16)*pi
        theta = spacing*omega
        tolerance = g.mul_2exp(one,1-p)
        budget = g.log(tolerance)

        def log_khat(xi):
            v = abs(xi)*pi/two
            if not v:
                return q(0)
            return g.log(two*v)-v-g.log(-g.expm1(-two*v))

        def score(lam):
            minus, plus = two*pi-theta, two*pi+theta
            a = g.log(theta/minus)+log_khat(minus/lam)
            b = g.log(theta/plus)+log_khat(plus/lam)
            hi, lo = max(a,b), min(a,b)
            return hi+g.log1p(g.exp(lo-hi))-log_khat(theta/lam)

        lo, hi = q("0.03"), q(3)
        if not score(lo)<budget<=score(hi):
            raise ValueError("p-bit lambda search did not bracket threshold")
        for _ in range(100):
            mid = (lo+hi)/two
            if mid==lo or mid==hi:
                break
            if score(mid)<budget:
                lo=mid
            else:
                hi=mid
        lam=lo
        gamma=lam/spacing
        centers=[-one+q(j)*spacing for j in range(-halo,n+halo+1)]
        slopes=np.full(width,float(gamma))
        biases=np.array([float(-gamma*c) for c in centers])
        meta={"N":n,"spacing":float(spacing),"lambda":float(lam),
              "rule_log_score":float(score(lam)),"rule_log_budget":float(budget),
              "distinct_centers":len(set(centers)),"distinct_biases":len(set(biases))}
    return slopes,biases,meta
