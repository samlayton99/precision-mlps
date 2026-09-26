"""expC13 shared protocol: formats, the target oracle, grids, the high-precision error meter, the two
network types with their p-bit forward passes, parameter counting and exact model storage.

SPEC.md in this folder is normative; this module implements its sections "Arithmetic", "Data and
the oracle", "Networks" and "Measurement". All model arithmetic goes through pfloat (correctly
rounded +, -, *, /, sqrt in the format, exact negation/abs/comparison/max, and pfloat's tanh built
from those operations). mpmath and wide pfloat formats appear only in the oracle and the meter.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import json
import os
from pathlib import Path

import mpmath as mp
import numpy as np
import pfloat

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(os.environ.get("EXPC13_OUT", ROOT / "results/checkpoint_C_geometry/expC13_five_method_comparison"))

EMIN, EMAX = -958, 959          # one exponent range for every p and every method
REF_BITS = 320                  # error meter; >= p + 64 for every p in the sweep
CHECK_BITS = 640                # stabilization re-measurement of reported errors
PARAM_BUDGET = 3073             # nonzero weights + biases of a 1024-neuron one-hidden-layer network
CHUNK = 1024                    # rows per forward-pass block (memory only; results do not depend on it)


def fmt(p: int) -> pfloat.Format:
    return pfloat.Format(int(p), EMIN, EMAX)


# ---------------------------------------------------------------- targets and the oracle

def _chirp(x):
    return mp.sin(8 * mp.pi * (x + 1) ** 2)


def _runge(x):
    return 1 / (1 + 25 * x * x)


def _sine(x):
    return mp.sin(4 * mp.pi * x)


def _exp(x):
    return mp.exp(x)


TARGETS = {"exp": _exp, "sine": _sine, "runge": _runge, "chirp": _chirp}
TITLES = {"exp": r"$e^{x}$", "sine": r"$\sin(4\pi x)$", "runge": r"$1/(1+25x^2)$",
          "chirp": r"$\sin(8\pi(x+1)^2)$"}


def _mpf(v) -> mp.mpf:
    """An exact value (Fraction, int, float, mpf) as an mpf at the current working precision."""
    if isinstance(v, Fraction):
        return mp.mpf(v.numerator) / v.denominator
    return mp.mpf(v)


def exact_values(target: str, points, bits: int) -> list:
    """f at the given exact points, computed by mpmath with `bits` of working precision."""
    f = TARGETS[target]
    with mp.workprec(bits):
        return [f(_mpf(v)) for v in points]


def _round_checked(values, F: pfloat.Format, bits: int):
    """Round reals known to about 2^-(bits-8) relative into F; ok[i] is False when the uncertainty
    interval of value i contains a rounding boundary (Ziv's test)."""
    out = pfloat.array(values, F)
    with mp.workprec(bits):
        slack = mp.mpf(2) ** (8 - bits)
        lo = pfloat.array([v - abs(v) * slack for v in values], F)
        hi = pfloat.array([v + abs(v) * slack for v in values], F)
    return out, np.asarray(lo == out) & np.asarray(hi == out)


def correctly_rounded(evaluate, n: int, F: pfloat.Format, bits: int) -> pfloat.PArray:
    """Ziv's strategy: evaluate(bits, indices) -> values at that working precision; entries whose
    rounding is not yet certain are re-evaluated at doubled precision (up to 64 times the start)."""
    idx = list(range(n))
    out, ok = _round_checked(evaluate(bits, idx), F, bits)
    todo = [i for i in idx if not ok[i]]
    limit = 64 * bits
    while todo:
        bits *= 2
        if bits > limit:
            raise ArithmeticError(f"value on or within 2^-{bits // 2} of a rounding midpoint")
        part, ok = _round_checked(evaluate(bits, todo), F, bits)
        out[todo] = part
        todo = [i for i, good in zip(todo, ok) if not good]
    return out


def round_values(values, F: pfloat.Format, bits: int) -> pfloat.PArray:
    """Round fixed reals known to about 2^-(bits-8) relative into F; raise if any rounding is uncertain."""
    out, ok = _round_checked(values, F, bits)
    if not ok.all():
        raise ArithmeticError("value too close to a rounding midpoint; raise the precision")
    return out


def sample(target: str, x: pfloat.PArray) -> pfloat.PArray:
    """The oracle: f at the (exact) format values x, correctly rounded once into x's format (Ziv's
    strategy: ambiguous entries are re-evaluated at doubled precision)."""
    pts = list(x.to_fractions().ravel())
    evaluate = lambda bits, idx: exact_values(target, [pts[i] for i in idx], bits)  # noqa: E731
    return correctly_rounded(evaluate, len(pts), x.fmt, x.fmt.p + 96).reshape(x.shape)


def constant(value, F: pfloat.Format) -> pfloat.PArray:
    """An exact number (int, Fraction) or an mpmath expression (callable, evaluated at high
    precision), rounded once into F: universal constants and external hyperparameters."""
    if callable(value):
        def evaluate(bits, idx):
            with mp.workprec(bits):
                return [value() for _ in idx]
        return correctly_rounded(evaluate, 1, F, F.p + 96).reshape(())
    return pfloat.array(Fraction(value), F)


# ---------------------------------------------------------------- grids (exact rationals)

def linspace_points(n: int) -> list[Fraction]:
    """-1 + 2i/(n-1), i = 0..n-1 (training grids, the reporting grid)."""
    return [Fraction(-1) + Fraction(2 * i, n - 1) for i in range(n)]


def midpoint_points(n: int) -> list[Fraction]:
    """-1 + (2i+1)/n, i = 0..n-1 (the validation grid, n prime; disjoint from the reporting grid)."""
    return [Fraction(-1) + Fraction(2 * i + 1, n) for i in range(n)]


def chebyshev_points(n: int) -> list:
    """First-kind Chebyshev nodes cos(pi (k + 1/2) / n) as callables for `constant`/rounding."""
    return [lambda k=k: mp.cos(mp.pi * (k + mp.mpf(1) / 2) / n) for k in range(n)]


def inputs(points, F: pfloat.Format) -> pfloat.PArray:
    """Grid points rounded once into F (callables are evaluated at high precision first)."""
    if points and callable(points[0]):
        def evaluate(bits, idx):
            with mp.workprec(bits):
                return [points[i]() for i in idx]
        return correctly_rounded(evaluate, len(points), F, F.p + 96)
    return pfloat.array(points, F)


VALIDATION_POINTS = 1021   # prime: the midpoints are not commensurate with any method's grid
REPORT_POINTS = 8001


# ---------------------------------------------------------------- the error meter

class Meter:
    """Relative L2 and L-infinity errors of p-bit outputs against f at the exact grid points, computed
    in a REF_BITS format. The meter only reads outputs; nothing flows back into a model."""

    def __init__(self, target: str, points, bits: int = REF_BITS):
        self.target, self.points, self.bits = target, points, bits
        self.R = pfloat.Format(bits, EMIN, EMAX)
        self.truth = pfloat.array(exact_values(target, points, bits + 64), self.R)
        self.norm = pfloat.sqrt(pfloat.sum(self.truth * self.truth))
        self.max = float(np.max(np.abs(self.truth.to_numpy(rounding=True))))

    def _lift(self, out: pfloat.PArray) -> pfloat.PArray:
        if out.fmt.p <= 53:
            return pfloat.array(out.to_numpy(), self.R)
        return out.astype(self.R)

    def errors(self, out: pfloat.PArray) -> dict:
        o = self._lift(out)
        if not np.all(np.isfinite(o.to_numpy(rounding=True))):
            return {"rel_l2": float("inf"), "rel_linf": float("inf")}
        r = o - self.truth
        l2 = pfloat.sqrt(pfloat.sum(r * r)) / self.norm
        # L-infinity needs a few digits only: the maximum of the correctly rounded binary64 residuals
        linf = float(np.max(np.abs(r.to_numpy(rounding=True)))) / self.max
        return {"rel_l2": float(l2.to_numpy(rounding=True)), "rel_linf": linf, "_l2": l2}

    def rel_l2(self, out: pfloat.PArray) -> float:
        return self.errors(out)["rel_l2"]


# ---------------------------------------------------------------- networks

def _nnz(a: pfloat.PArray) -> int:
    return int(np.count_nonzero(np.asarray(a != 0)))


@dataclass
class TanhNet:
    """y(x) = offset + sum_j readout_j tanh(slope_j x + bias_j), j ascending.

    Forward pass (SPEC "Networks"): z = add(mul(slope_j, x), bias_j); t = tanh_p(z); then
    s = offset, s = add(s, mul(t_j, readout_j)) for j = 0..W-1."""
    slope: pfloat.PArray
    bias: pfloat.PArray
    readout: pfloat.PArray
    offset: pfloat.PArray

    kind = "tanh"

    @property
    def fmt(self):
        return self.readout.fmt

    @property
    def width(self) -> int:
        return self.readout.shape[0]

    def params(self) -> int:
        return _nnz(self.slope) + _nnz(self.bias) + _nnz(self.readout) + _nnz(self.offset.reshape(1))

    def neurons(self) -> int:
        return self.width

    def depth(self) -> int:
        return 1

    def features(self, x: pfloat.PArray) -> pfloat.PArray:
        z = x.reshape(-1, 1) * self.slope.reshape(1, -1) + self.bias.reshape(1, -1)
        return pfloat.tanh(z)

    def forward(self, x: pfloat.PArray) -> pfloat.PArray:
        F = self.fmt
        w = pfloat.concatenate([self.offset.reshape(1), self.readout])
        out = []
        for s in range(0, x.shape[0], CHUNK):
            phi = self.features(x[s:s + CHUNK])
            ones = pfloat.ones((phi.shape[0], 1), F)
            # matmul accumulates 0 + 1*offset + phi_0 w_0 + ... left to right: offset first, exactly
            out.append(pfloat.matmul(pfloat.concatenate([ones, phi], axis=1), w))
        return pfloat.concatenate(out)

    def max_abs(self) -> float:
        vals = [pfloat.absolute(a).max() for a in (self.slope, self.bias, self.readout)] + [pfloat.absolute(self.offset)]
        return max(float(v.to_numpy(rounding=True)) for v in vals)

    def arrays(self) -> dict:
        return {"slope": self.slope, "bias": self.bias, "readout": self.readout, "offset": self.offset.reshape(1)}


@dataclass
class ReQULayer:
    """Sparse affine map followed (unless final) by ReQU: out_i = bias_i + sum_t w[i,t] h[idx[i,t]]
    over the neuron's inputs t = 0..fanin-1 in order (padding entries have w = 0 and are skipped)."""
    idx: np.ndarray          # (n_out, fanin) int
    w: pfloat.PArray         # (n_out, fanin)
    bias: pfloat.PArray      # (n_out,)
    mask: np.ndarray         # (n_out, fanin) bool: real connections


@dataclass
class ReQUNet:
    """A deep network with ReQU activation sigma(z) = max(z, 0)^2 on every hidden layer and a linear
    output layer. Input h^0 = x (one channel). Hidden layer: z = W h + b with rounded products and
    sums in the listed input order starting from the bias, then sigma(z) = mul(m, m), m = max(z, 0)."""
    layers: list

    kind = "requ"

    @property
    def fmt(self):
        return self.layers[-1].bias.fmt

    def forward(self, x: pfloat.PArray) -> pfloat.PArray:
        out = []
        for s in range(0, x.shape[0], CHUNK):
            h = x[s:s + CHUNK].reshape(-1, 1)
            for li, L in enumerate(self.layers):
                z = pfloat.ones((h.shape[0], 1), L.bias.fmt) * L.bias.reshape(1, -1)
                for t in range(L.idx.shape[1]):
                    cols = L.mask[:, t]
                    if not cols.any():
                        continue
                    term = h[:, L.idx[cols, t]] * L.w[cols, t].reshape(1, -1)
                    z_cols = z[:, cols] + term
                    z[:, cols] = z_cols
                if li < len(self.layers) - 1:
                    m = pfloat.maximum(z, 0)
                    z = m * m
                h = z
            out.append(h.reshape(-1))
        return pfloat.concatenate(out)

    def params(self) -> int:
        return sum(int(np.count_nonzero(np.asarray(L.w != 0) & L.mask)) + _nnz(L.bias) for L in self.layers)

    def neurons(self) -> int:
        return sum(L.bias.shape[0] for L in self.layers[:-1])

    def depth(self) -> int:
        return len(self.layers) - 1

    @property
    def width(self) -> int:
        return max(L.bias.shape[0] for L in self.layers[:-1])

    def max_abs(self) -> float:
        vals = []
        for L in self.layers:
            if L.mask.any():
                vals.append(float(pfloat.absolute(L.w[L.mask]).max().to_numpy(rounding=True)))
            vals.append(float(pfloat.absolute(L.bias).max().to_numpy(rounding=True)))
        return max(vals)

    def arrays(self) -> dict:
        out = {}
        for i, L in enumerate(self.layers):
            out.update({f"L{i}_idx": L.idx, f"L{i}_w": L.w, f"L{i}_bias": L.bias, f"L{i}_mask": L.mask})
        return out


# ---------------------------------------------------------------- exact storage and replay

def _raw(a):
    return a._v if isinstance(a, pfloat.PArray) else np.asarray(a)


def save_model(path: Path, net, p: int, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {k: _raw(v) for k, v in net.arrays().items()}
    np.savez(path, p=p, kind=net.kind, meta=json.dumps(meta), **arrays)


def load_model(path: Path):
    z = np.load(path, allow_pickle=False)
    F = fmt(int(z["p"]))
    wrap = lambda a: pfloat.PArray._wrap(np.array(a), F)  # noqa: E731
    if str(z["kind"]) == "tanh":
        return TanhNet(wrap(z["slope"]), wrap(z["bias"]), wrap(z["readout"]), wrap(z["offset"]).reshape(()))
    layers = []
    i = 0
    while f"L{i}_w" in z:
        layers.append(ReQULayer(np.array(z[f"L{i}_idx"]), wrap(z[f"L{i}_w"]), wrap(z[f"L{i}_bias"]),
                                np.array(z[f"L{i}_mask"])))
        i += 1
    return ReQUNet(layers)


def same_bits(a: pfloat.PArray, b: pfloat.PArray) -> bool:
    """Bit-for-bit equality of two arrays of one format (NaNs equal, signed zeros distinguished)."""
    if a.fmt != b.fmt or a.shape != b.shape:
        return False
    return np.array_equal(np.asarray(_raw(a)).view(np.uint8), np.asarray(_raw(b)).view(np.uint8))
