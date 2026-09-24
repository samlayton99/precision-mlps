"""Outward-rounded witness bounds for ideal fixed-feature empirical GD.

Binary64 inputs encode an exact empirical problem. This certifies neither
floating-point training trajectories nor continuum/generalization error.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from flint import arb, ctx


def exact(value):
    numerator, denominator = float(value).as_integer_ratio()
    return arb(numerator)/arb(denominator)


def norm_upper(values):
    return sum((abs(value).upper()**2 for value in values), arb(0)).sqrt().upper()


def certify(a, b, c, d, x, y, v, eta, N):
    ctx.prec = 128
    a, b, c, x, y, v = [np.asarray(z, dtype=np.float64) for z in (a, b, c, x, y, v)]
    if a.ndim != 1 or b.shape != a.shape or c.shape != a.shape or x.ndim != 1 or y.shape != x.shape or v.shape != x.shape:
        raise ValueError('Incompatible hidden-parameter or sample/witness shapes')
    if len(x) == 0 or not all(np.all(np.isfinite(z)) for z in (a, b, c, x, y, v, d, eta)):
        raise ValueError('Finite nonempty encoded inputs required')
    if int(N) != N or N < 0 or eta <= 0:
        raise ValueError('Nonnegative integer N and positive eta required')
    step = exact(eta); L = arb(len(a)+1); margin = 2-step*L
    if not margin > 0:
        raise ValueError('Analytic descent condition eta*(W+1) < 2 is required')
    aa, bb, cc = [[exact(z) for z in vector] for vector in (a, b, c)]
    vv = [exact(z) for z in v]; dd = exact(d); root_m = arb(len(x)).sqrt()
    residual, normalized_y = [], []
    atv = [arb(0) for _ in range(len(a)+1)]
    for xf, yf, vi in zip(x, y, vv):
        xi, yi = exact(xf), exact(yf)
        features = [(aj*xi+bj).tanh() for aj, bj in zip(aa, bb)]
        residual.append((sum((cj*feature for cj, feature in zip(cc, features)), dd)-yi)/root_m)
        normalized_y.append(yi/root_m)
        for j, feature in enumerate(features):
            atv[j] += vi*feature
        atv[-1] += vi
    atv = [value/root_m for value in atv]
    correlation = abs(sum((vi*ri for vi, ri in zip(vv, residual)), arb(0)))
    v_norm, y_norm = norm_upper(vv), norm_upper(normalized_y)
    if not v_norm > 0 or not y_norm > 0:
        raise ValueError('Nonzero witness and target required')
    r_norm, atv_norm = norm_upper(residual), norm_upper(atv)
    travel_factor = (arb(int(N))*step/margin).sqrt().upper()
    numerator = correlation-atv_norm*r_norm*travel_factor
    floor = ((numerator.lower()/(v_norm*y_norm).upper()).lower()
             if numerator.lower() > 0 else arb(0))
    intervals = dict(correlation=correlation, adjoint_witness_norm_upper=atv_norm,
                     initial_residual_norm_upper=r_norm, witness_norm_upper=v_norm,
                     normalized_target_norm_upper=y_norm, travel_factor_upper=travel_factor,
                     descent_margin=margin, numerator=numerator, relative_error_floor_lower=floor)
    return dict(precision_bits=ctx.prec, width=len(a), samples=len(x), through_update=int(N),
                eta=float(eta), eta_exact_ratio=list(float(eta).as_integer_ratio()),
                analytic_eigenvalue_upper=len(a)+1,
                intervals={key: value.str(45) for key, value in intervals.items()},
                certified_above={label: bool(floor > arb(1)/denominator)
                                 for label, denominator in (('0.01', 100), ('0.001', 1000), ('0.0001', 10000))},
                scope='Every integer update 0<=n<=N of exact-arithmetic frozen-feature GD for the encoded empirical problem; not FP64-training or continuum certification')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    with np.load(args.input) as source:
        inputs = {key: source[key].item() if key in ('d', 'eta') else source[key]
                  for key in ('a', 'b', 'c', 'd', 'x', 'y', 'v', 'eta')}
        inputs['N'] = source['N' if 'N' in source else 'updates'].item()
        result = certify(**inputs)
    result['input_sha256'] = hashlib.sha256(args.input.read_bytes()).hexdigest()
    result['helper_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
