import importlib.util
from pathlib import Path

import mpmath as mp
import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "c10_note_geometry", Path(__file__).with_name("note_geometry.py"))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.mark.parametrize("n,radius,lam,delta", [
    (64, 8, .25, .25), (128, 12, .13, .19),
    (256, 16, .05, .25), (512, 23, .25, .19)])
def test_outward_scales_enclose_independent_high_precision_formula(n, radius, lam, delta):
    alpha, scales = module.interval_envelopes(n, radius, lam, delta)
    with mp.workdps(120):
        h, ll, dd = mp.mpf(2) / n, mp.mpf(lam), mp.mpf(delta)
        zeta = mp.exp(-2 * ll)
        m = (radius + 1) // 2
        pp = [mp.mpf(1)]
        for j in range(1, m + 1):
            pp.append(mp.fprod(1 - zeta**k for k in range(1, j + 1)))
        aa = [h / (2 * (dd - mp.pi * h / (2 * ll)))] * (n + 2 * radius + 1)
        for i in range(1, m + 1):
            li = zeta ** (mp.mpf(i * (i + 1) - 1) / 2) / (pp[i - 1] * pp[m - i])
            li *= mp.fprod(1 + zeta ** (mp.mpf(j) - mp.mpf(".5"))
                          for j in range(1, m + 1) if j != i)
            addition = h * (mp.pi / (2 * ll) + 4 * mp.log(2) / mp.pi) * li / (2 * dd)
            aa[i - 1] += addition
            aa[-i] += addition
        aa = [1 + sum(aa)] + aa
        for actual_alpha, actual_scale, exact in zip(alpha, scales, aa):
            assert mp.mpf(float(actual_alpha)) >= exact
            assert mp.mpf(float(actual_scale)) >= mp.sqrt(exact)
            assert (mp.mpf(float(actual_scale)) / mp.sqrt(exact) - 1) < mp.mpf("3e-16")


def test_invalid_neighborhood_is_not_clipped_into_a_valid_envelope():
    assert module.interval_envelopes(64, 8, .1, .25) is None


def test_outer_correction_assignment_matches_merged_pr():
    # At the PR's reference bandwidth and analyticity radius, compare the
    # independent interval implementation with the original geometry function.
    # Execute just this pure constructor to avoid initializing JAX for a check.
    import ast
    from dataclasses import dataclass
    import math
    source = module.base.ROOT / "experiments/expD06_fixed_center_scales/core.py"
    tree = ast.parse(source.read_text())
    keep = [node for node in tree.body
            if isinstance(node, (ast.ClassDef, ast.FunctionDef))
            and node.name in ("Geometry", "geometry")]
    namespace = dict(np=np, math=math, dataclass=dataclass, LAMBDA_REF=.25)
    exec(compile(ast.Module(body=keep, type_ignores=[]), str(source), "exec"), namespace)
    for n in (64, 128, 256, 512):
        original = namespace["geometry"](n)
        actual, _ = module.interval_envelopes(n, original.radius, .25, .25)
        np.testing.assert_allclose(actual, original.alpha, rtol=2e-14)
