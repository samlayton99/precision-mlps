"""expD18 sanity gate: the QI optimizer's solve core reaches the fp64 floor
on a frozen geometry, on a synthetic noiseless task."""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


qi_opt = _load("t18_qi_opt", REPO / "experiments" / "expD18_qi_opt_tabular" / "qi_opt.py")
f04_model = _load("t18_f04_model", REPO / "experiments" / "expF04_qi_init_real_data" / "model.py")


def test_validated_solve_reaches_fp64_floor_on_frozen_geometry():
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(0)
    # noiseless 1-D sine; geometry = the repo construction (uniform grid + halo,
    # gamma = lambda*/h). This is the halo'd geometry, so the solve itself is
    # what is being tested. (The expF04 ridge init has no halo and floors near
    # 1e-7 by geometry, which is a property of the init, not the solver.)
    n = 2003
    x = torch.linspace(-1, 1, n).unsqueeze(1)
    y = torch.sin(math.pi * x).squeeze(1)
    n_int, halo = 128, 64
    h = 2.0 / n_int
    gamma = 0.25 / h
    centers = torch.arange(-halo, n_int + halo + 1, dtype=torch.float64) * h - 1.0
    width = centers.numel()
    model = f04_model.SimpleMLP(1, width, 1, activation="tanh").double()
    with torch.no_grad():
        model.fc1.weight.fill_(gamma)
        model.fc1.bias.copy_(-gamma * centers)

    ho = torch.zeros(n, dtype=torch.bool)
    ho[5::10] = True
    rel, tag = qi_opt.probe(model, x[~ho], y[~ho], x[ho], y[ho], x, y)
    assert rel < 1e-11, f"frozen-geometry solve should hit the fp64 floor, got {rel:.3e} ({tag})"


def test_expf04_ridge_init_geometry_floor_documented():
    # the expF04 halo-less ridge init at d=1: the probe lands orders above the
    # fp64 floor but far below standard init -- records the init's own limit
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(0)
    n = 2003
    x = torch.linspace(-1, 1, n).unsqueeze(1)
    y = torch.sin(math.pi * x).squeeze(1)
    model = f04_model.SimpleMLP(1, 256, 1, activation="tanh").double()
    f04_model.qi_ridge_init_(model, x, centers_per_dir=256, uniform_centers=True)
    ho = torch.zeros(n, dtype=torch.bool)
    ho[5::10] = True
    rel, _ = qi_opt.probe(model, x[~ho], y[~ho], x[ho], y[ho], x, y)
    assert 1e-9 < rel < 1e-4, f"expected the halo-less geometry limit, got {rel:.3e}"


def test_damped_correction_respects_floor():
    # with alpha ~ 1 the correction must barely move the readout;
    # with alpha ~ 1e-14 it must solve to the floor
    rng = np.random.default_rng(0)
    A = rng.standard_normal((500, 40))
    v_true = rng.standard_normal(40)
    y = A @ v_true
    v0 = np.zeros(40)
    v_hard = qi_opt.damped_correction(A, y, v0, alpha=1e-14)
    assert np.linalg.norm(A @ v_hard - y) / np.linalg.norm(y) < 1e-10
    v_soft = qi_opt.damped_correction(A, y, v0, alpha=1.0)
    assert np.linalg.norm(v_soft) < 0.6 * np.linalg.norm(v_hard)
