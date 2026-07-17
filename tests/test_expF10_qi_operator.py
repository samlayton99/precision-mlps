import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT / "experiments" / "expF08_darcy_sweep"))
sys.path.append(str(REPO_ROOT / "experiments" / "expF10_qi_operator"))

import qi_codec as qc

DARCY = "/scr/cdeng/continuous-mlps/data/fno_datasets_jax/darcy_train_16_jax.npz"


def _smooth(P):
    return np.sin(np.pi * P[:, 0]) * np.sin(np.pi * P[:, 1])


def test_codec_roundtrips_smooth_field():
    codec = qc.QICodec(W=576, lam=0.25)
    g = codec.grid(32)
    c = codec.encode(_smooth(g), 32)
    assert codec.rel_l2(codec.decode(c, g), _smooth(g)) < 1e-7


def test_codec_is_resolution_transferable():
    """Encode on 32^2, decode on 64^2 -- the property config A relies on."""
    codec = qc.QICodec(W=576, lam=0.25)
    c = codec.encode(_smooth(codec.grid(32)), 32)
    g64 = codec.grid(64)
    assert codec.rel_l2(codec.decode(c, g64), _smooth(g64)) < 1e-6


def test_rough_darcy_reconstruction_is_bounded():
    codec = qc.QICodec(W=576, lam=0.25)
    a = np.load(DARCY)["x"][0].astype(np.float64).ravel()
    c = codec.encode(a, 16)
    err = codec.rel_l2(codec.decode(c, codec.grid(16)), a)
    assert 1e-3 < err < 1e-1     # rough: represented, not exact (probe: ~4.5e-2)


import torch
import fno2d


def test_fno_forward_shape_and_backward():
    net = fno2d.FNO2d(width=16, modes=8, n_layers=3)
    x = torch.randn(2, 1, 48, 48, requires_grad=True)
    y = net(x)
    assert y.shape == (2, 1, 48, 48)
    y.sum().backward()
    assert x.grad is not None


import data as dd


def test_load_darcy_downsamples():
    a, u = dd.load_darcy("train", n=8, res=32)
    assert a.shape == (8, 32, 32) and u.shape == (8, 32, 32)
    assert np.isfinite(a).all() and np.isfinite(u).all()


import models as mo


def test_models_forward_backward():
    codec = qc.QICodec(W=128, lam=0.25)   # small W for a fast test
    Phi_out = codec.basis(codec.grid(16))  # [256, D]
    # A: coeff MLP
    A, kindA = mo.build_model("A", D=codec.D, Phi_out=Phi_out)
    assert kindA == "coeff"
    ca = torch.randn(4, codec.D)
    ua = A(ca)
    assert ua.shape == (4, 16 * 16)
    ua.sum().backward()
    # C: plain FNO (field in, field out)
    C, kindC = mo.build_model("C", fno_kw=dict(width=8, modes=6, n_layers=2))
    assert kindC == "field"
    xf = torch.randn(4, 1, 16, 16)
    yf = C(xf)
    assert yf.shape == (4, 1, 16, 16)


import run as g10


def test_smoke_train_one_config_returns_finite_loss():
    """Tiny end-to-end: train config C for 2 epochs on 16 instances at 16^2."""
    cfg = g10.SMOKE_CFG
    rec = g10.train_eval("C", cfg)
    assert np.isfinite(rec["test_rel_l2"])
    assert rec["test_rel_l2"] > 0
