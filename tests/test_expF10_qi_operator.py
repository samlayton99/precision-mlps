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
