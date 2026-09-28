"""The reduced frozen loss must reproduce physical-coordinate optimization."""
import numpy as np
import jax.numpy as jnp
import torch

from experiments.expD34_readout_race import frozen_readout_scale as fr, mechanism


def test_factored_adam_matches_direct_torch_updates():
    centers, x, y, A, R, rhs = fr.dictionary(16, .5, 128)
    # Avoid an exactly odd target's unstable, zero-gradient bias symmetry.
    # Tiny roundoff can pick opposite symmetry-related Adam oscillations there.
    y = y+.13*x*x+.07
    Q, _ = np.linalg.qr(A/np.sqrt(len(x)), mode='reduced')
    rhs = Q.T @ (y/np.sqrt(len(x)))
    c = torch.zeros(A.shape[1], dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.Adam([c], lr=.002, betas=(.9,.999), eps=1e-8, foreach=False)
    At, yt = torch.tensor(A), torch.tensor(y)
    for _ in range(100):
        optimizer.zero_grad()
        ((At @ c-yt).square().mean()/2).backward()
        optimizer.step()
    actual = fr.advance(fr.initial(1, A.shape[1]), jnp.array(R[None]), jnp.array(rhs[None]),
                        jnp.array([.002]), 100)
    np.testing.assert_allclose(actual[0][0], c.detach().numpy(), rtol=1e-10, atol=2e-14)


def test_spectral_gd_matches_direct_updates():
    centers, x, y, A, R, rhs = fr.dictionary(16, 1., 128)
    gamma = 8.
    _, _, cs = mechanism.frozen_curves(np.full(len(centers), gamma), -gamma*centers,
                                      x, y, x, y, horizons=[200])
    direct = np.zeros(A.shape[1])
    for _ in range(200):
        direct -= .002*(A.T @ (A @ direct-y)/len(x))
    np.testing.assert_allclose(cs[0], direct, rtol=1e-11, atol=2e-14)


def test_physical_weight_scale_is_separate_from_bias():
    centers, x, y, A, _, _ = fr.dictionary(16, .5, 128)
    h = 2/16
    c = np.r_[h*np.ones(len(centers)), 17.]
    row = fr.metrics(c, centers, 16, 4., x, y, A)
    assert row['all_rms_over_h']==1.
    assert row['halo_max_over_h']==1.
    assert row['bias']==17.
