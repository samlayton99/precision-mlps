"""QI-bandwidth spectral initialization for expF11 method (3).

The QI-resample operator a -> decode(encode(a)) = Phi Phi^+ a is low-pass; its
empirical radial frequency gain is used to shape the FNO's spectral-conv init so
the net starts biased toward the QI's frequency content (a simplification of a
full per-mode operator fit; the low-pass envelope is what carries the signal)."""
from __future__ import annotations

import numpy as np
import torch


def qi_resample_gain(codec, res, n_probe=48, seed=0):
    """Radial frequency gain |F(resample a)| / |F(a)| of the QI-resample, on
    random fields, as a 1-D profile indexed by integer radius."""
    rng = np.random.default_rng(seed)
    Pinv = codec.pinv(res)
    Phi = codec.basis(codec.grid(res))
    kx = np.fft.fftfreq(res) * res
    R = np.round(np.sqrt(kx[:, None] ** 2 + kx[None, :] ** 2)).astype(int)
    num = np.zeros(res)
    den = np.zeros(res)
    for _ in range(n_probe):
        a = rng.standard_normal((res, res))
        ar = (Phi @ (Pinv @ a.ravel())).reshape(res, res)
        fa, far = np.abs(np.fft.fft2(a)), np.abs(np.fft.fft2(ar))
        for r in range(res):
            m = R == r
            if m.any():
                num[r] += far[m].mean()
                den[r] += fa[m].mean()
    g = np.where(den > 0, num / np.maximum(den, 1e-12), 0.0)
    return g / max(g[0], 1e-12)          # normalize DC gain to 1


def qi_spectral_init(net, codec, res=64):
    """Scale each spectral-conv layer's per-mode weights by the QI-resample
    radial gain at that mode's (kx,ky), in place."""
    g = qi_resample_gain(codec, res)
    with torch.no_grad():
        for sp in net.specs:
            m = sp.modes
            env = np.zeros((m, m), dtype=np.float32)
            for i in range(m):
                for j in range(m):
                    r = int(round((i ** 2 + j ** 2) ** 0.5))
                    env[i, j] = g[min(r, len(g) - 1)]
            e = torch.tensor(env, device=sp.w1.device)[None, None, :, :, None]
            sp.w1.mul_(e)
            sp.w2.mul_(e)
