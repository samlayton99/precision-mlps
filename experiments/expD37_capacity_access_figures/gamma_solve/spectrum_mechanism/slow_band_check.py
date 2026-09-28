"""Apply the collaborator note's positive-band test to our existing geometry.

No GD trajectory is run. These are FP64 spectral evaluations, not interval
certificates. This uses no polynomial approximation.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.linalg import svd

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from experiments.expD37_capacity_access_figures.gamma_solve.core import (
    Geometry, features, target, weights, hitting_time,
)


def main():
    geometry = Geometry()
    x, c = geometry.arrays()
    y = target(x, "mixed") / np.sqrt(len(x))
    rows = []
    for gamma in [4., 8., 16., 64.]:
        J = features(x, c, gamma)/np.sqrt(len(x))
        u, s, vh = svd(J, full_matrices=False, lapack_driver="gesvd")
        p = weights(u, y)
        rates = np.r_[.5*(s/s[0])**2, 0.]
        bands = []
        for lower, upper in [(1e-9, 1e-6), (1e-14, 1e-11)]:
            fraction = float(p[(rates > lower) & (rates <= upper)].sum())
            necessary = (int(np.ceil(np.log(np.sqrt(fraction)/.01)/(-np.log1p(-upper))))
                         if fraction > 1e-4 else 0)
            bands.append(dict(lower_rate=lower, upper_rate=upper,
                              target_energy_fraction=fraction,
                              necessary_steps_to_one_percent=necessary))
        keep = rates[:-1] >= 1e-14
        coefficient = vh[keep].T @ ((u[:, keep].T@y)/s[keep])
        direct_fit_error = float(np.linalg.norm(J@coefficient-y)/np.linalg.norm(y))
        rows.append(dict(gamma=gamma, bands=bands,
                         spectral_forecast_steps_to_one_percent=hitting_time(rates, p),
                         rank32_ratio=float(2*rates[31]),
                         explicit_resolved_readout_relative_error=direct_fit_error,
                         explicit_readout_norm=float(np.linalg.norm(coefficient)),
                         retained_readout_modes=int(keep.sum()),
                         svd_relative_reconstruction_error=float(np.linalg.norm(J-(u*s)@vh)/np.linalg.norm(J))))
    result = dict(N=geometry.N, m=len(x), W=len(c), halo_per_side=geometry.halo,
                  target="sin(2*pi*x)+0.5*sin(6*pi*x)+0.25*sin(10*pi*x)",
                  learning_rate="0.5/lambda_max(K)", relative_tolerance=.01,
                  gd_executed=False, interval_certified=False,
                  band_selection="The 1e-9..1e-6 band matches the collaborator note; the 1e-14..1e-11 band was selected during this diagnostic.",
                  rows=rows)
    out = ROOT / "results/checkpoint_D_optimizers/expD37_capacity_access_figures/gamma_solve/spectrum_mechanism/data/slow_band_check.json"
    out.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
