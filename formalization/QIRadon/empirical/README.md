# Construction experiments

These are known-function constructions, with no training or fitted readouts.
The tanh experiments use the explicit finite-contour halo correction, which
is part of the main mathematical construction and proof. Its sharper scalar
width is not yet formalized in this Lean package.

Run with Python, NumPy, SciPy, and Matplotlib:

```
python empirical/run.py --suite all
python empirical/plot.py
```

`config.json` fixes the targets and resolution schedules before the production
run. Results retain all configurations, including loss of accuracy. The test
points are independent of the construction. Sampled errors are not certified
continuous suprema. Dashed width curves are analytic upper bounds, with no
constants fitted to the errors. The precision test isolates readout storage
precision; it is not a simulation of the entire coefficient calculation in
reduced precision.

The scaling figure uses only `exp(-3*sum(x_k^2))` in dimensions 1 through 4.
A separate robustness figure uses five Gaussian-based analytic families and
has no precision panel. Dashed precision curves use the exact-construction
bound plus readout rounding, with no measured error floor added. The other
tests check a nonanalytic target with the general spectral construction,
ReLU's algebraic limit, and direction/center allocation.

`python empirical/lambda_consistency.py` checks the exact connection between
the tanh Fourier-ratio score and the corrected pole bound, then compares five
lambda choices on the canonical Gaussian profiles in d=1..4 at two widths.
These are scalar-profile tests, not full angular reconstructions. The encoder's
optional lambda argument preserves the default geometry used in the main figures.

See `findings/scratchpad.md` for run-by-run progress, `results/summary.md` for
results, and `run.py` for the reproducible construction and figures.
