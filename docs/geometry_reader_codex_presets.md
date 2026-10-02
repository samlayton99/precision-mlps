# Geometry Reader — Codex preset evidence

Six numeric presets are bundled in `experiments/expG02_geometry_reader_codex/presets/catalog.json`. The catalog is independent of the ignored historical result files once exported. `presets/build_catalog.py` reconstructs the formula presets, copies the saved arrays, computes a fresh fp64 readout, and records the checkpoint SHA-256 hashes. It performs no training.

The requested **Xavier-to-sine-floor run exists**. The source is `results/checkpoint_D_optimizers/expD04_varpro/varpro_corrected/06_sine_xavier_seeds/geometry_sine_xavier_N256_s0.npz`, with matching `.json` metadata and `trace_sine_xavier_N256_s0.json`. It starts from ordinary Xavier slopes with zero hidden biases. Its initial least-squares error is `0.05334829700828216`; its final error is `6.654869951238788e-15`. A fresh solve of the exported raw geometry reproduces the latter value on this machine. The method was 200 VarPro Adam warmup steps followed by Gauss–Newton, capped at 40 iterations and stopping at 28. This is not evidence that ordinary joint Adam alone reaches the floor. The longer 1,000-Adam/300-GN comparison finishes at `3.2673950700389988e-12`.

| Preset | Numeric source | Width | Fresh sine refit relative L2 |
|---|---|---:|---:|
| Clean QI, λ=0.25 | Uniform QI formula with N=144 interior points, R=12 | 168 | 4.60e-14 |
| Xavier, original floor-run seed | expD04 `theta0` | 461 | 5.33e-2 |
| Xavier → sine floor | expD04 final `theta` | 461 | 6.65e-15 |
| Scaled Xavier, mean λ=0.25 | expD28 saved initial `a[0], b[0], v[0]` | 177 | 4.37e-6 |
| Uniform plus six soft neurons | expC06 threshold 0.75, uniform centers, interpolation endpoint | 269 | 2.80e-15 |
| Regular clusters of eight | expC04 placement routine, compression ratio 0.75 | 269 | 3.20e-8 |

These fresh metrics use each entry's saved settings and a `1e-13` relative singular-value cutoff. Their purpose is to verify the export and show useful differences, not establish new general research claims. The original expD28 evaluation uses midpoint samples; the catalog's fresh comparison evaluates on 8,192 endpoint-inclusive samples after fitting the same 1,024 midpoint training samples. The other catalog entries use 2,003 endpoint-inclusive training samples and 4,001 endpoint-inclusive evaluation samples. Last digits can vary with LAPACK.

## Coordinates and readouts

The model is `sum(v_j * tanh(a_j*x + b_j)) + output_bias`. Centers are `c_j=-b_j/a_j`; displayed dimensionless scales are `lambda_j=abs(a_j)*h`, where h is the **nominal** interior spacing. The app counts N interior points and uses `h=2/(N-1)`; historical source files count intervals and use `h=2/N`. Their saved spacing is retained, and the catalog supplies `n_interior` separately. It is not a local gap between dragged centers. Slopes and hidden biases retain their original signs and neuron ordering. Flipping a negative tanh slope and its readout sign preserves the function, but it also changes the coordinates used by Adam; the catalog therefore preserves the raw arrays.

The learned floor checkpoint stores `theta0`, `theta`, and `v`. Its saved `v` was a least-squares/VarPro readout, not a trained Adam readout. The original output bias was omitted. The catalog retains `readout.saved` with `saved_kind="least_squares"` and a null saved bias, and separately provides a fresh `readout.solved` plus `solved_bias`. Xavier initial readouts are included where they can be recovered exactly. No missing Adam trajectory or readout is invented.

For the tanh derivative guide, the existing checkpoint-A application uses `(h/2) f'(c_j)`. This guide corresponds to positive-oriented tanh features; when plotting raw signed readouts, multiply the guide by `sign(a_j)`, or show sign-corrected readouts consistently. A one-off arbitrary amplitude fit should not be mistaken for the repository's derivative rule.

## Cleaning

The app’s clean preset now uses Sam’s requested `R=max(10,sqrt(N))`: N=144 interior points, R=12, W=168. Other historical presets retain their original arrays.

The historical reference used by expC05 has uniform centers `-1+k*h` for `k=-halo,...,N+halo`, with `a=0.25/h` and `b=-a*c`. `default_halo` chooses `max(ceil(35/(2*0.25)), int(0.4*N))`; for N=128 this is 70 centers per side. The clean operation should sort/match centers to their uniform destinations before interpolation and preserve slope orientation. Interpolating positive magnitudes in the existing sign pattern avoids dragging half the features through zero bandwidth, a failure mode explicitly isolated in expC05.

The floor geometry is deliberately unusual: 242/461 centers fall inside [-1,1], and their median absolute slope is only `2.1125767403065643`, compared with the clean λ=0.25 slope of 32 at the source’s N=256 intervals (257 interior points). Uniformity and λ=0.25 are useful reference constructions, not a necessary characterization of every geometry that can fit this sine target.

## Additional evidence found

The older expD02 tuning records also contain scaled-Xavier sine results. `stage1_tune.json` reports a 64× inner-scale, 50,000-step N=512 run with best refitted error `1.5602192814819956e-13`, while its trained readout ends at `4.7045331567702983e-4`. Those records contain metrics rather than the needed checkpoint geometry, so they were not substituted for the actual saved expD04 floor state.

The soft-neuron recipe is reproduced from `experiments/expC06_soft_neuron_interp/run_threshold.py`: protect the six scaled-Xavier magnitudes below 0.75, set the other magnitudes equal, and normalize their sum so mean λ remains 0.25. The original writeup warns that reported improvements near the numerical floor require more seeds. The catalog presents one concrete example, without claiming a universally better initialization.
