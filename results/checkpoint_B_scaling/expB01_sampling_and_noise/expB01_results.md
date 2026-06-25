# expB01 -- Sampling and noise in the least-squares readout (1D)

**Status: conclusions approved by Sam.**

## TL;DR

- Precision is governed by the centers, not the sample points: uniform centers reach the floor with any sampling; random centers plateau with any sampling.
- Jittering the $x$ sample positions is harmless. $y$-noise sets a hard floor near the noise magnitude, recoverable only at the $1/\sqrt{n}$ rate (according to statistical theory).

## Question

Is reaching machine precision controlled by the centers or the sample points, and how does uniform-center recovery degrade under $x$-jitter and $y$-noise?

## Experiment design

All conditions are overdetermined least squares ($W$ centers $<$ samples $<$ eval points). Train/eval grids are chosen *prime* ($N_\text{train}=1031$, $N_\text{eval}=7919$) so they never align -- an aligned eval grid would sample near-zero error at the training points and hide the true error. Centers reuse expC04's sizing (uniform QI grid + halo fixes $W$, span, $h=2/N$; random centers are uniform draws over the same span); every condition uses $\gamma=\lambda/h$ at each swept $\lambda$, and error is always measured against the *clean* target. Three studies: (1) geometry x sampling, clean ({uniform, random} centers x {uniform, random} samples); (2) uniform centers under $x$-jitter (random sample positions) and $y$-noise (additive Gaussian, std $10^{-3}$); (3) sample-count scaling at fixed geometry ($N=64$, $\lambda=0.25$), sweeping samples over $\{256,\dots,131072\}$ across noise levels $\sigma\in\{0,10^{-8},\dots,10^{-2}\}$. Best over the $\lambda$ sweep, mean over 3 seeds.

**Code & data.** `experiments/expB01_sampling_and_noise/` (`run.py`, `config.yaml`). Figures: `error_vs_width_geometry.png`, `error_vs_width_noise.png`, `error_vs_samples_noise.png`.

## Results

- **Centers decide precision.** Both uniform-center conditions sit at the floor (~$10^{-13}$) regardless of sample type; both random-center conditions plateau ~$10^{-10}$--$10^{-11}$ regardless. Random samples neither rescue random centers nor spoil uniform ones.
- **$x$-jitter is harmless** -- random sample positions on clean $y$ stay at the floor.
- **$y$-noise sets a hard floor** near the noise level, ~10 orders above the clean floor; adding data recovers it only as $\sigma\,n^{-1/2}$ (slope ~$-0.5$, no bending toward the floor). Reaching machine precision from $10^{-2}$ noise would need ~$10^{23}$ samples.

### Figures

- **`error_vs_width_geometry.png`** -- two panels (rel $L_2$, $L_\infty$) vs width; four curves = {uniform, random} centers x {uniform, random} samples. The two uniform-center curves sit at the floor, the two random-center curves plateau above it, independent of sample type.
- **`error_vs_width_noise.png`** -- all uniform centers; clean curves at the floor, noisy curves at a much higher plateau near the noise magnitude.
- **`error_vs_samples_noise.png`** -- error vs sample count, one line per noise level: each noisy line is a straight $1/\sqrt{n}$ descent (evenly spaced by $\sigma$); the clean line stays flat at the floor.

## Conclusions

Reaching machine precision is a property of the centers, not the samples; $x$-jitter is harmless; $y$-noise sets a hard $\sigma\,n^{-1/2}$ floor. The construction's noise-free assumption is load-bearing for any precision claim. (Approved by Sam.)

## Open questions

None -- the live precision question is the geometry (Checkpoint C).
