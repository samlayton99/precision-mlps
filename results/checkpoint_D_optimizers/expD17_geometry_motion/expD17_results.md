# expD17 -- Does the geometry even move? Plain-Adam geometry motion, QI init vs standard

**Status: draft -- pending Sam's review.**

## TL;DR

- **No hard gradient death.** The bitwise-zero geometry gradient lasts exactly one step (the zero readout), then the geometry moves in every cell. Half the total motion lands within the first $\sim$50-1000 steps, then the run coasts.
- **The QI geometry moves far less in its own units**: relative drift 0.2-2.4% (1-D) and 2-86% (2-D/PINN) vs 50-1500% for standard init. Caveat: $\|g_0\|$ is 15-200$\times$ larger for QI ($\gamma=O(N)$), so *absolute* motion is comparable; the relative lens is the stated metric.
- **Which way the motion goes depends on where the geometry starts.** At the floor (all 1-D, gauss\_bump) motion *damages* it by 2-7 orders (worst: sine\_8pi $5.9\times10^{-14}\to1.2\times10^{-7}$). Off the floor (sine2d, mixed2d) the same gradient flow *improves* the QI geometry 150-1000$\times$. The geometry is not frozen; it refines where residual signal exists.
- **Inverse problems work from QI init, and work better.** Parameter recovery beats standard init 33$\times$ (burgers $\nu$) and 7$\times$ (bratu $\lambda$); field error is 4-22$\times$ better on all three. allencahn\_k defeats both arms.
- Tabular: both arms move plenty and land at the same noise-floored probe; QI slightly better everywhere.

## Question

Under plain Adam, does the first-layer geometry move at all, and by how much, when the network starts from a QI-family init versus standard init? This is the direct test of the killed-gradient fear (PROGRAM_FRAMING §6), and it gates the inverse-problem application class.

## Experiment design

Plain Adam only (no solves, no finishers), one hidden tanh layer everywhere, **readout initialized to zero in every arm** so the run starts with prediction $\equiv 0$ and the geometry gradient exactly zero (it flows through the readout). Two arms per problem: `standard` (Glorot / PyTorch default) vs `qi` (the QI-family init for that class). 4 problem classes $\times$ 3 problems $\times$ 2 arms = 24 runs, 5000 full-batch Adam steps each (cosine LR, warmup 100), fp64, CPU, single seed.

- **1-D interpolation** (sine / runge / sine\_8pi, $N=128$, $W=269$): `qi` = the construction geometry, uniform grid + halo, $\gamma=\lambda^*/h$ at $\lambda^*=0.25$, $b=-\gamma c$; `standard` = Glorot. Data 2003/4001 equispaced train/eval; lr $3\times10^{-3}$.
- **2-D interpolation** (gauss\_bump / sine2d / mixed2d on the unit disk, radon\_tensor with 570 ridges): `qi` = expE01's Radon geometry at that cell's best $\lambda$ (0.26 / 0.08 / 0.12), $\gamma=\lambda/h_\text{ref}$, $h_\text{ref}=2.8/\sqrt N$; `standard` = Glorot on the same architecture. Data 8000 area-uniform disk samples, eval on a 120$\times$120 disk-clipped grid; lr $3\times10^{-3}$.
- **2-D nonlinear inverse PINNs** (three manufactured problems on $[-1,1]^2$, 518 ridges, each with ONE learnable scalar PDE parameter, log-parameterized, started $10\times$ low, trained jointly by the same Adam): burgers\_nu $u u_x + u u_y = \nu\,\Delta u + f$ ($\nu^*=0.1$, $u^*=\sin\pi x\cos\pi y$); bratu\_lam $\Delta u + \lambda e^u = f$ ($\lambda^*=1$, $u^*=(1-x^2)(1-y^2)$); allencahn\_k $\Delta u + k(u-u^3) = f$ ($k^*=5$, $u^*=\sin\pi x\sin\pi y$). Loss = mean-square PDE residual on a 28$\times$28 interior collocation grid + boundary fit (256 points) + sparse interior data fit (100 points of $u^*$), equal weights; derivatives by autograd. `qi` = radon\_tensor at $\lambda=0.15$ (no tuned reference exists for this class; stated choice); `standard` = Glorot; lr $3\times10^{-3}$.
- **Tabular regression** (superconductivity / sarcos / parkinsons from the expF04 all20 cache, width 256, tanh, targets min-max normalized): `qi` = the all20 best arm `scaled_psqrt` (ridge-bundle init `qi_ridge_init_layer_`, `centers_per_dir` $=\sqrt{256}=16$, random centers from the data projections); `standard` = PyTorch default Linear init. Full-batch, lr $10^{-3}$ (expF04's lr).

**Metrics.** The geometry vector $g_i$ = first-layer weight and bias concatenated (readout excluded; the PDE scalar excluded and logged separately). Tracked every iteration: drift $\|g_i-g_0\|_2/\|g_0\|_2$ and per-step motion $\|g_i-g_{i-1}\|_2/\|g_0\|_2$. Per run, the two lstsq probes of PROGRAM\_FRAMING §7.1: pre\_probe = eval rel $L_2$ of a truncated-SVD readout solve ($[\Phi,\mathbf 1]$, rcond $10^{-13}$) on the *init* geometry; post\_probe = the same on the *final* geometry (trained readout thrown away); plus the run's own final eval rel $L_2$. For the PINN class the probe solves the data-fit block only (boundary + interior data rows), a stated caveat since it ignores the PDE rows. Eval rel $L_2$ is $\|\hat f - f\|_2/\|f\|_2$ on the eval grid (test set for tabular; dense $61\times61$ grid vs $u^*$ for PINNs).

**Code & data.** `experiments/expD17_geometry_motion/` (`run.py`, `config.yaml`; tests `tests/test_expD17_geometry_motion.py`: QI-init 1-D probe at the floor, drift exactly 0 at init, readout zeroed). Data: `results/checkpoint_D_optimizers/expD17_geometry_motion/data/{interp1d,interp2d,pinn_inverse,tabular}.jsonl`. Figures: `figures/expD17_drift_from_init.png`, `figures/expD17_step_size.png`.

## Results

Full table (drift at 5000 steps; probes; run error). pre $\to$ post is the **geometry score**: what one solve would give on the init vs the final geometry.

| class / problem | arm | drift end | pre\_probe | post\_probe | run error |
|---|---|---:|---:|---:|---:|
| 1-D sine | standard | 0.51 | 5.3e-2 | 3.1e-1 | 9.2e-1 |
| 1-D sine | qi | 6.7e-3 | 3.7e-14 | 7.2e-12 | 5.0e-4 |
| 1-D runge | standard | 9.1 | 1.5e-1 | 1.7e-4 | 2.0e-2 |
| 1-D runge | qi | 1.8e-3 | 2.5e-14 | 2.5e-11 | 1.2e-4 |
| 1-D sine\_8pi | standard | 0.68 | 9.7e-1 | 9.7e-1 | 9.9e-1 |
| 1-D sine\_8pi | qi | 2.4e-2 | 5.9e-14 | 1.2e-7 | 7.4e-3 |
| 2-D gauss\_bump | standard | 7.4 | 1.5e-4 | 6.8e-10 | 3.0e-3 |
| 2-D gauss\_bump | qi | 1.8e-2 | 2.2e-14 | 8.3e-12 | 1.3e-4 |
| 2-D sine2d | standard | 7.0 | 2.8e-3 | 6.6e-8 | 5.1e-2 |
| 2-D sine2d | qi | 0.69 | 4.8e-8 | 3.2e-10 | 2.7e-2 |
| 2-D mixed2d | standard | 15.1 | 2.7e-2 | 6.7e-4 | 1.8e-2 |
| 2-D mixed2d | qi | 0.86 | 5.5e-5 | 5.5e-8 | 1.2e-2 |
| PINN burgers\_nu | standard | 4.9 | 5.9e-3 | 4.1e-5 | 6.5e-2 |
| PINN burgers\_nu | qi | 0.29 | 2.4e-5 | 2.6e-5 | 9.3e-3 |
| PINN bratu\_lam | standard | 5.0 | 2.3e-8 | 5.9e-7 | 3.9e-2 |
| PINN bratu\_lam | qi | 0.23 | 1.5e-5 | 4.3e-5 | 6.0e-3 |
| PINN allencahn\_k | standard | 2.8 | 2.0e-2 | 4.6e-4 | 1.7e0 |
| PINN allencahn\_k | qi | 0.26 | 7.2e-6 | 1.9e-4 | 8.0e-2 |
| tab supercond. | standard | 1.6 | 2.3e-1 | 2.0e-1 | 2.0e-1 |
| tab supercond. | qi | 0.72 | 2.8e-1 | 1.9e-1 | 1.9e-1 |
| tab sarcos | standard | 1.4 | 1.9e-1 | 1.5e-1 | 1.6e-1 |
| tab sarcos | qi | 0.42 | 3.0e-1 | 1.5e-1 | 1.5e-1 |
| tab parkinsons | standard | 2.5 | 9.0e-1 | 6.4e-1 | 6.7e-1 |
| tab parkinsons | qi | 0.77 | 8.2e-1 | 6.1e-1 | 6.1e-1 |

- **The dead zone is one step, everywhere.** With the readout at zero the geometry gradient is bitwise zero at step 1 in all 24 runs; from step 2 the geometry moves. Half the total 5000-step drift is accumulated by step 35-240 in the interpolation/PINN classes (tabular: 650-1100); the cosine decay then freezes it.
- **Relative motion is 3-800$\times$ smaller from QI init**, most extreme in 1-D (0.18-2.4% vs 51-913%). Normalization caveat: $\|g_0\|$ is 413 (qi) vs 2.0 (standard) in 1-D, so equal absolute motion reads $\sim$200$\times$ smaller relatively; the QI arm's *absolute* motion is comparable to standard's.
- **Motion damages a floor geometry, improves an off-floor one.** Every QI geometry that starts at the fp64 floor ends worse (sine $3.7\times10^{-14}\to7.2\times10^{-12}$, runge $\to2.5\times10^{-11}$, sine\_8pi $\to1.2\times10^{-7}$, gauss\_bump $\to8.3\times10^{-12}$). Both QI geometries that start off the floor end better (sine2d $4.8\times10^{-8}\to3.2\times10^{-10}$, mixed2d $5.5\times10^{-5}\to5.5\times10^{-8}$). From standard init in 2-D, Adam's geometry learning is strong (gauss\_bump probe improves $2\times10^5\times$) yet still lands 1-4 orders above the QI init's untouched floor.
- **Inverse-problem payload** ($\hat p$ vs $p^*$, relative error): burgers\_nu $1.0\times10^{-3}$ (qi) vs $3.4\times10^{-2}$ (standard); bratu\_lam $1.2\times10^{-2}$ vs $8.3\times10^{-2}$; allencahn\_k fails in both arms ($\hat k=0.89$ vs $32.8$, $k^*=5$) though the qi field error is 22$\times$ better ($8.0\times10^{-2}$ vs $1.7$). The parameter moves through the QI-initialized network; gradient flow is alive where it matters.
- **Tabular**: post-probes converge to the same noise-floored value per task regardless of init (qi $\le$ standard by a few percent), and each run's own error matches its post-probe, i.e. Adam solved the readout to the data's floor on whatever geometry it ended with.

### Figures

- **`figures/expD17_drift_from_init.png`** -- 4$\times$3 grid (rows = class, cols = problems), x = iteration, log y = $\|g_i-g_0\|/\|g_0\|$, blue = standard, red = QI, rows share the y scale. Look for: red 1-3 decades below blue in every panel; both saturating early; the 1-D row's red lines flat at $10^{-3}$-$10^{-2}$.
- **`figures/expD17_step_size.png`** -- same layout, y = $\|g_i-g_{i-1}\|/\|g_0\|$. Look for: no dead plateau after step 1 (motion is continuous, not gated); the spikes are the cosine-schedule/loss-landscape events, mirrored in both arms; motion collapses at the end as the LR decays.

## Additional details

- The standard-arm 1-D sine cell failed to train (run error 0.92, post-probe worse than pre): with the zero readout and this LR the run never leaves the early regime. Single seed; expD16 (Glorot readout) reached $10^{-2}$ from the same init, so this is a zero-readout artifact, not a property of the target. It does not affect the drift comparison, which is the cell's purpose.
- The PINN probe fits only the boundary+data rows, so bratu\_lam's standard arm shows a tiny pre\_probe ($2.3\times10^{-8}$: $u^*$ is a degree-4 polynomial, easy for any wide tanh basis on 356 points) while its PDE solution is still poor. Geometry scores within the PINN row should be read with that caveat.
- The 2-D qi arms for sine2d/mixed2d use the expE01 best-$\lambda$ geometries whose pre-probes ($4.8\times10^{-8}$, $5.5\times10^{-5}$) are far above the gauss\_bump floor; that is what makes them the off-floor test cases.

## Conclusions

*Pending Sam's review.* The killed-gradient fear is quantitatively real but not absolute: from QI-family inits the geometry receives 1-3 decades less relative motion than from standard init, yet it is never frozen (the exact dead zone is the single zero-readout step), it still refines geometries that start off the floor, and the inverse-problem parameter trains through it, better than from standard init on 2 of 3 problems. The cost of the motion is one-sided at the floor: every floor-quality geometry was damaged by training (2-7 orders), which is the preservation problem the QI optimizer exists to solve.

## Width scaling

The same protocol at three widths per class, with per-neuron $\gamma$/center/update instrumentation, is in **`width_scaling/expD17w_results.md`**. Headline: the relative-drift gap between the inits is a normalization effect of $\gamma=O(N)$ (absolute motion is init-independent, median std/QI ratio 0.86), QI relative drift falls like $1/W$ while standard stays flat, Adam moves the 1-D median bandwidth by under $0.12\%$ and preserves the center grid, and per-row update magnitude is set by the learning-rate schedule rather than by $\gamma$. It also answers this file's third open question below.

## Open questions

- Multi-seed: the standard-arm cells (especially 1-D sine) need seeds before their probe deltas are trusted.
- allencahn\_k defeats both arms (parameter off by $6\times$ to $-5.6\times$); is it the equal loss weighting, or the $10\times$-low start?
- The absolute-vs-relative drift lens: is motion in units of $\|g_0\|$ the right gauge when $\gamma=O(N)$ inflates $\|g_0\|$? A function-space drift (the probe error as a trajectory, not just pre/post) would disambiguate; it costs one solve per eval point on top of the current logging.
