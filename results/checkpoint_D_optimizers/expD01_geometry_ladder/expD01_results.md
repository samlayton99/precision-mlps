# expD01 -- Geometry ladder: can an optimizer solve the readout on frozen ideal geometry?

**Status: draft (newly authored) -- pending Sam's review.**

## TL;DR

- The easiest rung: freeze the geometry in the correct regime so only the convex linear readout trains. Adam still can't reach it -- it stalls at ~$10^{-3}$ while lstsq on the *same* $\Phi$ reaches ~$10^{-13}$.
- It's not weight blowup (trained weights stay $O(1)$). It's first-order descent failing on the ill-conditioned $\Phi$. Solve the readout directly.

## Question

With the geometry fixed in the correct regime, how close does a standard optimizer get to the exact readout (the global optimum of a convex problem)?

## Experiment design

Freeze $\gamma$ and the uniform centers ($\lambda^*=0.25$, halo $=\texttt{default\_halo}(N,0.25)$, $\gamma=\lambda^*/h$); $\Phi_{ik}=\tanh(\gamma(x_i-c_k))$ is then constant and precomputed once, so training optimizes only the readout on $\text{pred}=\Phi v+\text{bias}$ -- a convex MSE problem whose global optimum is the lstsq solution. Train from random Xavier init by Adam (peak LR $0.1$, short warmup then cosine decay, full-batch MSE, fp64, 50k steps) and compare to that lstsq optimum on the same $\Phi$. Grid: targets sine, sine_8pi, runge; widths $\{32,64,128,256\}$; 3 seeds. (Simplified Phase 1 = level 3 of the ladder; levels 4--7 relax the geometry and are future work.)

**Code & data.** `experiments/expD01_geometry_ladder/` (`run.py`, `config.yaml`). Data: `phase1_data.json`. Figures: `error_vs_width.png`, `convergence_{sine,sine_8pi,runge}.png`.

## Results

- **Adam stalls ~10 orders above the exact solve.** The trained readout floors at ~$10^{-3}$ rel $L_2$ on smooth targets; lstsq on the same $\Phi$ reaches ~$10^{-13}$--$10^{-14}$ wherever resolved.
- **Not weight blowup.** Trained outer weights stay $O(1)$ ($\max|v|\sim0.5$), comparable to lstsq's. The stall is first-order descent on an ill-conditioned convex problem ($\mathrm{cond}(\Phi)\sim10^{19}$), not a runaway-weight pathology.

### Figures

- **`error_vs_width.png`** -- 3 targets (rows) x {rel $L_2$, $L_\infty$} (cols); the trained curve sits flat at ~$10^{-3}$ while the lstsq curve descends to the floor with width.
- **`convergence_<target>.png`** -- 4 widths (rows) x {rel $L_2$, $L_\infty$} (cols), error vs Adam step: the trajectory drops fast to ~$10^{-3}$ then plateaus (with spikes), never approaching the lstsq optimum marked on the axis.

## Conclusions

*Draft, pending Sam.* On the easiest rung -- correct geometry, convex readout -- Adam stalls at ~$10^{-3}$ while lstsq reaches the floor. The barrier is first-order optimization on the ill-conditioned $\Phi$, not weight blowup. The practical answer is to solve the readout directly.

## Open questions

- Levels 4--7: relax the geometry (free centers, free $\gamma$) with the readout solved or trained, to localize where precision is lost.
