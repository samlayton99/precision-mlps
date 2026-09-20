# expI02 -- a first prior on what the QI blocks are good for (design)

Status: approved by Sam 2026-09-05 (chat), with amendments: fp32 where precision is not the question; iso-params / iso-FLOPs and relative scaling as the main comparison axes; depth and residual streams included; few, carefully chosen figures.

## Goal

A speculative prior, not a definitive test: (1) the best configuration and scaling of the compositional QI block (coordinates, preconditioning, band control, init, optimizer, readout, rank, depth); (2) head-to-head against standard MLPs at matched parameters and matched FLOPs on analytic high-dimensional targets, structured regression, Fashion-MNIST, and three PINNs. Everything runs on the Mac mini (M4, 10 cores, 16 GB; MPS has no fp64).

## The corrected model (Sam's notes, 2026-09-05)

- **Fixed geometry at every depth.** Bank centers uniform on $[-1,1]$, $N$ cells, $h=2/N$, $\gamma=\lambda_\star/h$, $\lambda_\star=0.25$. Nothing moves. Changing coefficients or direction norms changes where data land on the mesh (utilization), never $\lambda$.
- **No forward normalization.** No unit-direction normalization, no range tracking, no EMA, no LayerNorm. Layer-1 projection $V$ is free.
- **Band control = calibration + weak penalty.** One calibration pass at init: per channel, rescale coefficients and set the bias so $\mathbb E[z_k]=0$, $\mathrm{sd}(z_k)=s_\star=0.4$. During training a weak hinge penalty on the actual channel values: $R_{\rm band}=\sum_k\big([|\mu_k|-0.25]_+^2+[0.15-s_k]_+^2+[s_k-0.6]_+^2+\mathbb E[(|z_k|-0.9)_+^2]\big)$, weight $\beta$ (default $10^{-2}$, swept once).
- **Coefficient coordinates.** Mixer coefficients $C_{r,:,k}=P\,\theta_{r,:,k}$ with $P\in\{I,\ D,\ W_{\rm cap},\ DW_{\rm cap}\}$: $D$ = function-value coordinates ($w_j=(a_j-a_{j-1})/2$); $W_{\rm cap}$ = target-independent whitening of the bank Gram on uniform occupancy of $[-1,1]$, amplification capped at $\kappa$ (default $10^3$). Implemented as a transform of the bank, $\tilde H=HP$.
- **Init.** Every profile starts as the same smooth random function (degree-3 Chebyshev, coefficients $\xi_m/(m+1)$) projected onto the bank, so coordinate arms differ only in optimization. Rank-$S$: $a$ random unit rows, $\Phi$ smooth. Never zero.
- **Readout.** Solved (QR then SVD, rcond $10^{-14}$) for regression and linear PDEs; trained (in $P$ coordinates) for cross-entropy. The solved-head error is logged along every curve as the diagnostic.
- **Optimizers.** Adam (lr $5\times10^{-3}$, cosine, warmup 50); heavy-ball momentum ($0.9$, lr $=1/L_0$ from power iteration at init, cosine); mixed (Adam on $V$, momentum on $\theta$). Head: trained / re-solved every 250 steps / re-solved every step (VarPro). Gauss-Newton only as an optional finisher, its own column.
- **Depth.** Stack of bank+mixer layers on the channels; residual $z_{\ell+1}=z_\ell+\Delta_\ell$ with $\Delta_\ell$ calibrated to sd $0.1$ at init.

## Studies

Precision: fp64 CPU for A0, A1, A2, B1, B4. fp32 (CPU or MPS, whichever is faster) for B2 noisy data and B3.

| id | what | knobs | read |
|---|---|---|---|
| A0 | fixed-feature ladder, frozen 1-D bank, 4 targets, 1000 steps | $P\in\{I,D,W,DW\}$ $\times$ {GD, momentum, Adam} | whitening vs coordinates vs optimizer, against the lstsq floor |
| A1 | mechanism ablation on the two-level block, $d=3$, targets gauss / fast waves / composition / three bumps, base config $(M,N_1,K,N_2)=(6,32,2,64)$, 3000 steps | old recipe $\to$ +fixed mesh/calibration/penalty $\to$ +smooth init $\to$ +$D$ $\to$ +$W$ $\to$ +momentum/mixed $\to$ +VarPro; then $\beta$, $\kappa$, rank $\{1,2,\text{full}\}$ on the winner; 3 seeds on the top three | which mechanisms matter, the solved-head gap, +GN column |
| A2 | scaling with the winning recipe, $d=4$ and $5$, fast waves / composition / product peak | $N_1$, $N_2$, $M$, $K$, depth $\{2,3,4\}$, residual on/off; MLP width ladder on the same axes; oracle floor | error vs params per knob; block scaling vs MLP scaling |
| B1 | analytic $d=5$: gauss, fast waves, composition, product peak, three bumps, random ridges (control) | block (winner), shallow ridge-QI, MLP@params, MLP@FLOPs; 3 seeds | iso-params / iso-FLOPs |
| B2 | Friedman1 ($d=10$, $\sigma=1$), Lorenz flow map (dysts, $3\to3$), UCI concrete, UCI energy | block, MLP@params, ridge regression; 3 splits | RMSE vs the linear baseline |
| B3 | Fashion-MNIST, fp32, 10 epochs, batch 256 | block, tanh MLP@params, ReLU MLP@params, logistic regression; two sizes | test accuracy vs epoch and vs params; wall clock |
| B4 | 1-D Poisson (oscillatory manufactured $u$), 2-D Poisson (manufactured), Burgers (Cole-Hopf oracle) | block vs MLP@params, Adam on the PINN loss, final head solve on the linear ones | rel $L_2$ vs step |

Discipline: same data, seed, step budget, and final solve for every model in a comparison; counts from one function per model; MLP widths chosen by that function; no per-task tuning of $\lambda$, lr, rcond; test scored on the inner 0.9 ball for analytic targets; single seed for sweeps, three seeds for every head-to-head number; error axes fixed; trajectories recorded.

## Figures (six, no more)

1. `fig1_ladder.png` -- A0: one panel per target, error vs step, lines = the four $P$ arms, solid momentum / dotted Adam, dashed grey lstsq floor.
2. `fig2_mechanisms.png` -- A1: (a) the recipe ladder, x = cumulative recipe step, y = solved-head test error, one line per target, hollow markers = +GN, ticks = oracle floor; (b) the solved-head gap: trained-head and solved-head error vs step for the winning recipe.
3. `fig3_scaling.png` -- A2: one panel per target, error vs parameters, one line per knob swept, depth/residual as separate lines, MLP width ladder in black.
4. `fig4_headtohead.png` -- B1 + B2: dot plot, tasks on x, models as markers with seed spread; panel (a) analytic rel $L_2$, panel (b) data RMSE normalized by the linear baseline.
5. `fig5_fashion.png` -- B3: (a) test accuracy vs epoch at matched params, (b) test accuracy vs params.
6. `fig6_pinn.png` -- B4: one panel per PDE, rel $L_2$ vs step, block vs MLP, final-solve endpoints as markers.

Legends above axes, never inside. Fixed log axes $[10^{-15},10^{1}]$ for precision panels.

## Code

`experiments/expI02_block_prior/`: `qi2.py` (model family: bank, coordinates, mixer, band penalty, calibration, stack, MLP baseline, optimizers, fit loop with the solved-head diagnostic, Gauss-Newton finisher), `tasks.py` (analytic targets, Friedman1, Lorenz, UCI loaders, Fashion-MNIST IDX loader, PDE definitions with autograd residuals), `run.py` (studies as subcommands with `--quick`, resumable JSON in `results/checkpoint_I_depth_theory/expI02_block_prior/`, `--plot`). Tests in `tests/test_expI02_block_prior.py`. The library `qiblocks.py` is not edited; solvers are imported from it.

## Predictions (falsifiable)

1. Capped whitening is the largest optimization gain in the block ($\ge2$ orders at 3000 first-order steps on the $d=3$ targets); function-value coordinates $\le1$ order and can hurt under Adam.
2. Momentum beats Adam on whitened coefficients; Adam stays better on $V$; mixed wins.
3. Calibration + weak penalty matches range tracking within $3\times$; failure mode is channel collapse toward constant.
4. Block beats MLP by $\ge3$ orders on factorizable $d=5$ targets at matched params and FLOPs; ties or loses on random ridges.
5. Friedman1: both at the noise floor; Lorenz: block 1-2 orders ahead; UCI: parity.
6. Fashion-MNIST: within one point of the matched MLP.
7. Linear PDEs with head solve: block $\ge3$ orders ahead; Burgers: both near $10^{-3}$, MLP possibly better.
