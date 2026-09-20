# expD16 -- Optimizer zoo on the correctly initialized QI geometry

**Status: draft -- pending Sam's review.**

## TL;DR

- The paper's recipe (Adam warm start, second-order finisher) reproduces in-repo: the ordering is SSBroyden $>$ NNCG $>$ L-BFGS $>$ Adam $\gg$ SPSA in every one of the 24 cells.
- QI-initializing the geometry helps every gradient-based optimizer (median $\sim15\times$ for Adam and for Adam$\to$SSBroyden; up to 3 orders on sine and sine\_8pi), but no optimizer comes near the construction floor: best cell overall is $3\times10^{-9}$ vs the $\sim4\times10^{-14}$ lstsq floor on the same init.
- SPSA is not competitive at any perturbation size tested (never below $3.5\times10^{-2}$); at $a=10^{-1}$ it diverges from the QI init.

## Question

Starting from the paper's construction geometry (vs standard Xavier init), which end-to-end optimizers train to (or hold) high precision -- and does the correct initialization change the ordering?

## Experiment design

Standard-parameterization tanh MLP (QIMlp, `layer_type="standard"`, fp64, CPU): $f(x) = b_0 + \sum_k v_k \tanh(w_k x + b_k)$, width $W = N + 2\,\text{halo} + 1$ with $\text{halo} = \texttt{default\_halo}(N, 0.25)$. Two inits, identical parameterization so only the starting point differs:

- **qi**: the construction geometry -- $w_k = \gamma = \lambda^*/h$ ($\lambda^* = 0.25$, $h = 2/N$), $b_k = -\gamma c_k$ on the uniform grid+halo $c_k = -1 + kh$; readout at Glorot init (the optimizer must find the readout -- QI-initializing it too would leave nothing to optimize).
- **xavier**: Glorot uniform everywhere (expD02's `xavier`), the paper's direct-training baseline.

All optimizers train **all** parameters (train-both, matching the paper's end-to-end protocol), full-batch fp64 MSE on $n_{train} = 2003$ equispaced points (prime, $> W_{\max} = 461$, so overdetermined -- the repo convention, not the paper's $n = 256$); metric is eval rel $L_2 = \|\hat f - f\|_2/\|f\|_2$ on a misaligned prime grid ($n_{eval} = 4001$), logged every 10 iterations (finishers: every accepted step). Shared iteration wall $T = 3000$; the three finishers branch from **one** shared Adam warm segment ($T_{Adam} = 2000$, lr $3\times10^{-3}$, warmup 100, cosine), so their trajectories are directly comparable:

- **adam** -- plain Adam, cosine over the full 3000.
- **adam$\to$ssbroyden** -- SSB-II quasi-Newton with strong Wolfe (`src/training/ssbroyden.py`, dense inverse Hessian, defaults lr 0.5/c1 $10^{-3}$/c2 0.3), 1000 steps.
- **adam$\to$nncg** -- Nystrom Newton-CG (Rathore et al. 2024): rank-100 Nystrom sketch of the loss Hessian (exact autograd HVPs) refreshed every 25 steps, Nystrom-preconditioned CG (25 iterations, Steihaug bailout) on $(H + \mu\lambda_{\max}I)d = -g$ with $\mu = 10^{-8}$, Armijo backtracking; 1000 steps.
- **adam$\to$lbfgs** -- torch LBFGS, strong Wolfe, history 100, one iteration per outer step; 1000 steps.
- **spsa** -- gradient-free SPSA (Spall 1992), Rademacher two-sided perturbation, standard gain decays; gains tuned by a $4\times4$ $(a, c)$ pre-sweep per init (600 iterations, sine, $N{=}64$): $c$ is flat across $10^{-2}$-$3\times10^{-1}$ (set to $10^{-1}$, comfortably above fp64 loss-difference noise), $a$ is the sensitive knob ($3\times10^{-2}$ qi / $10^{-1}$ xavier; $10^{-1}$ diverges from qi). Full 3000 iterations (two loss evals each).

Grid: 4 targets (sine, exp easy; runge, sine\_8pi difficult) $\times$ $N \in \{64, 128, 256\}$ $\times$ 5 optimizers $\times$ 2 inits, one seed per target (trajectory study). Sanity gate run before any training (`tests/test_expD16_optimizer_zoo.py`): the qi init geometry + truncated-SVD lstsq readout reaches $3.6\times10^{-14}$ rel $L_2$ on sine at $N{=}128$ (the floor), the xavier control $5.3\times10^{-2}$.

**Code & data.** `experiments/expD16_optimizer_zoo/` (`run.py`, `nncg.py`, `spsa.py`, `config.yaml`); tests `tests/test_expD16_optimizer_zoo.py`; a `step_callback` hook was added to `src/training/ssbroyden.py` (non-breaking). Data: `data/trajectories_{qi,xavier}.jsonl` (60 rows each, full traces). Figures: `figures/expD16_qi_init.png`, `figures/expD16_xavier_init.png`.

## Results

Final eval rel $L_2$ (median over the 12 cells per init):

| init | adam | adam$\to$ssbroyden | adam$\to$nncg | adam$\to$lbfgs | spsa |
|---|---:|---:|---:|---:|---:|
| qi | $8.3\times10^{-3}$ | $2.0\times10^{-7}$ | $9.3\times10^{-6}$ | $9.8\times10^{-4}$ | $1.9\times10^{-1}$ |
| xavier | $1.3\times10^{-1}$ | $2.9\times10^{-6}$ | $2.4\times10^{-5}$ | $2.8\times10^{-3}$ | $8.2\times10^{-1}$ |

- **The ordering is uniform.** SSBroyden is the best finisher in 23/24 cells (NNCG takes xavier/sine at $N{=}256$); L-BFGS lands 1-3 orders above NNCG; plain Adam sits at $10^{-3}$-$10^{-2}$ from qi init and often near $10^0$ from xavier at this budget; SPSA never leaves $O(10^{-1})$.
- **QI init helps, most where Adam alone fails.** On sine the qi/xavier gap for Adam$\to$SSBroyden is $10^3\times$ at $N{=}64$; on sine\_8pi the xavier runs barely move during the Adam segment while the qi runs descend. The exception is exp, where xavier Adam$\to$SSBroyden reaches the experiment-best $3.1\times10^{-9}$, slightly beating its qi counterpart.
- **Nobody reaches the floor.** The best trained cell ($3\times10^{-9}$) is $\sim5$ orders above the lstsq floor the qi geometry supports ($3.6\times10^{-14}$, sanity test) -- end-to-end gradient training does not hold the construction's precision even when initialized at it, consistent with expD01/expD02.

### Figures

- **`figures/expD16_qi_init.png`** -- 4 targets (rows) $\times$ $N \in \{64,128,256\}$ (cols); x = iteration (0-3000), y = eval rel $L_2$ (log, fixed $10^{-16}$-$10^1$ on all 12 panels); one color per optimizer, grey dotted line = the Adam$\to$finisher handoff at 2000. Read the vertical drop at the handoff and the finisher separation after it: red (SSBroyden) lowest everywhere, then green (NNCG), purple (L-BFGS), blue (Adam) flat, brown (SPSA) at the top. The empty bottom half of every panel is the point -- no trajectory approaches machine precision.
- **`figures/expD16_xavier_init.png`** -- identical layout and axes for the control. Compare panel-by-panel against the qi figure: Adam segments stall higher (sine\_8pi never leaves $10^0$), finisher floors sit 1-3 orders above their qi counterparts except exp.

## Additional details

- Iteration is the x-axis, not cost: one SSBroyden/L-BFGS iteration is a few function-gradient evals (line search), one NNCG iteration is $\sim25$ HVPs plus an amortized rank-100 sketch, one SPSA iteration is 2 loss evals. At matched iterations NNCG is the most expensive line by roughly an order of magnitude.
- NNCG's Armijo line search compares loss values, which the repo's kill list caps near $10^{-10}$; at the $10^{-5}$ floors reached here that guard never binds.
- Total wall clock for both grids: $\approx49$ minutes on CPU (fp64; MPS is fp32-only and was not used).

## Conclusions

*Draft, pending Sam.* Data-obvious: under the paper's two-stage protocol the finisher ordering is SSBroyden $>$ NNCG $>$ L-BFGS $>$ Adam $\gg$ SPSA on both inits; the correct QI geometry init improves every gradient-based line (median $\sim15\times$, up to 3 orders) but no end-to-end optimizer approaches the $10^{-14}$ floor that an exact readout solve reaches on the same init.

## Open questions

- Does a longer finisher budget (SSBroyden was still descending in several qi cells at iteration 3000) close any of the remaining $\sim5$ orders, or does it plateau like the paper's Fig. 3?
- The exp anomaly: why does the xavier geometry match/beat qi there for the second-order finishers?
- Extensions already scaffolded by the builder/registry split in `run.py`: 2-D (expE01 Radon geometry) and 2-layer variants, and seed-averaging (single seed here).
