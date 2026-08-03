# expD14 iteration 3 -- the Adam handoff

**Status: draft, pending Sam. 165 new cells at 4000 steps; `adam` and uncooled `rentry` baselines from iteration_2 at identical settings.**

## TL;DR

- **The expressibility signal banks machine epsilon on `qi` with no hand throttles.** Cooling Adam's geometry step by $f_\perp = \|r_\perp\|/\|r\|$ (the fraction of the residual the current features cannot express, free from the solve's own SVD) reaches $2.2\times10^{-16}$, $1.6\times10^{-15}$, $1.3\times10^{-16}$ on the three `qi` targets with the $10^{-14}$ floor preserved over 4000 steps -- iteration_0's flagship result, recovered by one measured scalar.
- **The same signal is the first to survive `datagap`.** Every other arm damages the initially-correct geometry by 3-4 orders; `perp` holds within $2$-$6\times$ of the do-nothing optimum and *reaches* $10^{-6}$-$10^{-7}$ during the run.
- **But it over-cools `rand`, for a measured and instructive reason.** On stock random init the features can *fit* 92% of the train residual while generalizing at $0.43$, so $f_\perp$ reads $0.075$ and freezes the geometry that most needs to learn. Train-expressibility conflates "the features are right" with "the features can overfit the sample."
- **The other cooling rules fail asymmetrically.** The line-search cap and the SNR gate cannot protect `qi` (the early residual is large, so neither binds before the damage is done); the SNR gate additionally costs `rand` five orders by cooling during active learning; the cosine schedule confirms that no-signal cooling loses on every case it wasn't tuned for.

## Question

One knob: how should Adam's geometry step cool, so that a correct geometry is not eroded (the `qi`/`datagap` failure of iteration_2) while a wrong geometry still learns at full speed (the `rand` success of iteration_2 that must not be given back)?

## Experiment design

Frozen from iteration_2: the $r_{\text{entry}}$-damped direct solve on the Method-C-discovered $L$, stock Adam moments, floor trajectory scored, same grid ($N{=}128$, 4000 steps, 3 seeds on random-init cases). Only the geometry step's length varies:

| arm | rule | signal class |
|---|---|---|
| `ls` | $\ell_A = \min(\|\Delta_A\|, t^\star)$, $t^\star = -(r\cdot Jp_A)/\|Jp_A\|^2$ | residual scale (1 JVP) |
| `snr` | $\ell_A \mathrel{*}= \hat s_{\text{geo}}$ | learning persistence (free) |
| `ls_snr` | both | -- |
| `cos` | cosine decay to 0 | no signal (control) |
| `perp` | $\ell_A \mathrel{*}= f_\perp = \|r - UU^\top r\|/\|r\|$ | expressibility (free from the solve's SVD) |

## Results

**Floor after the run (= that arm + terminal solve), medians; the two baselines and the uncooled arm included:**

| case / target | adam | rentry (uncooled) | ls | snr | perp | floor$_0$ |
|---|---|---|---|---|---|---|
| `qi`/`sine_8pi` | $4.2\times10^{-8}$ | $2.6\times10^{-8}$ | $8.1\times10^{-8}$ | $2.2\times10^{-8}$ | $\mathbf{4.4\times10^{-14}}$ | $5.5\times10^{-14}$ |
| `datagap`/`sine_8pi` | $1.2\times10^{-2}$ | $7.0\times10^{-1}$ | $1.6\times10^{0}$ | $1.4\times10^{0}$ | $\mathbf{5.5\times10^{-6}}$ | $9.3\times10^{-7}$ |
| `rand`/`sine_8pi` | $1.4\times10^{-4}$ | $\mathbf{7.5\times10^{-7}}$ | $1.2\times10^{-6}$ | $6.8\times10^{-2}$ | $3.6\times10^{-1}$ | $4.3\times10^{-1}$ |

And the reached errors on `qi`: `perp` $2.2\times10^{-16}$ / $1.6\times10^{-15}$ / $1.3\times10^{-16}$ against $10^{-11}$-$10^{-5}$ for every other arm. The full 7-arm $\times$ 15-cell tables are in the JSONL; the scorecard figure carries them.

**Why `perp` works where it works.** Its cooling trace on `qi`/`datagap` reads exactly zero for the first $\sim$1000 steps (the start error is fully expressible, the geometry does not move on it) while $\alpha = r_{\text{entry}}$ ladders to $10^{-15}$; by the time $f_\perp$ releases ($\sim$0.8 late, the leftover residual being pure noise), Adam's own step has decayed to $10^{-16}$ and no damage occurs. The handoff Sam asked for -- learn early, exploit the solve once ready -- is exactly what this trace shows, in the direction "already ready."

**Why it fails on `rand`, precisely.** $f_\perp$ is measured against the numerically live span of the feature matrix on the *training* rows. 269 random-slope tanh features fit $\sin(8\pi x)$ on 2003 train points to a 7.5% residual while the truncated solve generalizes at 43% -- so $f_\perp = 0.075$ and the geometry crawls. The signal needed is the *generalization* version of the same quantity: the residual fraction on **held-out** points after the solve. On paper it gets every regime right where train-$f_\perp$ gets `rand` wrong: `rand` held-out $\approx 0.43$ (free), `qi` $\approx 10^{-14}$ (frozen), `datagap` with the hold-out drawn from the *kept* (gappy) training distribution $\approx 10^{-6}$ (frozen, correctly), `clustered` large (free, correctly -- the middle train points are inexpressible). Cost: one forward on a held-out train split per step or per cadence.

### Figures

- **`H1_scorecard.png`** -- same design as iteration_2's G1 (light = reached, dark = after terminal solve, black line = step-0 baseline), now 6 arms including the two baselines. The `qi` and `datagap` columns show `perp`'s dark bars on the line and everyone else's above it; the `rand` column shows the inversion.
- **`H2_perp_story.png`** -- `qi`/`datagap`/`rand` on `sine_8pi`: floors and reached errors for `adam`/`rentry`/`perp` (top), and `perp`'s cooling factor with $\alpha$ overlaid (bottom). The trace to look at: cooling pinned at zero while $\alpha$ ladders down, then releasing harmlessly late (`qi`, `datagap`); cooling stuck at $0.075$ strangling the learning (`rand`).

## Conclusions

*Unsigned, pending Sam.*

1. Expressibility is the right *kind* of signal for the handoff: it is the only one that distinguishes the inits at step 1, and it delivers the full pipeline on `qi` (machine epsilon banked, floor intact, everything self-clocked) and the only acceptable behavior yet seen on `datagap`.
2. Measured on the training rows it has one precisely-identified defect: it cannot tell "features right" from "features overfit the sample," which freezes stock-random learning.
3. No arm currently wins all five regimes; the uncooled `rentry` remains best where geometry must be learned.

## Open questions

- The held-out variant $f_{\text{gen}}$ (residual fraction on a train-distribution hold-out after the solve): the sign-check above says it repairs `rand` without giving back `qi`/`datagap`. One knob, one run, directly falsifiable -- the natural iteration_4.
- Whether $f_{\text{gen}}$ should also gate the *solve* depth (it is a generalization clock; over-solving an overfitting span is how `datagap`'s reached error diverged from truth in iteration_2).
- Width scaling and 2-D remain untouched.
