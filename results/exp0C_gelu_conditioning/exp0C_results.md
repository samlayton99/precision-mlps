# Exp0C -- GELU vs tanh feature-matrix conditioning

**Question:** how does the conditioning of the feature matrix $\Phi$ change if we use a GELU activation instead of tanh, and what is the best bandwidth $\lambda$ for a GELU lstsq fit?

Code: `experiments/exp0C_gelu_conditioning/` (`run.py`, `config.yaml`, `gelu_phi.py`; tests in `tests/test_exp0C.py`). Run: `python3 experiments/exp0C_gelu_conditioning/run.py`.

GELU has no QI construction (its derivative $\mathrm{GELU}'(z) = \Phi(z) + z\,\phi(z)$ is a sigmoid, not a localized bump), so the QI-dependent metrics of exp0B (`ratio_full`, `ratio_row`, `qi_fit`) do not apply. This experiment studies what *is* defined for GELU: $\Phi$ conditioning and the lstsq fit, with tanh overlaid for contrast.

## Notation

Exact GELU activation $\mathrm{gelu}(z) = \tfrac{1}{2} z\,\big(1 + \mathrm{erf}(z/\sqrt{2})\big)$. On the fixed QI geometry ($W$ neurons at centers $c_k$, bandwidth $\gamma = \lambda/h$), the feature matrix is $\Phi_{ik} = \mathrm{gelu}\!\big(\gamma(x_i - c_k)\big)$, and $A = [\,\Phi \mid \mathbf{1}\,]$ is the augmented design matrix. "Effective rank" counts singular values above $10^{-13}\,\sigma_{\max}$; the null dimension is $(W+1) - \mathrm{rank}$; the effective condition number is the ratio of the largest to the smallest *kept* singular value (the full $\sigma_{\max}/\sigma_{\min}$ is often $\infty$ for GELU because its smallest singular values underflow to $0$).

## Experiments run (exactly as set up in the code)

fp64 throughout; eval grid $N_{\text{eval}} = 2048$.

1. **Step 1 -- $\lambda$ sweep.** Target $\sin(\pi x)$, width $N = 128$. For $\lambda \in \mathrm{linspace}(0.1, 0.5, 21)$ on the same geometry, solve the lstsq readout for GELU and (for contrast) tanh, and record eval $L_\infty$ and the conditioning. The best $\lambda$ is the GELU eval-$L_\infty$ minimizer.
2. **Step 2 -- width/target sweep at the best $\lambda$.** Widths $N \in \{32, 64, 96, 128\}$, targets {sine, sine_8pi, runge, sine_mixture, exp, abs_cubed}. Conditioning (cond, rank, null) is target-independent, so it is computed once per width for GELU vs tanh; the lstsq fit residual ($\|A\beta - y\|/\|y\|$) and eval $L_\infty$ are per target.

## Results

`data.json` holds `step1` (per-$\lambda$ rows), `best_lambda`, `cond_sweep` (per-width conditioning for both activations), and `err_sweep` (per-target, per-width GELU/tanh fit and eval error).

**Step 1.** Neither activation has a *sharp* $\lambda$ optimum on this (low-frequency) target -- both lstsq curves are flat-bottomed until the high-$\lambda$ fp64-cancellation wall (consistent with exp0D/exp01, where lstsq has a wide flat bottom and no fixed optimum). GELU's error is flat at $\sim 6\times10^{-13}$ to $5\times10^{-12}$ across the range; its minimum happens to fall at $\lambda = 0.220$ ($6.3\times10^{-13}$), but that is a representative point in the flat band, not a true optimum -- we adopt it as the fixed $\lambda$ for step 2. tanh is flat at $\sim 1\times10^{-13}$ over $\lambda \in [0.1, 0.3]$ then rises to $\sim 2\times10^{-9}$ by $\lambda = 0.5$.

**Step 2** (at $\lambda = 0.22$):

| $N$ | $W$ | GELU rank | GELU null | tanh rank | tanh null |
|---|---|---|---|---|---|
| 32 | 193 | 26 | 168 | 44 | 150 |
| 64 | 225 | 41 | 185 | 76 | 150 |
| 96 | 257 | 56 | 202 | 106 | 152 |
| 128 | 289 | 71 | 219 | 138 | 152 |

GELU effective rank grows like $\approx 0.43\,N$ while tanh grows like $\approx N$; GELU's null dimension grows with $N$ (168 to 219) while tanh's stays roughly constant ($\approx 150$). Effective condition numbers are comparable in magnitude ($\sim 10^{12}$).

Per-target eval $L_\infty$ at $N = 128$ (the precision check): **tanh reaches the fp64 floor** -- runge $1.3\times10^{-14}$, sine $5.4\times10^{-13}$, exp $9.3\times10^{-13}$ -- consistent with exp01's lstsq fp64 ($\sim 10^{-13}$). **GELU is 1-3 orders worse** -- sine $4.0\times10^{-12}$, sine_8pi $7.2\times10^{-11}$, runge $3.7\times10^{-9}$ -- a consequence of its lower-rank feature matrix. abs_cubed (only $C^1$) never reaches $\varepsilon$ for either ($\sim 10^{-7}$).

### Figures

**`gelu_lambda_ushape.png` (step 1).** Eval $L_\infty$ vs $\lambda$; GELU solid, tanh dashed; dotted vertical line marks the GELU minimum (a representative point in a flat band, not a sharp optimum). *How to read:* both are flat-bottomed in $\lambda$ until the high-$\lambda$ fp64-cancellation wall; GELU does not beat tanh's best.

**`gelu_vs_tanh_conditioning.png` (step 2).** Four panels vs width. *How to read:* panels 1-3 (condition number, effective rank, null dimension) are target-independent, so each shows GELU (solid) vs tanh (dashed); the dotted line in panel 2 is the full column count $W+1$. Panel 4 is the eval $L_\infty$ per target, GELU (solid) vs tanh (dashed) -- tanh hits the fp64 floor, GELU is 1-3 orders worse. The takeaway: GELU sits well below tanh in rank and above it in null dimension.

## Conclusions (pending Sam's review -- NOT yet approved)

*Proposed, for discussion -- do not treat as final.*

- GELU yields a **more rank-deficient** $\Phi$ than tanh on the same geometry (effective rank $\approx 0.43N$ vs $\approx N$; null dimension growing with $N$ vs roughly constant). It does not reduce the coefficient underdetermination of exp0B -- it enlarges it.
- GELU's lstsq error is **insensitive to $\lambda$** (flat $\sim 10^{-12}$) where tanh has a sharp optimum, but GELU does not reach tanh's best precision.
