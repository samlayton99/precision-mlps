# expA04 -- GELU vs tanh: feature-matrix rank regime

**Status: conclusions approved by Sam (O(N)/O(1) framing).**

## TL;DR

- The activation sets the rank regime: tanh has an $O(1)$ null space (rank $\approx N$); GELU has an $O(N)$ null space (rank $\approx 0.4N$).
- tanh reaches the fp64 floor on smooth targets; GELU is 1--3 orders worse. GELU enlarges the readout underdetermination of expA03 rather than fixing it.

## Question

How does the feature matrix's conditioning and rank change under GELU instead of tanh, and what's the best bandwidth for a GELU lstsq fit?

## Experiment design

On the fixed QI geometry ($W$ centers, $\gamma=\lambda/h$), build the feature matrix for GELU ($\mathrm{gelu}(z)=\tfrac12 z(1+\mathrm{erf}(z/\sqrt2))$, so $\Phi_{ik}=\mathrm{gelu}(\gamma(x_i-c_k))$) and, for contrast, tanh, fp64, augmented with a bias column. *Step 1:* sweep $\lambda\in\mathrm{linspace}(0.1,0.5,21)$ on $\sin\pi x$ at $N=128$ and pick the GELU eval-$L_\infty$ minimizer. *Step 2:* at that $\lambda$, sweep widths $\{32,\dots,128\}$ and 6 targets, recording effective rank, null dimension, and condition number (geometry-only, per width) and the lstsq fit + eval $L_\infty$ (per target). Effective rank counts singular values above $10^{-13}\sigma_{\max}$; the effective condition number uses only the kept singular values (GELU's smallest underflow to 0).

**Code & data.** `experiments/expA04_activation_conditioning/` (`run.py`, `gelu_phi.py`; tests in `tests/test_expA04_activation_conditioning.py`). Data: `data.json`. Figures: `gelu_lambda_ushape.png`, `gelu_vs_tanh_conditioning.png`.

## Results

- **Different rank regimes.** tanh rank tracks the width ($\approx N$) with a flat null dimension (~150) -- $O(1)$. GELU rank grows at only $\approx 0.4N$ with a null dimension that grows with width (168$\to$219) -- $O(N)$. Condition numbers are comparable (~$10^{12}$); rank, not conditioning, separates them.
- **tanh reaches the floor; GELU doesn't.** At $N=128$, tanh hits ~$10^{-13}$--$10^{-14}$ on smooth targets; GELU is 1--3 orders worse. `abs_cubed` reaches neither floor.
- **Bandwidth is a flat knob** for both -- no sharp $\lambda$ optimum until the high-$\lambda$ cancellation wall.

### Figures

- **`gelu_lambda_ushape.png`** -- eval $L_\infty$ vs $\lambda$, GELU solid, tanh dashed. Both flat-bottomed until the high-$\lambda$ wall; GELU never matches tanh's best.
- **`gelu_vs_tanh_conditioning.png`** -- four panels vs width: condition number, effective rank (dotted = full column count), null dimension (the headline -- GELU rises with $N$, tanh stays flat), and per-target eval $L_\infty$ (tanh at the floor, GELU 1--3 orders above).

## Conclusions

GELU yields a more rank-deficient $\Phi$ than tanh -- $O(N)$ null space vs $O(1)$ -- which costs 1--3 orders of precision. tanh is the right activation; GELU is a documented contrast, not a path forward. (Approved by Sam.)

## Open questions

None.
