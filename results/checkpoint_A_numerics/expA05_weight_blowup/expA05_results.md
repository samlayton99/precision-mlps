# expA05 -- Weight-blowup study: QI vs lstsq readout norm

**Status: draft -- pending Sam's review.**

## TL;DR

- No weight blowup: once the target is resolved, the readout norm *decays* with width (a power law, log-log slope ~$-0.5$ to $-1$) for both QI and lstsq.
- QI is not the minimum-norm readout -- it carries ~$1.25\times$ the min-norm lstsq vector (~$1.0\times$ for runge), consistent with expA03's null-space result.
- At small $N$ (target unresolved) norms do explode ($10^6$--$10^7$) for both methods -- a resolution failure, not an optimizer pathology.

## Question

Does the construction blow up its readout weights as width grows, and is the QI readout the minimum-norm solution on its geometry?

## Experiment design

On one fixed geometry per width -- QI centers + halo, $\gamma=\lambda/h$ at $\lambda=0.25$ (so $\gamma=O(N)$), $K_c=160$ -- take two readout vectors: (a) QI's outer coefficients from the fp64 construction, and (b) the fp64 min-norm least-squares solve of $\Phi v + b = y$ (`numpy.lstsq`). Record the readout norm $\|v\|_2$ (and $\max|v|$) and the eval error ($L_\infty$, rel $L_2$ on 2048 points) across 20 log-spaced widths $N\in[32,512]$ for 4 targets (sine, runge, exp, sine_mixture). Both fp64 -- the norm is identical to mpmath (e.g. $\|a\|_2=1.275$ at $N=64$ in both), so extended precision is unnecessary here.

**Code & data.** `experiments/expA05_weight_blowup/` (`run.py`, `config.yaml`). Data: `data.json`. Figures: `weight_blowup.png` ($\|v\|_2$), `weight_blowup_linf.png` ($\max|v|$).

## Results

- **Resolved regime:** the readout norm decays with width for both methods -- a power law, log-log slope ~$-0.5$ to $-1$ (e.g. sine $\|v\|_2$: $1.2\to0.26$ QI as $N:64\to512$). QI sits a constant ~$1.25\times$ above the min-norm lstsq (~$1.0\times$ for runge). The construction does not blow up its outer weights with width.
- **Undersampled regime (small $N$):** when the grid can't resolve the target, norms explode to $10^6$--$10^7$ (sine_mixture, runge at $N=32$), and lstsq is *not* smaller there -- for runge/mixture it is $10$--$60\times$ larger than QI (its "min-norm" fit is a poor, high-norm one). Norm and error correlate cleanly across ~7 decades, so this is a resolution failure.

### Figures

- **`weight_blowup.png`** -- 4 rows (targets) x 2 cols. *Left:* $\|v\|_2$ vs $N$ (log-log), QI orange, lstsq blue -- flat/decreasing = no blowup, a small-$N$ rise flags undersampling. *Right:* $\|v\|_2$ vs eval $L_\infty$ scatter, shaded light$\to$dark for $N$ low$\to$high -- points sliding up-and-right are the blowup regime; the low-error band at small norm is the resolved floor.
- **`weight_blowup_linf.png`** -- same layout for $\max|v|$ (the single largest coefficient). Same story; only the QI--lstsq constant-factor gap differs.

## Additional details

- The QI fp64 error floors at $L_\infty\sim10^{-10}$ from convolution cancellation (a construction-precision artifact, not weight-driven), so the QI-vs-lstsq vertical gap in the resolved scatter is that fp64 effect, not blowup. The blowup signal is the norm (left column) and the up-and-right trend at small $N$.

## Conclusions

*Pending Sam.* The construction does not blow up its readout weights -- once resolved, the norm decays with width (a power law) rather than growing. QI is not min-norm, but only ~$1.25\times$ above it, i.e. the null-space freedom of expA03, not a pathology. (Minor open question: whether that ~$1.25\times$ ratio -- ~$1.0\times$ for runge -- has a target-independent explanation.)
