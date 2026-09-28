# expC11 — Runge 25 on the true-p precision sweep

**Status: all 46 precisions complete and validated; pending Sam's review.**

## TL;DR

This comparison adds Runge 25 to panel (c), using the existing true-p model and SVD solver at every integer precision from 8 to 53 bits. Runge follows a similar descending trend with lower error through most of the range, reaching $2.50\times10^{-14}$ at 53 bits. The chirp measurements and their reference line are reused. No line is fitted to Runge.

## Question

What does the full working-precision sweep look like for $f(x)=1/(1+25x^2)$ alongside chirp?

## Experiment design

The total width is $W=1024$, with 24 halo centers per side, hence $N=975$ interior intervals. Both targets use the same 4,801 training points and 8,001 evaluation points on $[-1,1]$. Runge uses the existing refined bandwidth selector with frequency scale $\bar\omega=5$, tolerance $2^{1-p}$, and search interval $[0.03,3]$. Chirp retains its saved bandwidths from the same rule with $\bar\omega=16\pi$.

The unchanged expC11 arithmetic has a $p$-bit significand and exponent range $[-958,959]$. Geometry, tanh features, the ported reference LAPACK DGELSS solve, stored coefficients, and sequential readout all use this arithmetic. The solver cutoff is $2^{1-p}$, matching the primary chirp curve. Bandwidth selection and supplied target samples are computed offline and rounded at the input boundary. The observer computes relative $L^2$ error in binary64 against the target at the original evaluation points. Thus the sweep isolates significand precision; its 24-bit point does not use IEEE FP32's narrower exponent range.

**Code & data.** Runner: `experiments/expC11_true_precision_law/runge_comparison.py`; renderer: `experiments/expC09_bandwidth_figures/combined.py`; arithmetic specification: `experiments/expC11_true_precision_law/SPEC.md`. This result directory holds `data/config.json` (configuration, predictions, source hashes), `data/measurements.jsonl`, `data/summary.json`, `data/validation.json`, `data/models/p*.npz`, and `figures/all_targets_chirp_runge25_precision_law_true.png`. Reproduce with `.venv/bin/python experiments/expC11_true_precision_law/runge_comparison.py --workers 8`; it resumes completed measurements.

## Results

Runge starts with relative error above one at 8–10 bits, then decreases overall, with local increases at a few precisions. Its error is $9.57\times10^{-6}$ at 24 bits and $2.50\times10^{-14}$ at 53 bits, versus chirp's $6.81\times10^{-5}$ and $1.04\times10^{-13}$ respectively. These are about sevenfold and fourfold reductions. The final two Runge points are close ($2.66\times10^{-14}$ at 52 bits); two points alone do not establish an asymptotic floor. No new convergence-rate or intercept fit was computed.

All 46 saved models passed the representability and error-recomputation checks. Saved-model evaluation at 8, 24, and 53 bits reproduced every output bit for bit. All production-source hashes still match the run manifest. The sweep took approximately eleven minutes with eight workers.

### Figures

- `all_targets_chirp_runge25_precision_law_true.png`: the current three-panel figure with Runge 25 added in purple to panel (c), alongside chirp in teal. The dashed reference belongs only to chirp. Panels (a) and (b) retain their measurements and styling.

## Additional details

At $p=8,9,10,11$, the refined selector returns the upper search limit $\lambda=3$; these are bounded-rule choices, not located equality roots. At $p=12,\ldots,53$, the selected bandwidth satisfies the stated score equation. Each status is checked before running the sweep and saved with the predictions.

The arithmetic sources match the hashes recorded for the original chirp sweep. The existing arithmetic and independent-replay tests pass (202 tests). The runner also checks that every saved coefficient, prediction, and singular value is representable at its prescribed precision, recomputes every saved error, and replays the saved models at $p=8,24,53$ bit for bit.

## Conclusions

In this measured sweep, Runge 25 exhibits an overall decline in error as working precision increases and is more accurate than chirp through most of the range. Inclusion in the final figure is pending Sam's review.

## Open questions

Whether to retain Runge in the final figure is a presentation choice for Sam.
