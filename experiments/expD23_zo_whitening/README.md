# ZO w/ Whitening

This experiment asks whether Chaubard-style CD-RGE can fit a precision MLP
when its fixed feature matrix is whitened before optimization. It sweeps hidden
widths 8--1024 for `sin(2*pi*x)` on `[-1, 1]` and compares:

- **global whitening:** one SVD over all initialized feature columns;
- **partial whitening:** independent SVDs over contiguous blocks of at most 32
  feature columns.

Both transforms depend only on inputs and initialized features. Targets are
used only in scalar loss evaluations made by the two-sided zero-order
optimizer. There is no target-dependent least-squares coefficient solve,
autograd, or backpropagation.

The headline sweep uses `rcond=0`: no singular direction is deliberately
discarded. This makes loss of numerical rank or high-width instability visible
rather than silently converting nominal width into a much smaller effective
model.

That distinction should not be oversold: global whitening computes the inverse
geometry of the entire feature matrix and therefore performs much of the
matrix-factorization work associated with a least-squares method. The honest
claim is **zero-order optimization after target-independent whitening**, not
"raw zero-order optimization reaches machine precision."

## Protocol

For each width and whitening method, sixteen gamma values from 0.5 to 256 are
swept directly using the same 1,024-step selection budget. The relation
`lambda = gamma*h` is derived only for reporting; it is not imposed during
optimization or selection. The best midpoint-validation gamma is rerun
independently for 8,192 steps. Only then is a
16,385-point offset test grid evaluated. Every step uses eight Rademacher perturbations,
`lr=epsilon`, initial `epsilon=0.01`, momentum 0.9, and halves both effective
scales every 1,024 steps. The JSON records the precise evaluation counts.

Run and plot with:

```bash
uv run --extra dev python experiments/expD23_zo_whitening/run.py
uv run --extra dev python experiments/expD23_zo_whitening/plot.py
```

Machine-readable outputs and the three-panel lambda/gamma/error figure are in
`results/`. The figure stops at width 128 to focus on the precision regime;
the JSON and CSV retain the complete requested sweep through width 1024.

## Headline result

Global whitening reaches held-out relative L2 error `3.56e-13` at width 64
and `4.55e-13` at width 128. These are direct-gamma results: gamma is selected
without imposing `gamma=lambda/h`, and lambda is reported afterward.

This is not monotone scaling. With no singular-value truncation, global
whitening becomes numerically unstable by width 512 and the run remains at the
zero predictor. Block whitening is much less effective: it reaches `6.02e-4`
at width 64 and also fails at widths 512 and 1024 under this fixed budget.
Thus the experiment demonstrates a precision regime, not a general claim that
whitening makes CD-RGE width-stable.
