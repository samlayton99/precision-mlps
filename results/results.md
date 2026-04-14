# Results

## Exp01 -- Lambda Tradeoff

U-shaped error-vs-lambda confirmed. Viable regime: lambda ~ 0.20-0.35. Optimal at lambda = 0.25.

- Left side (lambda < 0.15): ill-conditioning, errors explode, worse with larger N.
- Right side (lambda > 0.5): aliasing-limited, width-independent, ~1e-3.
- Optimal lambda is target-independent (~0.25), as theory predicts.

To do: finer sweep around 0.225-0.275.

## Exp0A -- QI vs Lstsq (fair comparison)

Four methods on identical geometry (gamma, centers, halo), 4 targets, N={32,64,96,128}, lambda={0.20,0.25,0.30}:

| Method | Arithmetic | Typical best L_inf | Notes |
|---|---|---|---|
| lstsq mpmath SVD | mpmath (50 dps) | **2e-16 to 9e-16** | Best overall, true machine eps |
| QI construction | mpmath (50 dps) | 2e-15 to 6e-15 | ~10x worse than lstsq mpmath |
| lstsq fp64 | fp64 | 1e-13 to 1e-14 | Limited by Phi conditioning in fp64 |
| QI construction | fp64 | 1e-8 to 1e-12 | Worst -- convolution cancellation kills it |

Key findings:
- **Geometry is what matters, not the weight-computation method.** Given the right (gamma, centers), lstsq finds weights as good or better than the QI convolution formula.
- **QI fp64 is the worst method** due to cardinal coefficient cancellation (~300 magnitude, alternating sign). lstsq fp64 on the same geometry beats it by orders of magnitude.
- **QI's advantage was purely arithmetic**: mpmath convolution, not a better formula. When lstsq gets mpmath too, it wins.
- Phi is rank-deficient (rank ~N+12 out of N+2*halo+1), but truncated SVD handles this cleanly.

## Future
- how does truncated svd work on fp64 for lstsq
- Exp02: gradient descent from the QI solution -- does the optimizer drift away?
- Fix width, vary gamma, see where trained optimal lambda goes.

