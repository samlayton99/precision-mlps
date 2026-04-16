# Results

## Exp01 + Exp0A -- Lambda Tradeoff and QI vs Lstsq

Consolidated sweep: 4 targets, N={16,32,64,96,128}, lambda=0.01-1.0 (coarse) + 0.22-0.28 (fine). Three precision combos: fp64/fp64, mpmath QI/fp64 lstsq, mpmath/mpmath. See `consolidated_linf.png`.

**Conclusions:**

1. **Both QI and lstsq reach machine epsilon** given sufficient arithmetic precision (mpmath). lstsq mpmath: 2e-16. QI mpmath: 4e-16 to 2e-15.
2. **Lstsq is strictly better than QI** on identical geometry. It directly minimizes the residual rather than using a fixed convolution formula. lstsq mpmath beats QI mpmath in 48/48 configs.
3. **Lstsq fp64 is comparable to QI mpmath** (~1e-13 vs ~1e-15). The lstsq fp64 floor is catastrophic cancellation in fp64 arithmetic, not a method limitation -- it scales with precision.
4. **Optimal lambda ~ 0.23-0.26** for both methods. U-shaped curve confirmed: ill-conditioning below 0.15, aliasing above 0.5.
5. **QI is more sensitive to lambda than lstsq.** QI error is dominated by the aliasing term (exponential in lambda). lstsq error is dominated by arithmetic precision, making it nearly lambda-independent in the viable regime.

**Implication:** Geometry (gamma, centers) is the hard part. Given correct geometry, even fp64 lstsq gets 1e-13 without special arithmetic. The research question reduces to whether an optimizer can discover the geometry.

## Future
- Exp03: geometry ladder -- progressive constraint relaxation to pinpoint where precision is lost.
- Exp02: gradient descent from QI solution -- does the optimizer drift away?
- Fix width, vary gamma, see where trained optimal lambda goes.

