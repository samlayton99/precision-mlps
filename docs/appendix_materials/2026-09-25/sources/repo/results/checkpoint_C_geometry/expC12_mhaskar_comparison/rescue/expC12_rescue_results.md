# Classical numerical rescues for the Mhaskar construction — data-obvious

## TL;DR

- Compensated dot products greatly reduce readout arithmetic error. For the saved 53-bit model, relative error against a reference dot product falls from $0.139$ to $5.02\times10^{-17}$ on the diagnostic grid.
- Better readout arithmetic alone does not rescue the chirp approximation. Symmetric coefficient assembly and compensation leave its held-out error near $0.96$.
- Three Richardson levels improve the best checked 53-bit error from $0.959$ to $0.940$, with degree 10 and 51 neurons. This is a modest improvement, not accurate chirp recovery.

## Question / hypothesis

Can classical stable coefficient assembly and summation materially improve the Mhaskar baseline while retaining $p$-bit model operations and storage?

## Experiment design

The target, domain, width budget, observations, and validation/reporting grids are those of the strict comparison: the chirp on $[-1,1]$, at most 1024 neurons, 1024 validation midpoints, and a separate 8001-point reporting grid. All production primitives and every correction register retain $p$ significant bits. The models still store one $p$-bit value per scalar parameter. Compensation uses extra arithmetic and temporary storage; it is not a free change in accuracy or a native hardware benchmark.

The readout study holds the 46 saved strict Mhaskar models fixed and compares six algorithms: sequential summation, magnitude-sorted summation, pairwise summation, Neumaier compensation, Dot2, and magnitude-sorted Dot2. Dot2 follows [Ogita, Rump, and Oishi, Accurate Sum and Dot Product](https://www.tuhh.de/ti3/paper/rump/OgRuOi05.pdf), Algorithm 5.3. It corrects both multiplication and summation errors using error-free transformations implemented with ordinary $p$-bit operations. The implementation uses Dekker/Veltkamp splitting and Knuth TwoSum, with no FMA or wider floating-point temporary. Neumaier alone corrects additions after each product has already been rounded.

The construction study computes each normalized binomial stencil from its central coefficient outward, mirrors coefficients to preserve parity exactly, and merges contributions using compensated summation. Chebyshev projection and polynomial-coefficient accumulation use compensated sums; the Taylor-derivative recurrence uses compensated dot products. Every completed coefficient is rounded to one $p$-bit value. The same 2210 degree/step candidates are searched at seven representative precisions: 8, 16, 24, 32, 40, 48, and 53. Validation uses the strict comparison's certified partial-residual pruning, preserving the full-grid minimizer. The held-out grid never selects parameters. QUILL is also evaluated with Dot2 at those seven precisions using its fixed saved model.

The higher-order study combines centered-step approximants using Richardson extrapolation, derived from their even-power step-error expansion. Starting from $R_0(h)$, form $R_\ell(h)=[4^\ell R_{\ell-1}(h/2)-R_{\ell-1}(h)]/(4^\ell-1)$. Levels 1, 2, and 3 are tested at 24 and 53 bits, along with the original degree/step choices whenever the resulting union of neurons fits the width budget. Construction coefficients and the merged readout are computed at $p$ bits. The final object remains a flat tanh network; no factored or high-precision evaluation is substituted. [Fornberg's finite-difference work](https://www.colorado.edu/amath/sites/default/files/attached-files/mathcomp_88_fd_formulas.pdf) was also consulted; its general grid-weight algorithm was not needed for the explicit equispaced binomial stencil used here.

A separate diagnostic compares readout algorithms with a 256-bit reference dot product of the **same already-rounded weights and activations**, on 257 points. This reference is used only to assess arithmetic accuracy. It never supplies a model coefficient, prediction in the reported precision experiment, or model-selection criterion. Thus it isolates accumulation/product error without silently repairing the model's inputs to that dot product.

**Code & data**

- Algorithms: `experiments/expC12_mhaskar_comparison/robust.py`, `robust_kernels.c`.
- Main rescue runner: `experiments/expC12_mhaskar_comparison/rescue.py`.
- Higher-order study: `experiments/expC12_mhaskar_comparison/extrapolation.py`.
- Reference-only diagnostic: `experiments/expC12_mhaskar_comparison/rescue_diagnostics.py`.
- Tests: `tests/test_mhaskar_rescue.py`.
- Measurements: `data/readout_only.json`, `data/summary.json`, `data/extrapolation.json`, `data/readout_error_audit.json`.
- Candidate measurements and bounds: `data/search_p*.npz`; saved candidate construction arrays: `data/construction_p*.npz`.
- Saved models: `models/`; provenance and validation: `data/config.json`, `data/validation.json`, `data/additional_validation.json`.
- Figures: `figures/rescue_summary.png`, `figures/readout_error_diagnosis.png`, `figures/numerical_rescue_comparison.png`, `figures/extrapolation_check.png`.

## Results

| Method | Error at 24 bits | Error at 53 bits |
|---|---:|---:|
| Original strict construction/readout | $0.960274$ | $0.958814$ |
| Original model, Dot2 readout | $0.959841$ | $0.958164$ |
| Symmetric/compensated construction and Dot2 | $0.959895$ | $0.958167$ |
| Also cancel leading step errors | $0.958140$ | $0.940119$ |

These are relative errors on the 8001-point held-out grid. The 53-bit higher-order selection uses degree 10, three extrapolation levels, and 51 neurons, compared with degree 7 and 15 neurons in the original selected model. The 24-bit higher-order selection uses degree 6 and 31 neurons. All stay within the maximum width budget.

More accurate readout does not guarantee lower error against the target when the rounded model is already inaccurate: rounding in the sequential readout can accidentally partly compensate other errors. At 16 bits, for example, the rebuilt compensated model is slightly worse on the held-out grid. The study retains such outcomes rather than selecting results on the reporting grid.

### Figures

- **Rescue summary:** the left panel isolates dot-product arithmetic error in fixed saved models; the right panel shows complete-model target error. The higher-order variant has only two measured precisions, displayed as separate diamonds. The two panels use the explicitly stated diagnostic and reporting grids, respectively.
- **Readout diagnosis:** shows how Dot2 follows the reference dot product while both retain nearly the same large error against the chirp. This distinguishes accurate accumulation of existing model data from accurate approximation of the target.
- **Numerical rescue comparison:** compares all six readout choices, the seven construction reruns, and QUILL with equal access to compensated dot products. The first two panels deliberately zoom in near unit relative error.
- **Higher-order check:** compares the original, compensated, and extrapolated constructions at the two tested precisions. It does not represent an unmeasured full precision sweep.

## Additional details

The combined verification suite passes 99 tests. New tests recover known sum and product residuals lost by ordinary arithmetic, compare Dot2 with an independent reference, verify exact coefficient parity and $p$-bit storage, check low-degree network approximation rates, and verify that extrapolation cancels the appropriate even-power error terms. Saved improved models replay their reported errors, and all stored parameters and outputs satisfy the prescribed precision.

The large reduction in dot-product error does not identify a single remaining cause of the total approximation error. Finite-step approximation, polynomial degree, rounded coefficients, and rounded activation values all precede the final accumulation. The reference-dot diagnostic establishes that repairing accumulation alone is insufficient for these saved models. It is not a precision lower bound for every Mhaskar-style implementation.

## Conclusions

Classical compensation successfully repairs much of the readout arithmetic error, and higher-order centered differences modestly improve the tested complete construction. These changes do not produce an accurate chirp approximant within the tested $p$-bit, flat-network implementation.

## Open questions

- Which remaining error source dominates after higher-order step cancellation: coefficient construction, activation rounding, or finite-step approximation?
- Would a structured evaluation of the stencil improve accuracy enough to justify a separate comparison from ordinary flat-MLP inference?
