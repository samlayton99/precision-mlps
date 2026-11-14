# Independent audit of the final QUILL solver revision

Date: 2026-11-10. Scope: the reviewer's fitting-method criticism only.

## Verdict

**Ready after two specific reporting fixes.** The proposed revision is a sound response to the claim that the experimental fitting method differs from the analyzed method. It acknowledges the difference, attaches the guarantee to reference-scaled recovery, and supplies a controlled comparison demonstrating useful empirical agreement. It does not need to prove universal equivalence of the solvers.

I found no mathematical defect in the sharper same-witness approximation bound or the weighted-polar numerical bounds, and no mismatch between the final figure's two arms and its stated geometry/backend. The numerical claims in the proposed subsection are correct. The two needed additions are the actual analytic magnitude bounds used in the theoretical solve and a complete disclosure of the original experiment backends. Neither requires new fits or changes to the figure.

This review was performed without reading previous independent-review reports. I inspected all nine blocks in `docs/reports/quill_solver_revision_2026-11-10/COPY_PASTE_EDITS.txt`, the supplied submission (18), the matching appendix source, final data and source snapshots, and the rendered figure. Shared-context synchronization failed with a transport error; the local context installation check passed. This does not affect the inspected repository evidence.

## Required fixes, ranked

### 1. Specify the target-dependent magnitude bounds used in the theorem arm

**Priority: P2, reproducibility.** The proposed Appendix C subsection specifies delta and Delta, but never specifies B. B enters the witness budget S=BL and the cutoff through b_W/S. It also determines the coefficient and continuous-error allowances. It cannot simply be left implicit because the experiment is presented as a concrete realization of the theoretical recipe.

Location: `COPY_PASTE_EDITS.txt:248-259`, step 9, immediately after the paragraph specifying geometry and sampling. Implementation: `experiments/expC14_theory_solver_comparison/theory_parameters.py:80-88` (identical in the final theoretical run's saved source).

Insert:

```latex
For the five targets $e^x$, $\sin(4\pi x)$,
$\sin(24\pi x)$, the chirp, and the wave packet,
respectively, we use outward upper bounds on
\[
e^{89/16},\qquad \cosh(16\pi),\qquad
\cosh(96\pi),\qquad \cosh(420\pi),\qquad
e^{128}\cosh(32\pi)
\]
as the analytic magnitude bounds $B$.
These follow by enclosing $U_\delta(I_\Delta)$ in
$|\operatorname{Re}z|\le1+\Delta+\delta$,
$|\operatorname{Im}z|\le\delta$.
The bounds are stored in the auxiliary MPFR arithmetic,
including its extended exponent range.
```

These are precisely the code's formulas at delta=4 and Delta=9/16. For the chirp, the imaginary part of its phase is bounded by 16*pi*(2+Delta+delta)*delta=420*pi. For the packet, the Gaussian magnitude is at most exp(8*delta^2), and the sine magnitude is at most cosh(8*pi*delta). The chirp bound exceeds FP64 range; the model arrays do not. The existing separation between model arithmetic and auxiliary bound verification is accurate.

### 2. Disclose that the controlled raw arm also differs in backend from the original width/bandwidth experiments

**Priority: P2, interpretation of the fitting-method response.** The proposed text already says that the new DGEJSV comparison is separate from the precision-controlled DGELSS experiment. The original width/bandwidth experiments use DGELSD. State both, so a reader cannot mistake the new raw arm for a replay of every original experimental implementation. This is a disclosure repair, not a demand for another comparison.

Location: `COPY_PASTE_EDITS.txt:245-246`, step 9. Replace the final sentence of that paragraph with:

```latex
This controlled comparison uses a common DGEJSV backend;
the original width and bandwidth experiments use DGELSD,
and the precision-controlled experiment above uses DGELSS.
It therefore compares the two recovery prescriptions under
matched numerical conditions rather than replaying the
original experiments.
```

Evidence for the original FP64 driver: `experiments/expC09_bandwidth_figures/convergence.py:103-104`, `run.py:78`, `analytic_packet.py:42`, `smooth_absolute.py:42`, and `appendix_bandwidth.py:88`. The submission's Appendix C explicitly specifies the precision-controlled DGELSS driver at printed lines 1869-1880.

## Mathematical assessment

### Sharper approximation bound: valid, same witness

Step 5 combines the horizontal and pole estimates (A.24)-(A.25) with the boundary-correction estimate (A.42). The signed decomposition of the already-constructed network is

\[
f-q^\star=(\mathcal B_h^{\rm eff}f-q_{\rm bd})+
\mathcal P_h[F_x]-\mathcal H_h[F_x].
\]

Taking absolute values gives the proposed three-term expression. The density bound M_sigma=B/(delta-sigma-d) is already established in the appendix. Its denominator is positive under admissibility. Choosing m=floor((R+1)/2) is the same choice used for the original witness and its envelopes. Both the old simplified bound and the unsimplified sum bound this identical q-star, so taking their minimum is valid.

No coefficients, correction slots, alpha_j, D, L, S, or K_ref change. In particular, the reference-coordinate budget remains ||D^{-1}v-star||_2<=BL. Steps 6 and 7 correctly update Ean and the identity for R_X. The formation allowance remains epsilon_E=Ebar-Esharp>=0, so Delta_rec and all export/input/evaluation allowances retain their roles. Since Esharp<=Erep, the existing headline analytic rate and constants remain valid for the original theorem's qualifying runs. This is a tighter sufficient cutoff within the theorem, not a substitution of the practical cutoff.

The numerical improvement is substantial but legitimate: at N=256 the old Erep/B bound is 7.6033e-6 while the selected sharp bound is 9.7897e-19. Such a large difference comes from retaining the strong boundary estimate instead of its deliberately coarse conversion to a simple width-dependent rate.

### Weighted-polar bounds: correct formulas and constants

Write A=(I+H)^{1/2}, C=(I+G)^{1/2}, so Uhat=U0*A and Vhat=V0*C. Then

\[
A\widehat\Sigma C-\widehat\Sigma
=(A-I)\widehat\Sigma C+\widehat\Sigma(C-I).
\]

Taylor's second-derivative bounds for sqrt(1+t) and (1+t)^(-1/2), over |t|<=e<1, are respectively e^2/[8(1-e)^{3/2}] and 3e^2/[8(1-e)^{5/2}]. The spectral theorem applies because H and G are symmetric. The proposed a_W formula follows with the displayed weighted norms and ||C||<=sqrt(1+oV). No commutation of H and Sigma is assumed.

For labels, U0^T*ytilde=(I+H)^(-1/2)q. The triangle inequality yields the stated d_p, including ||p-q||. The existing y0 construction and division-rounding bound then yield b_W. The export comparison still uses the original polar discrepancy allowance; the tighter formulas do not silently remove it.

The manuscript-level derivation is sufficient. It is an alternative sufficient realization of the existing nearby-problem interface, as the proposal says. No change to (A.65) is needed.

## Numerical implementation and confidence

The final theoretical run uses the saved sources in `results/checkpoint_C_geometry/expC14_theory_solver_comparison/option_c_sharp_bound_64_768/source_snapshot/`. Source checksums match their provenance. The relevant current solver, parameter, sharp-bound, affine-grid, product-enclosure, and Jacobi files also match that snapshot. The later difference in `readout_norms.py` is plotting only. The raw-arm snapshot has its own provenance and uses the same DGEJSV call and pairwise projection/export; it does not use the DGELSD `actual` arrays incidentally saved by `option_c.py`.

I checked these points:

- The geometry satisfies all four admissibility inequalities at all 12 resolutions; M>=W+1 and the stated ellipse/sampling inequalities also hold.
- gamma=N/8 and beta_j=N/8-j/4 are exactly representable. The affine training routine forms exact rational-grid preactivations through the integer phase (i/s-j)/4, avoiding rounded-center ambiguity at non-dyadic N. Evaluation midpoints similarly have exactly representable affine preactivations at these sizes.
- Both final arms use A/sqrt(M) or AD/sqrt(M), the same normalized target labels, DGEJSV with the same options, rounded FP64 divisions, and explicit pairwise projection/export. The only intended solver-prescription changes are D and the cutoff rule.
- The final theoretical solve chooses an upward threshold from a_W+(Ebar+b_W)/S_lower. I independently recomputed all 72 saved target/width lower-cutoff inequalities at 512-bit precision; they pass. The figure uses 60 of these cases, omitting the sixth mixture target.
- The fixed-point product scheme enforces the exact integer-dot-product bit budget. Its decomposition tails, compensated reconstruction errors, matrix multiplication formation allowance, and norm-bound errors are retained. The Frobenius fallback avoids dropping entries when exact normalization is unavailable.
- The code propagates errors from the projection residual and Gram defect to the weighted label correction. The coefficients are not refined by these diagnostics.
- The non-dyadic continuous-evaluation route uses (A.81) with the valid maximum-residual upper bound sqrt(M)*R_X. It does not incorrectly substitute sampled RMS for sampled maximum error.

The focused command

```text
.venv/bin/python -m pytest -q experiments/expC14_theory_solver_comparison/test_tight_products.py experiments/expC14_theory_solver_comparison/test_option_c.py experiments/expC14_theory_solver_comparison/test_dense_grid.py
```

passed **20 tests**. These include independent high-precision residual-enclosure checks, polar-comparison checks, mixed-exponent/subnormal cases, exact affine geometry, large analytic constants, and cutoff checks.

This supports the stated numerical validation under the specified arithmetic assumptions. It is not a complete formal verification of Python, NumPy, BLAS/LAPACK, MPFR, or the full theorem. I did not rerun the whole sweep or independently replay every large factorization. Offline 160-bit target preparation is separate from the coefficient solve; its small error is covered in the formation allowances rather than supplying high precision to fitted coefficients.

Meeting these numerical conditions does **not** give a useful near-machine-precision continuous-error certificate here. For example, the saved A.81 bound at N=768 is about 3860 for exp and 8.09e573 for the chirp. With p=1 and these global analytic bounds, the transfer allowances are extremely loose. The proposal correctly calls the plotted errors measurements and explicitly excludes certified continuous-domain errors. Preserve that statement.

## Figure and quantitative claims

All input hashes recorded in the final panel's provenance match the current saved inputs. All 120 plotted readout norms match direct sums of absolute saved coefficients, including bias.

The actual hidden widths are:

```text
N:  64  128  196  256  320  384  448  512  576  640  704  768
W:  87  161  237  301  371  439  509  577  643  711  779  847
```

The renderer uses these W values at ordinary numeric positions. Axis ticks such as 256 and 512 are tick locations, not a replacement of W by N. Solid curves are the raw fixed-cutoff arm and dashed curves are the reference-scaled sharp-bound arm. The y-axis abbreviation does not alter the metric: the code computes the norm of evaluation residuals divided by the norm of target values, and the panel title and caption identify relative error. The bias is included in the right panel as the caption states.

For every tested W>=301 and all five plotted targets:

| Quantity | Verified maximum |
|---|---:|
| Theoretical-arm measured relative error | 5.6151491423e-13 (sine24, N=512) |
| Raw-arm measured relative error | 4.4818384613e-14 (chirp, N=256) |
| Absolute norm difference divided by raw norm | 16.3837% (exp, N=640) |
| Absolute norm difference divided by theory norm | 19.5939% (exp, N=640) |
| Evaluation-grid doubling change, over all plotted widths/arms | 0.3159192% (theory exp, N=576) |

Thus the thresholds below 6e-13, below 20%, and below 0.32% are all correct. The 20% statement remains true under either natural denominator convention.

“Comparable accuracy” should mean that both attain high absolute accuracy in the stated regime. It should not mean numerically equal errors: the largest theory/raw error ratio there is about 32.32 (sine24, N=512). This is visible in the figure and does not invalidate the stated threshold-based comparison.

## Optional wording improvements, not required repairs

1. At `COPY_PASTE_EDITS.txt:230`, begin “For prescribed neurons, we use unscaled truncated SVD ...”. The convenience argument is already scoped to the readout solve and is correct; this addition makes clear that selecting a dictionary/geometry is a separate issue.
2. At lines 280-283, replace “with comparable accuracy and coefficient magnitudes” by “with high accuracy and similar coefficient magnitudes”. This avoids a reader interpreting “comparable” as equal error within a small factor.
3. At lines 55-56, use “Under the numerical conditions of Appendix A, reference-scaled recovery returns ...”. The full theorem already states these conditions, but making them explicit in the headline summary would help.
4. When interpreting the new figure, say that non-dyadic widths use the appendix's uniform bound. They are not instances of the cell-preserving RMS-rounding argument with unchanged RMS allowances. The current protocol correctly identifies (A.81); no new theorem is needed for the restricted experimental claim.

No universal solver-equivalence theorem, all-target sweep, new backend experiment, or new tight continuous-error certificate is required to answer this specific criticism. Applying the two reporting fixes, retaining the existing guarantee/measurement distinction, and compiling the full manuscript are sufficient next steps.

## Short explanation for Sam

The revision works for this review concern. The sharper bound and numerical derivation are legitimate, and the final comparison supports using raw SVD as a convenient practical method in the tested regime. It does not transfer the theorem to raw SVD. Before sending it, include the actual analytic B bounds and state that the new comparison uses DGEJSV while the original experiments used DGELSD/DGELSS. The reported numerical thresholds are correct; no new experiment is needed.
