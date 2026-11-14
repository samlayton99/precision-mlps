# Independent review of the fitting-method discrepancy

Date: 2026-11-09. Scope: the reviewer's statement that the experimental fitting method differs from the method analyzed in the theory. This review does not assess the separate objections about gradient descent, optimizers, LLM integration, precision formats, or Chebyshev comparisons. It was conducted from a fresh task brief without the earlier conversation. Existing writeups were treated as claims to check against the paper, code, and raw results.

**Verdict: qualified yes for a carefully narrowed correction; no for the proposed justification that the two methods “work the same.”** The new experiments provide useful evidence that a method covered by the general recovery framework can achieve very high measured accuracy. This can repair the misleading presentation without proving a new theorem for the practical solver, provided the manuscript explicitly limits the guarantee to the reference-scaled method. The experiments do not transfer that guarantee to unscaled SVD. Moreover, the later sweeps directly contradict numerical equivalence across the tested widths. The correction is not complete in the supplied manuscript.

The useful claim is that the practical unscaled SVD is simpler and empirically effective, while the theoretical method is a separate, more conservative method that reaches a comparable accuracy floor in specified regimes. “Simpler” is well supported as an implementation statement: the practical arm omits reference envelopes, their scaling, and the numerical-bound machinery. No runtime comparison is needed for that statement, but no speedup claim has been established.

## 1. The manuscript's algorithm identification is materially wrong and needs a direct correction

In the supplied PDF, Section 3.1, lines 183–187, says “our implementation uses a reference-scaled SVD procedure” and displays an unrestricted least-squares argmin in Equation (4). Appendix A, lines 739–742, instead specifies reference-scaled **truncated** SVD and explicitly says it is not an unregularized least-squares minimizer. Appendix C, lines 1878–1880, correctly says the empirical solver uses the unscaled matrix and that the recovery theorem specifies a separate algorithm. These statements cannot all serve as the description of one implemented fitting method.

The appendix theorem assumes an analytic target on a complex neighborhood of an enlarged real interval. Its parameters are: N interior intervals; spacing h=2/N; R=ceil(a_H sqrt(N)) exterior centers on each side; W=N+2R+1 hidden neurons; bandwidth lambda; slope gamma=lambda/h; analytic-neighborhood radius delta; enlarged-interval margin Delta; and a bound B on the target throughout that complex neighborhood. The reference envelopes alpha_j bound the magnitudes of the coefficients of a theoretical approximant. They define D=diag(sqrt(alpha_j)), L=sqrt(sum alpha_j), and S=BL.

The recovery method factors F=AD/sqrt(M), where A contains sampled features and M is the number of samples. It retains singular values above an absolute threshold tau satisfying

`tau*S >= Ebar + a_W*S + b_W`.

Here Ebar bounds the theoretical approximation error; a_W and b_W bound perturbations in the matrix and labels of an exact nearby comparison problem. These are not the practical relative cutoff `2**-52` applied to A. In rank-deficient or truncated problems, column scaling can also change which coefficient solution is selected. Shared feature geometry is not shared fitting mathematics.

The main-text Theorem 3.1 fixes a_H=1 (R=ceil(sqrt(N))); the successful first comparison uses a_H=2.75. Appendix A.1 expressly permits general a_H, so this is a valid generalization within that appendix, but it is not a test of the displayed main-text special case at the same widths.

Sources: [supplied manuscript](</Users/sam/.codex/attachments/65f68427-bc80-4873-ac87-e0fdd36ac6c0/QI_MLPs___ICLR_2027_Submission (18).pdf>), pp. 4–5, 14, 20–24, and Appendix C; [construction appendix](/Users/sam/my-repos/research/collaborations/precisionMLPs/docs/appendix_notes/paper_v5_replacements/construction_appendix.tex:44), especially lines 71–102, 626–637, and 755–805; [actual comparison solve](/Users/sam/my-repos/research/collaborations/precisionMLPs/experiments/expC14_theory_solver_comparison/option_c.py:153).

## 2. The complete data refute “works the same” across width

The first sweep is genuinely encouraging. With a_H=2.75, delta=1, N in {128,256,512,1024,2048}, and Jacobi SVD, the largest ratio of theoretical-method error to unscaled-method error is 2.7534. For N>=256, both methods have measured relative errors below 6e-13 on all six targets. The theoretical method is sometimes better: at N=1024, sine24 has errors 2.13e-14 versus 3.26e-13. These facts support comparable **measured accuracy on this particular sweep**.

They do not describe the later sweeps:

| Sweep and target | N | Unscaled relative error | Theoretical-method relative error |
|---|---:|---:|---:|
| a_H=2, delta=2; Gaussian-windowed sine | 64 | 4.77e-13 | 5.08e-1 |
| a_H=2, delta=2; chirp | 128 | 2.84e-10 | 7.98e-4 |
| a_H=1.375, delta=4; exponential | 32 | 6.13e-15 | 1.00 |
| a_H=1.375, delta=4; sine4 | 128 | 1.14e-14 | 1.08e-1 |
| a_H=1.375, delta=4; chirp | 256 | 7.00e-13 | 5.27e-4 |
| a_H=1.375, delta=4; chirp | 512 | 1.24e-13 | 4.46e-13 |

In the dense sweep the theoretical solver retains zero singular directions at N=32 and 48, returning the zero network. Its errors of one are expected, not solver failures. At N=64 it retains one direction, versus 522 at N=512. The original analytic-error term forces a large cutoff at small widths. The unscaled solver can retain directions that this sufficient theoretical criterion removes.

For example, the first sampled width achieving exponential error below 1e-12 is N=32 for unscaled SVD but N=464 for the theoretical method in the dense sweep. For the Gaussian-windowed sine the corresponding first sampled widths are 64 and 416. These are empirical threshold crossings, not proven minimum widths; they still demonstrate a substantial practical difference.

The distinction between N and actual hidden width W matters. The first sweep ranges from W=193 to W=2299. The dense sweep ranges from W=49 to W=577. Within a pair the two methods have identical W, so the comparison is fair. Across sweeps, the same N does not mean the same W or the same analytic constants.

Raw sources: [first-sweep N512](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_C_geometry/expC14_theory_solver_comparison/option_c_jacobi/N512.json), [first-sweep N1024](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_C_geometry/expC14_theory_solver_comparison/option_c_jacobi/N1024.json), [N64 sweep N64](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_C_geometry/expC14_theory_solver_comparison/option_c_64_512/N64.json), [N64 sweep N128](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_C_geometry/expC14_theory_solver_comparison/option_c_64_512/N128.json), and the dense sweep's [N32](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_C_geometry/expC14_theory_solver_comparison/option_c_32_512_dense/N32.json), [N128](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_C_geometry/expC14_theory_solver_comparison/option_c_32_512_dense/N128.json), [N256](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_C_geometry/expC14_theory_solver_comparison/option_c_32_512_dense/N256.json), and [N512](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_C_geometry/expC14_theory_solver_comparison/option_c_32_512_dense/N512.json). I recomputed sweep summaries from all 30, 24, and 186 raw rows, respectively.

## 3. The general theorem is retained, but the numerical justification has new content

The implementation faithfully retains several important parts of the original framework:

- The tanh reference envelopes include both outermost-halo corrections and the bias coordinate; the approximation bound is the original A.51 with prefactor 128*pi*B/(lambda*delta), not the 512-prefactor final recovery bound.
- All three sweeps satisfy the listed geometry conditions, M>=W+1, and the cell-ellipse sampling conditions with s=8, p=1, and ellipse parameter rho=2. Here s is samples per interior interval and p is the local polynomial degree used by the sample-to-continuous bound.
- The six targets are entire functions, so the enlarged complex neighborhoods exist. The code's rectangle bounds give admissible B values, although some are enormous.
- Projection, retained-coordinate division, and coefficient export follow the prescribed truncated-SVD structure. The cutoff uses a lower enclosure of S and an upper approximation-error bound. The saved cutoff slack is nonnegative in every reviewed row; there is no empirical-error substitution or norm guard that changes the solve.

However, `option_c.py` lines 61–113 do not simply evaluate Equation A.63. They use sharper, weighted polar-factor bounds. Let Uhat and Vhat be computed singular vectors, H=Uhat^T Uhat-I and G=Vhat^T Vhat-I their orthogonality defects, and Sigma the diagonal singular-value matrix. The code bounds H*Sigma and Sigma*G rather than multiplying every defect by the largest singular value. For label projection it similarly bounds H*q, where q=Uhat^T*y is the exact projection of the stored labels.

These formulas follow from Taylor bounds for the matrix functions sqrt(I+H) and (I+H)^(-1/2), with second-order remainders. I checked their algebra against the polar decomposition and found no structural mismatch with the required nearby-problem bounds A.61–A.62. Equation A.63 is expressly introduced as a **sufficient realization**, so a sharper valid realization does not require changing the recovery theorem. Nevertheless, establishing the sharper realization is a new numerical lemma/proof obligation. It should be supplied, not hidden by saying the proof is entirely unchanged.

The change is numerically material. At N=128 in the first Jacobi sweep, the saved weighted a_W is 1.774e-14; inserting the saved defects into the displayed A.63 formula gives approximately 4.740e-14, even with the same residual estimate. The tighter quantities feed the cutoff and can therefore change the returned fit.

The enclosure machinery is also substantive: it uses fixed-point chunks of FP64 inputs, exact digit matrix products within checked dimension/exponent limits, compensated reconstruction, and norm bounds. Its correctness depends on the stated conventional FP64 matrix-product arithmetic, round-to-nearest behavior, and range assumptions. Tests support this machinery; they do not replace its mathematical justification.

Sources: [parameter enclosures](/Users/sam/my-repos/research/collaborations/precisionMLPs/experiments/expC14_theory_solver_comparison/theory_parameters.py:45), [weighted bounds](/Users/sam/my-repos/research/collaborations/precisionMLPs/experiments/expC14_theory_solver_comparison/option_c.py:61), [cutoff](/Users/sam/my-repos/research/collaborations/precisionMLPs/experiments/expC14_theory_solver_comparison/option_c.py:116), [matrix-product enclosures](/Users/sam/my-repos/research/collaborations/precisionMLPs/experiments/expC14_theory_solver_comparison/tight_products.py:1), and PDF pp. 21–22. The existing option-C writeup contains a useful derivation; it is not yet part of the supplied construction TeX.

## 4. Passing numerical conditions is not certification of the plotted accuracy

The figures measure relative error on 16,384 independent uniform midpoint samples, with a second 32,768-point midpoint grid. Doubling the grid changes errors by at most 1.4063% in the first sweep, 0.1535% in the N64 sweep, and 0.2817% in the dense sweep. This is useful convergence evidence for the sampled error estimate. It is not a validated continuous-domain error bound.

The theorem bounds an absolute continuous root-mean-square error, and separately coefficient magnitude. Its full error allowance includes a sample-to-continuous term. At p=1 and rho=2, that term is proportional to B+T, where T is the certified coefficient bound. It is not small merely because measured residuals are small.

Concrete examples make the distinction unavoidable. In the first sweep at N=128, the chirp's certified sampled-error upper bound is about 1.04e65 while its measured relative error is 3.09e-10. In the dense sweep at N=512, the reported A.81 uniform-error upper bound is about 4204.8 for the exponential and 8.81e573 for the chirp; the measured relative errors are 1.98e-14 and 4.46e-13. All the checks can correctly pass while the guaranteed accuracy is vacuous for practical purposes.

For dyadic widths in the first two sweeps, exactly representable cell endpoints support the original RMS input-rounding route. For non-dyadic widths in the dense sweep, the code uses the original A.81 **uniform** bound, deriving a sampled maximum-residual bound from sqrt(M) times the RMS bound. That is a permissible existing route, but it is not literally the A.80 RMS calculation or the A.83/A.84 decomposition used to prove A.1. The dense experiment should identify its route precisely. Its figure remains an empirical relative RMS comparison.

The arithmetic disclosure also matters. Coefficients and SVD factors are FP64, but reference constants, cutoff bounds, and verification use 256-bit directed MPFR and FP64 expansions. The dense chirp uses bounds exceeding the binary64 exponent range. Targets are prepared offline at high precision, and tanh values are correctly rounded using MPFR. Thus this is not a reproduction of Appendix C's entire p-bit pipeline or its particular approximate tanh evaluator and sequential summation. The theorem permits validated auxiliary bounds; “all computations are FP64” would nonetheless be false.

Sources: [evaluation](/Users/sam/my-repos/research/collaborations/precisionMLPs/experiments/expC14_theory_solver_comparison/run.py:247), [dense uniform bound](/Users/sam/my-repos/research/collaborations/precisionMLPs/experiments/expC14_theory_solver_comparison/option_c.py:188), [exact affine grid](/Users/sam/my-repos/research/collaborations/precisionMLPs/experiments/expC14_theory_solver_comparison/affine_grid.py:1), and PDF pp. 22–24 and Appendix C.

## 5. The experiment is a fair paired comparison, but is not a general equivalence test

Both arms receive the same neurons, sample locations, labels, precision, and evaluation procedure within each pair. The unscaled arm uses the same `gelsd` relative-cutoff recipe as the existing convergence experiment, adapted to the matched geometry. The theoretical arm uses a different driver (`gejsv`), scaling, and cutoff. This fairly compares two complete procedures. It does not isolate the effect of reference scaling alone, and it does not reproduce every detail of the existing main-figure computation.

The first sweep's halo parameter was selected to make E_rep/B smaller than 1e-16 already at N=128. This is a legitimate analytic design choice, and the subsequent smaller-halo sweeps help expose its consequence. It is nevertheless a favorable regime by construction: it largely removes the conservative analytic-cutoff bottleneck before the first plotted point. Showing only that sweep as evidence that the solvers generally behave alike would be misleading in light of the later data.

Earlier GESVD/GESDD trials also matter. At N=256 their chirp error is about 8.30e-12 against 1.02e-13 for unscaled SVD, an approximately 82-fold gap. These are successful theorem-condition runs with worse approximation, not violations of the theorem. The matrix-only Jacobi probe is evidence for a principled driver choice: its code uses feature matrices and factorization bounds without target labels. It does not erase the fact that accurate behavior depends on driver and bound implementation. Six deterministic, smooth, one-dimensional, entire targets at lambda=1/4 do not establish generality over analytic target classes, noisy labels, different precision formats, or other activations.

Source: [Jacobi probe](/Users/sam/my-repos/research/collaborations/precisionMLPs/experiments/expC14_theory_solver_comparison/jacobi_probe.py:26), [GESVD N256](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_C_geometry/expC14_theory_solver_comparison/option_c/N256.json), [GESDD N256](/Users/sam/my-repos/research/collaborations/precisionMLPs/results/checkpoint_C_geometry/expC14_theory_solver_comparison/option_c_gesdd/N256.json), and the three option-C configuration files.

There is a smaller reproducibility issue: saved first-sweep and N64-sweep source hashes for `option_c.py` and `theory_parameters.py` no longer match current files. The dense sweep's listed hashes do match. In addition, manifests omit imported implementation dependencies such as `run.py` and the relevant arithmetic helpers from `posterior.py`. This does not itself invalidate results, but a release should retain the exact sources/dependencies for each run. The resume logic skips existing N files while writing a new manifest, so rerunning in an existing directory is not sufficient provenance repair.

## Necessary work versus optional stronger evidence

Necessary to make this correction defensible:

1. Correct the main-text solver description and Equation (4). Identify the practical truncated unscaled solve, and state that the theorem guarantees the separately specified reference-scaled procedure under its conditions. Make figure captions and theorem references consistent with that distinction.
2. Replace “works the same” with a regime-specific empirical statement. Disclose the substantial small-width gaps, either through the dense sweep or an equally explicit summary. Do not describe all sweeps by the favorable first-sweep ratio.
3. State the halo constant and actual width mapping for each comparison. Identify the general Appendix A theorem rather than silently treating a_H=2.75 as the main-text a_H=1 special case.
4. Include the sharper nearby-problem lemma/derivation and enough arithmetic detail to justify its enclosures. The recovery theorem can remain unchanged; its implementation evidence needs this additional justification.
5. Label plotted errors as measured relative grid errors, identify the A.81 route where used, and distinguish successful numerical-condition checks from tight error certificates. Disclose the auxiliary high-precision arithmetic and evaluator differences.
6. Freeze reproducible source snapshots and dependency versions for the retained results; repair provenance without overwriting historical runs.

Optional for this narrowly scoped correction: a new theorem for unscaled SVD; a tight continuous-error certificate for each saved network; broader targets, noise, activations, and precision sweeps; a same-driver factorial ablation separating scaling from cutoff and SVD backend; runtime comparisons; and a full rerun of the original figures with the theoretical method. These become necessary only if the manuscript retains corresponding stronger claims. If the desired claim is specifically that the theorem applies to the practical unscaled solver, a proof or a change to the claimed algorithm is necessary—comparative experiments cannot supply that implication.

## Minimum acceptable claim, as an internal analytical suggestion

> The recovery theorem concerns the reference-scaled truncated-SVD procedure in Appendix A. Our empirical construction uses a simpler unscaled SVD solve and is not covered by that recovery guarantee. On the six tested functions, an implementation satisfying the original recovery interface attains comparable measured accuracy in the specified larger-width/larger-halo regime. At smaller widths, its conservative theoretical cutoff can produce substantially larger errors. These experiments demonstrate practical realizability of the theoretical procedure and empirical effectiveness of the simpler solver; they do not establish solver equivalence or transfer the guarantee to the unscaled method.

A methods sentence should add that the theorem and analytic cutoff are retained, while sharper validated nearby-problem bounds are supplied, and state the mixed arithmetic used for validation.

## Review checks and limits

I re-extracted the supplied PDF and visually checked the complete pages containing A.61–A.70; read the relevant theorem, coefficient-envelope, input-rounding, and empirical-solver passages; inspected all specified solver/bound modules, three configurations, raw JSON results, manifests, and comparison figures; independently recomputed the aggregate numerical facts above; and ran the focused existing tests for option C, tight products, and the dense grid: **20 passed**. No expensive sweep was rerun, and no paper, code, or result was changed.

I found no structural reason the weighted numerical bounds cannot instantiate the original general recovery interface. This is a mathematical/code review supported by numerical tests, not a formal verification of every floating-point enclosure or a new audit of the entire analytic-witness proof. The central negative conclusion does not depend on such verification: the saved comparisons themselves disprove a broad same-behavior claim. A narrower, explicit separation of theorem and practical solver can address this one reviewer concern; it says nothing about whether the rest of the review has been answered.
