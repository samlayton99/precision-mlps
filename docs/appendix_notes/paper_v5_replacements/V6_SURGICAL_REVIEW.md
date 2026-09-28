# Version 6: surgical review

Source: `QI_MLPs___ICLR_2027_Submission (6).pdf`, 42 pages. Page numbers below are PDF page numbers. Review only: no manuscript or TeX files were changed.

## What survived correctly

The construction reordering, activation-specific bandwidth formulas, solver description, and new finite-kernel proof are present. The internal mathematical references in the reviewed appendix resolve. Theorem 3.2's appendix (A.8, pp.38–41) retains the correct normalization, Fourier constants, correction signs, interlacing indices, ratio bounds, and conversion to residual/step bounds. Independent review found no consequential mathematical integration error there. This was not a fresh proof audit of every unchanged construction lemma.

The circuit correction was not incorporated: A.9 still contains the old text. No existing labels need renaming.

## Fix these first

### 1. Restore the conditions in main Theorem 3.1

**Page 5. Find:** `The computed network` inside Theorem 3.1.

The preceding version explicitly required admissibility and numerical implementation conditions. That clause is absent in version 6, although the appendix theorem still requires them. Replace the beginning of that sentence with:

```tex
Under the admissibility, sampling, and numerical conditions of
Theorem~\ref{thm:qi-numerics-explicit}, the computed network
```

Keep the rest of the sentence and both displayed inequalities.

### 2. Restore the nonzero sampled-target hypothesis

**Page 6, Theorem 3.2. Find:** `Fix the hidden geometry`

Replace that opening with:

```tex
For a nonzero sampled target, fix the hidden geometry
```

Equation (9) divides by the squared sampled target norm. The appendix correctly requires this to be nonzero. The informal theorem need not repeat the entire matrix definition.

### 3. Use the same upper-bound notation in main and appendix

**Page 6, Section 3.4 only:** Equation (9), the definition immediately following it, and the following slow-decay paragraph use `\rho_i` for the upper bound. Appendix A.8 defines `\rho_i=\mu_i/\mu_1` as the exact ratio and uses `\overline\rho_i` for the upper bound.

Change the main-text upper-bound symbols to `\overline\rho_i`. Leave the exact-ratio definition in the appendix unchanged. This is a local notation fix; do not run a project-wide replacement.

### 4. Apply the circuit correction

**Page 42, A.9. Find:** `For each circuit node`

The old recurrences still say `E_v <= ...`, while the proof concludes that E_v is a polynomial with nonnegative coefficients. The recursive equalities establish that assertion; arbitrary inequalities do not.

Use the already supplied `circuit_error_appendix.tex` as the replacement for this subsection. If making the correction by hand instead:

- Define B_v as a bound on the exact intermediate value.
- Say that E_v is defined recursively, with E_v=0 at input and constant nodes.
- Change the two recursive inequalities to equalities.
- Keep the separate assumption that approximate operands remain in the prescribed domains.

The final output-error inequality remains an inequality.

### 5. Repair existing missing references locally

There are 11 printed `??` references in this PDF. Most are in the main text. These references were not caused by renaming the retained appendix labels.

| Location / nearby text | Correct available destination |
|---|---|
| p.5, Theorem 3.1: `The full proof and formal theorem statement` | `Appendix~\ref{app:construction-theory}` |
| p.5, bandwidth formula: `gives the derivation` | `Appendix~\ref{sec:qi-bandwidth}` |
| p.5, slopes `grow approximately linearly with width` | `Appendix~\ref{sec:qi-bandwidth}` |
| p.6, Theorem 3.2: `The formal statement, proof, and numerical validation` | `Appendix~\ref{app:note-spectrum}` |
| p.32, `Figure ... complements Table` | Replace `Table~\ref{tab:comparison}` with `\ComparisonTable{}`. The comparison file already defines this macro and successfully uses it in its opening paragraph. |

The references to the unprovided joint-training and Boolean appendices require the intended material; they cannot all be repaired merely by changing a label. Theorem 3.3 remains outside this proof-writing pass.

### 6. Complete the numerical settings in A.8

**Page 41. Find:** `The saved checks increase quadrature order`

Insert this sentence immediately before it:

```tex
The calculation uses $\vartheta=\arctan(\pi/(2\gamma h))$,
retains enough lattice terms to make $\delta_\gamma\le10^{-18}$,
and retains $\lceil18/(\gamma h)\rceil$ absent centers per side.
Actual feature-SVD ratios above $10^{-18}$ are treated as numerically resolved.
```

These are existing saved-method settings, not a new experiment. They are recorded in `2026-09-25-revised/02_finite_kernel_ratios.tex` under the lattice allowance, exterior centers, and numerical calculations. Keep the current refinement-check and floating-point qualifications after this insertion.

## Eight safe compression edits

Together these save roughly 140 words. No equations, labels, hypotheses, or numerical qualifications need changing.

### C1. Construction: shorten the first boundary roadmap

**Page 15, construction file. Find:** `Since $\mathcal B_h f$ depends only`

Replace the paragraph through `will be corrected together with this functional.` with:

```tex
We correct $\mathcal B_h f$ together with the endpoint contribution
introduced by discretization.
```

### C2. Construction: delete the repeated boundary roadmap

**Page 17. Find:** `Thus the correction must treat both`

Delete the complete sentence ending `midpoint endpoint term.` The preceding equation and C1 already establish this.

### C3. Construction: shorten the shared-assumptions introduction

**Page 20. Find:** `The recovery argument below is conditional`

Replace through `before applying the same bounds.` with:

```tex
Recovery requires the following dictionary properties, verified above
for tanh and in Section~\ref{app:gelu-silu} for GELU/SiLU.
```

Keep the following M >= n assumption and all of the displayed conditions.

### C4. Bandwidth: delete the repeated figure signpost

**Page 34. Find:** `shows the width dependence`

Delete the complete sentence:

```tex
Figure~\ref{fig:bw-width-rules} shows the width dependence
of the refined bandwidth selections.
```

The width-dependence paragraph already introduces that figure on the next page.

### C5. Solver: merge its introduction

**Page 36. Find:** `The precision experiment controls arithmetic`

Replace the opening paragraph through `after each elementary operation.` with:

```tex
The precision experiment uses a binary format with $p$ significand bits,
including the leading bit, and rounds every elementary operation during
feature construction, readout recovery, and evaluation.
```

### C6. Solver: delete the repeated precision sentence

**Page 38. Find:** `The decomposition and subsequent`

Delete the sentence ending `performed at $p$ bits.` The preceding solver paragraph already specifies working arithmetic throughout. Keep the following sentence distinguishing this unscaled empirical solver from the reference-scaled recovery theorem.

### C7. Solver: shorten the external error-evaluation paragraph

**Page 38. Find:** `Reported relative errors are measured afterwards`

Replace through `additional precision to the fitted model.` with:

```tex
Relative errors are evaluated afterwards in binary64 against the target
at the original, unrounded grid locations, so they include input quantization.
This finite-grid evaluation does not affect model precision.
```

### C8. Kernel proof: delete the literal restatement

**Page 39. Find:** `Thus contributions at different frequencies`

Delete the sentence ending `change by different factors.` The preceding sentence already specifies the multiplier gain at low and high frequencies. Keep the matrix-ordering statement and finite-network qualification.

## One layout fix that improves the reading order

A bandwidth figure separates the feature-definition equation on p.36 from its explanation on p.38. Finish the bandwidth floats before starting the solver subsection. If `placeins` is available, put `\FloatBarrier` at the end of the bandwidth file, after its two figure environments. If it is not already loaded, `\usepackage{placeins}` is required in the preamble. This is optional; it does not affect the mathematical result.

## Remaining issues that are not replacement regressions

1. **Main Eq.(4) versus the guaranteed solver:** Eq.(4) defines an unregularized minimizer. The appendix guarantees reference-scaled truncated-SVD recovery. Those descriptions still need to be reconciled; arbitrary minimizers do not inherit the coefficient bound.
2. **Main Corollary 3.1.1:** the appendix gives a fixed-design, above-allowance result over a certified width range. It does not alone prove a uniform O(log(1/epsilon)) statement while lambda changes with epsilon.
3. **Section 3.3:** the formula sets a frequency-dependent aliasing score; it is not derived as the minimizer of readout amplification plus boundary cost. Replace `by balancing aliasing effects with readout amplification and boundary-correction cost` with `by setting a frequency-dependent aliasing score equal to this tolerance`.
4. **Figure 3(c) and the new Adam paragraph:** the current panel is acquired bandwidth. It does not display the denominator/tracking intervention that the paragraph cites it for. The intended intervention plot/protocol still needs to be inserted, or that citation removed until it exists. The caption remains a placeholder even though the current graphic already has three panels.
5. **Numerical-count scope:** A.8's example is width 153, while its text also mentions the separate main width-512 experiment. Its configuration cannot substitute for the latter's missing protocol. Preserve the existing distinction.

## Obvious draft residue outside the appendix

The title remains the conference-template title; Table 2 contains an author-to-self placement comment; Table 3 repeats the Gemma three-digit row and lacks its five-digit row; the abstract's 1,152 evaluations do not match the later 250 examples x nine settings (=2,250); the disclosure/ethics/reproducibility sections still contain template instructions. These are visible draft issues, not consequences of the appendix replacements. Restore the actual results/counts rather than guessing a replacement row or number.

## What not to cut

Keep the successful-run and numerical hypotheses, the fixed-design qualification on the width corollary, the representative-frequency heuristic limitation, the distinction between the empirical and guaranteed solver, and the analytic-versus-floating-point certification distinction. These specify the claims' scope. Compression that removes them would strengthen claims without supplying proofs.
