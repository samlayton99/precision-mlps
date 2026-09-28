# Independent appendix structure review

Reviewed September 26, 2026. Read-only review of the current 36-page submission, with the three supplied LaTeX attachments used to resolve formulas and locate passages. No paper files were edited or compiled.

## Recommendation

Finish the tanh theorem before presenting the GELU/SiLU construction. The readable order is **setup → tanh reference network → common recovery and continuous error → GELU/SiLU → bandwidth → comparisons**. This reduces twelve peer subsections to six, without discarding the proof's useful intermediate results.

Moving all of current A.7 downstream is valid only after making the recovery argument explicitly conditional on a short dictionary interface. As currently written, A.8 invokes the GELU/SiLU normalization and coordinate envelopes, and A.9 invokes their complex-growth lemma. Moving the section without changing those statements would leave unproved prerequisites in the common proof.

### Exact mapping

| Proposed subsection | Current material | Treatment |
|---|---|---|
| A.1 Setup and theorem | A.1, p. 15 | Keep the analytic class, geometry, admissibility, sample grid, algorithm specification, and theorem. Use one short roadmap. |
| A.2 Tanh reference network | A.2–A.6, pp. 16–21 | Keep this proof order: local representation; signed midpoint rule; correction in existing halo; coefficient control; assembled reference error. Use paragraph labels within one subsection, retaining labeled lemmas/propositions. |
| A.3 Recovery and continuous error | A.8–A.9, pp. 26–30; most of A.10, p. 30 | State the common interface below; prove recovery, sample-to-continuous transfer, and evaluation bounds; assemble the tanh theorem; give the fixed-allowance width corollary. |
| A.4 Exact GELU and SiLU | A.7, pp. 21–25; the GELU/SiLU conclusion of A.10, p. 30 | Keep the polynomial/deconvolution/halo/activation-lifting order. Finish with normalized envelopes and complex-growth verification, then invoke A.3 and give current (A.121). |
| A.5 Bandwidth selection | A.12, pp. 33–36 | Keep pole bound → compact score → practical frequency/tolerance choices. Fold the short width-dependence discussion into the end. |
| A.6 Comparison bounds | A.11, pp. 31–33 | Keep the common arithmetic assumptions and each implementation's derivation. Method names can be paragraph labels. |

For source-level precision: move construction lines 1449–1463, including `eq:psi-computed`, to the end of the later GELU/SiLU subsection. Keep construction lines 1421–1447 and 1465–1493 with recovery. The generic width corollary is an implication about an established error bound; the later activation subsection can apply it after verifying its own constants and allowances.

The bandwidth/comparison swap has no proof dependency cost. A.12 uses the tanh density, midpoint residues, and corrected error decomposition. A.11 uses the analytic class and finished approximation/recovery result; A.12 never uses a comparison derivation. Keeping bandwidth ahead of comparisons also puts the support for Section 3.3 before the independent Table 1 details.

## What must precede common recovery

Use the existing notation, but state these as hypotheses for a general prescribed dictionary, rather than asserting that all three activations have already been checked:

1. Native features are normalized as `varphi_j = phi_j/m_j`, with `m_j >= 1`, bias normalization one, and `|varphi_j| <= 1` on the real domain. The readout dimension includes the bias and any affine units, and `M >= n`.
2. There exists a reference network with uniform approximation error at most `E_W^an` and positive deterministic coordinate envelopes `|v_j^star| <= B alpha_j`, whose sum is bounded by a width-independent `K_ref`. These yield `||a^star||_2 <= S`, the sampled residual bound, and `||Da||_1 <= L||a||_2` in (A.93).
3. For the continuous-error part, the target and normalized features are holomorphic on the chosen cell ellipses and `|varphi_j(z)| <= C_ell` there, with `C_ell` independent of width. Retain the sampling/ellipse conditions (A.104). A real-axis feature bound alone does not provide this complex bound.
4. The stated numerical comparison, export, cutoff, input-rounding, and activation-evaluation assumptions hold: (A.94)–(A.100), (A.112)–(A.115). These are additional run conditions, not consequences of the reference-network construction.

This interface already contains exactly the inequalities used by the proofs. It does not require a new approximation result.

For tanh, immediately verify the first three hypotheses using `m_j=1`, the reference theorem (A.51), the coordinate envelopes (A.91) and their construction, and the displayed tanh bound for preactivation imaginary part at most pi/4. Thus the entire tanh theorem closes before A.4.

For GELU/SiLU, keep the following *proofs* with the later activation construction, and verify the interface there before applying recovery:

- Uniform reference errors (A.78)–(A.80).
- Native weights and affine corrections (A.81)–(A.82), then normalization (A.83), coordinate envelopes (A.84), and their bounded sum (A.85). Native coefficient control alone is insufficient for unbounded GELU/SiLU features; the proof needs the normalized coordinates, including the bias and two affine units.
- Lemma A.3.1, (A.86)–(A.88), for holomorphic feature bounds on cell ellipses. For SiLU retain the pole-free strip restriction. At `v_0=pi/4`, it gives the `C_ell` used by current A.9.

Concretely, construction lines 1030–1043 should cease treating (A.83), (A.84), and the activation error formulas as already proved. Lines 1265–1267 should use the stated complex-bound hypothesis, with the tanh verification nearby. The later GELU/SiLU section then discharges those hypotheses using construction lines 943–1016. A forward reference announcing that verification is harmless; silently using the result before establishing or assuming it is not.

If the theorem statements and proof wording are to remain unchanged, preserve the existing dependency order A.7 → A.8 → A.9 instead. The proposed reordering specifically includes the interface change above.

### Remaining dependencies worth preserving

- A.2–A.3 establish the exact signed error decomposition. The corrected boundary functional is `B_h f - V_h[F_x]`, not just the representation boundary term. A.4 must correct both contributions.
- A.4's canceled boundary moments are `O(Bh)`; the prescribed rational denominators realize the correction in existing halo slots. A.5's coefficient bound is what makes recovery stable. Approximation error alone does not replace it.
- A.7 uses the signed midpoint identity from A.3, but its contours have no kernel poles. Its polynomial deconvolution, exterior-density estimate, and different halo scalings remain needed: GELU uses `R_G=O(sqrt(N))`; this uncorrected SiLU construction uses `R_S=Theta(N)`.
- The SVD bound controls normalized coefficient size and sampled error. Cell analyticity and local polynomial sampling then transfer the result to continuous RMS. Input rounding and evaluation produce the final computed-network guarantee. None of these stages can be replaced by an appeal to an accurate least-squares fit alone.

## Six specific trims

| Passage / search string | Location | Recommended trim and why it is safe |
|---|---|---|
| `Proof outline.` | PDF p. 15; construction lines 103–108 | Remove this second roadmap after keeping one short opening paragraph (lines 27–31) updated to the final order. It contributes no hypothesis or proof step. |
| `The analytic tanh witness` and `Theorem assembly and width dependence` | PDF pp. 20–21, 30; construction lines 523 and 1421 | Remove their status as separate subsections, retaining the reference theorem/proof at the end of the tanh subsection and the assembly/corollary at the end of recovery. These short closing steps belong to the arguments they complete. |
| `The polynomial and its derivatives occur only` / `The analytical high derivatives and complex target values` | PDF pp. 21, 30; construction lines 568–570 and 1459–1463 | State once, beside the sampled algorithm, that it uses real target samples and deterministic envelopes. Remove the repetition, while retaining the successful-run/numerical-condition qualification. |
| `At fixed observation density, the projected-vector allowance` | PDF p. 28; construction lines 1238–1241 | Compress to one sentence stating the pairwise projection contribution's `O(u sqrt(W) log W)` rate. Keep the restriction to that contribution in the sentence; do not turn it into a bound for the whole solver. |
| `We retain the geometry` and `eq:bandwidth-exponent` | PDF p. 33; bandwidth lines 10–19 | Drop the repeated geometry list and make `2 pi d/h = pi^2/lambda` inline if needed. The same definitions are already fixed, and (A.136) immediately introduces the exponent again. No reference to this equation label occurs in the supplied sources. |
| `Finite precision floor.` | PDF p. 6, opening paragraph; compare `Attainable accuracy.` on p. 5 | Delete the duplicate paragraph. The previous page already explains recovery/evaluation allowances and the precision-dependent floor, and cites the same Figure 2 observations. Preserve the actual numerical assumptions in the appendix. |

These cuts remove repeated navigation and explanation. The lengthy approximation and numerical arguments should not be deleted merely because they look technical: they support the paper's central finite-precision claim.

## Material and qualifications to retain

Keep the uniform reference bounds and the maximum-residual certificate (A.117), even though the main Section 3 theorem uses RMS. The arithmetic-circuit statement on p. 9 requires uniform local multiplier accuracy; an RMS statement alone would not supply it. This observation does not verify the separate circuit theorem.

Keep the fixed-design/certified-width-range qualification in the width corollary (construction lines 1467–1493). It explicitly says that varying bandwidth with accuracy requires tracking the constants, admissibility, and allowances. Main Corollary 3.1.1 currently sounds broader by saying to choose lambda for the target tolerance and then asserting logarithmic width. That scope mismatch should be resolved rather than deleting the appendix qualification. The conditional polynomial numerical allowance in the QUILL comparison row likewise must remain (comparison lines 218–229).

Keep the representative-frequency heuristic warning and its convexity explanation (bandwidth lines 143–162). The centroid is not an upper-bound-preserving replacement for the full frequency sum. Keep the admissible bracket/infeasibility rule, nonperiodic endpoint treatment, amplitude weighting, and distinction between an aliasing budget and total-error minimization.

Keep the distinction between reference-scaled truncated SVD and an unregularized least-squares minimizer (construction lines 73–75). Main equation (4), p. 4, currently describes an unregularized argmin. The algorithm to which the theorem applies needs to be made consistent across these locations.

The broken GELU/SiLU reference on p. 15 has a concrete source cause: construction lines 30 and 105 reference `sec:gelu-silu`, whereas line 565 defines `app:gelu-silu`. This is a label mismatch, separate from missing proofs.

## Theorems 3.2 and 3.3 in the supplied PDF

Theorem 3.2 is stated on p. 6. The nearby paragraph gives the usual fixed-matrix GD intuition, but the promised explicit gamma-dependent eigenvalue-ratio bounds and finite-geometry allowances are absent from the appendix. The p. 7 plot caption does not provide that proof. A.12's pole/aliasing estimate concerns the constructed quasi-interpolant, not the sampled optimization kernel, so it cannot fill this gap by relabeling.

Theorem 3.3 is stated on p. 8. Its promised recurrence for `B_n`, coefficients, population-concentration bounds, target-coupling bounds, and coarse-tracking bounds are absent. The text and figure caption say coefficients were evaluated along stored trajectories, but do not define the conditions or prove the comparison theorem. This is missing mathematical material, not merely an unresolved cross-reference.

I checked the whole current appendix, pp. 15–36: it ends with the bandwidth figures and contains only A.1–A.12. This finding concerns the supplied PDF and attachments, not whether the missing proofs exist elsewhere. The coordinator is handling that source search. Once recovered, they belong in separate proof appendices for Sections 3.4 and 3.5; adding empty headings now would not repair their absence.

## Primary sources

- [Current 36-page paper](</Users/sam/.codex/attachments/409e26ef-2d5c-488f-9a03-8a7aa5cf21d0/QI_MLPs___ICLR_2027_Submission (4).pdf>).
- [Construction source](</Users/sam/.codex/attachments/64cd712b-3008-4cc9-a232-9b2604bb6260/Pasted text.txt>), 1493 lines.
- [Comparison source](</Users/sam/.codex/attachments/b26d7459-258d-40e0-ac0b-ef4f0c8e83c0/Pasted text.txt>), 229 lines.
- [Bandwidth source](</Users/sam/.codex/attachments/77b01ccd-76a7-4aa1-bf1d-04f4b351fa2a/Pasted text.txt>), 214 lines.

Personal context check passed at revision `1a626ad59f8931c5592c625afdfd0da80cb0bb5a`; sync failed without changing local work. The review relies on the supplied current paper and local guidance, not a claim of refreshed remote context.
