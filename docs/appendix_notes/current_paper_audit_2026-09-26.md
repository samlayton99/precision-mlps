# Current paper and appendix audit — 26 September 2026

## Scope

Reviewed the 36-page `QI_MLPs___ICLR_2027_Submission (4).pdf` supplied in this conversation, together with the current construction, comparison, and bandwidth LaTeX attachments. Page references below refer to that PDF. The review concentrates on Section 3 and its appendix. Separate mathematical, notation, and empirical-provenance reviews were combined here. Paper sources and figures were not edited, and no new training was run.

No confirmed internal algebraic error was found in the three supplied appendix proofs. That is a review finding, not a formal verification certificate. Several main-text claims exceed or differ from what those appendices establish. Other promised proofs are absent from this PDF.

## First pass: corrections with low implementation cost

### 1. The trillion-step example names the wrong experiment

**Location/search:** p6, Section 3.4, `9.20`.

The `9.20 × 10^11` necessary steps and `5.85 × 10^-4` least-squares error come from a **153-hidden-neuron network**, with 263 training points and a mixed-sine target whose highest frequency is `10π`. The draft attributes them to **W=512**; its Figure 3 target reaches `14π`.

**Repair:** present the numbers as a separate W=153 example with its correct protocol, or replace them with verified results for the actual W=512 experiment. The spectral step counts are computed from the frozen-kernel recurrence; they are not executed trillion-step training. The numerical endpoints have refinement checks but no directed-rounding certificate.

**Source:** [finite-kernel note, numerical protocol and explicit warning](2026-09-25-revised/02_finite_kernel_ratios.tex), especially its final section. That source explicitly warns against assigning these numbers to the W=512 example.

### 2. Repair references and captions

| Location / search | Correction |
|---|---|
| p5, Figure 2(b), `Equation 5` | Cite Equation 8, the bandwidth selector; Equation 5 only defines lambda. |
| p5, Figure 2(c), `O(log(1/ε)) error scaling` | The plotted error behaves approximately as `E_p = C 2^-p`; equivalently the required bit count is logarithmic in inverse error. The legend currently labels the wrong dependent quantity. |
| p7, Figure 3, `frozen-readout` | Geometry is frozen and readout weights are trained. |
| p7, Figure 4, `Needs to be updated to a 3-panel figure` | Replace the placeholder with a real caption and protocol. Current image has two panels. |
| p15, opening and proof outline, `sec:gelu-silu` | Use the defined label `app:gelu-silu`. |
| p6, `Appendix XX` | The recovery and continuous-transfer material is A.8–A.10. |

Other resolvable references: the main construction proof points to A.2–A.6 and A.8–A.10; the bandwidth derivation is A.12; Figure 3's output-bound theorem is Theorem 3.2. Missing optimization/circuit appendices require actual content, not merely a label replacement.

### 3. Match Figure 4's statistic to the prose

**Locations:** p6 final paragraph; p7 Figure 4; p8 Theorem 3.3.

The figure shows **RMS bandwidth and its 99th percentile**. The prose calls them mean bandwidths, while Theorem 3.3 defines an arithmetic mean. These are different statistics. Name the plotted summaries accurately and state how any comparison to the theorem's mean is being made. Supply initialization, rates, seeds, shading meaning, and the frozen-geometry comparison protocol.

### 4. The multi-activation bandwidth prescription is missing

**Locations:** p6 Section 3.3 promise; A.12 and Figures 6–7, pp33–36.

A.12 derives tanh's selector. Figures 6–7 also use GELU, SiLU, and erf, but their selection formulas are not stated in this appendix. Add the general derivative-kernel transform rule, the refined replica score, the derivative orders, and the frequency convention used in the plots. Use the existing derivation rather than creating a new one.

**Reusable source:** [general-rule insertion](bandwidth_figures/bandwidth_rules_insert.tex). Tanh/erf use the first derivative; exact GELU/SiLU use the second. The plotted fixed-halo experiments should not be presented as automatically satisfying every assumption of the separate activation-extension theorem.

## Claim/proof mismatches requiring decisions

### 5. The main construction defines a different solver from the guaranteed one

**Locations:** p4 Equation 4; p15 A.1; pp26–28 A.8.

Equation 4 defines an unregularized least-squares minimizer. A.1 expressly states that the analyzed algorithm is **reference-scaled truncated SVD**, not an unregularized minimizer. A.8 supplies the scaling and threshold requirements used to prove recovery and coefficient control.

**Repair:** retain least squares as the underlying problem, but identify the scaled, truncated solver as the computed output covered by the theorem. State separately which solver each empirical figure actually used. A default SVD cutoff does not automatically satisfy the certified algorithm's assumptions.

### 6. The main logarithmic-width claim is broader than the appendix result

**Locations:** p5 Corollary 3.1.1 and Table 1; p30 Corollary A.4.1; p33 A.11.5.

The main corollary chooses bandwidth for the desired tolerance and states logarithmic width. A.10 proves geometric refinement **for fixed design, above a bounded allowance, within a certified width range**. It explicitly does not claim a joint epsilon-to-zero result while the design changes.

This distinction affects the constants: `c_app = min(πδ, λ a_H²)/8`, the approximation prefactor contains `1/λ`, coefficient envelopes depend on lambda, and fixed sampling-transfer parameters leave an analytic allowance. These dependencies cannot be held constant while changing the design without an additional argument.

**Repair available now:** carry the fixed-design, allowance, and admissible-range qualifications into the main claim and table. A stronger joint asymptotic statement requires uniform design/precision/sampling choices. The review does not establish that such a stronger result is impossible; it establishes that the supplied corollary does not prove it.

### 7. The main bandwidth explanation overstates the selector

**Searches:** p6 `optimal relative bandwidth`, `Equating aliasing error`, `recovery terms decrease`.

A.12 correctly identifies substitution of a representative mean frequency as a **heuristic**, not a bound-preserving replacement of the full spectrum and not a minimizer of total measured error. The main text presents it as equating actual aliasing error to the tolerance. The asserted monotonic decrease of the full recovery allowance also needs justification.

**Repair:** describe selecting bandwidth by a representative-frequency aliasing score, and distinguish that score from a guaranteed output error. Define epsilon_eff as a dimensionless tolerance, not a bit count.

The limitation is measurable. At tanh, Gaussian target, N=32, the selected score is `2^-52` but measured relative error is `1.85666 × 10^-12`. The independent full-spectrum first-replica calculation gives `1.85652 × 10^-12`. This supports the replica mechanism and identifies the loss from substituting one mean frequency. It does not validate that approximation at all widths or below the numerical floor.

**Source:** [existing aliasing audit](bandwidth_figures/aliasing_audit.md).

### 8. Sufficient comparison bounds are described as necessities

**Locations:** p3 related work, p5 comparison discussion; A.11 and Table 1 caption.

The appendix and caption correctly give sufficient bounds for specified implementations, not lower bounds on all possible implementations. Main prose saying the alternatives “require” superlinear working precision is stronger.

**Repair:** attribute the sufficient precision bounds to the analyzed realizations. Do not turn them into unavoidable requirements.

## Material missing from the assembled PDF

### 9. The optimization proofs and protocols are not yet present

The PDF ends at A.12 on p36. It does not include the promised finite-kernel ratio proof and protocol for Theorem 3.2, nor the precise assumptions and proof for Theorem 3.3. The circuit appendix promised for Theorem 4.1 is also absent.

For Section 3, bringing in the existing proof/protocol material matters more than adding another descriptive plot. In particular:

- Theorem 3.2 needs the explicit rate bounds, finite-geometry allowances, target-weight handling, and numerical scope of evaluated certificates.
- Theorem 3.3 needs the recurrence and assumptions behind its trajectory-dependent coefficients. The claim of at least 97.7% agreement covers ten continuations; Figure 5 displays only two. Include the saved run summary and diagnostic-time definition.
- Section 3.5 invokes numerical readout refits and slope-rescaling controls for the W512 learned dictionaries. Their actual current protocols/results are absent. Add them before relying on the interventions in the argument.

### 10. Uniform circuit accuracy needs an explicit bridge

The main construction theorem controls RMS error; the circuit theorem assumes uniform scalar approximation. A.117 gives a separate **validated maximum-residual** certificate. It expressly does not permit replacing the maximum residual by the same numerical RMS error. The circuit appendix must identify which uniform result and assumptions supply its premise. This is a handoff item for the circuit portion of the paper.

## Notation changes worth making

1. **Define `A_gamma` before Theorem 3.2:** entries, constant feature, sample normalization, and the loss convention matching the learning rate.
2. **Use one training-sample notation:** main optimization uses `Z`, construction appendix uses `X`. Keep continuous RMS, sampled RMS, and relative errors explicitly distinguished.
3. **Use one notation for the recovered readout and recovery allowance:** main and appendix alternate hatted/unhatted coefficients and superscripted/unsuperscripted recovery terms.
4. **Separate working precision from polynomial degree:** `p` denotes working bits in Figure 2(c) and a local approximation degree in A.9. Renaming the latter to `p_loc` is a small, useful change.
5. **Keep N and W distinct:** N counts interior cells; W includes halo neurons. The new bandwidth plots use N, while the main width plot uses W. Their captions should retain the conversion for their actual halo choices.

No native/normalized feature-symbol reversal was found in the current supplied source. `phi_j` and `varphi_j` are used consistently there. Likewise, sample count `M` and ellipse envelope `mathcal M` are distinct. Avoid a wholesale renaming of harmless local symbols.

## Useful additions using work already done

### Highest priority: four-panel method comparison

Insert the existing Runge/chirp by 24/53-bit comparison after A.11. It shows measured error versus total hidden-neuron budget, which the asymptotic table cannot show.

- [Figure](../../results/checkpoint_C_geometry/expC13_five_method_comparison/figures/appendix_tapout_fp32_fp64.pdf)
- [Caption](../../results/checkpoint_C_geometry/expC13_five_method_comparison/figures/appendix_tapout_fp32_fp64_caption.tex)

Preserve the interpretation: validation-selected budget envelopes; 24-bit significands with expanded exponent range rather than native binary32; QUILL reaches its plateau sooner, while ChebNet can reach a lower final error.

### High priority: precision implementation details, not necessarily another figure

Reuse [the true-p-bit note](2026-09-25-revised/01_true_pbit_implementation.tex) to document Figure 2(c). Specify construction and evaluation rounding, sample counts, SVD cutoff, expanded exponent range, and native-format checks. Distinguish this experiment's raw-feature solver from the certified scaled solver.

### Useful small table: bandwidth prediction versus full-spectrum calculation

Use the Gaussian example above to distinguish the representative-frequency score from the more complete Fourier calculation. A small table is sufficient; no new sweep is needed. The existing four-activation error figure can remain the broader empirical comparison.

### Optional: one geometry-learning control

The existing D25 shared-scale Adam versus SGD experiment separates improved geometry under evaluation-only readout refits from the still-inaccurate trained readout. It uses **supplied uniform centers**, so it is a scoped positive control, not a matched reproduction of the W512 joint-training experiment. Use it if the paper needs to delimit the assertion that slopes and centers “must” be acquired together. Do not insert the whole historical split-Adam/F+G experiment catalogue.

Source: [geometry/readout note](2026-09-25-revised/03_geometry_readout.tex); figure at `results/checkpoint_D_optimizers/expD25_scale_barrier/figures/adaptive_gamma_1.png`.

## Organization and compression

- **Delete the duplicate p6 paragraph `Finite precision floor`.** The immediately preceding p5 paragraph `Attainable accuracy` makes the same points with the same figure references.
- **Complete the tanh proof before the five-page activation extension, with an explicit dependency adjustment.** Current A.7 interrupts the sequence from witness construction (A.6) to recovery and continuous certification (A.8–A.10). A.8–A.9 currently invoke the GELU/SiLU coefficient envelopes and complex-ellipse bounds from A.7. Moving A.7 unchanged would leave those prerequisites downstream. State common recovery and transfer conditionally on normalized-feature bounds, a reference approximation with bounded coordinate envelopes, and the complex-ellipse bound; verify these immediately for tanh. The later activation extension can then verify the same conditions before invoking the common results. If preserving the existing statements verbatim, retain A.7 before A.8–A.9.
- **Give bandwidth selection and comparison bounds separate navigation.** They are useful standalone appendix topics, currently buried inside “Construction theory.” This can be done without rewriting their mathematics.
- **Compress standard auxiliary derivations selectively.** The inverse-estimate derivation in A.9 can cite its standard result and keep the sampling specialization. A.8's numerical-interface realization can be separated from its contract and recovery theorem.
- **Keep the substantive proof work.** Boundary correction/readout control in A.4–A.5 and the conditions needed by recovery are not redundant. Necessary qualifications about fixed design, mean-frequency heuristics, and trajectory-evaluated quantities should survive compression.

The main editorial problem is interrupted argument order and repeated summaries, not a need to rewrite every mathematical paragraph.

## Obvious full-paper leftovers outside the core appendix task

- Abstract: 1,152 evaluations; introduction and later table: 2,250. Verify the intended count.
- Table 2 repeats the full 3-digit Gemma row and omits a 5-digit Gemma row. Retrieve the correct data rather than simply changing the label.
- Table 2 caption describes parenthetical completion-only results that are not displayed.
- Template title remains on p1. Citation placeholders remain. Figure 4 has a draft instruction instead of a caption. The circuit section has an inline editing instruction. Ethics/reproducibility/acknowledgment template instructions remain in the rendered draft.

## Recommended order of work

1. Correct the misattributed numerical example, references/captions, and duplicate paragraph.
2. Align solver, bandwidth-selection, and fixed-design claims with the actual appendix statements.
3. Insert the missing optimization proofs/protocols and general activation formulas.
4. Add the comparison figure and precision implementation description; optionally the small bandwidth diagnostic table.
5. Reorder/compress the appendix after those interfaces are stable.

## Follow-up verification: missing optimization proofs

Reopened the original supplied PDF and checked all 36 pages. Theorem 3.2 is stated on p6 and Theorem 3.3 on p8; their promised proof material is not in the appendix. The acquisition-specific assumptions appear only in the main statement on p8. The PDF ends with Figure 7 on p36. A fresh independent reviewer confirmed the same absence.

For Theorem 3.2, existing material is available in `2026-09-25-revised/02_finite_kernel_ratios.tex`, including the Fourier identity, lattice/halo enclosure, interlacing and normalization bounds, and conversion to residual/time bounds. It needs integration and notation/protocol alignment, not substitution of a different experiment's numerical example.

For Theorem 3.3, the local `03_geometry_readout.tex` contains different finite-horizon displacement bounds; those do not supply this theorem's population-growth recurrence or its target-coupling/concentration/coarse-tracking assumptions. The exact current source has not been located in the local text sources and likely relevant PDFs checked in this follow-up. This is not a claim that it does not exist elsewhere.

The required acquisition appendix must define the joint parameter norm, initial and updated B_n, every comparison coefficient and assumption, the valid time interval, and Q_n's bound on the non-affine output; then prove the parameter comparison, output-error bound, and mean-bandwidth consequence. The empirical protocol should separately state which coefficients are evaluated along saved trajectories.
