# Section 3.5 writing and integration record

The proposed subsection explains a conditional mechanism for slow joint feature acquisition: weak low-order target coupling and controlled population concentration limit joint parameter growth, which limits non-affine output and leaves an explicit output-error floor. The existing long-horizon Adam/GD comparison motivates this question; the theorem is checked on a separate set of GD continuations. Figure 4's revision remains a separate deliverable.

**Notation correspondence with the source mechanism note.**

| This note | Meaning and source correspondence |
| --- | --- |
| $f_\perp$ | Target after removing its affine projection; $g$ in the source. |
| $s=\|\vartheta\|_2$ | Joint norm of slopes, hidden biases, and readouts; $\sqrt M$ in the source. The output bias is omitted. |
| $B_n$ | Growing scalar upper bound on $s_n$; $b_n$ in Corollary 4a. |
| $\mathcal Q_n(B)$ | Bound on non-affine output at candidate radius $B$; $Q(n,B^2)$ in the source. |
| $\mathcal V_n(B)$ | Bound on the compensated fine gradient; $V(n,B^2)$ in the source. |
| $\overline\lambda$ | Mean of $h|\gamma_j|$ across neurons, bounded above by bandwidth RMS. |

## Deliverables

- [Main-text LaTeX](section35_feature_acquisition.tex): empirical motivation, compact output-error theorem, proof sketch, validation, and return to the construction.
- [Full proof and evidence appendix](section35_population_appendix.tex): exact discrete-GD proof, global remainder constants, target dependence, evolving structural conditions, and empirical protocols.
- [Standalone PDF](../output/pdf/section35_population_note.pdf) and [wrapper](section35_population_note.tex): six pages, with the proposed subsection on page 1.
- [Combined Sections 3.4–3.5 PDF](../output/pdf/section34_spectrum_note.pdf) and [wrapper](section34_spectrum_note.tex): Figure 3 and the existing frozen-readout result followed by the new subsection and both proofs.
- [Selected evidence and source hashes](../output/diagnostics/section35_population/evidence.json): ten native-step GD comparisons and the completed width-512 training endpoints. This is an extraction of existing records, not a new experiment.

The source result is Corollary 4a and its output-error consequence (18d) in `precision-mlps/docs/d34_population_balance_mechanism.md`, as of commit `7a12986`. The source checkout advanced during review; its later changes concern the separate reinforcement analysis, not the population comparison used here. The evidence file records the inspected source hashes and checkout revision.

## Claim boundaries

The scalar recurrence allows the population to grow. It uses polynomial output, Jacobian, target-coupling, concentration, and coarse-tracking bounds at each iterate; it does not freeze the features or assume that the generated-output correction dominates the target force. Its proof applies to exact tanh GD with global remainder bounds. The output-error conclusion needs joint control of readouts and hidden parameters, not just small slopes.

The numerical checks interpolate diagnostics sampled along the continuations. They support a conditional explanation of those trajectories, not a forecast from initial parameters or a certified upper enclosure between saved samples. The legacy mechanism cohort has width 705 and ten continuations across six targets. It is distinct from the five-million-update width-512 comparison, and it does not establish the same mechanism for Adam.

All main-text endpoint comparisons use relative training RMS error. Dense-grid refit assays are identified separately. The reported minimum theorem floors and executed endpoint errors are different summaries; their difference is not presented as trajectory-wide tightness. Conditional width scalings are confined to the appendix and are not claimed as measured laws.

## Three writing reviews

1. **Argument:** organized the subsection around useful feature acquisition and output accuracy. Chose the growing comparison over a contraction claim, retained the joint slope/center/readout issue, and ended with the construction supplying that geometry directly.
2. **Mathematics and evidence:** checked the coarse-compensation decomposition, global tanh and Jacobian constants, monotone discrete recurrence, and reverse-triangle output bound. Made the generated-output contribution explicit, retained tracking, defined the positive part, and distinguished observed coefficients from uniform structural assumptions. Checked the ten comparison rows and the selected training endpoints against their saved records.
3. **Density and integration:** kept the main theorem to the output-error statement and its bandwidth consequence; moved coefficients and full derivations to the appendix. Removed duplicated joint-training discussion and obsolete percentile/figure references from the Section 3.4 appendix. Rendered the PDFs to inspect equations, page breaks, and figure placement. No word-count target was used.

## Figure 4 handoff

The drop-in text references `fig:note-training`; the review wrappers resolve it to external Figure 4 and explicitly disclose that the artwork is pending. Its intended ordering remains width scaling (a), joint training error (b), and mean bandwidth with neuron-spread shading (c), as recorded in [the pending figure note](section34_pending_figure_updates.md). The new theorem does not depend on the unfinished width sweep. Do not carry the wrapper's external-label declaration into the paper: use the actual figure label there.

No experiments were launched for this writing task. The earlier Figure 4 run launch was paused before execution; it is not a running background dependency of this note.

## Build and verification

From this checkout's root:

```sh
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=tmp/pdfs/section35 docs/section35_population_note.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=tmp/pdfs/section34_compact docs/section34_spectrum_note.tex
```

Final PDFs are copied from those build directories into `output/pdf/`. Both documents compile without undefined references, duplicate labels, or overfull boxes. Hyperref reports deliberately unlinked external manuscript figures. The wrappers provide review layout; the drop-in subsection and appendix contain no forced page breaks. The standalone includes a small review-only disclosure that should not be inserted into the manuscript.
