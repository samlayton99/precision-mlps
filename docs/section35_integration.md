# Section 3.5 writing and integration record

Sections 3.4 and 3.5 now form a single argument: small bandwidth can delay learning an accurate readout already available in a fixed dictionary; joint training must acquire useful features to overcome that obstruction. The proposed Section 3.5 explains a conditional mechanism for slow acquisition: weak low-order target coupling and controlled population concentration limit joint parameter growth, which limits non-affine output and leaves an explicit output-error floor. The existing long-horizon Adam/GD comparison motivates this question; the theorem is checked on a separate set of GD continuations. Figure 4's revision remains a separate deliverable.

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

- [Main-text LaTeX](section35_feature_acquisition.tex): empirical motivation, compact output-error theorem, paired validation figure, and return to the construction.
- [Full proof and evidence appendix](section35_population_appendix.tex): exact discrete-GD proof, global remainder constants, target dependence, evolving structural conditions, and empirical protocols.
- [Standalone PDF](../output/pdf/section35_population_note.pdf) and [wrapper](section35_population_note.tex): the proposed subsection and its validation pair on page 1, followed by the discrete-GD proof and evidence appendix.
- [Combined Sections 3.4–3.5 PDF](../output/pdf/section34_spectrum_note.pdf) and [wrapper](section34_spectrum_note.tex): Figure 3 and the existing frozen-readout result followed by the new subsection and both proofs.
- [Selected evidence and source hashes](../output/diagnostics/section35_population/evidence.json): ten native-step GD comparisons and the completed width-512 training endpoints. This is an extraction of existing records, not a new experiment.
- [Population/output figure](../output/diagnostics/section35_population/figures/population_output.pdf) and [all-ten tightness figure](../output/diagnostics/section35_population/figures/output_tightness.pdf), also available as 600-DPI PNGs. The main pair uses mixed sine and smooth step; the latter has the largest principal population-growth allowance. Dashed bounds remain distinguishable from solid executed curves and hollow markers.
- [Matched-time curves](../output/diagnostics/section35_population/figures/curves.csv) and [verification](../output/diagnostics/section35_population/figures/facts.json): all ten saved summaries are reproduced, and the smallest bound/error ratio over the 500-update diagnostic grid is 0.9771404.

The source result is Corollary 4a and its output-error consequence (18d) in `precision-mlps/docs/d34_population_balance_mechanism.md`, as of commit `7a12986`. The source checkout advanced during review; its later changes concern the separate reinforcement analysis, not the population comparison used here. The evidence file records the inspected source hashes and checkout revision.

## Claim boundaries

The scalar recurrence allows the population to grow. It uses polynomial output, Jacobian, target-coupling, concentration, and coarse-tracking bounds at each iterate; it does not freeze the features or assume that the generated-output correction dominates the target force. Its proof applies to exact tanh GD with global remainder bounds. The output-error conclusion needs joint control of readouts and hidden parameters, not just small slopes.

The numerical checks interpolate diagnostics sampled along the continuations. They support a conditional explanation of those trajectories, not a forecast from initial parameters or a certified upper enclosure between saved samples. The legacy mechanism cohort has width 705 and ten continuations across six targets. It is distinct from the five-million-update width-512 comparison, and it does not establish the same mechanism for Adam.

All main-text endpoint comparisons use relative training RMS error. Dense-grid refit assays are identified separately. The reported minimum theorem floors and executed endpoint errors are different summaries; their difference is not presented as trajectory-wide tightness. The new tightness figure instead compares bound and error at matching diagnostic times. No new width-scaling claim is made.

## Integration with submission draft (24)

Keep the frozen-readout theorem, proof sketch, and spectral figure in Section 3.4. Its final paragraph now asks whether joint training acquires useful features within the training budget. Place the joint-training figure after Section 3.5's empirical opening; the drop-in source marks this location. Use `fig:note-training` for that figure, replacing the duplicate `fig:note-spectrum` label in the pasted draft. The spectral figure alone owns `fig:note-spectrum`.

Section 3.5 then introduces the cubic target coupling, states the existing output-error theorem, and displays its completed validation figure. The closing sentence connects supplying the feature geometry and readout to Section 4's arithmetic circuits. The heading retains slow acquisition rather than asserting a universal failure of self-reinforcement. The theorem allows positive growth and eventual escape.

The mean-slope wording is consistent with the agreed Figure 4 revision. The current source comment reserves panels (a) width scaling, (b) output error, and (c) mean bandwidth; the old RMS/99th-percentile image is not relabeled or included. The combined review retains working optimization-figure numbers 3--5 and references draft (24)'s construction as Figure 1. The manuscript must determine its own numbering through the actual figure environments.

## Three writing reviews

1. **Argument:** separated the fixed-dictionary access question from the joint-acquisition question. Moved the empirical interpretation to Section 3.5, retained the slope/center refit evidence, and closed with the construction's role in supplying primitives for arithmetic circuits.
2. **Mathematics and evidence:** preserved both theorem statements and full proofs. Checked the selected training endpoints and the minimum matched-time bound/error fraction against the saved evidence. The cubic discussion describes the small-parameter mechanism; the appendix retains the exact-tanh global remainder proof. The text identifies trajectory-evaluated coefficients and distinguishes the GD theorem from the Adam observations.
3. **Density and integration:** removed the repeated joint-training paragraph from Section 3.4, retained the figures and inline captions, and made the construction-to-training-to-circuits progression explicit. Rendered the combined and standalone notes, kept each main subsection with its figure on one page, and moved the review-only pending-figure notice to the appendix opening. No word-count target was used.

## Figure 4 handoff

The drop-in text references `fig:note-training`; the review wrappers resolve it to external Figure 4 and disclose at the appendix opening that the artwork is pending. Its intended ordering remains width scaling (a), joint training error (b), and mean bandwidth with neuron-spread shading (c), as recorded in [the pending figure note](section34_pending_figure_updates.md). The new theorem does not depend on the unfinished width sweep. Do not carry the wrapper's external-label declaration into the paper: use the actual figure label there.

No training was needed to add these theorem-validation plots. The separate Figure 4 width sweep has now been launched; its status and resource limits are in [the execution record](figure4_width_sweep_execution.md). Its results are not used in the current theorem figures.

## Build and verification

From this checkout's root:

```sh
python experiments/expD36_frozen_gamma_probe/section35_population_figures.py \
  --evidence ../precision-mlps/results/checkpoint_D_optimizers/expD34_readout_race/population_output/evidence \
  --output output/diagnostics/section35_population/figures
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=tmp/pdfs/section35 docs/section35_population_note.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=tmp/pdfs/section34_compact docs/section34_spectrum_note.tex
```

Final PDFs are copied from those build directories into `output/pdf/`. Both documents compile without undefined references, duplicate labels, or overfull boxes. Hyperref reports deliberately unlinked external manuscript figures. The wrappers provide review layout; the drop-in subsections contain no forced page breaks or float barriers. Both wrappers include a small review-only disclosure at the appendix opening that should not be inserted into the manuscript.

The plot revision removes the optional gradient-flow, signed-balance, and conditional width-scaling discussion from this note. The full proof of the stated discrete-GD theorem remains. The original mechanism note retains those additional results.
