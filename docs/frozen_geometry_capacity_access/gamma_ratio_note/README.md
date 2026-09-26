# Gamma ratio note

Compact presentation: **23 September 2026**.

- [Compact PDF: revised introduction, theorem, figure, and full appendix](gamma_ratio_note_compact.pdf)
- [Compact source with appendix](gamma_ratio_note_compact.md)
- [Main passage only](gamma_ratio_note_compact_main.md)

The main argument and one theorem occupy the first page, followed by the figure on the second page. It opens with the small-gamma training-delay objective, chooses the learning rate, shows the gradient descent residual update and exact error formula, and explains why ratio upper bounds give necessary training times. The row inner products of `B_gamma` introduce the same-sized continuous matrix `H_gamma`; sample centering cancels constant tails and makes its whole-line integral finite. The Fourier identity starts directly with `lambda_i(H_gamma)`. Appendix A explains the equivalent restricted coordinates used in the proofs. Normalization is justified by the constant-direction Rayleigh lower bound from the bias and neuron means, without assuming the largest eigenvalue stays constant. The finite endpoint construction is in Appendix E, equation (E0), using `lambda_i(S_gamma)` rather than beta. The main discloses this corrected-matrix spectrum calculation. The figure compares gamma 4's necessary count with gamma 64's sufficient count, establishing separated training-time intervals in the example. It reuses saved results; no training or new eigensolves were run. The full Revision 4 derivation is retained in Appendices A–G. Universal finite-ratio monotonicity is not claimed, and numerical evaluations remain checked rather than certified.

Expanded version: **Revision 4, 23 September 2026**, preserved unchanged.

- [PDF](gamma_ratio_note_v4.pdf)
- [Versioned source](gamma_ratio_note_v4.md)
- [Mathematical review](revision4_math_review.md)
- [Argument review](revision4_argument_review.md)
- [Figure calculations and numerical checks](../../../results/checkpoint_D_optimizers/expD37_capacity_access_figures/gamma_solve/spectrum_mechanism/note_interval_figures/MANIFEST.md)

`gamma_ratio_note.md` and `gamma_ratio_note.pdf` contain the expanded Revision 4. Earlier versioned PDFs are preserved. Revision 4 replaces the anchored lower-ratio theorem with a two-sided interval at each gamma and applies its endpoints directly to the full target-weighted error and training-time bounds. Exact inequalities and checked, uncertified numerical evaluations are distinguished throughout.

`build_pdf.mjs` renders Markdown and KaTeX to printable HTML. Optional arguments select a source filename and output HTML filename; defaults are `gamma_ratio_note.md` and `.build/note.html`. It requires `katex` and `markdown-it`, with `GAMMA_NOTE_NODE_MODULES` selecting their installation if needed. Print the resulting HTML to A4 PDF in Chromium with browser headers disabled; the document supplies page numbers. Scratch renders and page images belong in `.build/`.
