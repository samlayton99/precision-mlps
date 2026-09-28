# Bandwidth appendix package

The two requested figures and the appendix text are ready. The mathematical comparison is against `choosing_optimal_lambda_overleaf (2).pdf`, supplied September 26, and the supplied LaTeX subsection beginning `\subsection{Selection of the relative bandwidth}`.

## Exactly where to insert it

1. In the paper appendix, find `\subsection{Selection of the relative bandwidth}` with label `sec:qi-bandwidth`. This is the subsection supplied with the comment “Append immediately after comparison_bounds_appendix.tex, within Appendix A.”
2. Replace that **whole subsection**, through the final `\end{figure}` belonging to `fig:bandwidth-appendix-diagnostic`, with the contents of **`bandwidth_appendix_complete.tex`**. Keep it in its current position, immediately after the comparison-bounds appendix and before the next subsection. If it already lives in a separate input file, replace that file's contents instead.
3. Copy **`figures/bandwidth_width_rules.pdf`** and **`figures/bandwidth_error_rules.pdf`** into the paper's `figures/` directory.
4. Ensure the preamble loads `amsmath`, `amssymb`, `graphicx`, `booktabs`, and `tabularx`. Retain the paper's existing proposition/proof environments and citation package. The replacement preserves the original `JavedTrefethen2014` citation and references to the preceding representation proof.

For an input-file layout, the appendix location should read:

```latex
\input{comparison_bounds_appendix}
\input{bandwidth_appendix_complete}
% Next existing appendix subsection/input follows here.
```

Do not also input `bandwidth_rules_insert.tex`: its content is already included in the complete replacement. That smaller file is supplied only to show the new material. Its exact insertion point is after the paragraph ending “as in \qeqref{eq:Pbound}” below `eq:bandwidth-exponent`, immediately before `\subsubsection{Frequency-dependent pole bound}`. The complete replacement also removes the obsolete diagnostic figure and updates its references.

The figure macros can be overridden before the input if your image folder differs:

```latex
\newcommand{\BandwidthWidthFigure}{your/path/bandwidth_width_rules.pdf}
\newcommand{\BandwidthErrorFigure}{your/path/bandwidth_error_rules.pdf}
```

New figure labels are `fig:bw-width-rules` and `fig:bw-error-rules`. The original tanh equations, proposition, proof, and their labels are retained. The standalone preview replaces references to the preceding paper material with explicit descriptions, so it compiles by itself; use the **complete insertion**, not the preview source, inside the paper.

## Agreement with the note

The exact general first-pair rule is equation (3) of the supplied PDF. Substituting `K=tanh'` and `r=1` gives `sinh(s)[csch(a-s)+csch(a+s)]`, with `a=pi^2/lambda` and `s=pi*omega/(N*lambda)`. The current paper's compact score is `2 exp(-a) sinh(2s)`. Its existing comparison inequality rigorously controls that approximation.

The general score is `|Khat(2pi/lambda)|/|Khat(0)|`. The new text derives the integration factors and explains why the denominator is the transform at the retained frequency. It gives transforms for tanh, exact GELU, SiLU, and erf, with derivative orders 1, 2, 2, and 1. The finite-domain tanh theorem is not silently extended to other activations.

At fixed tolerance the refined root does **not** converge to the constant general-rule root. Page 6 of the supplied note explicitly says this. Its normalized small-angle limit and its threshold-sensitivity argument are included. The function enters the scalar rule only through its chosen frequency, so curves are horizontal rescalings in `N`. The revised figure now uses `N` directly; total neuron count remains `W=N+2R+1`.

## Figures and data

- `bandwidth_width_rules`: exactly 1×3, tanh/GELU/SiLU; the eight original targets; linear N axis through 8,192 (saved calculations extend through 16,384). The shared legend is centered below the plots, with the general rule in its rightmost column. Each panel has one dashed general-rule reference (approximately 0.25 for tanh). Curves are omitted where the representative frequency violates 2*omega/N < pi. The appendix lists every frequency input, including the chirp/packet approximations and the fixed-resolution log-cosh DFT estimate.
- `bandwidth_error_rules`: 4×4, tanh/GELU/SiLU/erf; mixed sine (2,6,10), Gaussian `exp(-20x^2)`, exponential sine `exp(sin(3pi x))`, and `sech(5x)`. These differ from the eight main-figure targets. Each panel has N=32,48,64,96,128,256 and x limits shared within each column: 0.05–0.5 for tanh and 0.05–1 for GELU, SiLU, and erf; vertical lines mark the general rule and red diamonds mark the directly measured refined selections.
- `bandwidth_fourier_ratios`: optional separate 1×3 diagnostic of `|Khat(2pi/lambda_refined(N))|/|Khat(0)|`. Kept separate to preserve the requested main figure layout. The displayed ratio range is limited; `data/width_predictions.json` also records log10 ratios without exponential underflow.
- All figures are supplied as PNG and vector PDF. `data/` holds the values and provenance.

The error curves reuse 1,440 archived FP64 observations and add 2,400 under the same protocol. Ninety-six additional fits evaluate the refined selections. Centers are fixed, readouts use SVD least squares, training/evaluation grids have 2003/4001 points, and no curves are smoothed. This is a test of practical selection, not a proof of measured-error optimality.

## Checks

- Independent numerical Fourier integrals agree with all four analytical transforms at the checked frequencies (absolute error below 3e-13).
- The note's tanh examples are reproduced: N=32 → 0.2004780; N=128 → 0.2555275; N=512 → 0.2718935.
- On the width-figure tanh points, the rigorous relative difference bound between exact and compact **scores** is at most 3.29e-28. This is not a measured floating-point root difference.
- A repeated archived fit agrees with its saved error.
- All displayed roots are actual bracketed roots; none is a capped search endpoint.
- The standalone PDF compiles with no warnings or unresolved references; the new formulas received an independent review.

## Reproduction in the repository

```sh
MPLCONFIGDIR=/tmp/precisionmlps-mpl .venv/bin/python experiments/expC09_bandwidth_figures/appendix_bandwidth.py
.venv/bin/python experiments/expC09_bandwidth_figures/build_bandwidth_bundle.py
```

The second command assembles the exact supplied appendix from its attachment path. The completed insertion and figures in this package can be used in Overleaf without running either command.

## Independent aliasing audit (September 26 refinement)

Sam questioned whether moving the selected bandwidth right with width spends too much aliasing error. A fresh agent with full context audited the note, implementation, full target spectra, and measured curves. See `aliasing_audit.md` for methods and quantitative comparisons. The selector is unchanged in these figures.

The rightward drift is consistent with the score: the integration-frequency prefactor decreases with N, allowing larger lambda at the same modeled tolerance. However, replacing a target spectrum with one average frequency is not an error-budget-preserving operation. At N=32, the Gaussian target's full-spectrum tanh alias calculation predicts 1.8565e-12 relative error and the measured error is 1.8567e-12, although the average-frequency score is 2^-52. This is evidence that the shortcut can miss aliasing, not an implementation discrepancy in the Fourier multiplier. The displayed dots remain heuristic selections, not precision guarantees or measured-error minimizers.

The displayed error-width selection is now N=32,48,64,96,128,256. Earlier N=512 calculations remain cached and are used in the dated audit; they are not displayed in the current figure. `data/selected_refined_marker_fits.json` contains only the 96 currently displayed selections.
