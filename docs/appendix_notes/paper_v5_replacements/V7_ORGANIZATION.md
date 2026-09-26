# Organized appendix for paper version 7

This package replaces the five existing appendix files and adds one file for the readout-optimization proof. It preserves the existing labels and mathematical displays. Do not change the main-text theorems or prose for this reorganization.

## 1. Replace five files; add one

Unzip the download. Its `sections/7_appendix/` hierarchy matches your project. Copy the contents of each supplied file into the matching existing file, replacing the entire old contents. Keep your existing folders and filenames.

| Appendix | Destination in your paper | Action |
|---|---|---|
| A: Construction theory | `sections/7_appendix/01_Construction/09_26_sl.tex` | Replace whole file |
| B: Bandwidth selection | `sections/7_appendix/03_bandwidth/092526_jh.tex` | Replace whole file |
| C: Precision-controlled computation | `sections/7_appendix/04_pbit_calculations/sl_09_25.tex` | Replace whole file |
| D: Comparison bounds and experiments | `sections/7_appendix/02_Comparison_table/092526_jh.tex` | Replace whole file |
| E: Frozen-geometry readout optimization | `sections/7_appendix/06_readout_optimization/sl_09_26.tex` | Create this folder and file |
| F: Error propagation for compiled arithmetic circuits | `sections/7_appendix/05_circuits/0925_cd.tex` | Replace whole file |

The folder numbers need not match the appendix letters. The order of `\input` commands below determines the printed order. Nothing needs to be uploaded next to `main.tex`.

## 2. Replace only the appendix input block in main.tex

Cmd+F for `\appendix`. Replace the `\newpage` immediately before it, the `\appendix` line itself, and the five appendix input lines through `05_circuits/0925_cd` with the following block. Keep the bibliography commands above it and `\end{document}` below it.

```tex
\clearpage
\appendix
\numberwithin{equation}{section}

\input{sections/7_appendix/01_Construction/09_26_sl}

\clearpage
\input{sections/7_appendix/03_bandwidth/092526_jh}

\clearpage
\input{sections/7_appendix/04_pbit_calculations/sl_09_25}

\clearpage
\input{sections/7_appendix/02_Comparison_table/092526_jh}

\clearpage
\input{sections/7_appendix/06_readout_optimization/sl_09_26}

\clearpage
\input{sections/7_appendix/05_circuits/0925_cd}
```

There is now exactly one `appendix`, in this block. The construction replacement no longer contains it. The same block controls appendix equation numbering. `clearpage` finishes pending figures before the next major appendix; no new package is required.

`APPENDIX_BLOCK.tex` contains this same block for easy copying. It is a copy source, not an additional file that you must upload or input.

## 3. Keep the figures and remove no other material

Leave these existing images where they are:

- `figures/appendix/bandwidth_width_rules.png`
- `figures/appendix/bandwidth_error_rules.png`
- `figures/appendix/comparison_results.png`

The bandwidth and comparison files already include their figures. Do not add a second input of `bandwidth_figures_only.tex`. Do not also include the old standalone `theorem32_appendix.tex`: the proof is now in Appendix E. Replacing the solver file removes its previous embedded copy automatically.

## Why this order

A proves the construction and its recovery guarantee. B chooses its bandwidth. C specifies how the precision experiment actually computes its features and readout. D derives the comparison-table entries and then shows the measured comparison. E proves the frozen-geometry training bound. F treats error propagation through circuits.

Each topic now has its own top-level appendix. Construction retains its internal proof order: setup, tanh reference construction, recovery, then GELU/SiLU extensions. Readout optimization follows its proof dependencies: residual recurrence, continuous-center representation, finite-center corrections, eigenvalue bounds, error/time bounds, numerical example.

## What changed

- Promoted the formerly nested bandwidth, solver, comparison, optimization, and circuit material to separate appendices.
- Moved the optimization proof out of the solver into its own file.
- Moved bandwidth and solver before the comparisons.
- Made the readout proof's six stages actual subsections, and promoted the bandwidth/comparison child headings accordingly.
- Removed a few repeated roadmap sentences. All 211 existing labels and all 192 displayed formulas are preserved, as are every theorem, proof, and figure environment.
- Kept the circuit correction that defines the recursive error majorants by equality, which its polynomial proof requires.
- Retained the numerical example's qualifications, clarified the main theorem's rate-bound notation, and supplied the archived lattice/exterior-center settings for that example.
- Reused the existing comparison-table macro instead of a broken local table reference.

Automatic references will acquire their new appendix numbers. No labels have been renamed. Existing unresolved placeholders elsewhere in the paper are a separate task; this reorganization does not fix or alter the main text. It does not add a proof of Theorem 3.3.

## Validation

An independent source review confirmed label, displayed-equation, and environment preservation. The six files compile together, with the theorem counters from the supplied main.tex. The editor's built-in compiler was unavailable, so the local check produced TeX layout output (XDV), not a replacement PDF or editor tab. It used a temporary article wrapper and compile-only placeholders for external main-paper references and bibliography entries. There were no undefined references or duplicate labels in that check; one harmless underfull-line warning remains. Recompile the full ICLR project to check final pagination.
