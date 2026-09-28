# Five research notes for Section 3

Revised 25 September 2026. These are source notes to clean up and fold into a future appendix. They are not standalone papers, and this folder does not assemble the appendix itself.

| Note | PDF | Editable LaTeX | What it adds |
|---|---|---|---|
| 1. True-p-bit implementation | [PDF](01_true_pbit_implementation.pdf) | [LaTeX](01_true_pbit_implementation.tex) | The arithmetic contract, operation order, solver cutoff, native-format checks, and completed expC11 evidence. This complements the supplied scaled-FP64 recovery proof; it describes a different implemented computation. |
| 2. Finite-kernel ratios | [PDF](02_finite_kernel_ratios.pdf) | [LaTeX](02_finite_kernel_ratios.tex) | The center-integral decomposition and its exact Fourier multiplier gain; explicit construction and gamma dependence of beta, delta, and tau; largest-eigenvalue normalization; and two-sided finite-network ratio and training-time bounds. Includes the saved mechanism diagnostics and figures. |
| 3. Geometry and readout | [PDF](03_geometry_readout.pdf) | [LaTeX](03_geometry_readout.tex) | The exact loss split, evidence that increasing the geometry learning rate or amplifying the approximation-floor gradient improves geometry, the Fourier/readout signal calculation, and finite-horizon movement budgets. |
| 4. Choosing a Practical Bandwidth | [PDF](04_activation_bandwidth.pdf) | [LaTeX](04_activation_bandwidth.tex) | Sam's supplied explanation: opposing error terms, an aliasing budget, cancellation of the fitted readout in Fourier ratios, and the general and frequency-dependent selection rules. Includes the matching proof appendix, the existing figure, and an additional appendix giving the derivative-transform formulas for tanh, sigmoid, exact GELU, SiLU, Gaussian, and sech-squared activations. |
| 5. Tapout comparisons | [PDF](05_tapout_comparisons.pdf) | [LaTeX](05_tapout_comparisons.tex) | The recent five-method error-versus-neuron curves show what the asymptotic table omits: the size needed before improvement slows, attainable finite-precision error, and whether more neurons or more bits help. Includes the arithmetic protocol, size-counting distinctions, threshold sensitivity, and a common-error comparison at the plotted cap-grid resolution. |

## Changes from the earlier packet

- The bandwidth explanation follows Sam's supplied `choosing_optimal_lambda_overleaf (2).pdf`, keeping its prose and argument order closely. Its matching proof appendix is retained. An additional short appendix preserves the general activation formulas; the finite-domain guarantee remains specific to the tanh construction.
- The geometry note leads with the successful learning-rate interventions. It distinguishes actual trained-readout error from the error after a least-squares refit, and separates exact identities, measured behavior, and a still-conditional causal account of geometry stalling.
- The neighboring/readout-conditioning note is excluded.
- The finite-kernel note now explains each term before the theorem. Its appendix derives the gamma-dependent Fourier gain, accounts for changing eigenvectors, constructs the retained grid and exterior-center corrections, and bounds every unretained term. It also shows what controls the largest eigenvalue. The interval theorem and existing numerical results are retained; an independent mathematical review checked the expanded explanation.
- The implementation note retains its reviewed content. All documents remain research notes.
- The fifth note uses the preferred error-versus-neuron curves, split across two pages for readability; it excludes the tapout-summary plot. It corrects two overstatements in the older experiment prose: the sensitivity to a stricter tapout threshold is larger than reported, and common-error dominance holds on the plotted cap grid rather than for every exact candidate neuron count. Notes 1–4 were not changed when adding this note.

These additions are complementary, not disjoint sentence by sentence. They restate definitions needed for standalone reading. In particular, the finite-kernel note expands an argument already appearing in the current draft, and the bandwidth note shares the tanh special case with the supplied material. Its additional content is the general activation derivation and selector, not a second finite-tanh proof.

## Evidence and limitations

All numerical figures and training outcomes are recovered from existing work. No training or experimental sweep was run for this revision. The PDFs distinguish proved inequalities from ordinary floating-point evaluations and reported experimental results. The finite-kernel note explains why higher-frequency contributions receive larger multiplicative gains, then uses finite-network intervals to check ratio improvement in the studied regime; it does not claim universal ratio monotonicity. Value bounds on corrections are not used as bounds on their derivatives. The bandwidth note does not transfer a finite-domain tanh theorem to arbitrary activations. The geometry note does not claim that the Fourier argument alone proves failure of joint training.

Each note identifies its primary sources. The broader recovered source inventory remains in `docs/appendix_materials/2026-09-25/` in the repository. This packet includes the figures needed to compile these five documents, rather than copying the entire historical archive.

## Rebuilding

All five PDFs were compiled directly from their accompanying LaTeX. With `tectonic` or `pdflatex` installed, run `bash build.sh` from this folder. Standard LaTeX packages are required; no bibliography download or experiment data is needed. The `figures/` directory must remain alongside the source files.

`MANIFEST.json` records the packet's file hashes. The ZIP contains this folder's five document PDFs, five LaTeX sources, figures, build script, and index; compilation logs and temporary files are omitted. PDFs inside `figures/` are the bandwidth figure and the two layouts of the saved tapout curves.
