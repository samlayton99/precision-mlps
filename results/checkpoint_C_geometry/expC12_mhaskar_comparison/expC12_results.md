# Mhaskar and QUILL under matched parameter/output quantization — data-obvious

**Superseded:** the current experiment is [the strict p-bit construction/inference comparison](strict/expC12_strict_results.md). This report and its measurements are retained as the earlier parameter/output-only quantization control.

## TL;DR

- For the chirp, QUILL's relative error falls to $1.13\times10^{-13}$ at 53 bits. The tested Mhaskar specialization stays near $0.96$ across the precision sweep.
- Direct Chebyshev evaluation reaches $3.72\times10^{-15}$ at degree 128. The polynomial-to-tanh conversion is the limiting stage in this implementation.
- Both methods construct and evaluate in FP64; all model parameters and final predictions are explicitly rounded to $p$ significant bits. Intermediate arithmetic is not rounded to $p$ bits.

## Question / hypothesis

How does the finite-difference polynomial construction in Mhaskar (1996) compare with the existing QUILL precision-law experiment when both methods use FP64 construction and the same parameter/output quantization?

## Experiment design

The target is $f(x)=\sin(8\pi(x+1)^2)$ on $[-1,1]$. The precision sweep includes every integer $p$ from 8 through 53. Each method has a maximum width budget of 1024 activation neurons; the output bias does not count toward width. Mhaskar selects its degree within this budget, so the actual widths need not be equal.

For either method, construct an ordinary affine network $F_\theta(x)=\sum_j a_j\tanh(w_jx+b_j)+c$ in FP64. Then use $\theta_p=Q_p(\theta)$ and report $Q_p(F_{\theta_p}^{\mathrm{FP64}}(x))$, where $Q_p$ rounds binary significands to nearest with ties to even. Every hidden weight, hidden bias, readout coefficient, and output bias is rounded. Inputs and all intermediate evaluations remain FP64. Exponent range remains that of FP64. The relative error is $\|\widehat f-f\|_2/\|f\|_2$, computed in FP64 on 8001 uniformly spaced points.

QUILL uses 24 halo centers per side, so $N=1024-49=975$, spacing $2/N$, and the existing refined bandwidth rule with tolerance $2^{1-p}$ and representative angular frequency $16\pi$. The affine hidden parameters are $w_j=\lambda/(2/N)$ and $b_j=-w_jc_j$. The readout is fit to 4801 uniform samples by FP64 SVD with relative singular-value cutoff $2^{1-p}$, retaining the precision-dependent rank-selection policy of the existing experiment. Features and target samples are FP64 during this solve; completed parameters are then quantized.

The Mhaskar implementation specializes Lemma 3.2 of [Neural Networks for Optimal Approximation of Smooth and Analytic Functions](https://doi.org/10.1162/neco.1996.8.1.164). It uses a discrete Chebyshev projection on 4096 first-kind nodes and truncates to degree $d$. This is a practical polynomial front end; it does not reproduce the paper's filtered operator or unspecified theorem constants. After conversion to monomials $P_d(x)=\sum_r a_rx^r$, replace each monomial by

$$H_{r,h}(x)=\frac{\sum_{j=0}^{r}(-1)^{r-j}\binom rj\tanh\bigl(b+(j-r/2)hx\bigr)}{h^r\tanh^{(r)}(b)}.$$

Here $b=(\log 2)/2$. Compute scaled Taylor coefficients $c_r=\tanh^{(r)}(b)/r!$ in FP64 using the differential equation $y'=1-y^2$. Assemble stencil weights through direct FP64 recurrence, with logarithmic scaling only when intermediate coefficient ranges require it. Merge identical hidden slopes into one network, giving at most $2d+1$ neurons. There is no least-squares refit of the Mhaskar readout and no separately retained unrounded stencil scale during evaluation.

For each precision, select the degree and step with the smallest relative error on 1024 uniform midpoints disjoint from the reporting grid. The search contains 34 degrees from 0 to 511 and 65 logarithmically spaced steps from $10^{-4}$ to 4: 2210 candidates per precision. Degree choices are dense at small degrees and progressively coarser above 10. Models with nonfinite FP64 parameters are invalid candidates; 2087 candidates have finite parameters. All selected steps are interior to the sampled step interval. The reporting grid is never used for selection.

The source pages available through the [indexed preprint](https://citeseerx.ist.psu.edu/document?doi=694ad455c119c0d07036792b80abbf5488a9a4ca&repid=rep1&type=pdf) identify the divided-difference construction. This is an explicitly normalized centered tanh specialization, not a claim to reproduce an author-provided numerical implementation.

**Code & data**

- Implementation and configuration: `experiments/expC12_mhaskar_comparison/{construction.py,run.py,config.yaml,plot.py}`.
- Mathematical tests: `tests/test_mhaskar_comparison.py`.
- Reproduce: `.venv/bin/python experiments/expC12_mhaskar_comparison/run.py`.
- Saved observations and full candidate grid: `data/summary.json`, `data/mhaskar_search.npz`, `data/diagnostics.npz`.
- FP64 arrays containing the rounded parameters for every selected model: `models/{quill,mhaskar}_p*.npz`.
- Source fingerprints, precision protocol, and replay checks: `data/config.json`, `data/validation.json`.
- Figures: `figures/three_panel_mhaskar_comparison.png`, `figures/precision_law_comparison.png`, `figures/mhaskar_diagnostics.png`.

## Results

| Significant bits | QUILL relative error | Mhaskar relative error | Selected Mhaskar degree | Actual Mhaskar width |
|---:|---:|---:|---:|---:|
| 8 | $7.06\times10^{-2}$ | $0.982$ | 4 | 9 |
| 24 | $1.41\times10^{-6}$ | $0.961$ | 6 | 13 |
| 40 | $2.49\times10^{-11}$ | $0.957$ | 7 | 15 |
| 53 | $1.13\times10^{-13}$ | $0.960$ | 7 | 15 |

The selected Mhaskar models stay at low degree even though the search permits degree 511. The chirp's polynomial approximation needs substantially higher degree: its Chebyshev error is about $0.255$ at degree 64, $4.05\times10^{-4}$ at degree 80, and $3.72\times10^{-15}$ at degree 128. Over the sampled steps, converting higher-degree approximants into the merged tanh network does not preserve that accuracy in FP64.

### Figures

- **Three-panel comparison:** panels (a) and (b) retain the saved width and bandwidth observations; panel (c) replaces the previous arithmetic experiment with this matched quantization comparison. Purple is the selected Mhaskar construction, teal is QUILL, and the dashed reference has fixed slope $-1$ in $\log_2$ error versus bits, with its intercept fit over bits 16–40.
- **Standalone precision law:** enlarges panel (c), with identical measurements and reference line. The label $O(\log(1/\varepsilon))$ denotes the reference relation between required bits and inverse error; it is not a theorem inferred from this finite sweep.
- **Mhaskar diagnostics:** the left panel compares directly evaluated Chebyshev truncations on the reporting grid with the best sampled FP64 tanh-network validation error at each degree. The right panel shows held-out FP64 network error versus step for six degrees. Its vertical range focuses on errors from $10^{-4}$ through $10^{24}$; still larger errors lie above the displayed range. Small steps cause severe numerical amplification; large steps also change the approximation. Curves are measured results, not error bounds.

## Additional details

Fourteen construction tests check Taylor normalization, cancellation of lower moments, second-order decrease with difference step, Chebyshev coefficient recovery, merged-versus-separate stencil equivalence, and the exact parameter/output rounding protocol. All 92 saved models were reloaded: every parameter is exactly representable at its prescribed significand length, and replay reproduces the stored errors.

Selection uses a different grid from reporting. For example, at 53 bits the selected Mhaskar model has validation error about $0.951$ and held-out error about $0.960$. Strong cancellation makes small numerical changes relevant; the runner enforces matching feature layout and BLAS accumulation convention between its cached search and standalone evaluation. No higher-precision arithmetic is used to rescue construction or evaluation.

This experiment does not measure true $p$-bit operation-by-operation arithmetic. It does not establish a necessary precision cost or rule out different biases, polynomial front ends, step families, or numerical formulations of Mhaskar's method. Its budget comparison allows a method to leave neurons unused, rather than forcing a degree that has already become numerically unusable.

## Conclusions

For this chirp, implementation, parameter search, and quantization protocol, QUILL attains much smaller held-out error than the tested Mhaskar specialization. Direct Chebyshev approximation succeeds in FP64, while the tested conversion to a tanh network loses that accuracy.

## Open questions

- How much can alternative valid bias choices or equivalent stencil assembly conventions improve the FP64 construction?
- Does the observed separation persist on slower-varying targets where useful Chebyshev degrees are smaller?
