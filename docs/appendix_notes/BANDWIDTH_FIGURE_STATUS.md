# Bandwidth appendix figures

Current request: two publication figures and a LaTeX insertion, extending the supplied bandwidth appendix while preserving its rigorous tanh pole bound.

- Figure 1: eight targets from the supplied three-panel figure; tanh, exact GELU and SiLU panels; predicted relative bandwidth against total width on a logarithmic axis to approximately 10,000; horizontal lambda=0.25 and activation-specific general Fourier-ratio root. The user clarified that "ratios" refers to the Fourier-transform ratio of the activation derivative, not gamma/W.
- Figure 2: multiple target functions and activations, measured relative L2 error against lambda; general-rule vertical lines and refined-rule markers; reuse saved observations where possible.
- Deliver PNG and vector PDF figures, reusable data/source, and appendix LaTeX with a compiled review PDF.

Scalar convention: epsilon_eff=2^-52 unless superseded; gamma=lambda/h and h=2/N. Keep N (interior intervals) distinct from W=N+2R+1. Preserve exact GELU rather than the tanh approximation. Lambda=0.25 is a rounded FP64 tanh default, not an independent rule for all activations. The refined root can drift slowly with width; it is not proved to converge to a width-independent constant.

Sources:
- User appendix: /Users/sam/.codex/attachments/ec79f374-507d-411f-9ec0-b0e6e1024ae1/Pasted text.txt
- docs/appendix_notes/2026-09-25-revised/04_activation_bandwidth.tex
- docs/lambda_theorem_compatibility/choosing_optimal_lambda/choose_lambda.py and reproduce_figures.py
- docs/lambda_theorem_compatibility/choosing_optimal_lambda/source_data/
- experiments/expC09_bandwidth_figures/combined.py (reference style)

Completed after the tool runner recovered. Final scope follows Sam's correction: width figure exactly 1×3; error figure 4×4 for tanh/GELU/SiLU/erf and four targets distinct from the main figure. The Fourier ratios are supplied separately. The complete replacement appendix, smaller insertion, vector/PNG figures, data, compiled preview, and exact integration instructions are in `docs/appendix_notes/bandwidth_figures/`.

Read and compared the newly supplied `choosing_optimal_lambda_overleaf (2).pdf` (attachment 063e37bb). Its equation (3) is the exact general first-pair prescription; the current paper's compact tanh equation is its controlled reduction. Page 6 explicitly excludes a constant fixed-precision asymptote of the refined roots. This is stated in the package. Formula/data checks passed and an independent agent reviewed the finished math.

September 26 revision: removed hollow/white markers and panel letters, separated legends from labels, standardized type/spines, and retained one activation-specific horizontal reference. Width axis now N. Error sweeps now N=32,64,128,256,512, reusing archived data and incrementally adding missing curves/marker fits. The fresh full-context agent completed the independent aliasing audit at bandwidth_figures/aliasing_audit.md. It identified concrete failures of the representative-frequency shortcut while verifying the expected direction of root drift. Full-spectrum Gaussian alias estimates match all four N32 observations within 0.01%; higher N does not systematically worsen selections. Tick/legend collision checks passed for all three output figures, and the compiled appendix has no layout or unresolved-reference warnings.
