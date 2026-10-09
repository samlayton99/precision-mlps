# Read-only explanation audit, October 3, 2026

Coordinator: root. Requested by Sam after the Claude continuation. Scope: recover the mathematical procedure, audit claims against source and saved results, compare with the PINN literature, and explain everything in chat. No new numerical experiments, solver changes, commits, or research recommendations are authorized in this phase. Sam will decide next work only after understanding the methods.

Sources: this repository at ca6dc9d; the live sibling `quill-radon` repository; relevant local agent transcripts and the October 3 handoff. Local context check passed; cross-machine context sync failed transport, so this is a local audit, not a claim of a freshly synchronized remote inventory.

## Shared foundation before the results tour

- A known analytic directional profile can be encoded into an ordinary tanh ridge block using QUILL's complex-shift density and finite-contour correction inside the existing sqrt(N) halo. The formula needs the profile; it does not discover an unknown PDE solution.
- Radon quadrature supplies the angular leg. Finite directions and within-direction resolution are separate approximation limits. Supplied sparse ridge support is a stronger assumption than a generic high-dimensional target.
- The previous route 2 used independent function coordinates a, an explicit map E from those coordinates to flat tanh readouts, native PDE residuals, streamed linearized corrections, and ordinary-network audits. Ideal polynomial correction operators approximate actual neural derivatives. Whether each newer variant evaluates and accepts the actual neural residual must be traced in code.
- A million output neurons need not mean a million independently solved coefficients. Report directions M, centers N, total neurons B, independent coordinates P, samples Q, and every dense array separately.
- Standard PINNs numerically minimize equation/BC/IC/data discrepancies. Newton/Gauss–Newton and preconditioning are not disqualifying by themselves. Right coordinate transforms differ from changing residual weights or adding a penalty.
- Project constraints are stricter than the community's definition: one global hidden layer, no externally solved field silently serving as target, memory/scaling discipline, general equation setup, honest accuracy labels. Record any later explicit user revision with its source; agent summaries of revisions are not automatically authority.
- Distinguish field error, PDE residual, scaled residual, encoding error, refinement difference, parameter error, and stopping status. Manufactured forcing/BCs are legitimate verification inputs; interior truth entering a solve or tuning needs explicit disclosure. No small sampled residual is automatically a continuum error certificate.

## Literature refreshed

Raissi et al. 2019 (forward/inverse framework, doi:10.1016/j.jcp.2018.10.045); Dong and Li 2021 (fixed hidden features with linear/nonlinear least squares, arXiv:2012.02895); Krishnapriyan et al. 2021 (optimization failures, arXiv:2109.01050); PINNacle 2024 (broad benchmark, arXiv:2306.08827); Sirignano and Spiliopoulos DGM (meshfree high-dimensional PDEs in selected classes, arXiv:1708.07469); Fong and Saunders LSMR (operator products for least squares, arXiv:1006.0758); Shin et al. (residual-to-error analysis requires assumptions, arXiv:2010.08019).

## Work state

- Root: original theory, requirements, and primary literature refreshed; both audit reports read, selected native/slab/Poisson/Bratu sources checked directly, explanatory synthesis in progress.
- Results auditor: completed the mechanism/evidence report in `continuation_evidence.md`; scoped helper completed `theory_history_evidence.md` with direct transcript authority, construction metrics, and conditional-theory audit.
- Delivery: self-contained detailed chat explanation. Internal reports are supporting pointers, not a substitute for the answer.

## Scope clarification during the audit

Sam supplied Claude's own retrospective admission during this task. Permission to use a polynomial core was conditional on delivering the requested PINN capabilities; it did not authorize substituting arbitrary classical boundary-integral or time-stepping solvers and counting their compiled outputs as success on the original goal. The audit must preserve both the actual conditional permission and this clarification. Numerical findings from those routes may remain useful, but their purpose, information inputs, equation-specific choices, and compliance must be reported separately. This task remains explanation only.

## Confirmed distinctions for the chat explanation

- The original native route2 evaluates/accepts actual tanh residuals. Newer uses of the same route name often evaluate polynomial or Fourier fields first and compile afterward.
- The new uniform-grid FFT nonlinear readout probe is a real native result under special whole-line/analytic-forcing/cutoff assumptions. Its fast general-domain replacements failed; the successful disk alternative retained dense pseudoinverses.
- Profile/tube regularization repairs numerical compilation of an already solved field. It is a dense fitted prototype, not an inference-order generic PDE solver.
- Million-neuron Burgers and Allen–Cahn runs compile precomputed spectral fields. Burgers has an independent analytic comparison; Allen–Cahn's main metric is sampled encoding error against the input field, not a published full-grid comparison.
- The reported 6D Poisson near-floor number is a polynomial boundary discrepancy; no tanh export appears in that runner. The 6D nonlinear network result is about 1e-6, with a real physics-only polynomial solve before compilation.
- Duct flow reduces to scalar 2D Poisson. Bratu-disk setup uses the exact solution to select continuation/grid geometry; the rod-domain counterpart uses a linear proxy instead.
- Current code removes some full-matrix allocations but retains dense profile fits, blocked all-pairs direction work, quadratic coordinate work, or tensor grids depending on the method.
- The recent DSGNAR Burgers paper uses viscosity .01 and periodic input embedding, whereas the sharper archive case uses .01/pi. Printed literature scores are not matched benchmark results.
