# expD26 — readout freezing and fixed-dictionary GD spectrum

Coordinator: main task. Status: complete. Sam confirmed all four targets and the readout singular-value interpretation; completed runs use those choices.

Sam's direction: prioritize readout freezing and Sections 3–4 of the gamma-barrier note. Do not pursue a Section-5 theorem or introduce another optimizer intervention here.

## Authorized experiments

1. Ordinary raw-parameter GD from the same seeded Xavier initialization. Clone one joint-GD trajectory at steps 2, 10, 50, and 150; freeze every readout coefficient including output bias; continue geometry GD for 500 updates. Same constant learning rate 0.002 throughout. Four prior matched targets, each on [-1,1], width parameter 128 with 24 halo slots per side, 1,024 training and 8,192 independent evaluation midpoints. One requested 2×4 PNG per target: actual relative L2 error and mean absolute slope gamma, with freeze markers. Include continued joint GD as a paired control. Mean gamma includes all 177 neurons. Sparse, evaluation-only readout refits establish whether geometry quality changes; no refits enter training.
2. Hold uniform centers, width, samples, and feature scaling fixed; increase lambda linearly from 0.1 to 2 (77 values, gamma=lambda/h, h=1/64). Animate the singular-value spectrum of the normalized readout matrix, including the output-bias column. Use the exact fixed-matrix GD recurrence for predicted convergence, with target loading and numerical approximation floor shown separately. Common rate 0.002 is primary; a documented per-dictionary spectral rate can distinguish conditioning from a fixed-rate artifact. This is an offline diagnostic, not a proposed optimizer.

## Verification and practicality checklist

- Training costs one forward/backward per update, plain SGD state; no dense Jacobians or extra optimizer state.
- Independent dense evaluations and SVDs are explicitly offline measurements; their storage and compute are not claimed to be training costs.
- No Krylov memory parameter or k/d scaling. Fixed-dictionary SVD is a small-problem reference, not a scalable solver proposal.
- Standard scalar loss reductions only during training; offline spectra and evaluation add diagnostic passes, counted separately.
- Freezing is an exact flag, tested by unchanged readout arrays. Branch starts must exactly match joint GD. All GD updates should agree with independent gradient formulas on small test cases.
- No training decisions compare tiny loss differences. Use dtype-scaled tolerances in numerical checks; preserve the explicit 1e-13 diagnostic rank cutoff from prior runs.
- Do not infer a finite convergence time from floating-point null singular values. Distinguish rank cutoff, error outside the retained span, and error on excited singular directions.
- Compare SVD predictions with explicit readout GD on representative lambda values. All targets and lambda values use a common normalization.
- This does not promote a production optimizer; batching, real-data, and precision tests are deferred until a method exists. Geometry and readout freezing are causal controls, not a claimed general recipe.
- Falsification: increased gamma with unchanged/worse refitted error does not establish useful geometry learning; improved conditioning with a worse approximation floor does not establish better accuracy.

## Ownership

- Main task: shared configuration, experiment status, scientific integration, final figure review, writeup.
- Freeze worker: freeze.py, its focused tests, and freeze outputs.
- Spectrum worker: spectrum.py, its focused tests, and fixed-dictionary outputs.
- Independent reviewer: mathematical assumptions and convergence interpretation; read-only review.

Results are in results/checkpoint_D_optimizers/expD26_freeze_and_readout_spectrum, with figures/, data/, and one completed expD26_results.md writeup. Earlier outputs are preserved.

## Completion report

- Freeze worker: four joint trajectories and 16 exact frozen-readout continuations; four requested 2×4 PNGs and one offline-refit companion. All branches have lower mean gamma and worse actual error than their matched joint controls. Mean gamma does not measure individual displacement; every slope trajectory and travel statistic is retained.
- Spectrum worker: 77 uniform lambda values, 24-second uniformly paced GIF, convergence PNG, and CSV covering three accuracy thresholds and two documented rates. Larger lambda improves bulk singular values but worsens the numerical approximation floor at large scales.
- Independent review: no material implementation defect. Tiny refit changes in Xavier geometry need qualification because solved coefficient norms are enormous. Predictions concern the retained numerical span, not exact mathematical rank.
- Verification: seven focused tests passed together; first frozen updates match analytic gradients within 2.8e-17; explicit GD and SVD predictions agree within 3.44e-15 over eight lambda/rate cases and all four targets through 2,000 steps. All 1,738 finite first-step brackets checked; no reporting-cap hits. Figures and representative GIF frames reviewed.
- Main task: integrated mathematical interpretation and results, documented scope and remaining direction-sensitive Section-3 question, and added the experiment to the session catalogue. No new optimizer mechanism was introduced.
