# expG05 -- Geometry reader (Claude version)

**Status:** tool built 2026-09-26, pending Sam's use. It makes no research claims yet.

## TL;DR
- An interactive editor for 1-D tanh geometry. You drag the centers and slopes, and the least-squares readout is re-solved live in fp64. Adam runs can be recorded, paused, edited mid-run, and replayed.
- A separate version exists (Codex, expG02_geometry_reader_codex). The two were built independently.

## Question / hypothesis
This is a tool, not a test: a way to look at how geometry drives the least-squares floor, and at how Adam moves geometry when you intervene.

## Experiment design
- **Model**: $f(x)=\sum_k a_k\tanh(\gamma_k(x-c_k))+b$.
- **Local spacing**: $h_k=(c_{k+1}-c_{k-1})/2$ in sorted order, one-sided at the ends. The local $\lambda_k=|\gamma_k|h_k$ and the ideal slope is $\lambda^*/h_k$, with $\lambda^*=0.25$ for tanh.
- **Least squares**: $[\Phi,\mathbf 1]$ with $\Phi_{ik}=\tanh(\gamma_k(x_i-c_k))$, solved by `numpy.linalg.lstsq` (gelsd). Singular values below rcond·$s_{\max}$ are dropped; the default rcond is $10^{-13}$, matching `solve_readout`.
- **Adam**: full-batch MSE with the exact analytic gradient. It follows `torch.optim.Adam` semantics.
- **Metrics**: eval rel $L_2=\|\hat f-f\|/\|f\|$ and $L_\infty=\max|\hat f-f|$, on a prime-sized equispaced grid.
- **Verification** (`tests/test_expG05_geometry_reader.py`, 15 tests, all passing, including regressions for the review findings and Full reset):
  - The standard grid with halo $R=\max(10,\lceil\sqrt N\rceil)$ and least squares reaches rel $L_2$ 1-5e-14 on $\sin 2\pi x$ at W = 81-600; $L_\infty$ is 7e-14 to 7e-13 (largest at $x=-1$).
  - The gradient equals the complex-step derivative to 1e-13 relative.
  - 200 Adam steps match `torch.optim.Adam` to 1e-10.
  - Resampling to a new neuron count preserves the mean λ to 1e-12.
  - Clean at 100% returns the exact grid with $\lambda_k=0.25$.
  - The recorded trained preset reproduces its expD06 error (L2RE < 1e-10 with 8193 points).
  - Pausing and resuming, and forking from a saved replay frame, reproduce an uninterrupted run bit for bit.
  - An injected edit changes only the touched entries and is recorded as a flagged frame.
- A headless-browser pass (Playwright) exercised the following with no console errors:
  - drag, clean, and count change
  - play, pause, staging, the inject dialog, and inject
  - a 50k-step geometric run: 129 frames in 19 s at W=81 on the mini
  - live scrub and follow
  - a knob stopping a run
  - replay and seek to an intervention
  - a drag during replay forking a new run
  - loading the presets

**Code & data:** `experiments/expG05_geometry_reader_claude/` (`app.py` server, `engine.py` numerics, `presets.py` + `presets/*.json`, `static/` front end, `README.md` usage). Recordings: `results/checkpoint_G_generalization/expG05_geometry_reader_claude/recordings/` (gitignored).

## Results
None yet.

## Open questions
- None recorded yet.
