# Geometry reader — Codex version

Coordinator: Codex, Mac mini; requested runtime and saved recordings: laptop.
Updated: 2026-09-26.
Context revision: `1a626ad59f8931c5592c625afdfd0da80cb0bb5a` (local; context sync failed, installation check passed).

Outcome: an independent interactive geometry reader with synchronized plots, direct geometry/readout editing, float64 least squares, interruptible Adam, recorded interventions, and replay.

- Done: independent Codex app installed and running on the laptop at `http://127.0.0.1:8068`; computation and saved recordings remain on the laptop.
- Done: four aligned canvas plots, gamma/center/readout handles, mean-gamma control, ideal-lambda reference, zoom/pan, overlapping-neuron selection, cleaning/jitter and node resizing that preserves mean lambda.
- Done: staged edits, separate geometry/readout injection, inject/discard prompt, LS comparison by default, optional snapshot feedback, live interventions, fixed/geometric recording schedules, delay, exact replay/scrubbing and immutable branches.
- Done: six numeric presets with source evidence in [geometry_reader_codex_presets.md](geometry_reader_codex_presets.md), including the ordinary-Xavier sine geometry reaching about 6.65e-15 after VarPro Adam then Gauss–Newton.
- Done: native `Geometry Reader Codex.app` launcher built under the laptop repository's `results/checkpoint_G_interactive/geometry_reader_codex/`; recordings use its `runs/` subfolder.
- Verified: all 75 engine/controller/HTTP tests pass on the laptop. An actual 50,000-step small-model run reached the exact endpoint with 38 persisted geometric snapshots.
- Verified in browser against the laptop engine: all six source-setting imports, no JavaScript errors, no horizontal layout overflow, gamma and center dragging, exact discard, pending edits surviving LR changes, resize invariant, readout injection, live Adam intervention, autosave, playback from the end, bidirectional scrubbing, branching without changing the original, shared-axis zoom, and selecting one of 461 coincident Xavier centers.
- Independent review: fixed preset alias/grid mismatches, lost previews after setting changes, replay-at-end behavior, click-only mutation, and offscreen ideal-gamma guide.
- Boundaries: only `experiments/expG02_geometry_reader_codex/`, its new test file, and Codex-specific documentation; no changes to or inspection of the parallel Claude version.
- User clarifications incorporated: gamma vertical axis; explicit injection; LS comparison by default; standard lambda cleaning rule; exact preservation of mean lambda when node count changes.
- Usage and implementation notes: [experiment README](../experiments/expG02_geometry_reader_codex/README.md).
- Blocked: none. Changes remain uncommitted.

## Corrected halo rule and full reset

- Sam corrected the rule to `R=max(10,sqrt(N))`. N now counts interior points including both endpoints. Default N=64 has R=10 and W=84; the clean preset uses N=144, R=12, W=168. Halo counts round upward to whole points per side. Resizing retains mean lambda. At N=100, h=2/99, R=10, total=120, and ideal gamma=12.375 for lambda=0.25.
- Historic presets and recordings keep their saved arrays. Explicit custom extents remain available.
- Full reset stops training/replay and restores the complete default experiment, optimizer, recording settings and plot controls; it clears pending changes/current timeline and retains saved runs.
- Verified: all 69 tests pass on the laptop. An isolated browser instance confirmed full reset during live training/replay, complete restoration of experiment/view controls, preservation of saved recordings, and both sides of the max-halo rule; no JavaScript errors.
- Deployed to the running laptop app. Saved and restored its current geometry/readout previews and discard baseline; both pending flags and all parameter arrays remained intact. Stopped the temporary verification server afterward.

## Interior-point count correction

- N is the integer number of interior reference points, including both endpoints. The control shows N separately from per-side halo and total width; h=2/(N-1), R=max(10,ceil(sqrt(N))), W=N+2R. N=100 therefore yields 120 total points and ideal gamma=12.375 at lambda=0.25.
- Preserved mean lambda and center-displacement profiles during resizing. Readout interpolation scales with reference spacing. Historical source counts are translated from intervals to points without changing their stored arrays.
- All 75 regression tests pass on the laptop. Isolated browser checks verified the numeric count field and slider (100 and 101), halo rounding, mean-lambda preservation, preset counts and the sine-floor refit, complete reset including live training/replay, retained saved runs, and no JavaScript errors.
- Laptop app updated with its current parameters, preview flags, and discard baseline preserved exactly. An update backup is retained under its results directory.
