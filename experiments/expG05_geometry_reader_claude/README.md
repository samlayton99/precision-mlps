# Geometry reader (Claude version)

An interactive editor for the geometry of $f(x)=\sum_k a_k\tanh(\gamma_k(x-c_k))+b$. You drag centers and slopes, the least-squares readout is re-solved live, and you can record Adam runs on the current geometry, intervene mid-run, and replay the recording. Everything is computed in fp64 numpy on a local server, and the browser only draws.

## Run it (laptop)

```
cd <this folder>
~/venv/precisionMLPs/bin/python app.py          # opens http://127.0.0.1:8790
```

Options: `--port N`, `--recordings DIR`, `--no-browser`. The only dependency is numpy. Recordings go to `~/my-repos/research/collaborations/precisionMLPs/results/checkpoint_G_generalization/expG05_geometry_reader_claude/recordings/`, or to the repo that contains this folder, one directory per run (`meta.json` + `frames.npz`). A deleted run moves to `recordings/_trash/`.

## Layout

- **Top knobs**: target, training data, solve (rcond, λ*), and display toggles. Changing any of the first three ends a live run or a replay.
- **Geometry plot** (big). The y-axis is $|\gamma_k|$ (log by default). The dots drag up and down. The teal dots on the bottom rail are the centers; they drag sideways. The purple diamond on the y-axis is the mean γ (geometric mean on a log axis), and dragging it scales every γ. The dashed orange curve is the ideal slope $\lambda^*/h_k$, where $h_k$ is the local spacing, so it is flat for uniform centers. Hollow dots mark negative γ.
- **Neuron count slider** rescales the current arrangement to the new count. Each neuron keeps its local $\lambda_k=|\gamma_k|h_k$, then all γ are rescaled so the mean λ is exactly preserved. As a result γ grows as the neurons pack closer.
- **Clean / Jitter** each take a strength from 0 to 100%:
  - Clean moves that fraction of the way toward the standard geometry. Centers go to the uniform grid with the chosen halo (default $R=\max(10,\lceil\sqrt N\rceil)$); γ goes to $\lambda^*/h_k$ at the current centers (interpolated geometrically).
  - Jitter centers at 100% adds noise with a standard deviation of one local spacing.
  - Jitter γ at 100% adds noise with standard deviation 1 in $\log|\gamma|$.
  - Undo and redo cover every edit.
- **Readout plot**: filled dots are the working (Adam) readout and drag up and down. Hollow dots are least squares; triangles mark least-squares coefficients too large to fit the scale. The red curve is $f'(x)\,h(x)/2$, which the readout should track (as it does for QI).
- **Fit** and **signed residual** plots use the readout chosen in the Display toggle. The residual axis is $\mathrm{sign}(r)(\log_{10}|r|+16)$; the grey line is the other readout.
- All four plots share the x-axis:
  - The mouse wheel zooms x.
  - Shift-wheel, or the wheel over a y-axis, zooms y.
  - Dragging the background pans.
  - Double-click resets the view.

**Full reset** (top right, click twice) ends any run or replay and restores every default: target, data, geometry, readout, and settings. Recordings are kept.

## Training and recording

- **Play / Pause / Stop**. The light is red and pulsing while training, dim red when paused, green during a replay, and grey when there is no run.
  - Play on a finished run extends it by *steps*.
  - Stop ends the run and keeps its end state as the editable geometry. The next Play continues from it with the Adam moments carried over.
- **Snapshots**: at each snapshot the loop does a least-squares solve on the current geometry, evaluates both readouts, and saves a frame before taking the next Adam step.
  - Spacing is either every *k* steps or geometric, where gap $i$ is $\min(\text{first}\cdot\text{ratio}^i,\ \text{max gap})$. The preview line shows how many frames a run will make.
  - *Delay per frame* slows a live run or a replay so it can be watched.
- **Edits during a run are staged**. They are drawn orange, with the run's state as grey ghosts, and the fit shows the edited model.
  - *Inject edits* writes only the entries you touched into the run. It records a flagged frame, marked in red on the scrubber.
  - *Inject lstsq readout* replaces the Adam readout with the least-squares one.
  - Pressing Play with staged edits opens a dialog: inject, or discard.
- **Replay**: the saved-runs list lets you open, rename or delete a run. The scrubber, the arrow keys, or a click on the history plot move through frames.
  - On a frame where an edit was injected, the plots flash red, the moved points turn orange next to grey ghosts of their old positions, and playback holds for one second.
  - Any edit during a replay ends it and continues from that frame. The next Play then records a new run, a fork of the replayed one that keeps its Adam moments.
- **Adam settings**:
  - lr, a separate geometry-lr multiplier, β1, β2, ε, full batch or minibatch
  - constant or cosine lr schedule
  - γ trained directly or as $\log\gamma$
  - per-group freezing (centers, γ, readout, bias)
  - readout init: zero, Xavier or least squares

  These settings are locked while a run exists. They match `torch.optim.Adam`, which is checked in the tests.

## Presets

The source of each preset is in its hover text and in `presets/*.json` (`source` field).

- **Generated at the current count:**
  - uniform with λ* and halo $R=\max(10,\lceil\sqrt N\rceil)$, or with no halo
  - Xavier (Glorot uniform, read as $c=-b/w$, $\gamma=|w|$)
  - Xavier slopes rescaled to median λ* (expD05)
- **Exact arrays from runs.** No run in the repo takes Xavier all the way to the fp64 floor on sine. The best are the expD06 `newton_handoffs` runs, all at N=512 on the target $\sqrt2\sin 2\pi x$:
  - the Xavier starting point
  - Adam 5.3M steps then Gauss-Newton: L2RE 3.2e-11, the best trained sine in the repo
  - SSBroyden: 5e-10
  - Newton: collapsed slopes
- **Graded meshes (expH02):** half-Gaussian, bimodal, Beta(2,5) (the failure case), Chebyshev
- **Irregular:** cascade (expG04), random and clustered (expC04), and a slope-monitor spike mesh (expH04)

`presets/_make_presets.py` regenerated the JSON files from the repo.

## Tests

`tests/test_expG05_geometry_reader.py`: run `uv run --extra dev python -m pytest -q tests/test_expG05_geometry_reader.py`.
