# expF04 uniform-center rerun implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rerun the complete two-layer expF04 benchmark with equally spaced ridge centers at widths 256 and 512, isolated from the sampled-center outputs.

**Architecture:** Add a separate experiment runner by copying the established `all20_2layers` protocol. Change both QI initializer calls from `uniform_centers=False` to `uniform_centers=True`, and use distinct experiment and result directory names. Do not alter the existing runner.

**Tech Stack:** Python, PyTorch, NumPy, pytest, Matplotlib.

## Global constraints

- Use all 16 cached datasets, both activations, all three initialization regimes, and seeds 0–2.
- Preserve 50 epochs, batch size 128, Adam at learning rate $10^{-3}$, and best-evaluation-loss tracking.
- Run widths $N=256$ and $N=512$.
- Write outputs under `results/checkpoint_F_applications/expF04_qi_init_real_data/all20_2layers_uniform/`.
- Leave existing sampled-center code and results untouched.

---

### Task 1: Add and run the isolated uniform-center experiment

**Files:**
- Create: `tests/test_expf04_uniform_centers.py`
- Create: `experiments/expF04_qi_init_real_data/all20_2layers_uniform/run.py`

**Interfaces:**
- Consumes: `SimpleMLP2`, `qi_ridge_init_layer_`, and the existing `data/cache_all20` files.
- Produces: `apply_init(model, scheme, xtr, p)` with uniform centers for `qi1` and `qi2`, plus JSON and figures for each width.

- [ ] **Step 1: Write the failing behavior test**

The test imports the new runner, replaces `qi_ridge_init_layer_` with a recording function, and asserts that `baseline` makes no call, `qi1` calls `fc1` once with `uniform_centers=True`, and `qi2` calls both hidden layers with `uniform_centers=True`.

- [ ] **Step 2: Verify the test fails because the runner is absent**

Run: `uv run pytest tests/test_expf04_uniform_centers.py -q`

Expected: failure while loading the missing `all20_2layers_uniform/run.py`.

- [ ] **Step 3: Add the minimal isolated runner**

Copy the existing experiment protocol, change the experiment/output labels to `all20_2layers_uniform`, and use:

```python
qi_ridge_init_layer_(model.fc1, xtr, centers_per_dir=p, uniform_centers=True)
qi_ridge_init_layer_(model.fc2, h1, centers_per_dir=p, uniform_centers=True)
```

- [ ] **Step 4: Verify the focused test passes**

Run: `uv run pytest tests/test_expf04_uniform_centers.py -q`

Expected: all focused tests pass.

- [ ] **Step 5: Run both widths**

Run:

```bash
uv run python experiments/expF04_qi_init_real_data/all20_2layers_uniform/run.py --no-wandb --width 256
uv run python experiments/expF04_qi_init_real_data/all20_2layers_uniform/run.py --no-wandb --width 512
```

Expected: 288 completed configurations per width, one JSON result file and two figures per width.

- [ ] **Step 6: Compare uniform and sampled centers**

Compute geometric-mean best-evaluation-loss ratios, configuration wins, paired uniform-to-sampled ratios, and the largest per-task changes from the two JSON pairs.

- [ ] **Step 7: Verify repository diagnostics**

Run the focused test again, inspect lints on the new Python files, and confirm both result JSON files contain 288 rows.
