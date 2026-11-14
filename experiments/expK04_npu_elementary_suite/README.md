# Small NPU elementary-function screen

Current scope: `1/sqrt(x)` as a native control, then `1/(sqrt(x)+x)`,
`exp(-x)/sqrt(x)`, and `-log(x)/sqrt(x)`, all on **[0.0001,1]**. The originally
drafted wider suite was paused before training and reduced at the user's request.
Unused targets in `problem.py` were not measured.

This corrects the earlier benchmark mismatch: train on the actual positive
interval, not QUILL's auxiliary C17 continuation. All outputs are physical-dx
relative L2, with a general metric valid for every target. Importance-weighted
log sampling preserves that objective. Inputs to models remain physical x.

The core NPU implementation is the verified port in `src/models/neural_power_unit.py`.
The initial small screen tries shallow 6/32-unit models and a small stacked model
at two Adam rates. It records best and final Adam performance, then a separately
reported L-BFGS polish. Native scalar controls use fixed/learned gates. A generic
five-parameter compositional model was added as a follow-up so an easily
representable arithmetic composition is not tested only with a large network.
All parameter initializations are ordinary random Glorot, with the documented
zero-exponent native control; no correct exponent/coefficient initializes a run.

Selection uses seed-0 validation only. A fresh seed checks each selected recipe.
This is an illustrative screen, not a reliable estimate of success probabilities
or an exhaustive claim about what NPU can learn. Source snapshots preserve the
original screen before adding the optional model-factory seam for the compact
follow-up.

```sh
~/venv/quill-radon/bin/python -m pytest -q tests/test_npu_elementary_suite.py
~/venv/quill-radon/bin/python experiments/expK04_npu_elementary_suite/run.py
~/venv/quill-radon/bin/python experiments/expK04_npu_elementary_suite/compact_check.py
~/venv/quill-radon/bin/python experiments/expK04_npu_elementary_suite/analyze.py
```

Results live in `results/checkpoint_K_practical_recipes/expK04_npu_elementary_suite/`.
See `results.md` and `all_runs.csv` for actual outcomes. Existing results are
preserved; source-hash checks deliberately reject reusing a changed original
specification. The two final selected-recipe rechecks were executed with the
same saved configs at seed 1 and are saved in `selected_recipe_rechecks/`.
