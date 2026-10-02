# Task selection and training evidence

Reviewed September 29, 2026. The purpose is to find regression problems with a substantial learnable nonlinear component, then test initialization in a plain MLP with at most two hidden layers. A poor affine fit alone is insufficient: it can also indicate noise, bad features, or a numerical problem.

## Published evidence

[Holzmüller et al., *Better by Default*](https://arxiv.org/html/2407.04491v3), Tables D.7 and D.11, report the following mean normalized RMSE across ten splits. Smaller is better; these are published results, not our reruns.

| Candidate | Published MLP-D nRMSE | Published RealMLP-TD nRMSE | Table |
|---|---:|---:|---|
| Airfoil Self-Noise | 0.308 | 0.180 | D.7 |
| Bike Sharing Demand | 0.244 | 0.228 | D.11 |
| Kin8nm | 0.302 | 0.242 | D.7 |
| Pol | 0.133 | 0.067 | D.7 |
| SARCOS | 0.132 | 0.117 | D.7 |
| Superconductivity | 0.308 | 0.305 | D.11 |

RealMLP uses three 256-unit hidden layers, additional feature embeddings, special parameterization, and 256 epochs with batch size 256. These numbers establish task potential, not performance guaranteed for our restricted architecture. The paper does not provide the matched affine baseline needed here. Its individual-task results also illustrate recipe sensitivity: on Elevators, the default MLP has nRMSE 0.745, versus RealMLP's 0.280. A weak initial MLP result is therefore insufficient grounds for rejecting a dataset.

[Gorishniy et al., *Revisiting Deep Learning Models for Tabular Data*, official California Housing tuning configuration](https://raw.githubusercontent.com/yandex-research/rtdl-revisiting-models/main/output/california_housing/mlp/tuning/0.toml) uses target standardization, AdamW, batch size 256, validation early stopping, and a learning-rate search from $10^{-5}$ to $10^{-2}$. Its architecture search includes widths up to 512 and depths beyond our limit. We adapt the optimizer family, scaling, batch size, and validation-based tuning; our fixed two-layer tanh architecture and schedule are local experimental choices.

[RealMLP's official implementation](https://raw.githubusercontent.com/dholzmueller/pytabkit/main/pytabkit/models/sklearn/default_params.py) confirms that its regression learning rates are attached to an NTK parameterization and other changes. Copying its large learning rate into an ordinary PyTorch MLP would not reproduce its recipe.

[Gal and Ghahramani, Section 5.3](https://proceedings.mlr.press/v48/gal16.pdf), provide a useful precedent for separating tuning from adequate final training: their small regression networks use 50 units, Adam, and batch size 32, and they increase the training iterations tenfold after selecting hyperparameters because the shorter runs have not converged. This supports using bounded pilots to locate a recipe without mistaking their endpoint for the task's attainable error.

## Task-specific interpretation

- **Airfoil:** a small physical regression problem. The [UCI source](https://archive.ics.uci.edu/dataset/291/airfoil%2Bself-noise) identifies five inputs and measured sound pressure from NASA wind-tunnel experiments. This tests whether the simple MLP can learn a strongly nonlinear low-dimensional response without requiring a large dataset.
- **Bike Sharing:** hourly counts with weather and calendar inputs. The [UCI source](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset) makes clear that `casual + registered = cnt`; those two count columns must be excluded. Give both affine regression and MLP identical categorical encodings, including hour-of-day, so the comparison is not driven by treating category codes as a linear quantity. Random row splits measure interpolation, not future-time forecasting.
- **Kin8nm:** the [original DELVE documentation](https://www.cs.toronto.edu/~delve/data/kin/desc.html) explicitly distinguishes fairly linear/nonlinear and medium/high-noise variants. `kin8nm` is nonlinear with medium noise, not noiseless. [Gal and Ghahramani's regression benchmark](https://proceedings.mlr.press/v48/gal16.pdf) provides independent neural-regression evidence. Do not interpret an eventual plateau as purely an optimization limitation.
- **Pol:** use the original 48-input OpenML regression task cached in D20. The RealMLP paper lists this 48-input version separately from the reduced 26-input Grinsztajn benchmark; their results are not interchangeable. We avoid inferring a physical interpretation from the short dataset name.
- **SARCOS:** joint-one torque from 21 state variables. **Do not use the supplied test file as a held-out set.** [RealMLP, appendix C.3.2](https://arxiv.org/html/2407.04491v3#A3.SS3.SSS2), explicitly excludes it because it duplicates training samples. Our source audit confirms that all 4,449 test inputs occur among the 44,484 training-source inputs, with 4,446 entire 28-column rows identical. The corrected experiment uses only the 44,484 distinct source rows, with an outer random 80/20 split (seed 0) and inner validation split (seed 3701). The earlier supplied-test runs are archived as invalid for generalization claims. No 30,000-row cap is applied.
- **Superconductivity:** include a larger, higher-dimensional materials task. The literature provides evidence of learnable structure, while its weaker neural error than the other candidates makes it useful for checking whether a longer training budget matters. A random row split is an interpolation benchmark, not a test on unseen chemical families.

## Selection and budget rules

Select the tasks from published evidence and ordinary-initialization validation pilots, without selecting for a QI advantage. Start at width 512 in each of two tanh hidden layers. Try learning rates $10^{-4}$, $3\times10^{-4}$, and $10^{-3}$ with otherwise identical conditions. Choose the rate by validation MSE, then use it unchanged for both initializations and both readout measurements.

The pilot has 10,000 updates per learning rate; the paired comparison has 20,000 updates and three initialization/minibatch seeds on a fixed split. These are diagnostic budgets, not a claim of convergence. A task with published neural success but poor local results receives a recipe/optimization investigation before rejection. Consider more updates before increasing width; if width is needed, try 1024 and then at most 2048, applying the chosen width to both initializations. SARCOS also receives fresh paired 100,000-update seed-0 runs, including a stretched flat/cosine schedule, with validation-selected checkpoints reported alongside final-step errors. The split correction invalidates and replaces its earlier pilots as well as its main and longer-budget comparisons.

Measure five quantities: affine input regression; the actually trained readout; an unregularized least-squares readout on current hidden features; an initial frozen-feature least-squares readout; and a validation-tuned ridge readout on those same frozen initial features. The requested main figure has only the first three estimator types (five lines across the two initializations). The frozen-feature ridge control goes in a companion figure. Improvement over it supports a benefit from learning the hidden representation, beyond simply fitting a linear output to an initial nonlinear dictionary.

Only affine input scaling is used in this first experiment. Quantile transforms and nonlinear feature embeddings can improve tabular models, but would change what the raw-input linear baseline means. The categorical expansion for Bike Sharing is shared explicitly by every estimator.

An exact-input overlap audit accompanies the corrected results. A separate, untuned sensitivity analysis evaluates the saved final models on test inputs absent from both fitting and validation data, using the same subset for MLP and affine regression. This is particularly relevant to repeated composition-derived inputs in Superconductivity; it is still not a grouped test of unseen chemical families.
