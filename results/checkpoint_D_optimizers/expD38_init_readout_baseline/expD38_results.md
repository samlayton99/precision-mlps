# expD38: initialization and readout baselines — Status: data-obvious, corrected comparison complete

**SARCOS correction:** all results below use disjoint splits of the training source. The earlier supplied-test results are invalid and have been archived. The supplied test inputs all duplicate source training inputs, as explicitly noted in *Better by Default*, appendix C.3.2. The other five tasks did not have this file-level overlap; repeated inputs are quantified separately.

## TL;DR

- All six tasks learn substantially beyond affine regression at width 512: ordinary standard-initialization training has 3.0–53.1 times lower mean test MSE across three seeds. The gain remains 2.8–52.8 times when excluding test inputs present in fitting or validation data.
- Training also beats validation-tuned ridge on frozen standard-initialization features by 2.3–18.0 times. Learning the hidden representation contributes on every task.
- QI lowers mean final test MSE on five tasks, but Kin8nm is effectively close across these seeds; SARCOS favors standard initialization in all three pairs. These are initialization/minibatch repeats on one split, not a statistical ranking across splits.
- The LS readout is a fitting diagnostic, not a guaranteed better predictor. Airfoil's QI mean final test MSE rises from 0.0495 to 0.1054 after the refit. Extending SARCOS to the tested 100k schedule also lowers fitting error while worsening validation and test error; 100k is not the default recommendation.

## Question / hypothesis

Can this restricted MLP learn substantially more than affine input regression and a readout fitted to frozen initial features? Once a credible ordinary-training baseline exists, does QI initialization change feature learning, or the gap between the trained and optimal linear readouts?

## Experiment design

The network is $f(x)=W_3\tanh(W_2\tanh(W_1x+b_1)+b_2)+b_3$, with two hidden layers of width 512 and a scalar output. Standard initialization uses Xavier uniform hidden weights with tanh gain $5/3$ and zero biases. Both arms start with exactly the same Xavier-initialized readout, gain 1, and zero output bias. The QI arm replaces both hidden-layer initializations with the existing F04 ridge-bundle construction, using $\lambda=0.25$, 22 centers per direction, uniform centers over each direction's robust projection range, and an independent initialization RNG. All parameters remain free during training; no directions or centers remain tied.

More precisely, this is a direction-adapted QI variant: each bundle $m$ shares a unit direction $u_m$, and its row norm is $\gamma_m=\lambda P_m/(2A_m)$, where $A_m$ is the 99.9th percentile of $|x\cdot u_m|$ and $P_m$ is the bundle's neuron count. Width 512 yields 23 bundles of 22 and one remainder of six. There is no single shared gamma across a layer. The initializer also has a spacing-calibration inconsistency: it computes nominal $h_m=2A_m/P_m$ but uses endpoint-inclusive centers with actual spacing $\Delta c_m=2A_m/(P_m-1)$. Thus the measured $\gamma_m\Delta c_m$ is 0.261905 for a full bundle and 0.3 for the remainder, despite the nominal configuration value 0.25. The reported runs retain this existing implementation; correcting it or testing a shared gamma requires a separately identified run.

At a checkpoint, let $H_\theta$ contain the current second-hidden-layer features. The diagnostic solves $\min_{w,b}\|H_\theta w+b-y\|_2^2$ on the complete fitting split using centered float64 SVD least squares, relative cutoff $10^{-12}$. It never writes the solution into the live network or optimizer. Singular-value extrema, numerical rank, coefficient norm, and training-error optimality checks are recorded. Thus four colored curves come from two training trajectories, not four training algorithms. The fifth, horizontal curve fits $\min_{a,c}\|Xa+c-y\|_2^2$ directly on the input features.

Inputs and targets are standardized using only the fitting split. Reported MSE is $n^{-1}\sum_i(\hat z_i-z_i)^2$, where $z=(y-\bar y_{\mathrm{fit}})/s_{\mathrm{fit}}$; raw-unit MSE is recovered by multiplying by $s_{\mathrm{fit}}^2$. No quantile transforms, polynomial features, or numerical embeddings are used. Bike Sharing uses shared one-hot encodings of season, month, hour, weekday, and weather, plus the continuous and binary predictors; identifiers and the target components `casual` and `registered` are excluded.

The outer split is the existing D20 split for Kin8nm and Pol and a seed-0 random 80/20 row split for the other tasks. Twenty percent of each outer training split is reserved for validation using seed 3701. SARCOS uses only the 44,484 distinct rows in its training source, without the earlier 30,000-row cap. Its supplied test file is excluded because it duplicates training-source inputs, as warned in [RealMLP, appendix C.3.2](https://arxiv.org/html/2407.04491v3#A3.SS3.SSS2). The original SARCOS pilot and comparison runs are archived and replaced. Data hashes, split hashes, scaling parameters, and split sizes are stored with each run. No held-out test score is evaluated by the learning-rate pilots.

Adam uses batch size 256, sampling with replacement, $\beta=(0.9,0.999)$, no weight decay, and float64 arithmetic. Learning rate is constant for the first 20% of updates, then cosine-decays to 1% of its initial value. Standard-initialization seed-0 pilots try $10^{-4}$, $3\times10^{-4}$, and $10^{-3}$ for 10,000 updates each. Minimum validation MSE selects one rate per task. Both initializations then use that rate, the same minibatch sequence within each seed, architecture, preprocessing, schedule, and 20,000-update budget. The comparison repeats initialization/minibatch seeds 0, 1, and 2 on a fixed data split. Shaded seed ranges describe these three runs; they are not confidence intervals or uncertainty over dataset splits.

The longer-budget check uses SARCOS because both seed-0 arms reach their best observed validation score at the 20,000-update endpoint. It starts fresh paired seed-0 runs with 100,000 updates and stretches the same schedule over that budget. It therefore tests a larger training budget with a correspondingly slower schedule, rather than appending 80,000 updates to the earlier checkpoint. Architecture, learning rate, batch size, and the two initialization constructions stay fixed. Both final-step and validation-selected scores are recorded.

The frozen-feature control fits ridge readouts to each initialization's step-zero features, selecting regularization by validation MSE from $\{0,10^{-10},10^{-8},10^{-6},10^{-4},10^{-2},1\}$. The objective is mean squared residual plus $\alpha\|w\|^2$ with an unpenalized intercept. This helps separate feature learning from overfitting an ill-conditioned initial least-squares system. Those control scores appear separately to preserve the requested five-line main panels.

The split audit counts exact input and input–target matches across splits. Its sensitivity check takes the existing test set and excludes every input vector also present in fitting or validation data. Saved final MLPs and the same fitted affine regression are evaluated on that shared subset, without retuning or selecting new checkpoints. This asks whether the nonlinear gain persists away from repeated inputs; it is not a replacement for a grouped evaluation by time or chemical family.

The Airfoil seed-0 weight-geometry diagnostic compares the two hidden matrices at initialization and after 20,000 updates. For row $i$ in layer $\ell$, its scale is $\gamma_{\ell i}=\|W_{\ell,i,:}\|_2$, excluding the bias. Direction retention is the signed cosine $c_{\ell i}=\langle W^{(0)}_{\ell,i,:},W^{(T)}_{\ell,i,:}\rangle/(\gamma^{(0)}_{\ell i}\gamma^{(T)}_{\ell i})$, matching the same row index throughout; no permutation or sign matching is applied. Initial weights are reconstructed using each saved run's recipe and seed, and their fitting/validation/test MSEs must reproduce the recorded step-zero errors. Final weights come from saved checkpoints. Layer-two directions refer to its fixed hidden-neuron coordinates; their cosine does not measure the change in the composed input-space feature function.

Training itself costs one minibatch forward/backward pass per update, with ordinary Adam state. Diagnostics perform complete-split forward passes and dense SVD outside the optimization loop; their cost is included in elapsed runtime. This experiment does not introduce or claim an Adam-cost least-squares optimizer.

**Code & data**

- Experiment: `experiments/expD38_init_readout_baseline/run.py`, `config.yaml`; requested axis views: `plot_comparison.py`; weight histograms: `plot_weight_geometry.py`; independent checks: `audit_readout.py`, `audit_splits.py`, `audit_qi_scales.py` in the same directory.
- Provenance and published comparison: `results/checkpoint_D_optimizers/expD38_init_readout_baseline/literature.md`.
- Run traces, diagnostics, final model and optimizer states: `results/checkpoint_D_optimizers/expD38_init_readout_baseline/data/{pilot,compare,long_budget}/`.
- Airfoil seed-0 hidden-row norms, cosines, summaries, histogram bins, and checkpoint identities: `results/checkpoint_D_optimizers/expD38_init_readout_baseline/data/weight_geometry/airfoil_seed0_steps20000.{npz,json}`.
- Selected recipes, comparison summaries, and audits: `results/checkpoint_D_optimizers/expD38_init_readout_baseline/{selected_recipe,summary,long_budget_summary,readout_audit,split_audit}.json`.
- Initial QI bundle ranges, norms, and actual/nominal center-spacing audit: `results/checkpoint_D_optimizers/expD38_init_readout_baseline/qi_scale_audit.json`.
- Invalid earlier SARCOS runs: `results/checkpoint_D_optimizers/expD38_init_readout_baseline/data/invalid_sarcos_supplied_test/`; corresponding figure snapshots: `figures/invalid_sarcos_supplied_test/` below the same experiment output directory.
- Figures: `results/checkpoint_D_optimizers/expD38_init_readout_baseline/figures/`.
- Scientific checks: `tests/test_expD38_readout_baseline.py`.

## Results

All 18 valid learning-rate pilots and 36 matched 20,000-update trajectories completed. The selected learning rate is $10^{-4}$ for Bike Sharing and Kin8nm, and $10^{-3}$ for the other four tasks, including the corrected SARCOS pilots. No width increase was needed to establish a substantial improvement over either baseline.

These are mean final-step held-out MSEs over seeds 0, 1, and 2 in standardized-target units. The improvement factor uses the ordinary trained standard-initialization model, without a readout refit. The frozen-feature comparison divides the mean frozen-control MSE by the mean trained MSE for standard initialization. All data splits are fixed across seeds.

| Task | Input OLS | Standard trained | QI trained | OLS / standard | Frozen ridge / standard |
|---|---:|---:|---:|---:|---:|
| Airfoil | 0.57194 | 0.07014 | 0.04952 | 8.15 | 3.14 |
| Bike Sharing | 0.32243 | 0.08302 | 0.06338 | 3.88 | 2.93 |
| Kin8nm | 0.59467 | 0.07350 | 0.07252 | 8.09 | 5.43 |
| Pol | 0.54315 | 0.01023 | 0.00512 | 53.08 | 17.96 |
| SARCOS | 0.07365 | 0.01118 | 0.01173 | 6.59 | 2.98 |
| Superconductivity | 0.27981 | 0.09210 | 0.08766 | 3.04 | 2.27 |

QI lowers mean final test MSE by about 29% on Airfoil, 24% on Bike Sharing, 50% on Pol, and 5% on Superconductivity; it wins each of the three paired seeds on those four tasks. Kin8nm has only a 1.3% mean difference and QI wins two of three pairs. On SARCOS, QI is about 5% worse and loses all three pairs. These task-specific results do not support universal initialization superiority.

The learned/readout-refit distinction matters most on Airfoil: its QI refit improves fitting error but substantially worsens mean test error. At seed 0, the test error rises from 0.0476 to 0.1864; the effect is sensitive to small singular directions, as the independent SVD audit shows. Across all 360 main-comparison LS evaluations, solved training MSE is no greater than the live trained-head MSE within the $10^{-8}$ numerical tolerance. The nominal solve never participates in training.

Superconductivity has 1,438 of 4,252 test inputs exactly present in the fitting split, although only 21 have an identical input–target pair. Excluding test inputs present in either fitting or validation leaves 2,686 rows; standard and QI MLPs still beat affine regression by factors of 2.78 and 2.95 respectively. Airfoil, Kin8nm, and corrected SARCOS have no exact fitting/test input matches. Bike Sharing has three and Pol nine; their exclusion has little effect on the improvement factors. This supports nonlinear generalization beyond exact repeated inputs, under the stated interpolation task.

For standard initialization, the minimum-validation checkpoints range from 5k–8k on Airfoil, 3k–4k on Bike Sharing, 6k–10k on Kin8nm, 10k–12k on Pol, 18k–20k on SARCOS, and 8k–12k on Superconductivity. QI reaches its minimum at 20k on all three SARCOS seeds. Thus 20,000 updates establish the requested baseline, while SARCOS remains the most direct test of a larger budget. The same step count is not the same number of passes through data: it corresponds to about 180 passes on corrected SARCOS versus 5,317 on Airfoil, with minibatch sampling with replacement.

Both corrected 100,000-update SARCOS runs completed. The following table compares final trained heads for the same paired seed 0; each budget uses its own stretched flat/cosine schedule.

| Initialization | Budget | Fitting MSE | Validation MSE | Test MSE |
|---|---:|---:|---:|---:|
| Standard | 20k | 0.004773 | 0.010492 | 0.011142 |
| Standard | 100k | 0.000188 | 0.014027 | 0.015803 |
| QI | 20k | 0.006351 | 0.010873 | 0.011843 |
| QI | 100k | 0.000755 | 0.013250 | 0.013885 |

The longer recipe fits much more closely but worsens both validation and test error. Within the 100k runs, minimum validation error occurs at 15k for standard and 22k for QI; even those minimum validation scores are worse than the corresponding 20k-budget endpoints. This does not isolate update count from schedule effects or prove that every longer recipe fails. It does establish that a blanket extension to this 100k recipe is unwarranted. Keep 20k as the diagnostic comparison budget and select predictive checkpoints using validation; investigate regularization or schedules before spending more updates on every task.

All 56 valid runs have saved final model, optimizer, scheduler, and minibatch-RNG states. Across the main and longer-budget comparisons, all 412 diagnostic LS evaluations satisfy the training-MSE optimality check. Each task's six main runs share identical data metadata and affine baseline scores.

On Airfoil seed 0, both methods increase median row scale and retain positive alignment with initialization for nearly all rows. QI has a smaller median rotation in each layer. The following entries are medians over 512 matched rows; norm growth is the median of each row's final/initial norm ratio, rather than the ratio of the two marginal medians.

| Hidden layer | Standard norm multiplier | QI norm multiplier | Standard cosine | QI cosine |
|---|---:|---:|---:|---:|
| First | 1.872 | 1.203 | 0.917 | 0.959 |
| Second | 1.125 | 1.067 | 0.915 | 0.954 |

The corresponding median rotations are $23.5^\circ$ and $23.8^\circ$ for standard initialization, versus $16.5^\circ$ and $17.5^\circ$ for QI. These describe the requested Airfoil seed-0 trajectories, not a multi-task or multi-seed conclusion about movement. They also do not measure whether different rows within an initial QI bundle remain mutually parallel.

The first-layer initial QI spread comes from variation between direction bundles. Full bundles have $A_m$ from 2.107 to 4.336, giving gamma from 0.634 to 1.305; the six-neuron remainder has $A_m=5.396$ and gamma 0.139. Every row within a bundle has exactly the same norm in the reconstructed weights, and the per-direction formula agrees within floating-point precision. This identifies the broad histogram as a consequence of the chosen projection-range adaptation and partial-bundle policy. It is separate from the center-spacing calibration inconsistency described above.

### Figures

- **Learning-rate pilots:** 2×3 validation-MSE trajectories, standard initialization only; one curve per learning rate and a gray dotted affine baseline. The requested view has logarithmic steps and MSE, with an MSE ceiling of $10^0$. Step zero is omitted; triangles mark sampled values above the ceiling. Full-range versions remain available.
- **Test comparison:** 2×3 panels with logarithmic gradient steps and standardized-target MSE, capped at $10^0$. Orange is QI, blue is standard; solid is the actual trained readout, dotted is the diagnostic LS readout. The gray dotted horizontal line is affine input regression. Lines average three seeds; shading spans their range, not a confidence interval. Width appears beneath each task name. Step zero is omitted and above-ceiling sampled means are marked with triangles. The main result is the large margin over the affine baseline, with task-dependent initialization differences.
- **Training comparison:** the same five-line layout and $10^0$ ceiling, with a lower limit of $10^{-4}$. This distinguishes fitting progress from test behavior and shows when the actual readout catches up with the diagnostic solve.
- **Validation comparison:** the same layout using the validation split. This identifies earlier minima or continued improvement without choosing checkpoints by test performance. Both full-range and late-training detail views are retained for all three splits.
- **Feature-learning controls:** two panels show validation and test error ratios to affine regression. Gray bars use frozen initial features with validation-tuned ridge; dark blue uses trained features with the actual head; light blue uses trained features with LS. Values are means over the three standard-initialization seeds. The learned representation contributes on every task.
- **Input-overlap controls:** the left panel gives the fraction of test inputs exactly present in fitting data. The right panel compares standard (blue) and QI (orange) trained-head MSE to affine regression after excluding test inputs present in fitting or validation; bars and whiskers show mean and range across three seeds. Both methods remain below the affine baseline on every task.
- **Airfoil readout sensitivity:** a 2×2 grid compares standard/QI columns and initial/final feature rows. Both axes are logarithmic, with identical limits and ticks across all four panels: relative singular-value cutoffs from $10^{-14}$ to $10^{-4}$, with shared endpoint padding, and standardized-target MSE from $10^{-4}$ to $10^{10}$. Blue is fitting error, orange is test error, and the gray vertical dotted line marks the nominal $10^{-12}$ cutoff. Independent classical SVD reproduces the large errors; excluding tiny singular directions changes the QI generalization result.
- **Airfoil gamma histograms:** a 2×2 grid places hidden layer one above layer two and initialization left of the 20k endpoint. Semi-transparent blue/orange histograms show standard/QI row norms, with 512 rows per histogram. All panels use the same 50 bins, linear axes, and percentage-of-rows vertical limits; the readout and biases are excluded. The first-layer scales start substantially larger under QI, while both second-layer distributions broaden during training.
- **Airfoil direction cosines:** a 1×2 grid shows hidden layer one on the left and layer two on the right. The horizontal axis is the signed cosine of each final row with its own initial row; vertical values are percentages of rows per bin. Blue/orange again identify standard/QI, with shared bins and axes and the full cosine range $[-1,1]$. Concentration near one indicates retention of the original direction.
- **Airfoil QI scale audit:** two panels explain the first-layer initial spread. The left plots row norm against neuron index, exposing constant norms within each 22-row bundle. The right plots each bundle's norm against its projected-data half-range, with the full-bundle formula as a gray dashed curve. Orange marks full bundles and red the six-row remainder; both panels share the gamma axis. This separates intended per-direction scale adaptation from the small remainder group's lower scale.
- **SARCOS longer-budget check:** three panels show fitting, validation, and test MSE through 100,000 updates, with the same five curve meanings. Both full-range and $10^0$-capped log-log views are retained. This is one paired seed with a fresh schedule stretched to 100k, not an average over the main comparison's three seeds.

## Additional details

Published benchmark numbers motivate task selection but do not share our exact splits, preprocessing, or two-layer constraint. Random row splits on Bike Sharing and Superconductivity are interpolation tests, not future-time or unseen-material-family evaluations. Kin8nm has medium output noise according to its original source, despite a prior local loader comment describing it as nearly noiseless. A validation plateau does not establish a noise floor or prove that width cannot help.

The SARCOS correction applies to D38. Earlier experiments using the supplied SARCOS test file, including the F04 loading path, require a separate overlap audit before their SARCOS scores can be used as held-out generalization evidence. Their results have not been silently rewritten here.

Six focused checks pass: rank-deficient affine least squares handles the intercept correctly; the two initialization arms have identical readout parameters; running the diagnostic leaves the next Adam update and RNG state bitwise unchanged; adding repeat seeds reuses completed runs while changes to the recipe or training budget invalidate that reuse; the invalid SARCOS protocol is rejected by run reuse and plotting filters; and a synthetic duplicate-test-file fixture produces disjoint fitting, validation, and test splits with the corrected loader.

An independent classical SVD (`gesvd`) reproduces the seed-0 Airfoil results from the original divide-and-conquer least-squares driver (`gelsd`). At the nominal relative cutoff $10^{-12}$, their largest relative MSE discrepancy is 0.12%, on the severely ill-conditioned initial QI features. At the trained QI endpoint, the readout coefficient norm is about $5.8\times10^6$, compared with 36.6 for standard initialization. A diagnostic sweep of singular-value cutoffs shows that suppressing tiny singular directions markedly reduces the QI readout's held-out error, while the standard endpoint is much less sensitive. This is evidence of conditioning and generalization sensitivity, not a justification for selecting a cutoff on the test set. The nominal curves retain the original cutoff throughout.

## Conclusions

The corrected protocol supplies six usable tasks and matched recipes for subsequent initialization/optimizer comparisons. The evidence supports learning useful nonlinear features at width 512 and retaining both readout measurements; it does not establish a noise floor or an initialization ordering across new data splits.

## Open questions

- Can regularization or schedule changes improve held-out error where further fitting overtrains the baseline?
- Does the initialization comparison persist across independent data splits and grouped task definitions?
- After fixing this baseline, what changes when only the second hidden layer receives QI initialization, or when a readout solve participates in training?
- How do the results change with consistent center-spacing calibration, balanced bundle sizes, and a clearly separated shared-gamma versus direction-adapted-gamma comparison?
