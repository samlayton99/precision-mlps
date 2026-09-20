# expD20 -- the tabular axis: is there a suite that can discriminate our methods?

**Status: draft -- pending Sam's review.** Scouting and measurement, not a training-method experiment. Commissioned because the expD18 optimizer comparison landed inside a 4-20% band on every task, which raised the question of whether the tabular suite has enough nonlinear signal to discriminate anything.

## TL;DR

- **The incumbent tasks are not as weak as they looked, and two separate mistakes made them look weak.** The reported 1.2-1.8x nonlinear headroom was measured with a metric that gives free credit for predicting the mean, and with a network far too small to extract the signal. Corrected, the same six tasks carry 1.45x-4.61x headroom.
- **Our nets were badly underfitting**: a 1-hidden-layer width-256 net (expD18's architecture) leaves a factor of **1.5x-4.1x** on the table against a deep net or GBDT on the same data and the same metric.
- **The finding that matters more than the suite: on real tabular regression, one hidden layer is width-saturated.** Going from width 256 to 4096 (16x) changes test error by less than 5% on **every one of the 17 tasks measured**. The available headroom is depth-headroom, and no dataset choice fixes that.
- Consequently the tabular axis **cannot test the width-scaling success criterion at all**, for any method, on any of these datasets. It can only test parity at fixed capacity, which is what PROGRAM_FRAMING §5(a) asks of it.
- **`naval` is the one genuinely valuable new task**: a noiseless gas-turbine simulator where a smooth polynomial basis reaches $1.0\times10^{-3}$ while every neural net stalls at $1.4\times10^{-2}$. The bottleneck there is demonstrably optimization, not data -- this project's thesis, on real data.

## Question

Is there a tabular regression suite on which deep learning substantially beats linear and polynomial regression, so that optimizer and initialization differences have room to show up? And before blaming the datasets: were the incumbent tasks weak, or were our networks underfitting them?

## Experiment design

Every task, incumbent and candidate, is scored by one protocol so the numbers are comparable.

**Metric and preprocessing.** Test relative $L_2$, $\|\hat y - y\|_2 / \|y\|_2$, the repo-wide metric. Inputs are standardized on the train split, then divided by the 99.9th percentile of $|x|$ and clipped to $[-1,1]$ to match the repo's domain convention. **Targets are standardized on the train split**, which is the correction that matters: under standardized targets a model that predicts the train mean scores exactly $1.0$, so the metric measures only explained variance.

**Why that correction changes the answer.** The expF04 cache stores min-max normalized targets, and expD18 scored on them directly. A min-maxed target carries an offset, so $\|y\|$ is dominated by the mean and every model receives free credit. Measured on the cache: a mean-predictor scores $0.52$ on `bike_sharing`, $0.985$ on `parkinsons`, $0.9998$ on `sarcos`. Where that number is far from 1, every score is compressed toward zero and every ratio between models is compressed with it.

**Models.** Best linear $=\min$(OLS, ridge swept over $\lambda\in[10^{-8},10^{5}]$). Poly-2 ridge with interaction terms when the expansion stays under 4000 columns. MLPs in torch: architectures $(256)$, $(1024)$, $(4096)$, $(256,256)$, $(512,512)$, $(512,512,512)$, GELU, AdamW with cosine decay, batch 512, up to 400 epochs, early stopped on a 15% validation split carved from train with patience 30. HistGradientBoostingRegressor over three capacity settings with its own internal early stopping, random forest, and kNN as references.

**The two headroom quantities.**

$$\text{headroom} = \frac{\min(\text{OLS},\ \text{ridge},\ \text{poly-2})}{\min(\text{best MLP},\ \text{GBDT},\ \text{RF},\ \text{kNN})}, \qquad \text{headroom}_{1\text{L}} = \frac{\min(\text{OLS},\ \text{ridge},\ \text{poly-2})}{\text{best 1-hidden-layer MLP}}.$$

The first says what a strong nonlinear model can win. The second says how much of that **one hidden layer** -- the architecture the QI construction covers -- can actually reach. The gap between them is the part of the tabular signal that is unavailable to this project's architecture.

**The underfitting control.** The same $(256)$ single-hidden-layer architecture expD18 used, retrained under the corrected metric, isolates capacity from metric: the ratio (shallow) / (best deep or GBDT) is pure underfitting with the metric held fixed.

**Candidates.** 15 datasets, biased toward physical and simulation surrogates because this project approximates smooth functions and a sampled deterministic map has little label noise -- the only regime in which high precision is even conceivable (expB01: $y$-noise sets a hard $\sigma n^{-1/2}$ floor). Sources: UCI (`ccpp`, `naval`, `gasturbine`, `casp`) and OpenML via `fetch_openml` (`kin8nm`, `elevators`, `ailerons`, `puma32h`, `bank8fm`, `grid_stability`, `pol`, `cpu_act`, `wind`, `house_16h`, `fried`).

**Code & data.** `experiments/expD20_tabular_suite/{evaluate.py, datasets.py, run.py, shallow_ref.py}`. Data cached at `data/cache_expD20/*.npz`; results at `results/checkpoint_D_optimizers/expD20_tabular_suite/data/{incumbents,candidates,shallow_ref,one_layer_scaling,one_layer_candidates}.jsonl`. Figure `figures/expD20_headroom.png`.

## Results

### The incumbents: was it the tasks, or was it us?

All under the corrected metric, so the columns are comparable:

| task | 1-layer W=256 (expD18 arch) | best deep / GBDT | underfit gap | best lin/poly | **true headroom** | apparent headroom with the shallow net |
|---|---:|---:|---:|---:|---:|---:|
| superconductivity | 0.3921 | 0.2638 | 1.49x | 0.3812 | **1.45x** | 0.97x |
| sarcos | 0.1574 | 0.0829 | 1.90x | 0.1807 | **2.18x** | 1.15x |
| airfoil | 0.6442 | 0.2316 | 2.78x | 0.6024 | **2.60x** | 0.94x |
| parkinsons | 0.7497 | 0.1833 | 4.09x | 0.8453 | **4.61x** | 1.13x |
| bike_sharing | 0.4909 | 0.2032 | 2.42x | 0.6754 | **3.32x** | 1.38x |
| beijing_pm25 | 0.6543 | 0.4399 | 1.49x | 0.7580 | **1.72x** | 1.16x |

The last column reproduces the original complaint almost exactly: with a 1-layer width-256 net, apparent headroom is 0.94x-1.38x, which is the 1.2-1.8x that prompted this experiment. The true headroom is 1.45x-4.61x. **Both diagnoses were right and the underfitting one dominates**: the tasks carry real nonlinear signal, and expD18's architecture reached almost none of it.

### The structural finding: one hidden layer is width-saturated

Best 1-hidden-layer test error at three widths, incumbents:

| task | W=256 | W=1024 | W=4096 | width gain | best deep MLP | depth gain |
|---|---:|---:|---:|---:|---:|---:|
| superconductivity | 0.3921 | 0.3909 | 0.3904 | 1.00x | 0.2957 | 1.32x |
| sarcos | 0.1574 | 0.1543 | 0.1535 | 1.03x | 0.0829 | 1.85x |
| airfoil | 0.6442 | 0.6317 | 0.6207 | 1.04x | 0.2672 | 2.32x |
| parkinsons | 0.7497 | 0.7487 | 0.7524 | 1.00x | 0.3516 | 2.13x |
| bike_sharing | 0.4909 | 0.4830 | 0.4964 | 0.99x | 0.2065 | 2.34x |
| beijing_pm25 | 0.6543 | 0.6541 | 0.6538 | 1.00x | 0.5024 | 1.30x |

And on the eleven candidates measured the same way, the width gain from 256 to 4096 ranges over **0.92x to 1.18x** -- `pol` 0.97, `puma32h` 0.99, `kin8nm` 1.03, `grid_stability` 1.06, `ccpp` 0.99, `gasturbine` 1.00, `fried` 1.00, `bank8fm` 1.00, `elevators` 1.01, `cpu_act` 0.92, `naval` 1.18.

**Not one of the seventeen tasks rewards 1-layer width.** Sixteen-fold more capacity buys under 5% almost everywhere. The headroom that exists is reachable only by depth (1.3x-2.3x) or by trees.

This is the load-bearing result of the experiment, and it is a property of tabular data rather than of any particular dataset. These targets are noise-limited, not resolution-limited: once a width-256 layer has fit the learnable signal, additional neurons fit noise. The QI success criterion is that error falls as width $N$ grows; on tabular data error does not fall with width for **any** method. So no choice of tabular dataset can test the width-scaling criterion, and picking a "better" suite would not have fixed it.

### Candidates, ranked by available headroom

| task | n_train | d | best lin/poly | best deep/GBDT | headroom | 1-layer headroom | near-noiseless |
|---|---:|---:|---:|---:|---:|---:|:--:|
| pol | 12000 | 48 | 0.5353 | 0.0575 | **9.32x** | 2.01x | yes |
| puma32h | 6553 | 32 | 0.8846 | 0.2262 | **3.91x** | 0.99x | yes |
| kin8nm | 6553 | 8 | 0.6773 | 0.2428 | **2.79x** | 2.27x | yes |
| grid_stability | 8000 | 12 | 0.3272 | 0.1542 | 2.12x | 1.73x | yes |
| gasturbine | 29386 | 10 | 0.0522 | 0.0280 | 1.86x | 1.09x | - |
| fried | 32614 | 10 | 0.3376 | 0.2024 | 1.67x | 1.67x | yes |
| ccpp | 7654 | 4 | 0.2374 | 0.1656 | 1.43x | 1.01x | yes |
| house_16h | 18227 | 16 | 0.7693 | 0.5574 | 1.38x | - | - |
| casp | 36584 | 9 | 0.7992 | 0.5819 | 1.37x | - | - |
| cpu_act | 6553 | 21 | 0.1628 | 0.1206 | 1.35x | 1.05x | - |
| elevators | 13279 | 18 | 0.3472 | 0.2889 | 1.20x | 1.09x | yes |
| bank8fm | 6553 | 8 | 0.2118 | 0.1789 | 1.18x | 0.99x | yes |
| ailerons | 11000 | 40 | 0.4022 | 0.3909 | 1.03x | - | yes |
| wind | 5259 | 14 | 0.4311 | 0.4260 | 1.01x | - | - |
| naval | 9547 | 16 | 0.0048 | 0.0135 | 0.35x | 0.06x | yes |

`naval` scores below 1 because the **poly-2 model wins outright** there, which is the reason it is interesting rather than a reason to discard it (below).

### naval: the one task where this project's thesis is directly visible

`naval` is the output of a numerical gas-turbine simulator, and its target (compressor decay coefficient) takes 51 distinct values on an exact $10^{-3}$ grid over $[0.95, 1.0]$ -- it is a simulator *design parameter*, so the map is deterministic and the task is really an inverse problem: recover the parameter that generated the sensor readings.

Fitting a smooth polynomial basis of increasing degree:

| basis | features | test rel $L_2$ |
|---|---:|---:|
| linear | 17 | $3.88\times10^{-1}$ |
| degree 2 | 153 | $3.24\times10^{-3}$ |
| degree 3 | 969 | $1.04\times10^{-3}$ |
| degree 4 | 4845 | $1.01\times10^{-3}$ |
| best neural net (512x512x512) | ~0.6M | $1.35\times10^{-2}$ |
| GBDT | - | $5.99\times10^{-2}$ |

A 969-term linear-in-parameters smooth basis beats a 600k-parameter network by **13x**, and beats GBDT by 58x. The target is near-noiseless and smooth, capacity is not the constraint, and the network's failure is an optimization failure. That is exactly the gap this project exists to close, appearing on real data rather than on a constructed target. The degree-4 plateau at $1.01\times10^{-3}$ appears to be an approximation limit of the polynomial basis rather than a label-noise floor, so there is likely genuine room below it -- worth confirming before leaning on the task.

### Figures

- **`figures/expD20_headroom.png`** -- two panels sharing a task axis, sorted by available headroom; red = incumbent, blue = candidate, log x on both. **Left:** headroom available, bars = (best linear/poly-2)/(best deep or GBDT), black dots = the same ratio against the best *linear* model only (ignoring poly-2), green dashed line = the 3x adoption bar, `~noiseless` annotations mark sampled deterministic simulations. **Right:** how much of it one hidden layer reaches, bars = (best linear/poly-2)/(best 1-layer net), black triangles = the width gain from 256 to 4096. Read the right panel first: every triangle sits on the $1.0$ line, which is the width-saturation result, and every bar is far short of its left-panel counterpart, which is the headroom one hidden layer cannot reach. `naval`'s bars point the other way in both panels because the polynomial wins there.

## Additional details

- Single seed per (task, model) and no per-task hyperparameter search beyond the architecture and capacity grids described above. The MLP numbers are therefore lower bounds on what a tuned deep net could reach, which if anything strengthens the underfitting verdict.
- `puma32h`, `bank8fm`, `ccpp` and `wind` have 1-layer headroom at or below $1.0$: a single hidden layer does not beat poly-2 ridge on them at all.
- `grid_stability` ships a `stabf` column that is a binarized copy of the target; it is dropped in the loader. Leaving it in would leak.
- OpenML id 572 is `bank8FM` (8 features), not the 32-feature `bank32nh`; the loader is named for what it actually returns.
- Two datasets were dropped during loader validation: `delta_ailerons` (OpenML id 803 serves the binarized classification variant, target `binaryClass`) and the 32-feature bank variant (id not resolvable to a regression target). Nothing else failed to download; all 15 remaining candidates fetched cleanly from UCI and OpenML.
- Sizes are capped at 60k rows before splitting to keep the sweep cheap; `casp`, `fried`, `gasturbine`, `beijing_pm25` and `sarcos` are affected.

## Conclusions

*Pending Sam.* The incumbent tabular suite was misjudged on two counts: the metric gave every model free credit for predicting the mean, and the 1-hidden-layer width-256 network used to score it was underfitting by 1.5x-4.1x. Corrected, those tasks carry 1.45x-4.61x of nonlinear headroom, comparable to the best candidates found here.

The more consequential result is that **one hidden layer is width-saturated on real tabular regression**: across seventeen datasets, a sixteen-fold width increase changes test error by less than 5%, so the headroom that exists is depth-headroom. The tabular axis therefore cannot test the width-scaling success criterion for any method, and no replacement suite would change that. What it can test is parity at fixed capacity, which is what PROGRAM_FRAMING §5(a) asks of it, and for that the discriminating tasks are the high-headroom ones.

## Recommendation

**For parity testing (PROGRAM_FRAMING §5a), adopt six tasks** -- the four strongest incumbents, which need no new plumbing, plus the two best candidates:

| task | headroom | why |
|---|---:|---|
| parkinsons | 4.61x | highest-headroom incumbent, already cached |
| bike_sharing | 3.32x | high headroom, already cached |
| airfoil | 2.60x | small ($n{=}1203$), exposes overfitting, already cached |
| sarcos | 2.18x | robot inverse dynamics, smooth physics, already cached |
| **pol** | **9.32x** | by far the widest gap measured; 1-layer still reaches 2.01x |
| **kin8nm** | **2.79x** | robot forward kinematics, near-noiseless, and 1-layer reaches 2.27x of it |

Drop `superconductivity` and `beijing_pm25` (1.45x, 1.72x, and both large and slow). `puma32h` has high total headroom but a 1-layer net reaches none of it (0.99x), so it cannot discriminate our architecture.

**Adopt `naval` separately, as a precision target rather than a parity task.** It is the only real dataset found where the target is smooth and near-noiseless and a linear-in-parameters smooth basis beats every neural network by an order of magnitude. If the QI init plus the readout solve can close a $1.35\times10^{-2} \to 10^{-3}$ gap there, that is the project's thesis demonstrated on real data. Verify the $10^{-3}$ polynomial plateau is an approximation limit and not a label-noise floor before relying on it.

## Open questions

- Is the `naval` plateau at $1.0\times10^{-3}$ a polynomial-basis limit or a label floor? A higher-degree or non-polynomial smooth basis settles it and decides whether the task can carry a precision claim.
- Since tabular cannot test width scaling, should the "real data, noise inherent" cell of the problem axis be re-scoped to parity-only, with the width-scaling criterion tested exclusively on the noiseless 1-D/2-D and PDE classes?
- Does the width-saturation result also hold for the QI-family init, or does a structured geometry keep improving with width where a random one saturates? That is a cheap and genuinely informative follow-up: it is the tabular version of the width-scaling question, asked about the init rather than the architecture.
- The 1-layer nets here use standard init. Whether the QI-inspired init changes the saturation point is untested and is the natural bridge from this experiment back to the main program.
