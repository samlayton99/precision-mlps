# expF04 two-layer uniform-center comparison

## Question

Does the scaled ridge-bundle initialization perform better when each bundle's centers are equally spaced across its projected input range, rather than sampled from observed projections?

The existing `all20_2layers` experiment uses projection-sampled centers. This follow-up changes center placement only.

## Experimental control

The new experiment lives in `experiments/expF04_qi_init_real_data/all20_2layers_uniform/` and writes results to the matching folder under `results/checkpoint_F_applications/expF04_qi_init_real_data/`.

It retains the existing protocol:

- 16 cached real datasets
- widths $N\in\{256,512\}$
- two hidden layers of width $N$
- activations `tanh` and `gelu`
- initialization regimes `baseline`, `qi1`, and `qi2`
- seeds $\{0,1,2\}$
- Adam with learning rate $10^{-3}$, batch size 128, and 50 epochs
- best evaluation loss over training as the primary metric

For `qi1`, the first hidden layer receives ridge-bundle initialization. For `qi2`, both hidden layers receive it. Each bundle contains $P=\lfloor\sqrt N\rfloor$ neurons.

## Scientific change

For every initialized layer and ridge direction $u_m$, let $t_i=x_i^\top u_m$ denote the observed projections and let

$$
A_m=\operatorname{quantile}_{0.999}(|t_i|).
$$

The bundle scale remains

$$
\gamma_m=\frac{\lambda P}{2A_m},
\qquad \lambda=0.25.
$$

The existing sampled-center experiment draws each center from the empirical set $\{t_i\}$. The new experiment instead places the $P$ centers at equally spaced locations in $[-A_m,A_m]$. Directions, scales, architecture, optimizer, data, and evaluation remain unchanged.

## Isolation

The runner has its own output directory, filenames, plots, and W&B tags. The existing sampled-center runner and its results remain untouched.

The separate runner intentionally duplicates the established experiment script. For a research experiment, this keeps the comparison auditable and prevents a command-line default from changing the meaning of an existing result.

## Verification

A focused test will import the new runner, replace the ridge initializer with a recording function, and verify:

1. `baseline` invokes no ridge initialization.
2. `qi1` initializes only `fc1` with `uniform_centers=True`.
3. `qi2` initializes both `fc1` and `fc2` with `uniform_centers=True`.

An import or smoke check will run before the full experiment.

## Comparison

For each width, the analysis will report:

- the geometric mean of per-dataset best-evaluation-loss ratios relative to `tanh-baseline`;
- the paired geometric mean of uniform-center loss divided by sampled-center loss for each activation and initialization regime;
- the number of datasets won by each configuration;
- per-task cases where the center choice materially changes the result.

Because center sampling consumes random numbers in the existing runner, later minibatch permutations are not identical between the two QI variants. Results are therefore compared over the same three seed labels as aggregate experimental outcomes, not as bit-for-bit paired training trajectories.
