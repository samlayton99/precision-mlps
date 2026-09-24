# Experimental protocol for Section 3.4

The experiment separates two questions. Frozen readouts test the slope-dependent GD theorem without changing its assumptions. Joint training tests whether Adam and GD acquire a geometry that permits accurate output within the measured budget. The same target, hidden-width budget, sample grids, arithmetic precision, and output-error definition connect the two experiments; the frozen-GD theorem is not applied to joint training or Adam.

## Target, geometry, and error

The primary target on $[-1,1]$ is

$$
f(x)=\frac{\sin(2\pi x)+\frac12\sin(6\pi x)+\frac14\sin(14\pi x)}{\sqrt{21/32}}.
$$

Its three components carry fractions $16/21$, $4/21$, and $1/21$ of the squared target norm. A second experiment uses $x^2$, normalized by its training-grid RMS, to check the behavior on the arithmetic primitive. Neither target is sampled from a small-slope network.

The total hidden width is $W=512$, including halo features: $N=467$, $R=22$, $h=2/467$, and centers $x_j=-1+jh$ for $j=-22,\ldots,489$. Frozen uniform dictionaries share these centers and differ only in their common slope $\gamma=\lambda/h$. Every model also has an output bias. Joint models retain the same total width but train their input slopes, hidden intercepts, readout, and output bias.

Training, validation, and resolution checks use 2,048, 4,096, and 8,192 equally spaced midpoints, respectively. The validation grid chooses optimizer recipes. The denser grid checks sensitivity to sampling; it is not an untouched test set. The error on any grid is the Euclidean norm of the output residual divided by the target norm. All computations use FP64 and full-batch half-MSE. The main figure plots raw output error, without an EMA or a tolerance-based success definition.

## Initialization, schedules, and selection

Frozen readouts, output biases, and Adam moments start at zero. For joint training, five paired seeds initialize every input slope and intercept uniformly in $[-\sqrt{6/513},\sqrt{6/513}]$. The readout uses an independent draw from the same interval and the output bias is zero. These parameters are shared across optimizer recipes, schedules, and the two targets.

The initial joint experiment runs two million updates. Adam tests initial learning rates $0.0002,0.002,0.02,0.05,0.1$; GD tests $0.0002,0.002,0.02,0.05,0.1,0.2,0.5,1$. Each rate is run with a constant schedule and with $\eta_n=\eta_0[1+\cos(\pi n/H)]/2$ for $0\le n<H$. Thus cosine decay spans the entire run. Adam uses $(\beta_1,\beta_2,\epsilon)=(0.9,0.999,10^{-8})$. A winning boundary rate receives one factor-three outward expansion. Divergent recipes remain in the record.

One recipe is selected globally per optimizer by median final validation error across all five seeds, requiring finite endpoints in every seed. The figure does not splice recipes, choose a separate recipe for each seed, or replace the final checkpoint with an earlier minimum. The best constant-rate recipe is also examined independently of the selected schedule. A greater-than-10% error reduction or 99th-percentile slope increase over the final fifth triggers a five-million-update follow-up with the best two rates per optimizer and schedule. Error reduction compares trailing windows of length $H/100$ ending at $0.8H$ and $H$. This trigger allocates computation; it does not establish convergence. A longer cosine experiment starts afresh with its new horizon.

Neither optimizer met that constant-run trigger on the primary target at two million updates. We added the five-million-update primary comparison as a separate duration check because the selected Adam cosine trajectory still improved by 20.60% over its final fifth. This extension advances the two strongest rates per schedule: Adam constant $0.02,0.002$, Adam cosine $0.05,0.02$, and GD $0.2,0.5$ for both schedules. It preserves the original initialization and selection rule, and retains the two-million-update results separately.

Quadratic Adam did meet the continuation trigger: the best constant recipe, $\eta=0.0002$, improved its trailing-window error by 30.82%. That rate also lay at the lower search boundary, prompting one expansion to $0.0002/3$ under both schedules before selecting continuation candidates.

The supporting quadratic extension advances the best rate per schedule across all five seeds, rather than the two rates used for the primary extension. This budget adjustment follows measured small-batch throughput. Its five-million-update joint curves are compared over the same horizon; the quadratic uniform readouts were executed for two million updates and are reported separately.

Every raw joint-training error and RMS slope is saved. Parameters are saved every 10,000 updates. The plotted slope statistics are the RMS and 99th percentile of $h|a_j|$, using the uniform reference spacing $h$ even though learned centers are not uniform. A uniform reference line does not assert that the same statistic is a necessary threshold for heterogeneous features. Plot reduction retains the full horizon, exact endpoints, and within-bin extrema so brief instability remains visible.

## Frozen-feature checks

The theorem comparison uses $\lambda\in\{1/32,1/16,1/8,1/4\}$ and executes actual GD at $\eta=1/(2\mu_1)$. Target projections come from the actual dictionary. The predicted lower curve uses the theorem's upper rate bounds; actual small eigenvalues determine numerical resolution and an independently identified exact-spectrum check, not the predicted decay rates. The evaluated lower curve conservatively omits modes whose computed relative eigenvalue is at most $10^{-18}$, together with the residual outside the computed left-singular-vector span. These nonnegative terms are retained in the exact-spectrum diagnostic. Predictions are evaluated before reading the training trajectory. Two center-quadrature resolutions check numerical sensitivity, and the saved GD error is compared with the bound at every update. These are checked floating-point evaluations of the analytic theorem, not directed-rounding numerical certificates.

Frozen Adam tests $\lambda\in\{1/32,1/16,3/32,1/8,1/4,1/2,1\}$ with rates $10^{-5},10^{-4},10^{-3},0.002,0.01,0.1$ and both schedules. Boundary-rate expansion is applied consistently across dictionaries. The main uniform reference is predeclared as $\lambda=1/4$. Its trajectory uses one complete recipe selected by final validation error.

Boundary winners at both ends of the initial frozen-Adam grid led to one shared expansion, adding $10^{-5}/3$ and $0.3$. All seven uniform dictionaries and all learned-feature assays receive those rates. The five-million-update uniform comparison uses the complete expanded grid.

A direct SVD readout fit independently tests attainable accuracy. The displayed reference cutoff is $10^{-12}$ relative to the largest singular value; cutoffs $10^{-14}$ and $10^{-16}$ provide a sensitivity check. Errors are recomputed from the fitted coefficients on all three grids, and coefficient norms are retained. A truncated-SVD residual is not called a capacity floor, and the best cutoff is not silently selected at each slope.

## Learned-feature and Adam diagnostics

Readout-restart experiments freeze the selected joint model's features at initialization, 200,000 updates, 600,000 updates, and the final checkpoint. They reset the readout and its moments. Separate final-checkpoint interventions multiply both slopes and hidden intercepts by four or sixteen, preserving each feature's center. These interventions test slope at the learned center placement; they do not reproduce the uniform-center theorem assumptions.

Each restart receives two million readout updates. This common diagnostic budget compares access through different features; it is additional computation after joint training, not an end-to-end optimizer comparison.

For each dictionary, spectral diagnostics report target and actual residual energy below relative eigenvalue cutoffs. The Adam diagnostic also forms the endpoint second-moment metric $J\operatorname{diag}[(\sqrt{\widehat v}+\epsilon)^{-1}]J^T$. Its target and residual projections use the same target-norm denominator as the raw kernel. This describes which directions remain weak after adaptive coordinate scaling. It is not an Adam convergence formula: momentum and the changing second moments remain part of the actual algorithm. Omitting the scalar learning rate leaves the relative spectrum defined even at the end of a cosine schedule.

The actual joint-training endpoints receive a separate check. Their readout Jacobian is $[\tanh(a_jx+b_j),\mathbf1]$; their full Jacobian additionally includes the columns $xc_j\operatorname{sech}^2(a_jx+b_j)$ and $c_j\operatorname{sech}^2(a_jx+b_j)$. Both are divided by $\sqrt m$. Adaptive metrics for these Jacobians use the saved second moments of the actual joint Adam run, restricted to the corresponding parameter blocks. This avoids substituting the moments of a restarted frozen readout for those of joint training.

The results, selected recipes, continuation decisions, and figure provenance are recorded alongside the completed runs. An improvement with longer training is reported as improvement; the experiment does not infer permanent optimization failure from a finite budget.

## Completed two-million-update adaptive-metric control

At the selected cosine Adam endpoints, over 99.998% of residual energy lies below relative eigenvalue $10^{-6}$ in the full Jacobian metric with the actual Adam scaling. Repeating the diagnostic on the best constant-rate recipe, $\eta=0.02$, gives a median of 98.763%, with seed range 47.671–99.309%. Thus weak local directions also matter without cosine cooldown, although the constant-run concentration is not uniform across seeds. These fractions divide by each endpoint's residual energy; the underlying cumulative curves divide by target energy.

For the constant runs, the median adaptive-trace-weighted contribution of $\epsilon$ to $\sqrt{\widehat v}+\epsilon$ is $7.71\times10^{-6}$ for the full Jacobian and $1.09\times10^{-6}$ for its readout block. The corresponding cosine values are about 24% and 23%. Counting coordinates with $\sqrt{\widehat v}\le\epsilon$ would overstate this effect, because many have negligible Jacobian columns. The constant-rate check therefore supports the weak-direction interpretation without relying on an epsilon-dominated endpoint metric. It does not turn that diagnostic into an Adam convergence theorem.

The completed [frozen-feature validation](section34_frozen_validation.md) reports five-million-update theorem tightness and the bandwidth sweep with Adam trajectories.

The [learned-geometry controls](section34_geometry_controls.md) compare readout restarts, direct fits, and isolated changes to slopes or centers. They also explain why a high slope percentile does not imply acquisition of a uniform high-precision dictionary.
