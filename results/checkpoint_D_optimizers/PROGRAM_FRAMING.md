# The program framing: two mechanisms, four axes, the 2x2, and the moonshot

**Status: program-level context, dictated by Sam 2026-08-12 and written out in full here. This is the document that should have existed before expD07. Read it with `D07-D15_SALVAGE.md` (what the campaign produced) and before designing any new optimizer or initialization experiment. Where `docs/ORIENTATION.md` frames things differently, this page is the intent.**

## 1. The two mechanisms

Training a network to high precision on a regression task decomposes into exactly two sub-problems. The split is settled for one hidden layer; what happens with multiple hidden layers and deep models is genuinely less certain, and in particular we do not know what stacking layers of the high-dimensional QI-inspired init does. Treat the depth story as open, not as an extension of the 1-layer story.

**The geometry.** Where approximation capacity sits: centers, bandwidths, ridge directions in 2-D and up, and (with the uncertainty above) whatever the layers below the readout compute in deep models. A geometry is *good* when a linear solve on top of it reaches the precision floor. The theory says a good geometry exists at every width ($\lambda^*\approx0.25$, uniform coverage, shared bandwidth: checkpoint C), placement decides everything (expB01: centers, not samples; expC04: off-grid never recovers), and it is a structured, measure-zero set that per-coordinate gradient pressure has no bias toward (expC synthesis §4).

**The readout solve.** The parameters the network is *exactly linear* in. On a fixed geometry this is a convex least-squares problem with a closed-form answer, and it is where the last ten digits live. Adam solves it reasonably well but cannot fundamentally finish it: even on a frozen correct geometry it bottoms out around $10^{-3}$ to $10^{-5}$ depending on setup (expD01; better than without the good geometry), and the trained weights visibly approach the lstsq solution's shape (the sampled-derivative structure, expA06) without reaching it, because finishing requires resolving singular directions at $(\sigma_1/\sigma_j)^2$ steps each and the spectrum is gapless. It gets close; it cannot get there. Hence the need to lobotomize Adam and insert a dynamic lstsq solver.

**Adam does decently at both and great at neither.** It finds genuinely useful geometries from standard init (a trained-then-solved net beats an init-then-solved net: expD02's training slices; expD14 iteration 2's scorecard), it preserves a good geometry it is handed (expD02 win #2), and it does okay on the readout as above. But it cannot finish the readout, and it cannot *place* the geometry on the structured set (expC §4, expD02: covering-but-random inits decay with width while the grid holds the floor).

## 2. The working hypothesis from the unrecorded generalization experiments

The interactive 1-D generalization work (expG01's explorer; never written up because it was interactive) produced the intuition the whole geometry side runs on, so it is stated here explicitly, **as a hypothesis from 1-D, possibly wrong**:

Put centers where there is data and make them sparse where there is not. Locally follow the $\lambda=0.25$ rule, and make center density track data density. When this was followed, the readout solve **generalized well in low-data regions and stayed precise in high-data regions**.

That is an information-theory statement: capacity allocation should match information density. It is the deep hypothesis behind the QI-inspired inits, behind the fear about high-dimensional geometry, and behind the moonshot in §6. Everything downstream that says "the right geometry over the right data" is leaning on this 1-D observation.

## 3. The 2x2: inits crossed with optimizers, and what dominates

There are two inits (standard/Xavier vs the QI family: exact 1-D, Radon 2-D, QI-inspired high-d from expF04) and two optimizers (standard Adam vs the lstsq-lobotomized optimizer, call it the **QI optimizer**). That is a 2x2 grid, and the program's state is a claim about which cells dominate where.

| | standard init | QI init |
|---|---|---|
| **Adam** | the baseline everywhere | preserves the geometry, cannot cash it out |
| **QI optimizer** | unclear whether it dominates Adam here | dominates on 1-D/2-D; the win condition cell for high-d |

What the evidence says so far, axis by axis:

- **Init axis.** QI init dominates standard init on 1-D, 2-D, and the tabular regression tasks (also apparently for two layers, but that is unexplored and must not get conflated into the 1-layer claim). Whether it holds on real-world tasks and large models is unknown, and there are two specific reasons to doubt it: the information-theory lens (§2: high-d coverage cannot be uniform, it has to follow the data) and the killed-gradient lens (a near-correct geometry may be a gradient dead zone).
- **Optimizer axis.** The QI optimizer dominates on 1-D/2-D with good QI init. It is not established that it dominates on standard init. It appears to do worse on tabular as-is, and it is very unlikely to dominate on real-world tasks as-is.

Success means solving both the right geometry and the right readout. The two levers that attack this are good inits and optimizers that can do least squares, and it is very clear why the combination works in the 1-D/2-D case. The ideal world is that our init and our optimizer dominate their standard counterparts on everything, including the information-compression moonshot cases, and reach machine epsilon when the problem admits it. That ideal is also very unlikely to be true as things stand; the grid is how we find out where it breaks.

## 4. The four axes, and how the experiments conflated them

Every experiment in this program lives at a point in a four-axis space. Naming the axes is what lets us stop conflating them.

| axis | values | state of evidence |
|---|---|---|
| **initialization** | Xavier; exact QI (1-D); Radon QI (2-D); QI-inspired ridge (high-d, expF04, best variant) | 1-D and 2-D solid; expF04 promising but has no writeup and has never met the solve |
| **input/output** | $1\to1$; $2\to1$; $\mathbb{R}^n\to\mathbb{R}$; $\mathbb{R}^n\to\mathbb{R}^m$ | $\mathbb{R}^m$ untested (expected cheap: shared geometry, per-coordinate solve) |
| **problem** | noiseless toy interpolation; noisy toy; real tabular regression (noise inherent); domain task (image generation, PINN, inverse problems, ...) | noiseless toy saturated; noise floor known ($\sigma n^{-1/2}$, expB01); real tabular touched by expD07/expF04; domain tasks untouched |
| **model** | 1-layer MLP; 2-layer MLP; conv/ResNet/U-Net; regression subtask inside a non-regression model (transformer FFN) | 1-layer saturated; 2-layer touched (expD07 dl, iteration 11 depth-2) and NOT settled; discovery validated on the complex architectures (expD15) but the *solve* never run there |

The conflations, named so they are not repeated:

1. **The success criterion conflated the readout fix with the moonshot.** "Machine epsilon from a non-construction init" requires geometry *discovery*, but the optimizer we built is a readout *solver*. Judging the solver against the discovery bar made working machinery look like failure.
2. **expD14 mixed "solve the readout" and "improve the geometry" in one loop**, with throttles on top, so the geometry effects were confounded (iteration 1 had to be spent clearing one such confound: the stun is intrinsic, not a throttle artifact).
3. **A single end-task error conflates the two mechanisms.** The de-conflated scoring reads **three comparisons** off every run, built from two lstsq probes: the solve on the *pre-train* geometry (raw init) and the solve on the *post-train* geometry (readout thrown away and re-solved).
   - **Geometry score of the optimizer** = post-train lstsq vs pre-train lstsq. Did training improve the geometry, and by how much? This must be run on standard init as the control and on our init, since the init is otherwise a confound.
   - **Readout score of the optimizer** = the optimizer's own final readout error vs the post-train lstsq. How close did the optimizer get to the solution available on the geometry it actually ended with?
   - **Was the optimizer worth running at all** = pre-train lstsq vs the optimizer's final readout error. If init + one solve beats the whole training run, training subtracted value.
4. **Init and optimizer usually varied together.** Most runs changed both at once; expD16 is the first clean init $\times$ optimizer factorial, and it still scores only end error, not the probes above.

## 5. What success actually looks like

Adam works at every point of the axis space, just not to machine epsilon. So the bar is *not* "beat Adam everywhere." It is:

**(a) Parity.** On every cell where machine epsilon is unattainable or irrelevant (noise-floored, real tabular, domain tasks), the QI optimizer lands in Adam's ballpark at Adam-class cost. Ideally better; same order is a pass.

**(b) Precision where it exists.** On every cell where the initialization provides a solvable geometry (1-D QI, 2-D Radon, and, if it holds up, the expF04 init), the QI optimizer reaches the dtype floor. Adam does not, ever (expD16: no gradient pipeline gets within 5 orders of the lstsq floor even when handed the geometry).

**The concrete win condition:** parity on all the standard-init cells, plus a clear improvement over Adam on the QI-inspired-init cells. That combination is a deployable optimizer with a genuinely new capability, and it does not require the moonshot.

Three ways to win, in increasing order of ambition:

1. **A better initialization regime.** Known hazard: the QI-type inits may kill the gradient signal (the geometry is already near-optimal in a thin basin, and the solve makes the residual orthogonal to what the features express), so this axis has to be paired with (2).
2. **The readout-recognizing optimizer.** Recognize and solve the least-squares block; default to Adam otherwise. This takes care of the 1-D and 2-D init cells outright, and the hypothesis is that it does the same on the expF04 init.
3. **A geometry-uncovering optimizer.** The moonshot: §6.

## 6. The moonshot, stated properly this time

The true moonshot is that our init and our optimizer **dominate on large real-world tasks**, on both axes of the 2x2 at once. Two specific mechanisms are feared to make that unlearnable as things stand: the init kills gradients (a near-correct geometry produces almost no geometry signal), and the optimizer locks in the geometry the moment it solves the lstsq (the stun result: solving exactly makes the residual orthogonal to everything the features express). Between them, the very things that buy precision may freeze learning.

So the ideal moonshot optimizer is one that can **discover the right geometry in high-dimensional space according to the generalization/information-theory hypothesis of §2, mitigated by the optimizer** (it must not stun learning while the geometry is still wrong), **encouraged by the initialization** (which supplies coverage in the right regime), **and then locked in to solve the readout when the time is right**. A lot needs to happen to get there, and a lot of conflating happens the moment we reach for it. Back at the roots it all comes down to two things: the right geometry over the right data, and the ability to do a good lstsq solve. Inits and optimizers are our first approach to getting both.

This is not just for a diffusion model or a transformer. Inverse problems need it too: learning parameters from measured data while interpolating over the dynamics requires the geometry to move under gradient flow, so a gradient-dead init or a geometry-locking solve would break exactly the application the method is best suited for. That is why the gradient-death question (§7, experiment 2) gates more than curiosity.

**Why the optimizer defaults to Adam:** Adam is the best geometry-finder we have from standard init. The design intent was always that the solve supplements Adam's learning rather than displacing it, so that whatever geometry-finding ability exists in the system is Adam's, undamaged. The stun result is why this default is load-bearing and why under-solving mid-run is a feature, not a compromise.

**The current division, honestly stated.** Init owns geometry, the optimizer owns the readout, and the conflation/moonshot is an optimizer that *also* helps with the geometry. That third thing stays on the table precisely because it is not settled that init dominates on everything, in particular not on inverse problems or on large models doing heavy information compression.

## 7. The de-conflated test matrix

What a clean campaign over these axes looks like, in order of information per unit work:

1. **Instrument the probes everywhere.** Every run in every future experiment logs the two lstsq probes (pre-train and post-train geometry) alongside its own error, so all three §4 comparisons can be read off. This one change de-conflates the two mechanisms in all subsequent data.
2. **The gradient-death measurement.** From QI-type init vs standard init, plain Adam, across problem classes (1-D, 2-D, nonlinear 2-D inverse PINNs, expF04 tabular): does the geometry move at all, and by how much? One run per cell, no new machinery. This is the direct test of the killed-gradient fear and it gates the inverse-problem application class.
3. **The win-condition experiment.** expF04's best init $\times$ {Adam, the QI optimizer} on the real tabular tasks, with standard init as the control arm, scored by the three comparisons. Parity on end error plus a better geometry or readout score on the QI-inspired arm is the win as defined in §5.
4. **Is the high-dim init good enough on the geometry side?** The same probes answer it: compare the pre-train lstsq on the expF04 init against the post-train lstsq on what Adam finds from Xavier. If Adam's own found geometry out-floors the init, the init is not yet carrying its weight, and the information-theory suspicion is confirmed with a number.
5. **$\mathbb{R}^m$ output.** Shared geometry, per-coordinate solve. Expected cheap; run it once to remove the caveat.
6. **Only then the moonshot**, with the coverage constraint of §2 built in from the start, and always ablated against "expF04 init + QI optimizer" as the null hypothesis.

## 8. Implications, and where we are (2026-08-12)

1. **Against the corrected criterion, the campaign is much closer to a win than the old criterion made it look.** The old bar demanded the moonshot. The restated bar has two parts, and both are substantially met: parity (the latch passes the dl litmus at 1.2-2.6x Adam) and precision on the 1-D/2-D QI-init cells ($10^{-16}$/$10^{-14}$ where Adam degrades the same init to $10^{-3}$). The only unmeasured cell in the win condition is the expF04 one, and every piece it needs already exists; they have just never been composed. That is one experiment, not a research program.
2. **expD16 strengthens the split.** Even the best classical second-order finisher (SSBroyden) stalls at $2\times10^{-7}$, five orders above the lstsq floor on the geometry it was handed. The readout problem is not "use a better general optimizer"; the solve is a categorically different operation.
3. **The three-comparison scoring is the biggest structural fix and it is nearly free.** It turns "is the high-dim init good enough?" from a suspicion into a single measured number (§7.4), and it makes every future run informative about both mechanisms separately.
4. **The gradient-death test is the cheapest missing measurement with the largest downstream reach**, because it gates the inverse-problem class (§6). It was never run in fifteen experiments.
5. **The moonshot's optimistic reading is free.** "Maybe expF04 init + QI optimizer already finds the geometry" falls out of the win-condition experiment at no extra cost, so the moonshot's null hypothesis gets tested before anything new is built. The pessimistic reading (geometry discovery needs its own mechanism) is currently supported: nothing in D07-D15 improves an already-decent geometry, and any real attempt must build in the §2 coverage constraint from the start, since no current signal even sees geometry damage where there is no data.

Execution order: probes as universal instrumentation, then the gradient-death test (§7.2), then the win-condition experiment (§7.3), then $\mathbb{R}^m$, then decide whether the moonshot needs its own mechanism. Items 2 and 3 are running as expD17 (geometry motion) and expD18 (QI optimizer on expF04 init) as of this writing.

## 9. Questions to explore

**(a) Is the information-theory / data-matching generalization hypothesis (§2) true, and how do you test it in high dimensions?** The 1-D statement is testable directly (match center density to data density, score generalization split by region). The high-dimensional test is the **random-direction Radon idea**: reduce the high-D hypothesis to a bundle of 1-D hypotheses. A ridge neuron is a point $(u, t)$ in Radon space (expE01): a direction $u$ and an offset $t$ along it. For any direction $u$, the data becomes a 1-D projected density $p_u(t)$ (the distribution of $u^\top x$ over the dataset), and the target becomes 1-D information content along that direction, measured by the **directional derivative** $\partial f/\partial u$ on the projection (this is what a ridge family's readout actually encodes: the sampled-derivative law of expA06, $v_k \approx \tfrac h2 f'(c_k)$, applied per direction). So "capacity matches information density" becomes, direction by direction: **offset density along $u$ should match the projected data density $p_u(t)$, weighted by the directional-derivative energy of the target along $u$**, with the local $\lambda=0.25$ rule setting each ridge's bandwidth from its local offset spacing. The test: draw random directions, build geometries that *match* vs deliberately *mismatch* this rule on synthetic high-D targets with nonuniform data, solve the readout, and score eval error split by data-rich vs data-poor regions. If the 1-D hypothesis is true in high-D, the matched geometry generalizes in the sparse regions and stays precise in the dense ones, and the mismatched one fails in exactly the regions where its allocation disagrees with $p_u$.

**(b) Does the QI init truly dominate?** Honest status: the current high-dimensional QI-inspired init was LLM-generated and is neither rigorously tested nor theoretically backed, and expD18 already measured that its untrained geometry loses to Adam-found geometry on all 6 tabular tasks. The suspicion is that it fails outright on the models we actually care about (transformers, diffusion models). Experiments that would test dominance: the §7.4 probe comparison run on each new domain (init's pre-train probe vs Adam-from-standard's post-train probe); width scaling of the init advantage on tabular (does it grow or shrink with capacity); a transformer-FFN swap-in (initialize just the FFN blocks with the ridge init, standard elsewhere); a small diffusion/U-Net swap-in; and a version of the init rebuilt from the (a) rule (data-density-matched offsets) rather than the current uniform projection sweep, which is the theoretically-motivated candidate to replace it.

**(c) What happens with multiple layers, and what is the theory there?** We are mostly single-layer. For depth, the split itself blurs: only the final readout is exactly linear, the inner layers are curved, and "geometry" for layer 2 means placement over *learned* features whose distribution changes during training, so the §2 hypothesis has no static data density to match. There is no construction theory for composed layers, the 2-layer evidence is thin and conflated (iteration 11's depth-2 bench, expF04's 2-layer variants, never cleanly separated), and it is not even known whether the right mental model is "each layer has its own QI geometry" or "depth is doing something the 1-layer theory does not describe." This needs its own theory work before its own experiments.

**(d) Does the init kill the gradient signal, and if it does, can we overcome it?** expD17 is the direct measurement. Suppose it does: the known partial remedies are under-solving (leaves residual for the geometry gradient; measured to keep learning alive), softening the init (deliberately imperfect geometry that leaves signal), and letting Adam keep stepping the solved block. Whether any of these amounts to a real mechanism for *discovering* good geometry, rather than just not destroying the signal that exists, is the moonshot question of §6, and it stays open until the (a) hypothesis gives geometry discovery a target to aim at: you cannot optimize toward "the right geometry" without a measurable definition of it, and (a) is the candidate definition.
