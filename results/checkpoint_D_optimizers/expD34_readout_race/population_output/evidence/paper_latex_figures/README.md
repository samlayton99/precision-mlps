# Publication figures: observations and a matched mechanism example

The main section uses two two-panel figures at 5.5-inch width, exported as
vector PDFs, editable SVGs, and PNG previews. No training was performed.
The saved-data selections and raw-error reductions from the spectrum note
are preserved. The opening scale plot contains RMS levels only.

`joint_acquisition_rms` compares the five-million-update width-512 joint and
fixed-geometry experiments. Joint curves use five paired seeds. The supplied
geometry has relative bandwidth 1/4. Fixed-Adam bands show within-bin
extrema; joint bands also span seeds. These are optimization observations.

`tracking_and_reinforcement` now uses one smooth-step checkpoint and a common
20k-to-120k update axis: width 705, seed 30, GD rate 0.002. Left: full-parameter
norms of effective fine and tracking gradients on GD. Right: signed integrated
feedback and relaxation on the paired effective-flow continuation, plus its
measured log-force change. The maximum sampled tracking/fine norm ratio is
0.001963. The maximum relative force difference between GD and effective flow
is 0.000795; the integrated identity defect is below $3.16\times10^{-6}$.
These comparisons support the reduction empirically, not by interval enclosure.

`tracking_transition` is the earlier width-177, five-seed mixed-sine slope-block
example, now a separate appendix figure. It is not spliced into the matched
mechanism pair. Every plotted coarse solve is resolved.

The source is
[population_narrative_figures.py](../../../../../../experiments/expD34_readout_race/population_narrative_figures.py).
Reproduce with its existing Modal entrypoint and a fresh `--output` directory;
`D34_SPECTRUM_ROOT` selects the sibling spectrum checkout. Input hashes,
selected recipes, endpoints, and checks are in [facts.json](facts.json).
Run `ap-f06MXBIhf2r7Z9ahTnZgx5` used Modal CPU, a 4 GiB hard memory limit,
and 339.03 MiB measured peak. Numerical arrays were processed remotely.
All three previews were inspected at their intended aspect ratios.
