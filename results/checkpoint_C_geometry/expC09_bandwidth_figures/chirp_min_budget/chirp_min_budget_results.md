# Chirp with sweep-minimum alias budgets — measured, retrospective

## TL;DR

- Each curve's minimum observed relative error sets its own alias tolerance.
- The refined predictions move right of the sampled minima. At FP64 widths 128, 192, and 256, errors at the predicted bandwidths exceed the original minima by approximately 1277, 173, and 47 times respectively.

## Question

Where does the refined rule place its bandwidth when supplied with the attainable error measured in the chirp sweep?

## Experiment design

The target remains $f(x)=\sin(8\pi(x+1)^2)$. For each original fixed-width, fixed-precision curve, set $e_{\mathrm{tol}}=\min_{\lambda\in\Lambda} E_{W,p}(\lambda)$ using its saved sweep observations. Freeze that tolerance before evaluating the new marker. The selector retains the mean local angular frequency $16\pi$, with interior spacing $h=2/N$ and $N=W-49$ for 24 halo nodes per side.

Panel (b) uses widths 96, 128, 192, and 256 at $p=53$; panel (c) fixes width 256 and varies $p=24,32,40,53$. Seven new fits measure errors at the predicted bandwidths directly, using the existing 4,801-point fit and 8,001-point evaluation grids and rounded-precision model. The original sweep remains over $[0.05,1.5]$. Only the selector's search interval expands to $[0.03,10]$ so out-of-sweep predictions are identified accurately rather than reported as search-boundary roots.

**Code & data**

- Runner: `experiments/expC09_bandwidth_figures/chirp_min_budget.py`; shared numerical routine: `run.py`; shared panel renderer: `sine24_panels.py`.
- Frozen tolerance provenance and predictions: `data/predictions.json`; comparisons: `data/comparison.json`; directly measured markers: `data/prediction_measurements.jsonl`.
- PNG: `figures/chirp_min_budget.png`.

## Results

At FP64 widths 128, 192, and 256, predicted bandwidths are approximately 0.399, 0.312, and 0.294, versus sampled minima near 0.183, 0.233, and 0.253. At width 256, the lower-precision predictions yield errors approximately 4.3, 8.9, and 17.3 times the original minima for $p=24,32,40$ respectively.

### Figures

- **Chirp two-panel PNG:** original width and precision curves with diamonds at the newly measured predicted bandwidths. The width-96 root is approximately 2.47, beyond the displayed sweep range, so its diamond is not visible.

## Additional details

All seven stored tolerances were checked against the original per-curve minima, and all seven roots satisfy the refined selector equation to numerical precision. These are retrospective inputs derived from the observed results. They do not provide an independent prediction of either the attainable accuracy or the optimal bandwidth. The chirp frequency summary remains a heuristic. No new halo or sampling sensitivity checks were performed for these seven probes; the original sweep controls remain available.

## Conclusions

Using the observed minimum as the alias budget does not align the chirp's refined predictions with its measured minima under the retained mean-frequency summary.

## Open questions

How much of this discrepancy comes from representing the chirp by one mean frequency remains untested here.
