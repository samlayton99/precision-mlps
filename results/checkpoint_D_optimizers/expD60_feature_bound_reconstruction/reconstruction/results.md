# Appendix reconstruction: useful refined bounds recovered

Completed four width-705 native-GD runs to 125k updates, using learning rate 0.002 and 2,048 midpoint training points. The analyzed continuation is 25k–125k. Three targets use the declared smaller, width-scaled initialization; P5 also has a PyTorch-default control.

A bound is useful only while its RMS-scale ceiling is below lambda=0.25 and its relative training-error floor is above 1%. Durations below count additional updates after 25k.

| Target / initialization | Generic useful steps | Projected useful steps | Refined useful steps | Final refined error floor | Final refined scale ceiling |
|---|---:|---:|---:|---:|---:|
| legendre_5 / glorot | 16,511 | 93,027 | 100,000 | 99.68% | 0.000370233 |
| localized_oscillation / glorot | 16,511 | 93,027 | 100,000 | 99.73% | 0.000355195 |
| square / glorot | 22,942 | 100,000 | 100,000 | 55.18% | 0.000788369 |
| legendre_5 / pytorch_default | 0 | 0 | 0 | 0.00% | Exceeded reference |

All three smaller-initialization cases retain useful refined bounds for the complete 100k-step continuation. None retains a useful generic bound throughout. Projected bounds cover the full interval for x² and about 93k additional updates for the other two functions. The default-initialization P5 control has zero useful duration because its error floor is zero at the starting checkpoint.

The generic median useful duration is 16.511k, consistent with the appendix’s reported 16.5k median. This reconstruction’s projected median is 93.027k and its refined median is 100k. These are three selected targets and one seed, not the original 23-target/two-seed count.

## Initialization comparison

| P5 initialization | Parameter norm at 25k | Parameter energy at 25k | RMS slope at 25k |
|---|---:|---:|---:|
| glorot | 2.4101 | 5.8085 | 0.053077 |
| pytorch_default | 21.9337 | 481.0863 | 0.584697 |

The paired P5 comparison changes the initialization family while holding width, target, sampling, GD rate, and duration fixed. The smaller initialization yields an informative refined bound; the default control does not. This demonstrates a substantial dependence on the starting parameter regime. It does not identify the original paper’s unspecified initialization.

## Verification and limitations

- Independent physical-B and normalized-q recurrences agree on all 12 case/family evaluations.
- The four analytical force bounds exceed the measured nonlinear-fitting gradient norm at all 1,036 saved states.
- There are no saved-state parameter-envelope violations before each evaluated ceiling crosses its reference.
- Keeping every other regular checkpoint changes no useful-duration counts; the largest relative change in an available final scale ceiling is 6.36e-5.
- Structural inputs are interpolated saved records, matching the paper’s empirical audit approach; this is not an outward-rounded, every-update certificate.
- The PDF does not specify initialization or the complete target/seed list. The chosen initialization is explicitly documented in the experiment README. No exact reproduction of the 32/46 count is claimed.

[Figure](appendix_reconstruction.png) · [PDF](appendix_reconstruction.pdf) · [Per-case results](summary.csv) · [Validation](validation.json)
