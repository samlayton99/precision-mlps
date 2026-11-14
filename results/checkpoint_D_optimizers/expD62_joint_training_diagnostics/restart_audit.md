# GD restart audit

Checked the original Appendix F.1/F.6 definitions against all 24 saved D62 GD runs at steps 0, 1,000, and 30,000. Recomputed each starting energy directly from the parameter snapshots; all 72 values match the recorded energy. An independent physical-B transcription matches all four force alternatives at each restart.

At restart t, B0 = sqrt(sum_j(gamma_j(t)^2 + beta_j(t)^2 + w_j(t)^2)). This is the total Euclidean norm of slopes, hidden biases, and readout weights, excluding output bias. There is no width average in B0. The resulting RMS-slope ceiling is B/sqrt(W). Each restart starts a new recurrence; later B values are not reset to measured norms.

Initialization is not separately inserted into a 30k restart. The actual 30k parameters determine B0. They can retain a norm close to initialization even after the affine-fitting transient has decayed. Similar total norm does not imply every individual parameter is unchanged.

Across these 24 runs, the change in restart norm from step zero ranges from -0.0201% to +0.0986% at 1k, and from -0.0051% to +13.1497% at 30k. “Roughly similar magnitude” is supported; “identical” is not.

For width 196 and learning rate 0.001:

| Function | Restart | B0 | E0 | R at restart | Additional updates until B > 1000 |
|---|---:|---:|---:|---:|---:|
| runge_100 | 0 | 11.415607 | 0.844711 | 3.96166 | 476 |
| runge_100 | 1000 | 11.420927 | 0.797488 | 0.000406446 | 513 |
| runge_100 | 30000 | 11.659301 | 0.672704 | 8.36684e-06 | 569 |
| sine_4pi | 0 | 11.415607 | 0.980879 | 1.57544 | 412 |
| sine_4pi | 1000 | 11.415595 | 0.980715 | 2.40002e-05 | 420 |
| sine_4pi | 30000 | 11.466019 | 0.971361 | 5.14898e-06 | 417 |
| square | 0 | 11.415607 | 0.708575 | 4.97185 | 564 |
| square | 1000 | 11.418841 | 0.637839 | 0.000620505 | 642 |
| square | 30000 | 11.790881 | 0.0593275 | 7.43761e-06 | 6272 |

For sine, the restart tracking norm falls from 1.575 at step zero to 2.40e-5 at 1k and 5.15e-6 at 30k. Nevertheless, B0, nonaffine residual, and concentration remain close enough that the ceiling reaches the arbitrary plotting cap after 412, 420, and 417 additional updates. Thus late failure is not explained just by an early affine transient. For x², the residual decreases substantially, extending the duration from 642 updates after a 1k restart to 6,272 updates after a 30k restart. The cap is a plotting cutoff, not a theorem threshold.

Every term added in F.6 is nonnegative. The recurrence does not contract below a special threshold. Small B can make its positive growth slow enough to remain useful over a chosen horizon; a larger B can produce rapid positive feedback. All 72 evaluated recurrences select the generic force allowance, proportional to E0*sqrt(chi6)*B^3/W, throughout the computed portion. The starting norm alone does not determine the outcome: residual, concentration, tracking, learning rate, width, and target loading also enter.

Paired width-705 P5 GD reconstruction (same training configuration, different initialization):

| Initialization | Norm at 0 | Norm at 1k | Norm at 30k |
|---|---:|---:|---:|
| Declared smaller, width-scaled reconstruction | 2.412637 | 2.410093 | 2.410092 |
| PyTorch default | 21.906635 | 21.906390 | 21.943308 |

This comparison directly shows persistent initialization-scale differences in the saved GD runs. The PDF does not specify the initialization used for its original appendix experiments. The smaller family is our declared reconstruction choice.

Starting at 1k is mathematically allowed if the theorem premises hold over the analyzed continuation; it is not intrinsically a post-transient guarantee. A 30k restart uses its own current parameters and suffix allowances. E0 is the largest recorded nonaffine residual from that restart to 50k. Structural inputs are interpolated between checkpoints; these evaluations are retrospective, not certified every-update enclosures.
