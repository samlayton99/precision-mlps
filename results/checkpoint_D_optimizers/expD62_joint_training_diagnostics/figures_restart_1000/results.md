# Joint-training diagnostics: completed sweep

All 24 cases reached 50,000 GD updates: widths 64, 196, 384, 768; targets x², sin(4πx), and 1/(1+100x²); learning rates 0.001 and 0.0001. Six matching width-196 prefixes were reused; 18 new runs were completed.

The measured parameter norm remains modest while the theorem ceiling exceeds the display cutoff of 1,000 between total steps 1,408 and 6,800. All four force alternatives in Appendix F.4 were evaluated; the generic allowance was smallest at every evaluated ceiling step in these cases. Independent direct-B calculations matched all 24 ceiling trajectories at their cutoff, and all observed parameter norms before the cutoff lay below their ceilings.

## What happens at 1k

The ceiling is initialized once at the actual parameter norm at step 1,000. It is then advanced at every individual GD update. It is never periodically reset. The nonlinear-residual allowance E0 is fixed at its maximum recorded value over 1k–50k, as in the PDF's uniform-residual hypothesis. It multiplies the force allowance; it is not an added parameter jump.

The growth allowance can rise rapidly because it depends on the ceiling itself, with a cubic dependence in the generic term. This can make the bound loose while training remains stable. The log time axis compresses the early rise.

As a diagnostic, replacing fixed E0 with the interpolated current residual for width-196 x² at learning rate 0.001 shifts the B>1000 crossing from step 1,642 to 1,667. Thus the fixed residual allowance is not the principal cause of the rapid rise in that example. This alternative is a time-varying residual-envelope evaluation, distinguished from the PDF's fixed-E0 formulation.

Measurements are saved at early checkpoints and every 500 updates thereafter. Chi and R are interpolated for the per-update ceiling recurrence. This is empirical evaluation, not certification of upper bounds between checkpoints. Only chi6 is plotted; chi4, chi10, and chi14 are also used in the ceiling calculation.

[Figure](figures/joint_training_diagnostics.png) · [PDF](figures/joint_training_diagnostics.pdf)
