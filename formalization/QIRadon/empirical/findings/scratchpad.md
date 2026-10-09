# Construction-rate experiments

## Before the first run — 2026-10-09

The generic verified tanh bound has exponent d+1. The sharper existing
finite-contour construction gives exponent d with fixed physical halo,
gamma=2*pi/r, lambda=4*pi/(r*n), and R=ceil(r*n/2). Its contour identity is
not formalized. The experiments will implement that construction, include all
halo neurons in width, and state this distinction in every relevant figure.

Five explicit Gaussian-mixture families permit exact canonical Radon profiles
in every dimension tested. This avoids hiding a fitted profile or another
solver. It is a restricted analytic benchmark family, not evidence for all
analytic functions. Full Gauss-Jacobi direction grids are used in d=1..4.

Additional tests: fixed-geometry readout rounding (the numerical perturbation
theorem); ReLU interpolation of a quadratic (the algebraic activation result);
and an independently bounded angular/scalar allocation compared against all
tested configurations. The last experiment must distinguish minimizing an
upper bound from minimizing actual error.

- 2026-10-09T17:20:57+00:00: Starting validation suite, declared config in config.json.

- 2026-10-09T17:21:00+00:00: Validation passed: positive normalized sphere rules and exact moments; constant profile; doubled contour quadrature. Refinement is a diagnostic, not an interval certificate.

- 2026-10-09T17:21:00+00:00: Starting width suite, declared config in config.json.

- 2026-10-09T17:21:00+00:00: Width run d=1, q=2, cells=32, W=49: relative errors [7.05e-07 3.53e-06 1.59e-06 1.56e-06 1.17e-05].

- 2026-10-09T17:21:00+00:00: Width run d=1, q=6, cells=48, W=73: relative errors [1.31e-09 6.58e-09 2.97e-09 3.11e-09 2.18e-08].

- 2026-10-09T17:21:00+00:00: Width run d=1, q=8, cells=64, W=97: relative errors [2.45e-12 1.22e-11 5.41e-12 5.57e-12 4.05e-11].

- 2026-10-09T17:21:00+00:00: Width run d=1, q=10, cells=80, W=121: relative errors [4.55e-15 2.32e-14 1.01e-14 1.08e-14 7.50e-14].

- 2026-10-09T17:21:00+00:00: Width run d=1, q=12, cells=96, W=145: relative errors [3.06e-16 9.07e-16 2.81e-16 3.69e-16 8.44e-16].

- 2026-10-09T17:21:00+00:00: Width run d=1, q=16, cells=128, W=193: relative errors [3.00e-16 6.43e-16 4.89e-16 5.90e-16 1.37e-15].

- 2026-10-09T17:21:00+00:00: Width run d=1, q=20, cells=160, W=241: relative errors [2.64e-16 8.82e-16 3.87e-16 5.98e-16 2.53e-15].

- 2026-10-09T17:21:01+00:00: Width run d=1, q=24, cells=192, W=289: relative errors [3.86e-16 1.64e-15 3.85e-16 7.67e-16 1.64e-15].

- 2026-10-09T17:21:01+00:00: Width run d=2, q=2, cells=32, W=98: relative errors [0.01 0.18 0.1  0.15 0.59].

- 2026-10-09T17:21:01+00:00: Width run d=2, q=3, cells=32, W=147: relative errors [0.   0.05 0.02 0.05 0.24].

- 2026-10-09T17:21:01+00:00: Width run d=2, q=4, cells=32, W=196: relative errors [1.63e-05 9.24e-03 2.89e-03 1.83e-02 7.93e-02].

- 2026-10-09T17:21:01+00:00: Width run d=2, q=6, cells=48, W=438: relative errors [9.33e-09 1.81e-04 4.69e-05 1.32e-03 8.36e-03].

- 2026-10-09T17:21:01+00:00: Width run d=2, q=8, cells=64, W=776: relative errors [3.86e-12 1.88e-06 3.80e-07 6.36e-05 5.02e-04].

- 2026-10-09T17:21:01+00:00: Width run d=2, q=10, cells=80, W=1210: relative errors [3.39e-15 1.22e-08 1.32e-09 2.02e-06 1.75e-05].

- 2026-10-09T17:21:01+00:00: Width run d=2, q=12, cells=96, W=1740: relative errors [7.73e-15 5.16e-11 5.70e-12 4.40e-08 5.92e-07].

- 2026-10-09T17:21:01+00:00: Width run d=2, q=16, cells=128, W=3088: relative errors [5.53e-15 1.35e-14 3.16e-15 7.30e-12 1.60e-10].

- 2026-10-09T17:21:01+00:00: Width run d=2, q=20, cells=160, W=4820: relative errors [1.63e-14 1.30e-14 2.33e-15 3.23e-15 3.67e-14].

- 2026-10-09T17:21:01+00:00: Width run d=2, q=24, cells=192, W=6936: relative errors [9.43e-15 2.07e-14 3.50e-15 4.35e-15 4.25e-15].

- 2026-10-09T17:21:01+00:00: Width run d=3, q=2, cells=32, W=196: relative errors [0.01 0.17 0.08 0.16 0.45].

- 2026-10-09T17:21:01+00:00: Width run d=3, q=3, cells=32, W=441: relative errors [0.   0.05 0.02 0.05 0.2 ].

- 2026-10-09T17:21:01+00:00: Width run d=3, q=4, cells=32, W=784: relative errors [1.76e-05 9.92e-03 2.03e-03 1.23e-02 6.26e-02].

- 2026-10-09T17:21:01+00:00: Width run d=3, q=6, cells=48, W=2628: relative errors [1.12e-08 2.26e-04 2.43e-05 9.97e-04 4.88e-03].

- 2026-10-09T17:21:01+00:00: Width run d=3, q=8, cells=64, W=6208: relative errors [4.34e-12 2.92e-06 1.70e-07 5.09e-05 2.12e-04].

- 2026-10-09T17:21:01+00:00: Width run d=3, q=10, cells=80, W=12100: relative errors [1.09e-14 2.20e-08 7.38e-10 1.91e-06 4.04e-06].

- 2026-10-09T17:21:01+00:00: Width run d=3, q=12, cells=96, W=20880: relative errors [1.22e-14 1.11e-10 2.90e-12 5.12e-08 1.72e-07].

- 2026-10-09T17:21:01+00:00: Width run d=3, q=16, cells=128, W=49408: relative errors [4.87e-15 2.32e-15 2.39e-15 3.55e-12 5.20e-11].

- 2026-10-09T17:21:01+00:00: Width run d=3, q=20, cells=160, W=96400: relative errors [1.99e-15 1.62e-14 3.42e-15 2.49e-15 6.78e-15].

- 2026-10-09T17:21:02+00:00: Width run d=3, q=24, cells=192, W=166464: relative errors [3.29e-15 6.09e-15 2.43e-15 1.45e-15 1.36e-15].

- 2026-10-09T17:21:02+00:00: Width run d=4, q=2, cells=32, W=392: relative errors [0.01 0.17 0.07 0.15 0.3 ].

- 2026-10-09T17:21:02+00:00: Width run d=4, q=3, cells=32, W=1323: relative errors [0.   0.06 0.01 0.05 0.16].

- 2026-10-09T17:21:02+00:00: Width run d=4, q=4, cells=32, W=3136: relative errors [2.48e-05 1.36e-02 1.57e-03 1.46e-02 5.91e-02].

- 2026-10-09T17:21:02+00:00: Width run d=4, q=6, cells=48, W=15768: relative errors [2.01e-08 3.94e-04 9.58e-06 7.01e-04 5.03e-03].

- 2026-10-09T17:21:04+00:00: Width run d=4, q=8, cells=64, W=49664: relative errors [8.10e-12 5.56e-06 1.30e-07 4.16e-05 1.98e-04].

- 2026-10-09T17:21:07+00:00: Width run d=4, q=10, cells=80, W=121000: relative errors [3.52e-15 4.60e-08 4.70e-10 1.34e-06 3.51e-06].

- 2026-10-09T17:21:13+00:00: Width run d=4, q=12, cells=96, W=250560: relative errors [3.64e-15 2.49e-10 5.22e-13 2.28e-08 1.17e-07].

- 2026-10-09T17:21:31+00:00: Width run d=4, q=16, cells=128, W=790528: relative errors [4.43e-15 3.87e-15 6.93e-16 3.58e-12 3.21e-11].

- 2026-10-09T17:22:16+00:00: Width run d=4, q=20, cells=160, W=1928000: relative errors [2.33e-15 7.17e-15 8.42e-16 7.14e-16 2.46e-15].

- 2026-10-09T17:23:46+00:00: Width run d=4, q=24, cells=192, W=3995136: relative errors [4.54e-15 9.31e-15 1.04e-15 1.26e-15 1.98e-15].

- 2026-10-09T17:23:46+00:00: Width suite completed in 166.0s. All configurations retained, including roundoff plateaus.

- 2026-10-09T17:23:46+00:00: Starting precision suite, declared config in config.json.

- 2026-10-09T17:23:46+00:00: Precision run d=1, W=193; float64 baseline 3.05e-16; readout+bias absolute mass 1.27.

- 2026-10-09T17:23:47+00:00: Precision run d=2, W=3860; float64 baseline 3.66e-15; readout+bias absolute mass 1.62.

- 2026-10-09T17:23:47+00:00: Precision run d=3, W=77200; float64 baseline 2.53e-15; readout+bias absolute mass 2.

- 2026-10-09T17:24:24+00:00: Precision run d=4, W=1544000; float64 baseline 1.46e-15; readout+bias absolute mass 2.21.

- 2026-10-09T17:24:24+00:00: Starting algebraic suite, declared config in config.json.

- 2026-10-09T17:24:24+00:00: Algebraic experiment: explicit ReLU interpolation of x^2 agrees with 1/n^2 to float64 accuracy. This tests the ReLU corollary, not the full finite-smooth Radon theorem.

- 2026-10-09T17:24:24+00:00: Starting allocation suite, declared config in config.json.

- 2026-10-09T17:24:24+00:00: Allocation grid q=2 completed for narrow and signed-mixture targets.

- 2026-10-09T17:24:24+00:00: Allocation grid q=3 completed for narrow and signed-mixture targets.

- 2026-10-09T17:24:24+00:00: Allocation grid q=4 completed for narrow and signed-mixture targets.

- 2026-10-09T17:24:24+00:00: Allocation grid q=6 completed for narrow and signed-mixture targets.

- 2026-10-09T17:24:25+00:00: Allocation grid q=8 completed for narrow and signed-mixture targets.

- 2026-10-09T17:24:25+00:00: Allocation grid q=10 completed for narrow and signed-mixture targets.

- 2026-10-09T17:24:25+00:00: Allocation grid q=12 completed for narrow and signed-mixture targets.

- 2026-10-09T17:24:25+00:00: Allocation grid q=16 completed for narrow and signed-mixture targets.

- 2026-10-09T17:24:26+00:00: Allocation grid q=20 completed for narrow and signed-mixture targets.

- 2026-10-09T17:24:26+00:00: Allocation narrow, eps=0.001: theory q=12, cells=42, W=780, error=9.26e-08; best tested W=294.

- 2026-10-09T17:24:26+00:00: Allocation narrow, eps=1e-06: theory q=19, cells=60, W=1729, error=7.78e-11; best tested W=610.

- 2026-10-09T17:24:26+00:00: Allocation narrow, eps=1e-09: theory q=26, cells=77, W=3068, error=9.85e-14; best tested W=1164.

- 2026-10-09T17:24:26+00:00: Allocation signed_mixture, eps=0.001: theory q=17, cells=48, W=1241, error=2.57e-08; best tested W=490.

- 2026-10-09T17:24:26+00:00: Allocation signed_mixture, eps=1e-06: theory q=24, cells=66, W=2424, error=1.93e-11; best tested W=732.

- 2026-10-09T17:24:26+00:00: Allocation signed_mixture, eps=1e-09: theory q=31, cells=83, W=3906, error=2.04e-14; best tested W=1552.

- 2026-10-09T17:33:30+00:00: Starting algebraic_kernel suite, declared config in config.json.

- 2026-10-09T17:33:30+00:00: Nonanalytic spectral construction B=4, W=43: max sampled error 0.00968; theorem bound 1.37; quadrature refinement 1.5204111486045799e-13.

- 2026-10-09T17:33:30+00:00: Nonanalytic spectral construction B=8, W=75: max sampled error 0.00128; theorem bound 0.423; quadrature refinement None.

- 2026-10-09T17:33:30+00:00: Nonanalytic spectral construction B=16, W=131: max sampled error 0.000162; theorem bound 0.119; quadrature refinement None.

- 2026-10-09T17:33:30+00:00: Nonanalytic spectral construction B=32, W=235: max sampled error 2.03e-05; theorem bound 0.0315; quadrature refinement None.

- 2026-10-09T17:33:30+00:00: Nonanalytic spectral construction B=64, W=435: max sampled error 2.54e-06; theorem bound 0.00811; quadrature refinement 3.9329147080495905e-12.

- 2026-10-09T17:33:30+00:00: Nonanalytic spectral construction B=128, W=829: max sampled error 3.18e-07; theorem bound 0.00206; quadrature refinement None.

- 2026-10-09T17:33:32+00:00: Nonanalytic spectral construction B=256, W=1611: max sampled error 3.97e-08; theorem bound 0.000519; quadrature refinement 2.0199499862430552e-10.

- 2026-10-09T17:38:59+00:00: Starting precision suite, declared config in config.json.

- 2026-10-09T17:38:59+00:00: Precision run d=1, W=193; float64 baseline 5.43e-16; readout+bias absolute mass 1.02.

- 2026-10-09T17:38:59+00:00: Precision run d=2, W=3860; float64 baseline 1.66e-14; readout+bias absolute mass 1.71.

- 2026-10-09T17:38:59+00:00: Precision run d=3, W=77200; float64 baseline 2.91e-15; readout+bias absolute mass 2.02.

- 2026-10-09T17:39:35+00:00: Precision run d=4, W=1544000; float64 baseline 4.65e-15; readout+bias absolute mass 2.4.

## Final organization and checks

The scaling figure now uses only exp(-3*sum(x_k^2)) in d=1..4. Its precision panel was rerun for the same family. The separate robustness figure keeps all five Gaussian-based families, with width as the horizontal axis and no precision panel. The uncorrected-halo ablation is not displayed. Dashed predictions include the corrected halo and do not add a measured numerical floor.

The corrected tanh coefficient formula and contour proof were moved into the main analytic proof. A short additional argument applies the correction to the already controlled entire sample preparation, giving linear scalar count for both the analytic and local algebraic tanh classes. This is a human mathematical extension; the contour and its substitution remain outside Lean. All 145 Lean files (143 modules plus root and audit) match the previous verified release byte for byte.

Numerical checks: analytic bounds held for all measured cases away from float64 roundoff; no error-fitted constants. ReLU interpolation matched its exact formula. The nonanalytic spectral construction respected its moment bound; the largest quadrature-refinement difference is 2.02e-10 at an error of 3.97e-8, so no extra digits are claimed. The sufficient allocation met all tolerances but cost roughly 2.6 to 2.8 times the best tested widths for the Gaussian shown.

## Figure polish — 2026-10-09

Inspected the PDF before and after rendering. Removed repeated titles and explanatory footers; shortened axes and captions; used shared legends and fonts sized for the printed width. Figures now appear alongside their discussion on pages 12–14, rather than on detached or crowded figure pages. No numerical runs, results, theorem statements, or Lean source changed. The final PDF has no layout or reference warnings.

## Final construction subsections — 2026-10-09

Added a concise admissible-lambda starting sweep, explicitly labeled as a proposed search procedure rather than a proven optimum or new benchmark. It preserves the certified schedules as defaults and requires recomputing the bound after geometry changes. Added the complex-shift tanh corollary, the imaginary-part readout prefactor lambda/pi, halo adjustments, and a single output bias. The shortcut is explicitly restricted to real centers; complex halo quadrature uses both shifts. The one-shift and existing two-shift baseline weights agreed in the sine and Gaussian checks. Rendered and inspected the additions; references and PDF layout pass, and all Lean source and recorded numerical results are unchanged.

## Fourier-ratio consistency check — 2026-10-09

Before running: compare the existing corrected encoder on the canonical scalar profiles of exp(-3*||x||^2), d=1..4, n=128 and 512, R=n/4. Test 2*pi/R, the basic Fourier-ratio root, 0.25, 0.30 and 0.40. All are admissible with delta=1. This isolates scalar behavior; no angular-network claim. The optional lambda argument preserves the existing default geometry.

- n=128, R=32, halo_balance, lambda=0.19635: scalar relative errors [5.551e-16 1.811e-15 1.122e-15 1.213e-15]

- n=128, R=32, kernel_ratio, lambda=0.244076: scalar relative errors [4.503e-16 1.191e-15 1.346e-15 7.773e-16]

- n=128, R=32, quarter, lambda=0.25: scalar relative errors [6.292e-16 1.025e-15 1.888e-15 1.492e-15]

- n=128, R=32, 0.30, lambda=0.3: scalar relative errors [2.553e-15 5.492e-15 8.155e-15 9.523e-15]

- n=128, R=32, 0.40, lambda=0.4: scalar relative errors [6.689e-12 1.300e-11 1.759e-11 2.212e-11]

- n=512, R=128, halo_balance, lambda=0.0490874: scalar relative errors [7.745e-13 4.361e-11 9.695e-12 3.038e-11]

- n=512, R=128, kernel_ratio, lambda=0.244076: scalar relative errors [7.976e-16 4.216e-15 6.744e-16 1.472e-15]

- n=512, R=128, quarter, lambda=0.25: scalar relative errors [9.584e-16 3.415e-15 5.226e-15 8.131e-16]

- n=512, R=128, 0.30, lambda=0.3: scalar relative errors [6.720e-16 1.924e-15 1.624e-15 3.590e-15]

- n=512, R=128, 0.40, lambda=0.4: scalar relative errors [1.456e-12 2.615e-12 3.298e-12 3.917e-12]

The corrected pole envelope is exactly (32*B/(pi*delta*n))*(1-exp(-2*pi^2/lambda))*A_K(lambda), where A_K=(pi^2/lambda)/sinh(pi^2/lambda). The note now starts the practical sweep at this Fourier-ratio threshold (0.244076 for 2^-52), keeping 0.25 explicitly in the candidate set. The 2*pi/R expression is retained only as the exact-arithmetic halo/alias balance. The n=512 check shows why this distinction matters in float64. The default encoder matches the previous release bit for bit on both checked geometries; the earlier figure data and all Lean files are unchanged. PDF labels and layout pass.

## Final editorial pass — 2026-10-09

Kept the section order, four main theorems, all supporting proofs, algorithm, lambda rule, tanh corollary, and three figures. Trimmed repeated qualifications and incidental storage arithmetic; replaced proof-assistant wording in the human argument. Corrected three stray M symbols to the already defined profile bound B_g, defined H_x within Theorem 2, clarified anchor derivative orders, separated the alias score notation from the kernel envelope constant, and fixed the boundary-extension reference. No assumptions or convergence claims were weakened. Removed a forced page break that left most of a page blank. The PDF is 39 pages rather than 41. Inspected the rendered introduction, main results, lambda subsection, empirical pages, proof opening, conclusion, and Lean appendix. Build and reference checks pass. All Lean source, recorded results, and figure files match the preceding release byte for byte; no new experiments or Lean modifications.
