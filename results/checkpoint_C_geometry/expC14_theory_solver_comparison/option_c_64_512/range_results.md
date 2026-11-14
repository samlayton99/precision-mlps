# Original-theorem comparison, N=64–512 — draft-pending-Sam

The requested range is N=64,128,256,512. Both methods were rerun with one shared admissible design: a_H=2, delta=2, Delta=9/16, lambda=1/4, R=ceil(2 sqrt(N)), M=8N+1. All other solver, cutoff, validation, and arithmetic choices follow option_c_results.md in the parent directory. Solid lines are simple SVD; dashed lines are the original theorem recipe; colors identify functions.

This is a new configuration, not a cropped copy of the previous comparison. At N=64 the previous a_H=2.75 violates N>=16 a_H^2, and previous delta=1 violates gamma*delta>=4*pi. The revised constants satisfy both conditions throughout the requested range and were chosen before observing the new errors. Both solvers use identical geometry at every N. The original E_rep bound and A.65 cutoff remain unchanged.

The theoretical procedure is substantially less accurate at N=64 and 128, where its analytic cutoff component is about 1.28e-3 and 9.37e-8, respectively. At N=256 and 512, both methods reach the high-accuracy numerical regime. The plot preserves this negative small-N result without changing the cutoff or selecting different parameters from test errors. All 24 original-condition/recovery checks pass. Five option-C tests pass, including a regression that rejects the old configuration at N=64 and accepts the new configuration at every requested N.

Data: N*.json, coefficients_N*.npz, summary.csv, summary.json. Outputs: comparison.png, comparison.pdf, comparison.svg. Source configuration: experiments/expC14_theory_solver_comparison/option_c_64_512.yaml. Reproduce with option_c.py --config pointing to that file and --output pointing to a fresh results directory. Plot using plot_option_c.py on the results directory.

Model fitting and evaluation remain FP64; parameter/enclosure verification uses 256-bit MPFR as in the prior Option C experiment. Errors are measured on independent midpoint grids, not asserted to equal a sharp continuous-error certificate.
