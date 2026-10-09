# Several-complex-variables library check

The pinned project is Lean 4.33.0 / Mathlib db584cd6d46c92f209a44c0f1c829460d327499d. A newer Mathlib toolchain alone did not establish that the missing finite-dimensional holomorphic-to-analytic interface was available: the inspected upstream CauchyIntegral theorem still has one complex input.

Public library [lean-SCV](https://github.com/bjbraams/lean-SCV), by Bastiaan J Braams, contains the relevant stronger theorem and analytic parameter-integral results. Read-only checkout inspected: d9ed6b74e6432eaf682020e3b960087cf9ec786f, requiring Lean 4.35.0-rc3. License Apache-2.0. Its earlier registered snapshot is not the same revision as this checkout; external verification claims must not be represented as our independent audit of this checkout.

The current coordinate-free declaration is `DifferentiableOn.analyticOnNhd_of_finiteDimensional`, in `SeveralComplexVariables/Analyticity.lean`. Its source closure consists of Analyticity, Osgood, CauchySeries, CauchyIntegral, Polydisc, and Reinhardt (about 1,640 lines). This is useful reusable infrastructure, not a one-line Mathlib version fix.

No external proof has been imported and no dependency was upgraded. Our direct checked `Radon/ParameterIntegral.lean` supplies complex C-infinity regularity from dominated analytic fibers. The checked smooth variants in `Quadrature/MixedCauchy.lean` give the required sharp bounds. Therefore the task can continue in the existing pinned environment without backporting SCV merely to convert equivalent regularity predicates.

The ignored `scratch/lean-scv` checkout is inspection material only. It is not a proof dependency and is not covered by the project's axiom audit.
