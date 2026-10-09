# Radon PINN solver theory — October 9, 2026

Coordinator: this Codex chat. Status: active theoretical investigation, not a completed solver or complexity theorem.

Sam's objective: preserve the actual fixed-geometry Radon neural representation and useful PINN capabilities while obtaining solve overhead approximately linear in direction count M, with a target such as O(M N^2). N is centers per direction; M grows with input dimension. Machine-epsilon-scale numerical accuracy remains an eventual requirement. Main work is theory, not numerical tuning.

## Admissibility contract

- Preserve general local nonlinear and coupled PDE declarations; retain data, inverse-parameter/field, initial/boundary/gauge constraints and scattered physical samples.
- Keep the actual one-hidden-layer ridge network. No external PDE solution followed by encoding may be presented as a native neural solve.
- Do not quietly restrict readouts to a low-degree, low-rank, sparse or known-solution family; add a prior; drop constraints; change the physical domain; or weaken the accuracy target.
- Invertible coordinate changes, exact reformulations and explicitly certified numerical approximations to the same operator are within scope. Their conditioning and all errors must be counted.
- Track total samples K, independent scalar constraints, derivative count, dimension, setup, per-application cost, iterations, memory, and validation. No M factor may be hidden in K, iteration counts or a preconditioner.
- Use roughly fixed lambda = gamma*h as the operating regime requested by Sam. Distinguish asymptotic construction schedules without replacing this regime.
- Well-posedness and local identifiability are explicit theorem hypotheses, not promises for every nonlinear equation. No global convergence or uniform accuracy claim without proof.
- If a promising route requires a questionable restriction or tradeoff, stop that route and ask Sam about the concrete choice before adopting it. Do not silently redefine success.

## Work packages

1. Exact change to cumulative profile coordinates, nonlinear residual factorization, local block structure, and error bounds.
2. Full sample-to-direction aggregation complexity, including possible fast transforms and hidden dimensional/sample costs.
3. Width-uniform stability, exact redundancies and near-cancellation; distinguish intrinsic physical ill-conditioning from coordinate defects.
4. Coordinator synthesis: proved statements, counterexamples, candidate transformations, cost ledger and any necessary question for Sam.

No new performance or accuracy sweeps are planned in this investigation. Small symbolic/algebraic checks may be used to audit derivations.

## Clarification from Sam

Sam rejected the abstract choice between arbitrary finite readouts and construction-defined coordinates as unclear. His criterion is preserving realistic solutions of useful PDEs. The coordinator must translate any proposed restriction into concrete lost physical capabilities (for example thin layers, oscillatory scales, nonlinear flow regimes or inverse-field recovery) before asking for a decision. Preserving every artificial readout combination is not itself the stated research objective. No particular restriction or regularity class has been approved; the abstract question is withdrawn, and the exact-coordinate investigation can continue without that choice.

Completed theory notes: `local_structure.md`, `aggregation.md`, `stability.md`, and coordinator synthesis `README.md`. These are partial mathematical results and explicit open problems, not the completed scaling theorem.

## Current continuation

Sam explicitly reaffirmed two hard requirements: the result remains a standard one-hidden-layer MLP (tanh is acceptable now; other theory-supported activations later), and the method must retain general PDE solution capability, including difficult nonlinear systems such as Navier–Stokes. No further input is needed to continue admissible theoretical work. Current focus: physical resolution and angular coupling, a native-neural operator perturbation framework, and a Navier–Stokes compatibility/error analysis. No restriction to symmetric domains, low-degree target families or low-Reynolds-number flows is being adopted as the general problem.

Completed continuation derivations: `angular_coefficients.md`, `angular_resolution.md`, `navier_stokes_compatibility.md`, and `operator_transfer.md`. Exact results include bounded-degree angular modulation of arbitrary local PDE Jacobians; three-bank directional Navier–Stokes contractions with matching adjoints; polar-cluster local synthesis rank; and a graph-norm perturbation theorem transferring reference stability to actual neural residuals. Unfinished: a complete fast nested angular/source–target factorization, a scalable physical preconditioner on general domains, and total nonlinear iteration/precision guarantees. No full-method success or general high-Re solve is claimed.

## Operation ladder and literature continuation

Sam requested an incremental investigation: one derivative equation, mixed derivatives, sums and constant coefficients, variable coefficients, forcing, products of linear measurements, and more general nonlinearities. Boundary/data/inverse constraints are carried through each step. Additional work examines what projected samples actually reveal and maps these formulations to primary literature. New scoped reports: `product_measurements_literature.md`, `projected_sampling_literature.md`, `nonlinear_pde_literature.md`, and coordinator `operation_ladder.md`. Literature assumptions and their actual applicability to deterministic ridge features must remain explicit; a named method is not a scaling guarantee.
