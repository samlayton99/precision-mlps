import QIRadon.Activations.PolynomialObstruction
import QIRadon.Activations.ReLUSpline
import QIRadon.Construction.AllocatedConstruction
import QIRadon.Construction.AutomaticConstruction
import QIRadon.Construction.GenericAnchoredConstruction
import QIRadon.Construction.LocalAlgebraicConstruction
import QIRadon.Construction.TanhCorollary
import QIRadon.Construction.UniformConstruction
import QIRadon.Numerics.NumericalPrecision
import QIRadon.Numerics.PreparationStability
import QIRadon.Validation.DeterministicSelection
import QIRadon.Validation.FiniteSampleObstruction
import QIRadon.Validation.StatisticalSelection

/-!
# General-kernel QI–Radon construction

These public endpoints import the complete verified machinery. See
`docs/argument.md` for the proof route and `docs/theorems.md` for theorem scope.
Historical and unfinished material is isolated under `archive/`.
-/
