import Mathlib.Analysis.InnerProductSpace.Dual
import Mathlib.MeasureTheory.Integral.Bochner.Basic
import Mathlib.Tactic

/-!
# Addition estimate for a transitive invariant finite-dimensional function space

The hypotheses describe the actual structure of an L2 function space: its norm
is the integrated squared evaluation, and the symmetry group acts transitively
on points and isometrically on the function space. They do not posit an L∞
estimate. That estimate is proved by taking the trace of the evaluation kernel.
Instantiation with spherical harmonic subspaces is a separate obligation.
-/

noncomputable section
open MeasureTheory
open scoped RealInnerProductSpace

namespace QIRadon.Addition

variable {X E ι : Type*} [MeasurableSpace X]
    [NormedAddCommGroup E] [InnerProductSpace ℝ E] [Fintype ι]

omit [MeasurableSpace X] in
/-- Transitive isometric covariance forces the diagonal kernel to be constant. -/
theorem evaluation_norm_sq_constant (b : OrthonormalBasis ι ℝ E)
    (ev : X → E →L[ℝ] ℝ)
    (htrans : ∀ x y, ∃ U : E ≃ₗᵢ[ℝ] E, ∀ v, ev y v = ev x (U v))
    (x y : X) : ‖ev x‖ ^ 2 = ‖ev y‖ ^ 2 := by
  obtain ⟨U, hU⟩ := htrans x y
  rw [(b.map U).norm_dual, b.norm_dual]
  apply Finset.sum_congr rfl
  intro i _
  rw [hU]
  rfl

/-- The constant diagonal is exactly the dimension when the underlying measure
is a probability measure and the Hilbert norm is the actual L2 norm. -/
theorem evaluation_norm_sq_dimension (b : OrthonormalBasis ι ℝ E)
    (ev : X → E →L[ℝ] ℝ) (μ : Measure X) [IsProbabilityMeasure μ]
    (hL2 : ∀ v, Integrable (fun x => (ev x v)^2) μ ∧
      (∫ x, (ev x v)^2 ∂μ) = ‖v‖^2)
    (htrans : ∀ x y, ∃ U : E ≃ₗᵢ[ℝ] E, ∀ v, ev y v = ev x (U v))
    (x : X) : ‖ev x‖ ^ 2 = (Fintype.card ι : ℝ) := by
  calc
    ‖ev x‖^2 = ∫ y : X, ‖ev x‖^2 ∂μ := by simp
    _ = ∫ y : X, ∑ i, (ev y (b i))^2 ∂μ := by
      apply integral_congr_ae
      filter_upwards with y
      rw [← b.norm_dual]
      exact evaluation_norm_sq_constant b ev htrans x y
    _ = ∑ i, ∫ y : X, (ev y (b i))^2 ∂μ := by
      exact integral_finsetSum _ (fun i _ => (hL2 (b i)).1)
    _ = ∑ i : ι, (1 : ℝ) := by
      apply Finset.sum_congr rfl
      intro i _
      rw [(hL2 (b i)).2, b.orthonormal.norm_eq_one i]
      norm_num
    _ = _ := by simp

/-- Uniform evaluation estimate (the addition-theorem bound), derived from
symmetry and L2 structure rather than assumed as a regularity hypothesis. -/
theorem evaluation_bound (b : OrthonormalBasis ι ℝ E)
    (ev : X → E →L[ℝ] ℝ) (μ : Measure X) [IsProbabilityMeasure μ]
    (hL2 : ∀ v, Integrable (fun x => (ev x v)^2) μ ∧
      (∫ x, (ev x v)^2 ∂μ) = ‖v‖^2)
    (htrans : ∀ x y, ∃ U : E ≃ₗᵢ[ℝ] E, ∀ v, ev y v = ev x (U v))
    (x : X) (v : E) :
    |ev x v| ≤ Real.sqrt (Fintype.card ι) * ‖v‖ := by
  have hn := evaluation_norm_sq_dimension b ev μ hL2 htrans x
  have hs : ‖ev x‖ = Real.sqrt (Fintype.card ι) := by
    rw [← hn, Real.sqrt_sq (norm_nonneg _)]
  simpa only [Real.norm_eq_abs, hs] using (ev x).le_opNorm v

end QIRadon.Addition
