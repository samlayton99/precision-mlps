import QIRadon.Quadrature.HomogeneousDimension

open MeasureTheory
open scoped BigOperators RealInnerProductSpace

namespace QIRadon.Gaussian

noncomputable section

def squaredRadiusPolynomial (n : ℕ) : MvPolynomial (Fin (n + 1)) ℝ :=
  ∑ i, MvPolynomial.X i ^ 2

theorem squaredRadiusPolynomial_homogeneous (n : ℕ) :
    (squaredRadiusPolynomial n).IsHomogeneous 2 := by
  change squaredRadiusPolynomial n ∈ MvPolynomial.homogeneousSubmodule _ ℝ 2
  apply Submodule.sum_mem
  intro i _
  exact MvPolynomial.isHomogeneous_X_pow (R := ℝ) i 2

theorem squaredRadiusPolynomial_eval (n : ℕ) (x : EuclideanSpace ℝ (Fin (n + 1))) :
    MvPolynomial.eval (fun i ↦ x i) (squaredRadiusPolynomial n) = ‖x‖ ^ 2 := by
  simp [squaredRadiusPolynomial, EuclideanSpace.real_norm_sq_eq]

def raiseHomogeneousPolynomial (n k : ℕ) (p : HomogeneousPolynomials n k) :
    HomogeneousPolynomials n (k + 2) :=
  ⟨squaredRadiusPolynomial n * p.val, by
    simpa [Nat.add_comm] using (squaredRadiusPolynomial_homogeneous n).mul p.property⟩

theorem raiseHomogeneousPolynomial_restriction (n k : ℕ) (p : HomogeneousPolynomials n k) :
    homogeneousRestriction n (k + 2) (raiseHomogeneousPolynomial n k p) =
      homogeneousRestriction n k p := by
  ext θ
  change MvPolynomial.eval (fun i ↦ θ.val i) (squaredRadiusPolynomial n * p.val) = _
  rw [map_mul, squaredRadiusPolynomial_eval]
  have hθ : ‖θ.val‖ = 1 := by simp
  rw [hθ, one_pow, one_mul]
  rfl

theorem continuousPolynomialSpace_raise (n k : ℕ) (f : ContinuousPolynomialSpace n k) :
    f.val ∈ ContinuousPolynomialSpace n (k + 2) := by
  obtain ⟨p, hp⟩ := f.property
  exact ⟨raiseHomogeneousPolynomial n k p, (raiseHomogeneousPolynomial_restriction n k p).trans hp⟩

/-- The lower-degree restriction space embeds without changing the actual L2
function: multiplication by squared radius equals one on the unit sphere. -/
def polynomialDegreeRaise (n k : ℕ) : PolynomialL2Space n k →ₗ[ℝ] PolynomialL2Space n (k + 2) where
  toFun v := ⟨v.val, by
    obtain ⟨f, hf⟩ := v.property
    exact ⟨⟨f.val, continuousPolynomialSpace_raise n k f⟩, hf⟩⟩
  map_add' := by intro v w; rfl
  map_smul' := by intro c v; rfl

theorem polynomialDegreeRaise_injective (n k : ℕ) :
    Function.Injective (polynomialDegreeRaise n k) := by
  intro v w h
  have h' := congrArg (fun z : PolynomialL2Space n (k + 2) ↦ z.val) h
  change v.val = w.val at h'
  exact Subtype.ext h'

def lowerDegreeSpace (n k : ℕ) : Submodule ℝ (PolynomialL2Space n (k + 2)) :=
  (polynomialDegreeRaise n k).range

def harmonicSubspace (n : ℕ) : (k : ℕ) → Submodule ℝ (PolynomialL2Space n k)
  | 0 => ⊤
  | 1 => ⊤
  | k + 2 => (lowerDegreeSpace n k)ᗮ

abbrev HarmonicSpace (n k : ℕ) := harmonicSubspace n k

theorem lowerDegreeSpace_finrank (n k : ℕ) :
    Module.finrank ℝ (lowerDegreeSpace n k) = (n + k).choose k := by
  unfold lowerDegreeSpace
  rw [← (LinearEquiv.ofInjective (polynomialDegreeRaise n k)
    (polynomialDegreeRaise_injective n k)).finrank_eq, polynomialL2_finrank]

/-- Exact dimension difference, derived from an actual orthogonal complement. -/
theorem harmonic_finrank_add (n k : ℕ) :
    Module.finrank ℝ (HarmonicSpace n (k + 2)) + (n + k).choose k =
      (n + (k + 2)).choose (k + 2) := by
  have h := Submodule.finrank_add_finrank_orthogonal (lowerDegreeSpace n k)
  rw [lowerDegreeSpace_finrank, polynomialL2_finrank] at h
  change Module.finrank ℝ ((lowerDegreeSpace n k)ᗮ) + _ = _
  omega

theorem harmonic_finrank_zero (n : ℕ) : Module.finrank ℝ (HarmonicSpace n 0) = 1 := by
  change Module.finrank ℝ (⊤ : Submodule ℝ (PolynomialL2Space n 0)) = 1
  rw [finrank_top, polynomialL2_finrank]
  simp

theorem harmonic_finrank_one (n : ℕ) : Module.finrank ℝ (HarmonicSpace n 1) = n + 1 := by
  change Module.finrank ℝ (⊤ : Submodule ℝ (PolynomialL2Space n 1)) = n + 1
  rw [finrank_top, polynomialL2_finrank]
  simp

def harmonicEvaluation (n k : ℕ) (θ : UnitSphere n) : HarmonicSpace n k →L[ℝ] ℝ :=
  (polynomialEvaluation n k θ).comp (harmonicSubspace n k).subtypeL

theorem harmonicEvaluation_L2 (n k : ℕ) (v : HarmonicSpace n k) :
    (∫ θ, harmonicEvaluation n k θ v ^ 2 ∂normalizedSphere n) = ‖v‖ ^ 2 :=
  polynomialEvaluation_L2 n k v.val

theorem harmonicEvaluation_square_integrable (n k : ℕ) (v : HarmonicSpace n k) :
    Integrable (fun θ ↦ harmonicEvaluation n k θ v ^ 2) (normalizedSphere n) :=
  polynomialEvaluation_square_integrable n k v.val

end
end QIRadon.Gaussian
