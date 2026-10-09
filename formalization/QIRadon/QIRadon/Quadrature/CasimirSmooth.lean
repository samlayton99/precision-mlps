import QIRadon.Quadrature.CasimirRotation

open MeasureTheory

namespace QIRadon.Casimir

noncomputable section
set_option backward.isDefEq.respectTransparency false

open Gaussian

def rotationFlow {n : ℕ} (i j : Fin (n+1)) (t : ℝ) : UnitSphere n ≃ₜ UnitSphere n :=
  if hij : i = j then Homeomorph.refl _ else sphereRotation (planeRotation i j hij t)

theorem rotationFlow_measurePreserving {n : ℕ} (i j : Fin (n+1)) (t : ℝ) :
    MeasurePreserving (rotationFlow i j t) (normalizedSphere n) (normalizedSphere n) := by
  by_cases hij : i = j
  · simpa [rotationFlow, hij] using MeasurePreserving.id (normalizedSphere n)
  · simpa [rotationFlow, hij] using sphereRotation_measurePreserving (planeRotation i j hij t)

@[simp] theorem rotationFlow_zero {n : ℕ} (i j : Fin (n+1)) (θ : UnitSphere n) :
    rotationFlow i j 0 θ = θ := by
  by_cases hij : i = j
  · simp [rotationFlow, hij]
  · rw [rotationFlow, dif_neg hij]
    apply Subtype.ext
    exact planeRotate_zero i j hij θ.val

/-- An actual derivative relation along every real rotation orbit. This is a
 calculus hypothesis on continuous functions, not a spectral decay assertion. -/
def HasRotationDerivative {n : ℕ} (i j : Fin (n+1))
    (F D : C(UnitSphere n, ℝ)) : Prop :=
  ∀ θ t, HasDerivAt (fun s ↦ F (rotationFlow i j s θ)) (D (rotationFlow i j t θ)) t

theorem polynomial_hasRotationDerivative (n : ℕ) (i j : Fin (n+1))
    (p : MvPolynomial (Fin (n+1)) ℝ) :
    HasRotationDerivative i j (polynomialOnSphere n p) (polynomialOnSphere n (generator i j p)) := by
  intro θ t
  by_cases hij : i = j
  · subst j
    simpa [rotationFlow, generator_self, polynomialOnSphere] using
      hasDerivAt_const t (polynomialOnSphere n p θ)
  · simp only [rotationFlow, dif_neg hij]
    convert planeRotation_polynomial_hasDerivAt i j hij p θ.val t using 1 <;> rfl

/-- Rotational integration by parts for genuinely differentiable continuous
 functions on the sphere, obtained from actual invariant integration. -/
theorem rotationDerivative_skew {n : ℕ} (i j : Fin (n+1))
    (F G D E : C(UnitSphere n, ℝ))
    (hF : HasRotationDerivative i j F D) (hG : HasRotationDerivative i j G E) :
    (∫ θ, D θ * G θ ∂normalizedSphere n) =
      -(∫ θ, F θ * E θ ∂normalizedSphere n) := by
  let A : ℝ → UnitSphere n → ℝ := fun t θ ↦
    F (rotationFlow i j t θ) * G (rotationFlow i j t θ)
  let B : ℝ → UnitSphere n → ℝ := fun t θ ↦
    D (rotationFlow i j t θ) * G (rotationFlow i j t θ) +
      F (rotationFlow i j t θ) * E (rotationFlow i j t θ)
  have hA (t : ℝ) : Continuous (A t) := by dsimp [A]; fun_prop
  have hB (t : ℝ) : Continuous (B t) := by dsimp [B]; fun_prop
  have hd : ∀ θ t, HasDerivAt (fun s ↦ A s θ) (B t θ) t := by
    intro θ t
    convert (hF θ t).mul (hG θ t) using 1 <;> rfl
  have hbound : ∀ θ t, ‖B t θ‖ ≤ ‖D‖ * ‖G‖ + ‖F‖ * ‖E‖ := by
    intro θ t
    dsimp [B]
    calc
      _ ≤ ‖D (rotationFlow i j t θ) * G (rotationFlow i j t θ)‖ +
          ‖F (rotationFlow i j t θ) * E (rotationFlow i j t θ)‖ := norm_add_le _ _
      _ ≤ _ := by
        simp only [norm_mul]
        gcongr
        · exact D.norm_coe_le_norm _
        · exact G.norm_coe_le_norm _
        · exact F.norm_coe_le_norm _
        · exact E.norm_coe_le_norm _
  have h := hasDerivAt_integral_of_dominated_loc_of_deriv_le
    (μ := normalizedSphere n) (F := A) (F' := B) (x₀ := (0 : ℝ))
    (s := Set.univ) (bound := fun _ ↦ ‖D‖ * ‖G‖ + ‖F‖ * ‖E‖)
    Filter.univ_mem (Filter.Eventually.of_forall fun t ↦ (hA t).aestronglyMeasurable)
    ((hA 0).integrable_of_hasCompactSupport (HasCompactSupport.of_compactSpace _))
    (hB 0).aestronglyMeasurable
    (Filter.Eventually.of_forall fun θ t _ ↦ hbound θ t)
    (integrable_const _) (Filter.Eventually.of_forall fun θ t _ ↦ hd θ t)
  have hinv : (fun t ↦ ∫ θ, A t θ ∂normalizedSphere n) =
      (fun _ : ℝ ↦ ∫ θ, F θ * G θ ∂normalizedSphere n) := by
    funext t
    exact (rotationFlow_measurePreserving i j t).integral_comp
      (rotationFlow i j t).measurableEmbedding (fun θ ↦ F θ * G θ)
  have hz := h.2
  rw [hinv] at hz
  have heq := hz.unique (hasDerivAt_const 0 _)
  simp only [B, rotationFlow_zero] at heq
  have hDG : Integrable (fun θ ↦ D θ * G θ) (normalizedSphere n) :=
    (D.continuous.mul G.continuous).integrable_of_hasCompactSupport
      (HasCompactSupport.of_compactSpace _)
  have hFE : Integrable (fun θ ↦ F θ * E θ) (normalizedSphere n) :=
    (F.continuous.mul E.continuous).integrable_of_hasCompactSupport
      (HasCompactSupport.of_compactSpace _)
  rw [integral_add hDG hFE] at heq
  linarith

end
end QIRadon.Casimir
