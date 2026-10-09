import QIRadon.Quadrature.CasimirSmooth

namespace QIRadon.Gaussian

noncomputable section

variable {ι : Type*} [Fintype ι] [DecidableEq ι]

omit [Fintype ι] in
theorem planeRotate_add (i j : ι) (hij : i ≠ j) (s t : ℝ) (x : EuclideanSpace ℝ ι) :
    planeRotate i j (s+t) x = planeRotate i j s (planeRotate i j t x) := by
  ext k
  by_cases hki : k = i
  · subst k
    simp only [planeRotate_apply, if_true, hij.symm, if_false, Real.cos_add, Real.sin_add]
    ring
  · by_cases hkj : k = j
    · subst k
      simp only [planeRotate_apply, hki, if_false, if_true, Real.cos_add, Real.sin_add]
      ring
    · simp [planeRotate_apply, hki, hkj]

end
end QIRadon.Gaussian

namespace QIRadon.Casimir

noncomputable section
set_option backward.isDefEq.respectTransparency false

open Gaussian

theorem rotationFlow_add {n : ℕ} (i j : Fin (n+1)) (s t : ℝ) (θ : UnitSphere n) :
    rotationFlow i j (s+t) θ = rotationFlow i j s (rotationFlow i j t θ) := by
  by_cases hij : i = j
  · simp [rotationFlow, hij]
  · simp only [rotationFlow, dif_neg hij]
    apply Subtype.ext
    exact planeRotate_add i j hij s t θ.val

/-- The group law transports a derivative at time zero to every point of the
 actual rotation orbit. This avoids adding a separate global-time hypothesis. -/
theorem hasRotationDerivative_of_zero {n : ℕ} (i j : Fin (n+1))
    (F D : C(UnitSphere n, ℝ))
    (hzero : ∀ θ, HasDerivAt (fun t ↦ F (rotationFlow i j t θ)) (D θ) 0) :
    HasRotationDerivative i j F D := by
  intro θ t
  have h0 := hzero (rotationFlow i j t θ)
  have hshift : HasDerivAt (fun s : ℝ ↦ s-t) 1 t :=
    (hasDerivAt_id t).sub_const t
  have h0' : HasDerivAt (fun s ↦ F (rotationFlow i j s (rotationFlow i j t θ)))
      (D (rotationFlow i j t θ)) ((fun s : ℝ ↦ s-t) t) := by simpa using h0
  have hcomp := HasDerivAt.comp (h := fun s : ℝ ↦ s-t) t h0' hshift
  have heq : (fun s : ℝ ↦ F (rotationFlow i j (s-t) (rotationFlow i j t θ))) =
      (fun s ↦ F (rotationFlow i j s θ)) := by
    funext s
    rw [← rotationFlow_add, sub_add_cancel]
  simpa only [Function.comp_def, one_mul, mul_one, heq] using hcomp

end
end QIRadon.Casimir
