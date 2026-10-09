import Mathlib

/-! Deterministic, pointwise certificates for finite-network evaluation and sample coverage.
These results do not assert recovery of a function from arbitrary finite data. -/

namespace QIRadon

noncomputable section

theorem finite_network_perturbation {ι : Type*} [Fintype ι]
    (v w f g δ : ι → ℝ) (bias bias' B : ℝ)
    (hf : ∀ i, |f i| ≤ B) (hfg : ∀ i, |g i - f i| ≤ δ i) :
    |(bias' + ∑ i, w i * g i) - (bias + ∑ i, v i * f i)| ≤
      |bias' - bias| + B * (∑ i, |w i - v i|) + ∑ i, |w i| * δ i := by
  have hterm (i : ι) : |w i * g i - v i * f i| ≤
      B * |w i - v i| + |w i| * δ i := by
    calc
      _ = |(w i - v i) * f i + w i * (g i - f i)| := by ring_nf
      _ ≤ |(w i - v i) * f i| + |w i * (g i - f i)| := abs_add_le _ _
      _ = |w i - v i| * |f i| + |w i| * |g i - f i| := by rw [abs_mul, abs_mul]
      _ ≤ |w i - v i| * B + |w i| * δ i :=
        add_le_add (mul_le_mul_of_nonneg_left (hf i) (abs_nonneg _))
          (mul_le_mul_of_nonneg_left (hfg i) (abs_nonneg _))
      _ = _ := by ring
  calc
    _ = |(bias' - bias) + ∑ i, (w i * g i - v i * f i)| := by
      rw [Finset.sum_sub_distrib]
      ring_nf
    _ ≤ |bias' - bias| + |∑ i, (w i * g i - v i * f i)| := abs_add_le _ _
    _ ≤ |bias' - bias| + ∑ i, |w i * g i - v i * f i| :=
      add_le_add le_rfl (Finset.abs_sum_le_sum_abs _ _)
    _ ≤ |bias' - bias| + ∑ i, (B * |w i - v i| + |w i| * δ i) := by
      exact add_le_add le_rfl (Finset.sum_le_sum (fun i _ => hterm i))
    _ = _ := by rw [Finset.sum_add_distrib, ← Finset.mul_sum]; ring

theorem activation_network_perturbation {ι : Type*} [Fintype ι]
    (φ : ℝ → ℝ) (v w p q η : ι → ℝ) (bias bias' B L : ℝ)
    (hL : 0 ≤ L) (hφ : ∀ i, |φ (p i)| ≤ B)
    (hlip : ∀ i, |φ (q i) - φ (p i)| ≤ L * |q i - p i|)
    (hpre : ∀ i, |q i - p i| ≤ η i) :
    |(bias' + ∑ i, w i * φ (q i)) - (bias + ∑ i, v i * φ (p i))| ≤
      |bias' - bias| + B * (∑ i, |w i - v i|) + ∑ i, |w i| * (L * η i) := by
  apply finite_network_perturbation v w (fun i => φ (p i)) (fun i => φ (q i))
    (fun i => L * η i) bias bias' B hφ
  intro i
  exact (hlip i).trans (mul_le_mul_of_nonneg_left (hpre i) hL)

/-- Coverage plus known regularity gives a genuine uniform-error certificate, even if
the candidate depends on the samples. No statistical independence is assumed. -/
theorem sample_cover_error_bound {X ι : Type*} [PseudoMetricSpace X]
    (D : Set X) (sites : ι → X) (y : ι → ℝ) (f Q : X → ℝ)
    (E noise Lf LQ radius : ℝ)
    (hLf : 0 ≤ Lf) (hLQ : 0 ≤ LQ)
    (hsites : ∀ i, sites i ∈ D)
    (hcover : ∀ x ∈ D, ∃ i, dist x (sites i) ≤ radius)
    (hf : ∀ x ∈ D, ∀ z ∈ D, |f x - f z| ≤ Lf * dist x z)
    (hQ : ∀ x ∈ D, ∀ z ∈ D, |Q x - Q z| ≤ LQ * dist x z)
    (hnoise : ∀ i, |y i - f (sites i)| ≤ noise)
    (hres : ∀ i, |y i - Q (sites i)| ≤ E) :
    ∀ x ∈ D, |f x - Q x| ≤ E + noise + (Lf + LQ) * radius := by
  intro x hx
  obtain ⟨i, hi⟩ := hcover x hx
  have hfi : |f x - f (sites i)| ≤ Lf * radius :=
    (hf x hx (sites i) (hsites i)).trans (mul_le_mul_of_nonneg_left hi hLf)
  have hQi : |Q (sites i) - Q x| ≤ LQ * radius := by
    rw [abs_sub_comm]
    exact (hQ x hx (sites i) (hsites i)).trans (mul_le_mul_of_nonneg_left hi hLQ)
  have hni : |f (sites i) - y i| ≤ noise := by
    rw [abs_sub_comm]
    exact hnoise i
  calc
    |f x - Q x| = |((f x - f (sites i)) + (f (sites i) - y i)) +
        ((y i - Q (sites i)) + (Q (sites i) - Q x))| := by ring_nf
    _ ≤ |(f x - f (sites i)) + (f (sites i) - y i)| +
        |(y i - Q (sites i)) + (Q (sites i) - Q x)| := abs_add_le _ _
    _ ≤ (|f x - f (sites i)| + |f (sites i) - y i|) +
        (|y i - Q (sites i)| + |Q (sites i) - Q x|) :=
      add_le_add (abs_add_le _ _) (abs_add_le _ _)
    _ ≤ (Lf * radius + noise) + (E + LQ * radius) :=
      add_le_add (add_le_add hfi hni) (add_le_add (hres i) hQi)
    _ = E + noise + (Lf + LQ) * radius := by ring

/-- Actual finite-candidate minimum-width selection, assuming each supplied
certificate has separately been proved sound. No global network optimum is claimed. -/
theorem exists_smallest_certified_candidate {J X : Type*} [Fintype J]
    (D : Set X) (f : X → ℝ) (Q : J → X → ℝ)
    (width : J → ℕ) (certificate : J → ℝ) (ε : ℝ)
    (hsound : ∀ j, ∀ x ∈ D, |f x - Q j x| ≤ certificate j)
    (hpass : ∃ j, certificate j ≤ ε) :
    ∃ j, certificate j ≤ ε ∧
      (∀ x ∈ D, |f x - Q j x| ≤ ε) ∧
      (∀ k, certificate k ≤ ε → width j ≤ width k) := by
  classical
  let passing : Finset J := Finset.univ.filter (fun j => certificate j ≤ ε)
  have hp : passing.Nonempty := by
    obtain ⟨j, hj⟩ := hpass
    exact ⟨j, by simp [passing, hj]⟩
  obtain ⟨j, hj, hmin⟩ := Finset.exists_min_image passing width hp
  have hc : certificate j ≤ ε := (Finset.mem_filter.mp hj).2
  refine ⟨j, hc, ?_, ?_⟩
  · intro x hx
    exact (hsound j x hx).trans hc
  · intro k hk
    exact hmin k (by simp [passing, hk])

end
end QIRadon
