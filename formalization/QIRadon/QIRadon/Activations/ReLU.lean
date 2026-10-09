import Mathlib

/-!
An activation obstruction for the actual finite one-hidden-layer ReLU network.
No approximation or piecewise-affine structure is assumed of the network.
-/

namespace QIRadon

noncomputable section

def relu (x : ℝ) : ℝ := max 0 x

def reluNetwork {W : ℕ} (v a b : Fin W → ℝ) (bias x : ℝ) : ℝ :=
  bias + ∑ j, v j * relu (a j * x + b j)

lemma relu_midpoint_of_same_sign (u v : ℝ)
    (h : (0 ≤ u ∧ 0 ≤ v) ∨ (u ≤ 0 ∧ v ≤ 0)) :
    relu u + relu v = 2 * relu ((u + v) / 2) := by
  rcases h with ⟨hu, hv⟩ | ⟨hu, hv⟩
  · have hm : 0 ≤ (u + v) / 2 := by positivity
    simp only [relu, max_eq_right hu, max_eq_right hv, max_eq_right hm]
    ring
  · have hm : (u + v) / 2 ≤ 0 := by linarith
    simp only [relu, max_eq_left hu, max_eq_left hv, max_eq_left hm]
    ring

lemma relu_affine_midpoint (a b l u : ℝ) (hlu : l ≤ u)
    (hroot : ¬ (l < -b / a ∧ -b / a < u)) :
    relu (a * l + b) + relu (a * u + b) =
      2 * relu (a * ((l + u) / 2) + b) := by
  by_cases ha : a = 0
  · subst a
    simp only [zero_mul, zero_add]
    ring
  have hc : a * (-b / a) + b = 0 := by
    field_simp
    ring
  have hout : -b / a ≤ l ∨ u ≤ -b / a := by
    by_cases hleft : -b / a ≤ l
    · exact Or.inl hleft
    · exact Or.inr (le_of_not_gt (fun hright => hroot ⟨lt_of_not_ge hleft, hright⟩))
  have hsign : (0 ≤ a * l + b ∧ 0 ≤ a * u + b) ∨
      (a * l + b ≤ 0 ∧ a * u + b ≤ 0) := by
    rcases le_total 0 a with hapos | haneg
    · rcases hout with hleft | hright
      · left
        constructor <;> nlinarith [mul_nonneg hapos (sub_nonneg.mpr hleft),
          mul_nonneg hapos (sub_nonneg.mpr hlu)]
      · right
        constructor <;> nlinarith [mul_nonneg hapos (sub_nonneg.mpr hright),
          mul_nonneg hapos (sub_nonneg.mpr hlu)]
    · rcases hout with hleft | hright
      · right
        constructor <;> nlinarith [mul_nonpos_of_nonpos_of_nonneg haneg
          (sub_nonneg.mpr hleft), mul_nonpos_of_nonpos_of_nonneg haneg (sub_nonneg.mpr hlu)]
      · left
        constructor <;> nlinarith [mul_nonpos_of_nonpos_of_nonneg haneg
          (sub_nonneg.mpr hright), mul_nonpos_of_nonpos_of_nonneg haneg (sub_nonneg.mpr hlu)]
  have hm : ((a * l + b) + (a * u + b)) / 2 =
      a * ((l + u) / 2) + b := by ring
  simpa only [hm] using relu_midpoint_of_same_sign (a * l + b) (a * u + b) hsign

lemma exists_empty_uniform_interval (W : ℕ) (roots : Fin W → ℝ) :
    ∃ i : Fin (W + 1), ∀ j : Fin W,
      ¬ (-1 + (i : ℝ) * (2 / (W + 1 : ℝ)) < roots j ∧
        roots j < -1 + ((i : ℝ) + 1) * (2 / (W + 1 : ℝ))) := by
  classical
  by_contra h
  push Not at h
  choose idx hidx using h
  have hs : 0 < 2 / (W + 1 : ℝ) := by positivity
  have hinj : Function.Injective idx := by
    intro i k hik
    by_contra hne
    have hneval : i.val ≠ k.val := fun heq => hne (Fin.ext heq)
    rcases lt_or_gt_of_ne hneval with hlt | hgt
    · have hnat : i.val + 1 ≤ k.val := hlt
      have hreal : (i : ℝ) + 1 ≤ (k : ℝ) := by exact_mod_cast hnat
      have hsep := mul_le_mul_of_nonneg_right hreal hs.le
      have hi := hidx i
      have hk := hidx k
      rw [hik] at hi
      linarith [hi.2, hk.1]
    · have hnat : k.val + 1 ≤ i.val := hgt
      have hreal : (k : ℝ) + 1 ≤ (i : ℝ) := by exact_mod_cast hnat
      have hsep := mul_le_mul_of_nonneg_right hreal hs.le
      have hi := hidx i
      have hk := hidx k
      rw [hik] at hi
      linarith [hk.2, hi.1]
  have hcard := Fintype.card_le_of_injective idx hinj
  simp only [Fintype.card_fin] at hcard
  omega

lemma reluNetwork_midpoint {W : ℕ} (v a b : Fin W → ℝ) (bias l u : ℝ)
    (hlu : l ≤ u) (hroot : ∀ j, ¬ (l < -b j / a j ∧ -b j / a j < u)) :
    reluNetwork v a b bias l + reluNetwork v a b bias u =
      2 * reluNetwork v a b bias ((l + u) / 2) := by
  have hsum : (∑ j, v j * relu (a j * l + b j)) +
      (∑ j, v j * relu (a j * u + b j)) =
      2 * ∑ j, v j * relu (a j * ((l + u) / 2) + b j) := by
    rw [← Finset.sum_add_distrib, Finset.mul_sum]
    apply Finset.sum_congr rfl
    intro j _
    have hj := relu_affine_midpoint (a j) (b j) l u hlu (hroot j)
    nlinarith [congrArg (fun z : ℝ => v j * z) hj]
  unfold reluNetwork
  linarith

lemma quadratic_error_on_affine_interval (Q : ℝ → ℝ) (l u E : ℝ)
    (hmid : Q l + Q u = 2 * Q ((l + u) / 2))
    (hl : |l ^ 2 - Q l| ≤ E) (hu : |u ^ 2 - Q u| ≤ E)
    (hm : |((l + u) / 2) ^ 2 - Q ((l + u) / 2)| ≤ E) :
    (u - l) ^ 2 / 8 ≤ E := by
  rcases abs_le.mp hl with ⟨hl₁, hl₂⟩
  rcases abs_le.mp hu with ⟨hu₁, hu₂⟩
  rcases abs_le.mp hm with ⟨hm₁, hm₂⟩
  nlinarith

/-- Every width-W scalar shallow ReLU network has uniform quadratic error at
least 1 / (2 (W+1)^2). Biases, slopes, readouts and the output constant are arbitrary. -/
theorem relu_quadratic_lower_bound {W : ℕ} (v a b : Fin W → ℝ) (bias E : ℝ)
    (hE : ∀ x ∈ Set.Icc (-1 : ℝ) 1,
      |x ^ 2 - reluNetwork v a b bias x| ≤ E) :
    1 / (2 * (W + 1 : ℝ) ^ 2) ≤ E := by
  obtain ⟨i, hi⟩ := exists_empty_uniform_interval W (fun j => -b j / a j)
  let step : ℝ := 2 / (W + 1 : ℝ)
  let l : ℝ := -1 + (i : ℝ) * step
  let u : ℝ := -1 + ((i : ℝ) + 1) * step
  have hden : (0 : ℝ) < W + 1 := by positivity
  have hs : 0 < step := by dsimp [step]; positivity
  have hi0 : (0 : ℝ) ≤ (i : ℝ) := by positivity
  have hi1 : (i : ℝ) + 1 ≤ (W + 1 : ℝ) := by exact_mod_cast i.isLt
  have hl0 : -1 ≤ l := by dsimp [l]; nlinarith
  have hprod : ((i : ℝ) + 1) * step ≤ 2 := by
    calc
      _ ≤ (W + 1 : ℝ) * step := mul_le_mul_of_nonneg_right hi1 hs.le
      _ = 2 := by dsimp [step]; field_simp
  have hu1 : u ≤ 1 := by dsimp [u]; linarith
  have hlu : l ≤ u := by dsimp [l, u]; nlinarith
  have hl : l ∈ Set.Icc (-1 : ℝ) 1 := ⟨hl0, hlu.trans hu1⟩
  have hu : u ∈ Set.Icc (-1 : ℝ) 1 := ⟨hl0.trans hlu, hu1⟩
  have hm : (l + u) / 2 ∈ Set.Icc (-1 : ℝ) 1 := by
    constructor <;> linarith
  have hmid := reluNetwork_midpoint v a b bias l u hlu hi
  have herr := quadratic_error_on_affine_interval (reluNetwork v a b bias)
    l u E hmid (hE l hl) (hE u hu) (hE ((l + u) / 2) hm)
  have hwidth : (u - l) ^ 2 / 8 = 1 / (2 * (W + 1 : ℝ) ^ 2) := by
    dsimp [u, l, step]
    field_simp
    ring
  rwa [hwidth] at herr

/-- The lower bound has an actual witness point, not only a conditional norm statement. -/
theorem relu_quadratic_error_witness {W : ℕ} (v a b : Fin W → ℝ) (bias : ℝ) :
    ∃ x ∈ Set.Icc (-1 : ℝ) 1,
      1 / (2 * (W + 1 : ℝ) ^ 2) ≤ |x ^ 2 - reluNetwork v a b bias x| := by
  obtain ⟨i, hi⟩ := exists_empty_uniform_interval W (fun j => -b j / a j)
  let step : ℝ := 2 / (W + 1 : ℝ)
  let l : ℝ := -1 + (i : ℝ) * step
  let u : ℝ := -1 + ((i : ℝ) + 1) * step
  have hs : 0 < step := by dsimp [step]; positivity
  have hi0 : (0 : ℝ) ≤ (i : ℝ) := by positivity
  have hi1 : (i : ℝ) + 1 ≤ (W + 1 : ℝ) := by exact_mod_cast i.isLt
  have hl0 : -1 ≤ l := by dsimp [l]; nlinarith
  have hprod : ((i : ℝ) + 1) * step ≤ 2 := by
    calc
      _ ≤ (W + 1 : ℝ) * step := mul_le_mul_of_nonneg_right hi1 hs.le
      _ = 2 := by dsimp [step]; field_simp
  have hu1 : u ≤ 1 := by dsimp [u]; linarith
  have hlu : l ≤ u := by dsimp [l, u]; nlinarith
  have hl : l ∈ Set.Icc (-1 : ℝ) 1 := ⟨hl0, hlu.trans hu1⟩
  have hu : u ∈ Set.Icc (-1 : ℝ) 1 := ⟨hl0.trans hlu, hu1⟩
  have hm : (l + u) / 2 ∈ Set.Icc (-1 : ℝ) 1 := by
    constructor <;> linarith
  have hmid := reluNetwork_midpoint v a b bias l u hlu hi
  have hwidth : (u - l) ^ 2 / 8 = 1 / (2 * (W + 1 : ℝ) ^ 2) := by
    dsimp [u, l, step]
    field_simp
    ring
  by_contra h
  push Not at h
  have he_l := abs_lt.mp (h l hl)
  have he_u := abs_lt.mp (h u hu)
  have he_m := abs_lt.mp (h ((l + u) / 2) hm)
  nlinarith

end
end QIRadon
