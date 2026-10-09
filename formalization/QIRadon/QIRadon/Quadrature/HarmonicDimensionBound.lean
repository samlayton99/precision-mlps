import Mathlib

namespace QIRadon.HarmonicDimensionBound

/-- The exact difference between homogeneous degrees k+2 and k, in ambient
dimension m+2. This is the dimension of the orthogonal harmonic layer once the
geometric subspace inclusion has been established. -/
theorem choose_difference (m k : ℕ) :
    (m + k + 3).choose (m + 1) - (m + k + 1).choose (m + 1) =
      (m + k + 2).choose m + (m + k + 1).choose m := by
  have h1 : (m + k + 3).choose (m + 1) =
      (m + k + 2).choose m + (m + k + 2).choose (m + 1) := by
    convert Nat.choose_succ_succ (m + k + 2) m using 1
  have h2 : (m + k + 2).choose (m + 1) =
      (m + k + 1).choose m + (m + k + 1).choose (m + 1) := by
    convert Nat.choose_succ_succ (m + k + 1) m using 1
  rw [h1, h2]
  omega

/-- The sharp growth exponent is ambient dimension minus two. -/
theorem choose_difference_bound (m k : ℕ) :
    (m + k + 3).choose (m + 1) - (m + k + 1).choose (m + 1) ≤
      2 * (m + 1)^m * (k + 3)^m := by
  rw [choose_difference]
  have h1 := Nat.choose_le_pow (m + k + 2) m
  have h2 := Nat.choose_le_pow (m + k + 1) m
  have hb1 : m + k + 2 ≤ (m + 1) * (k + 3) := by nlinarith
  have hb2 : m + k + 1 ≤ (m + 1) * (k + 3) := by nlinarith
  have hp1 := Nat.pow_le_pow_left hb1 m
  have hp2 := Nat.pow_le_pow_left hb2 m
  rw [mul_pow] at hp1 hp2
  calc
    _ ≤ 2 * ((m + 1)^m * (k + 3)^m) := by omega
    _ = _ := by ring

/-- Same bound using the degree k+2 as the binomial's lower argument. -/
theorem homogeneous_difference_bound (m k : ℕ) :
    (m + k + 3).choose (k + 2) - (m + k + 1).choose k ≤
      2 * (m + 1)^m * (k + 3)^m := by
  have h1 : (m + k + 3).choose (k + 2) = (m + k + 3).choose (m + 1) := by
    have h := Nat.choose_symm_add (a := k + 2) (b := m + 1)
    convert h using 1 <;> congr 1 <;> omega
  have h2 : (m + k + 1).choose k = (m + k + 1).choose (m + 1) := by
    have h := Nat.choose_symm_add (a := k) (b := m + 1)
    convert h using 1 <;> congr 1 <;> omega
  rw [h1, h2]
  exact choose_difference_bound m k

end QIRadon.HarmonicDimensionBound
