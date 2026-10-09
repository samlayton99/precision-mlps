import QIRadon.Validation.Certification
import QIRadon.Architecture.Basic
import QIRadon.Scalar.ScalarRate

/-! Finite arithmetic tolerance for the actual shallow network. The theorem
accounts for rounded hidden weights, biases, readouts, activation evaluations,
and final summation. This is a sufficient absolute-error/working-precision
certificate, not a claim about uncertified float64 or coefficient acquisition cost. -/

noncomputable section
open scoped BigOperators Real SchwartzMap
open MeasureTheory FourierTransform
namespace QIRadon.NumericalPrecision

theorem preactivation_error {d : ℕ} {a a' x : Fin d → ℝ} {b b' R δ : ℝ}
    (hR : 0 ≤ R) (ha : ∀ k, |a' k-a k| ≤ δ) (hb : |b'-b| ≤ δ)
    (hx : ∀ k, |x k| ≤ R) :
    |((∑ k, a' k*x k)+b')-((∑ k, a k*x k)+b)| ≤ ((d:ℝ)*R+1)*δ := by
  have he : ((∑ k, a' k*x k)+b')-((∑ k, a k*x k)+b) =
      (∑ k, (a' k-a k)*x k)+(b'-b) := by
    simp_rw [sub_mul, Finset.sum_sub_distrib]
    ring
  rw [he]
  calc
    _ ≤ (∑ k, |(a' k-a k)*x k|)+|b'-b| :=
      (abs_add_le _ _).trans (add_le_add (Finset.abs_sum_le_sum_abs _ _) le_rfl)
    _ ≤ (∑ _k : Fin d, δ*R)+δ := add_le_add
      (Finset.sum_le_sum (fun k _ => by
        rw [abs_mul]
        exact (mul_le_mul_of_nonneg_left (hx k) (abs_nonneg _)).trans
          (mul_le_mul_of_nonneg_right (ha k) hR))) hb
    _ = _ := by simp; ring

def errorAmplification (d W : ℕ) (R B V L : ℝ) : ℝ :=
  2+B*W+(V+W)*(L*((d:ℝ)*R+1)+1)

theorem errorAmplification_pos (d W : ℕ) {R B V L : ℝ}
    (hR : 0 ≤ R) (hB : 0 ≤ B) (hV : 0 ≤ V) (hL : 0 ≤ L) :
    0 < errorAmplification d W R B V L := by unfold errorAmplification; positivity

/-- Every error source is charged at absolute tolerance δ; the amplification
depends explicitly on dimension, width, activation scale and readout mass. -/
theorem rounded_network_error {d : ℕ} {ι : Type*} [Fintype ι]
    (φ : ℝ → ℝ) (a a' : ι → Fin d → ℝ) (b b' v v' : ι → ℝ)
    (bias bias' : ℝ) (x : Fin d → ℝ) (y : ι → ℝ) (output : ℝ)
    {R B V L δ : ℝ} (hR : 0 ≤ R) (hB : 0 ≤ B) (hL : 0 ≤ L)
    (hδ : 0 ≤ δ) (hδ1 : δ ≤ 1)
    (hx : ∀ k, |x k| ≤ R)
    (ha : ∀ i k, |a' i k-a i k| ≤ δ) (hb : ∀ i, |b' i-b i| ≤ δ)
    (hv : ∀ i, |v' i-v i| ≤ δ) (hbias : |bias'-bias| ≤ δ)
    (hV : (∑ i, |v i|) ≤ V)
    (hφ : ∀ i, |φ ((∑ k, a i k*x k)+b i)| ≤ B)
    (hlip : ∀ i, |φ ((∑ k, a' i k*x k)+b' i)-φ ((∑ k, a i k*x k)+b i)| ≤
      L*|((∑ k, a' i k*x k)+b' i)-((∑ k, a i k*x k)+b i)|)
    (heval : ∀ i, |y i-φ ((∑ k, a' i k*x k)+b' i)| ≤ δ)
    (hsum : |output-(bias'+∑ i, v' i*y i)| ≤ δ) :
    |output-Architecture.network φ a b v bias x| ≤
      errorAmplification d (Fintype.card ι) R B V L * δ := by
  let C := L*((d:ℝ)*R+1)+1
  have hC : 0 ≤ C := by dsimp [C]; positivity
  have hfeature (i : ι) : |y i-φ ((∑ k, a i k*x k)+b i)| ≤ C*δ := by
    have hp := preactivation_error hR (ha i) (hb i) hx
    have h := (abs_sub_le (y i) (φ ((∑ k, a' i k*x k)+b' i))
      (φ ((∑ k, a i k*x k)+b i))).trans
        (add_le_add (heval i) ((hlip i).trans (mul_le_mul_of_nonneg_left hp hL)))
    apply h.trans_eq
    dsimp [C]
    ring
  have hvsum : (∑ i, |v' i-v i|) ≤ (Fintype.card ι:ℝ)*δ := by
    simpa using Finset.sum_le_sum (s := Finset.univ) (fun i _ => hv i)
  have hv'sum : (∑ i, |v' i|) ≤ V+(Fintype.card ι:ℝ) := by
    have hh : ∀ i, |v' i| ≤ |v i|+δ := by
      intro i
      have h := (abs_sub_le (v' i) (v i) 0).trans (add_le_add (hv i) le_rfl)
      simpa only [sub_zero, add_comm δ] using h
    have hsumv := Finset.sum_le_sum (s := Finset.univ) (fun i _ => hh i)
    simp only [Finset.sum_add_distrib, Finset.sum_const, Finset.card_univ, nsmul_eq_mul] at hsumv
    exact hsumv.trans (add_le_add hV (mul_le_of_le_one_right (Nat.cast_nonneg _) hδ1))
  have hnet := finite_network_perturbation v v'
    (fun i => φ ((∑ k, a i k*x k)+b i)) y (fun _ => C*δ) bias bias' B hφ hfeature
  have hnet' : |(bias'+∑ i, v' i*y i)-Architecture.network φ a b v bias x| ≤
      δ+B*((Fintype.card ι:ℝ)*δ)+(V+(Fintype.card ι:ℝ))*(C*δ) := by
    apply hnet.trans
    rw [← Finset.sum_mul]
    exact add_le_add (add_le_add hbias (mul_le_mul_of_nonneg_left hvsum hB))
      (mul_le_mul_of_nonneg_right hv'sum (mul_nonneg hC hδ))
  have hh := (abs_sub_le output (bias'+∑ i, v' i*y i)
    (Architecture.network φ a b v bias x)).trans (add_le_add hsum hnet')
  apply hh.trans_eq
  dsimp [errorAmplification, C]
  ring

/-- A sufficient number of binary absolute-accuracy bits. This counts working
precision, not a fixed hardware mantissa independent of coefficient magnitude. -/
def workingBits (C ε : ℝ) : ℕ := ⌈Real.log (C/ε)/Real.log 2⌉₊

def binaryTolerance (b : ℕ) : ℝ := (2:ℝ)^(-(b:ℝ))

theorem binaryTolerance_pos (b : ℕ) : 0 < binaryTolerance b :=
  Real.rpow_pos_of_pos (by norm_num) _

theorem binaryTolerance_le_one (b : ℕ) : binaryTolerance b ≤ 1 := by
  unfold binaryTolerance
  exact Real.rpow_le_one_of_one_le_of_nonpos (by norm_num)
    (neg_nonpos.mpr (Nat.cast_nonneg b))

theorem workingBits_accuracy {C ε : ℝ} (hC : 0 < C) (hε : 0 < ε) :
    C*binaryTolerance (workingBits C ε) ≤ ε := by
  have hlog2 : 0 < Real.log 2 := Real.log_pos (by norm_num)
  have hceil : Real.log (C/ε)/Real.log 2 ≤ (workingBits C ε:ℝ) := Nat.le_ceil _
  have hb : Real.log (C/ε) ≤ (workingBits C ε:ℝ)*Real.log 2 :=
    (div_le_iff₀ hlog2).mp hceil
  have hpow : binaryTolerance (workingBits C ε) ≤ Real.exp (-Real.log (C/ε)) := by
    unfold binaryTolerance
    rw [Real.rpow_def_of_pos (by norm_num : (0:ℝ)<2)]
    apply Real.exp_le_exp.mpr
    nlinarith
  have hh := mul_le_mul_of_nonneg_left hpow hC.le
  rw [Real.exp_neg, Real.exp_log (div_pos hC hε), inv_div] at hh
  exact hh.trans_eq (by field_simp [hC.ne'])

/-- The actual inverse-kernel readout density has a derived magnitude bound.
No unmeasured coefficient condition number is inserted as an assumption. -/
theorem readoutDensity_bound (K : 𝓢(ℝ, ℝ)) {pHat : ℝ → ℂ} {ν B γ L : ℝ}
    (E : KernelRates.SpectralEnvelope (𝓕 (fun x : ℝ => (K x:ℂ))) ν)
    (hp : Integrable pHat) (hB : 0 < B) (hγ : 0 < γ)
    (hscale : (B/γ)^ν=L) (x : ℝ) :
    |Kernel.realReadoutDensity pHat K γ hγ.ne' B x| ≤
      (∫ ξ : ℝ, ‖pHat ξ‖)*((1/E.c)*Real.exp (E.A*L)) := by
  unfold Kernel.realReadoutDensity Kernel.cutoffDensity
  have hfirst := (Complex.abs_re_le_norm _).trans
    (Kernel.norm_synthesize_le
      (Kernel.bandSpectrum (fun ξ => pHat ξ /
        𝓕 ((Kernel.complexScaledKernel K γ hγ.ne'):ℝ → ℂ) ξ) B) x)
  apply hfirst.trans
  simp_rw [Kernel.complexScaledKernel_fourier K hγ]
  exact ScalarRate.band_density_integral_bound E hp hB hγ hscale

/-- Normalized positive angular quadrature avoids an artificial factor M in
the total readout magnitude; the factor here is the actual row width. -/
theorem normalized_spoke_readout_mass {M J : Type*} [Fintype M] [Fintype J]
    (α : M → ℝ) (ρ : M → J → ℝ) {c D : ℝ}
    (hα : ∀ m, 0 ≤ α m) (hmass : (∑ m, α m)=1)
    (hρ : ∀ m j, |ρ m j| ≤ D) :
    (∑ m, ∑ j, |α m*c*ρ m j|) ≤ (Fintype.card J:ℝ) * |c| * D := by
  have h : ∀ m, (∑ j, |α m*c*ρ m j|) ≤ α m*((Fintype.card J:ℝ) * |c| * D) := by
    intro m
    have hh := Finset.sum_le_sum (s := Finset.univ) (fun j _ =>
      mul_le_mul_of_nonneg_left (hρ m j) (mul_nonneg (hα m) (abs_nonneg c)))
    simp only [abs_mul, abs_of_nonneg (hα m)]
    apply hh.trans_eq
    simp
    ring
  have hs := Finset.sum_le_sum (s := Finset.univ) (fun m _ => h m)
  rwa [← Finset.sum_mul, hmass, one_mul] at hs

/-- The selected number of binary bits has an explicit finite upper bound.
Thus precision is charged logarithmically in both accuracy and amplification. -/
theorem workingBits_bound (C ε : ℝ) :
    (workingBits C ε:ℝ) < max 0 (Real.log (C/ε)/Real.log 2)+1 := by
  unfold workingBits
  by_cases h : 0 ≤ Real.log (C/ε)/Real.log 2
  · rw [max_eq_right h]
    exact Nat.ceil_lt_add_one h
  · have hh : Real.log (C/ε)/Real.log 2 ≤ 0 := (lt_of_not_ge h).le
    rw [max_eq_left hh, Nat.ceil_eq_zero.mpr hh]
    norm_num

end QIRadon.NumericalPrecision
