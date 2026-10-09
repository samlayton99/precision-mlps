import Mathlib

/-! Independent-validation certificates for a finite frozen candidate family.
The bounded Bernstein inequality is derived from the exponential series,
independent MGF factorization, and Mathlib's proved Chernoff inequality. -/

noncomputable section
open MeasureTheory ProbabilityTheory Set
open scoped BigOperators

namespace QIRadon.StatisticalSelection

set_option backward.isDefEq.respectTransparency false

theorem bernstein_exp_remainder {u : ℝ} (hu : |u| < 3) :
    |Real.exp u-1-u| ≤ u^2/(2*(1-|u|/3)) := by
  have hsum : HasSum (fun k : ℕ ↦ u^k/(k.factorial:ℝ)) (Real.exp u) := by
    simpa only [Real.exp_eq_exp_ℝ] using NormedSpace.expSeries_div_hasSum_exp u
  have htail := (hasSum_nat_add_iff' 2).mpr hsum
  have hhead : (∑ k ∈ Finset.range 2, u^k/(k.factorial:ℝ)) = 1+u := by
    norm_num [Finset.sum_range_succ]
  rw [hhead] at htail
  have hr : 0 ≤ |u|/3 := by positivity
  have hr1 : |u|/3 < 1 := by linarith
  have hgeom := (hasSum_geometric_of_lt_one hr hr1).mul_left (u^2/2)
  have hterm (k : ℕ) : ‖u^(k+2)/((k+2).factorial:ℝ)‖ ≤ (u^2/2)*(|u|/3)^k := by
    simp only [norm_div, norm_pow, Real.norm_eq_abs, Nat.abs_cast]
    have hfact : (2:ℝ)*3^k ≤ ((k+2).factorial:ℝ) := by
      exact_mod_cast (show 2*3^k ≤ (k+2).factorial by
        simpa [Nat.add_comm] using (Nat.factorial_mul_pow_le_factorial (m := 2) (n := k)))
    calc
      _ ≤ |u|^(k+2)/(2*3^k) := by gcongr
      _ = _ := by rw [pow_add, div_pow, sq_abs]; ring
  have h := htail.norm_le_of_bounded hgeom hterm
  have heq : Real.exp u-(1+u) = Real.exp u-1-u := by ring
  rw [heq, Real.norm_eq_abs] at h
  apply h.trans_eq
  simp only [div_eq_mul_inv, mul_inv_rev]
  ring


theorem bounded_integrable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}
    [IsFiniteMeasure μ] {X : Ω → ℝ} {B : ℝ}
    (hX : Measurable X) (hbound : ∀ᵐ ω ∂μ, |X ω| ≤ B) : Integrable X μ :=
  (integrable_const B).mono' hX.aestronglyMeasurable (by simpa only [Real.norm_eq_abs] using hbound)

theorem bounded_square_integrable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}
    [IsFiniteMeasure μ] {X : Ω → ℝ} {B : ℝ}
    (hX : Measurable X) (hB : 0 ≤ B) (hbound : ∀ᵐ ω ∂μ, |X ω| ≤ B) :
    Integrable (fun ω ↦ (X ω)^2) μ := by
  apply (integrable_const (B^2)).mono' (hX.pow_const 2).aestronglyMeasurable
  filter_upwards [hbound] with ω hω
  rw [Real.norm_eq_abs, abs_of_nonneg (sq_nonneg _)]
  simpa only [sq_abs] using (sq_le_sq₀ (abs_nonneg _) hB).mpr hω

theorem bounded_exp_integrable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}
    [IsFiniteMeasure μ] {X : Ω → ℝ} {B t : ℝ}
    (hX : Measurable X) (ht : 0 ≤ t) (hbound : ∀ᵐ ω ∂μ, |X ω| ≤ B) :
    Integrable (fun ω ↦ Real.exp (t*X ω)) μ := by
  apply (integrable_const (Real.exp (t*B))).mono' ((hX.const_mul t).exp.aestronglyMeasurable)
  filter_upwards [hbound] with ω hω
  rw [Real.norm_eq_abs, abs_of_pos (Real.exp_pos _)]
  exact Real.exp_le_exp.mpr (mul_le_mul_of_nonneg_left ((le_abs_self _).trans hω) ht)

/-- The classical bounded centered MGF estimate, with its variance term and
 Bernstein radius 3/B, is derived from the proved exponential remainder. -/
theorem bounded_centered_mgf {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}
    [IsProbabilityMeasure μ] {X : Ω → ℝ} {B v t : ℝ}
    (hX : Measurable X) (hB : 0 ≤ B) (ht : 0 ≤ t) (htB : t*B < 3)
    (hbound : ∀ᵐ ω ∂μ, |X ω| ≤ B) (hcenter : (∫ ω, X ω ∂μ)=0)
    (hsecond : (∫ ω, (X ω)^2 ∂μ) ≤ v) :
    mgf X μ t ≤ Real.exp (v*t^2/(2*(1-t*B/3))) := by
  let C := t^2/(2*(1-t*B/3))
  have hden : 0 < 1-t*B/3 := by linarith
  have hC : 0 ≤ C := by dsimp [C]; positivity
  have hpoint : ∀ᵐ ω ∂μ, Real.exp (t*X ω) ≤ 1+t*X ω+C*(X ω)^2 := by
    filter_upwards [hbound] with ω hω
    have htx : |t*X ω| ≤ t*B := by
      rw [abs_mul, abs_of_nonneg ht]
      exact mul_le_mul_of_nonneg_left hω ht
    have htail := bernstein_exp_remainder (htx.trans_lt htB)
    have hdenx : 0 < 1-|t*X ω|/3 := by linarith
    have hrem : (t*X ω)^2/(2*(1-|t*X ω|/3)) ≤ C*(X ω)^2 := by
      calc
        _ ≤ (t*X ω)^2/(2*(1-t*B/3)) := by gcongr
        _ = _ := by dsimp [C]; ring
    have h := (le_abs_self (Real.exp (t*X ω)-1-t*X ω)).trans (htail.trans hrem)
    linarith
  have hi := bounded_integrable hX hbound
  have hi2 := bounded_square_integrable hX hB hbound
  have hipoly : Integrable (fun ω ↦ 1+t*X ω+C*(X ω)^2) μ :=
    ((integrable_const 1).add (hi.const_mul t)).add (hi2.const_mul C)
  have h := integral_mono_ae (bounded_exp_integrable hX ht hbound) hipoly hpoint

  have heq : (∫ ω, 1+t*X ω+C*(X ω)^2 ∂μ) = 1+C*(∫ ω, (X ω)^2 ∂μ) := by
    have hi1 : Integrable (fun ω ↦ 1+t*X ω) μ := (integrable_const 1).add (hi.const_mul t)
    rw [integral_add hi1 (hi2.const_mul C), integral_const_mul]
    have hfirst : (∫ ω, 1+t*X ω ∂μ) = 1 := by
      rw [integral_add (integrable_const 1) (hi.const_mul t), integral_const_mul, hcenter]
      simp
    rw [hfirst]
  change mgf X μ t ≤ _ at h
  rw [heq] at h
  have hcv := mul_le_mul_of_nonneg_left hsecond hC
  calc
    mgf X μ t ≤ 1+C*(∫ ω, (X ω)^2 ∂μ) := h
    _ ≤ 1+C*v := by linarith
    _ ≤ Real.exp (C*v) := by simpa only [add_comm] using Real.add_one_le_exp (C*v)
    _ = _ := by congr 1; dsimp [C]; ring


/-- Independence enters through the proved factorization of the actual MGF. -/
theorem independent_mgf_bound {Ω ι : Type*} [MeasurableSpace Ω] [Fintype ι]
    {μ : Measure Ω} [IsProbabilityMeasure μ] {X : ι → Ω → ℝ} {B t : ℝ} {v : ι → ℝ}
    (hX : ∀ i, Measurable (X i)) (hindep : iIndepFun X μ)
    (hB : 0 ≤ B) (ht : 0 ≤ t) (htB : t*B < 3)
    (hbound : ∀ i, ∀ᵐ ω ∂μ, |X i ω| ≤ B)
    (hcenter : ∀ i, (∫ ω, X i ω ∂μ)=0)
    (hsecond : ∀ i, (∫ ω, (X i ω)^2 ∂μ) ≤ v i) :
    mgf (fun ω ↦ ∑ i, X i ω) μ t ≤
      Real.exp ((∑ i, v i)*t^2/(2*(1-t*B/3))) := by
  have heq := hindep.mgf_sum (t := t) hX Finset.univ
  have hf : (fun ω ↦ ∑ i, X i ω) = ∑ i, X i := by ext ω; simp only [Finset.sum_apply]
  rw [hf, heq]
  calc
    _ ≤ ∏ i, Real.exp (v i*t^2/(2*(1-t*B/3))) := by
      apply Finset.prod_le_prod
      · intro i _; exact mgf_nonneg
      · intro i _; exact bounded_centered_mgf (hX i) hB ht htB (hbound i) (hcenter i) (hsecond i)
    _ = _ := by
      rw [← Real.exp_sum]
      congr 1
      rw [← Finset.sum_div, ← Finset.sum_mul]

/-- Classical Bernstein upper tail for bounded independent centered variables,
 with the actual second-moment upper bounds and the standard 1/3 constant. -/
theorem bernstein_upper_tail {Ω ι : Type*} [MeasurableSpace Ω] [Fintype ι]
    {μ : Measure Ω} [IsProbabilityMeasure μ] {X : ι → Ω → ℝ} {B r : ℝ} {v : ι → ℝ}
    (hX : ∀ i, Measurable (X i)) (hindep : iIndepFun X μ)
    (hB : 0 ≤ B) (hr : 0 < r) (hV : 0 < ∑ i, v i)
    (hbound : ∀ i, ∀ᵐ ω ∂μ, |X i ω| ≤ B)
    (hcenter : ∀ i, (∫ ω, X i ω ∂μ)=0)
    (hsecond : ∀ i, (∫ ω, (X i ω)^2 ∂μ) ≤ v i) :
    μ.real {ω | r ≤ ∑ i, X i ω} ≤
      Real.exp (-r^2/(2*((∑ i, v i)+B*r/3))) := by
  let V := ∑ i, v i
  let D := V+B*r/3
  let t := r/D
  have hD : 0 < D := by dsimp [D,V]; positivity
  have ht : 0 < t := div_pos hr hD
  have htB : t*B < 3 := by
    dsimp [t]
    rw [div_mul_eq_mul_div]
    apply (div_lt_iff₀ hD).mpr
    dsimp [D,V]
    nlinarith
  have hint : Integrable (fun ω ↦ Real.exp (t*(∑ i, X i ω))) μ := by
    simpa only [Finset.sum_apply] using hindep.integrable_exp_mul_sum hX
      (s := Finset.univ) (fun i _ ↦ bounded_exp_integrable (hX i) ht.le (hbound i))
  have hc := measure_ge_le_exp_mul_mgf (X := fun ω ↦ ∑ i, X i ω) r ht.le hint
  have hm := independent_mgf_bound hX hindep hB ht.le htB hbound hcenter hsecond
  have hrem : 1-t*B/3 = V/D := by
    dsimp only [t]
    field_simp [hD.ne']
    dsimp [D]
    ring
  have he : -t*r+V*t^2/(2*(1-t*B/3)) = -r^2/(2*D) := by
    rw [hrem]
    dsimp only [t]
    field_simp [hD.ne', (show V ≠ 0 from hV.ne')]
    ring
  calc
    _ ≤ Real.exp (-t*r)*mgf (fun ω ↦ ∑ i, X i ω) μ t := hc
    _ ≤ Real.exp (-t*r)*Real.exp (V*t^2/(2*(1-t*B/3))) :=
      mul_le_mul_of_nonneg_left hm (Real.exp_pos _).le
    _ = _ := by rw [← Real.exp_add, he]

theorem bernstein_two_sided {Ω ι : Type*} [MeasurableSpace Ω] [Fintype ι]
    {μ : Measure Ω} [IsProbabilityMeasure μ] {X : ι → Ω → ℝ} {B r : ℝ} {v : ι → ℝ}
    (hX : ∀ i, Measurable (X i)) (hindep : iIndepFun X μ)
    (hB : 0 ≤ B) (hr : 0 < r) (hV : 0 < ∑ i, v i)
    (hbound : ∀ i, ∀ᵐ ω ∂μ, |X i ω| ≤ B)
    (hcenter : ∀ i, (∫ ω, X i ω ∂μ)=0)
    (hsecond : ∀ i, (∫ ω, (X i ω)^2 ∂μ) ≤ v i) :
    μ.real {ω | r ≤ |∑ i, X i ω|} ≤
      2*Real.exp (-r^2/(2*((∑ i, v i)+B*r/3))) := by
  have hp := bernstein_upper_tail hX hindep hB hr hV hbound hcenter hsecond
  have hnegind : iIndepFun (fun i ω ↦ -X i ω) μ :=
    hindep.comp (fun _ x ↦ -x) (fun _ ↦ measurable_neg)
  have hn := bernstein_upper_tail (X := fun i ω ↦ -X i ω) (fun i ↦ (hX i).neg) hnegind hB hr hV
    (fun i ↦ by simpa only [Pi.neg_apply, abs_neg] using hbound i)
    (fun i ↦ by simp only [integral_neg, hcenter i, neg_zero])
    (fun i ↦ by simpa only [Pi.neg_apply, neg_sq] using hsecond i)
  have hsub : {ω | r ≤ |∑ i, X i ω|} ⊆
      {ω | r ≤ ∑ i, X i ω} ∪ {ω | r ≤ ∑ i, -X i ω} := by
    intro ω hω
    by_cases hs : 0 ≤ ∑ i, X i ω
    · left; simpa only [mem_ofPred_eq, abs_of_nonneg hs] using hω
    · right; simpa only [mem_ofPred_eq, abs_of_nonpos (le_of_not_ge hs), Finset.sum_neg_distrib] using hω
  have hm := (measureReal_mono (μ := μ) hsub).trans (measureReal_union_le (μ := μ) _ _)
  exact hm.trans (by linarith)


def bernsteinCutoff (V B s : ℝ) : ℝ := Real.sqrt (2*V*s)+2*B*s/3

theorem bernsteinCutoff_pos {V B s : ℝ} (hV : 0 < V) (hB : 0 ≤ B) (hs : 0 < s) :
    0 < bernsteinCutoff V B s := by
  unfold bernsteinCutoff
  positivity

theorem bernsteinCutoff_exponent {V B s : ℝ} (hV : 0 < V) (hB : 0 ≤ B) (hs : 0 < s) :
    s ≤ (bernsteinCutoff V B s)^2/(2*(V+B*bernsteinCutoff V B s/3)) := by
  have hr := bernsteinCutoff_pos hV hB hs
  apply (le_div_iff₀ (by positivity : 0 < 2*(V+B*bernsteinCutoff V B s/3))).mpr
  have hsqrt : (Real.sqrt (2*V*s))^2 = 2*V*s := Real.sq_sqrt (by positivity)
  have hcross : 0 ≤ Real.sqrt (2*V*s)*B*s := by positivity
  unfold bernsteinCutoff
  nlinarith

theorem bernstein_confidence {Ω ι : Type*} [MeasurableSpace Ω] [Fintype ι]
    {μ : Measure Ω} [IsProbabilityMeasure μ] {X : ι → Ω → ℝ} {B s : ℝ} {v : ι → ℝ}
    (hX : ∀ i, Measurable (X i)) (hindep : iIndepFun X μ)
    (hB : 0 ≤ B) (hs : 0 < s) (hV : 0 < ∑ i, v i)
    (hbound : ∀ i, ∀ᵐ ω ∂μ, |X i ω| ≤ B)
    (hcenter : ∀ i, (∫ ω, X i ω ∂μ)=0)
    (hsecond : ∀ i, (∫ ω, (X i ω)^2 ∂μ) ≤ v i) :
    μ.real {ω | bernsteinCutoff (∑ i, v i) B s ≤ |∑ i, X i ω|} ≤ 2*Real.exp (-s) := by
  have h := bernstein_two_sided hX hindep hB (bernsteinCutoff_pos hV hB hs) hV hbound hcenter hsecond
  apply h.trans
  gcongr
  have hc := bernsteinCutoff_exponent hV hB hs
  simpa only [neg_div] using neg_le_neg hc

theorem bounded_mean_mem {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}
    [IsProbabilityMeasure μ] {Y : Ω → ℝ} {L : ℝ}
    (hY : Measurable Y) (_hL : 0 ≤ L) (hbound : ∀ᵐ ω ∂μ, Y ω ∈ Icc 0 L) :
    (∫ ω, Y ω ∂μ) ∈ Icc 0 L := by
  have habs : ∀ᵐ ω ∂μ, |Y ω| ≤ L := by
    filter_upwards [hbound] with ω hω
    simpa only [abs_of_nonneg hω.1] using hω.2
  have hi := bounded_integrable hY habs
  constructor
  · exact integral_nonneg_of_ae (hbound.mono (fun _ h ↦ h.1))
  · have h := integral_mono_ae hi (integrable_const L) (hbound.mono (fun _ h ↦ h.2))
    simpa using h

def empiricalMean {Ω : Type*} {n : ℕ} (Y : Fin n → Ω → ℝ) (ω : Ω) : ℝ :=
  (∑ i, Y i ω)/(n:ℝ)

/-- Automatic known variance upper bound from the bounded loss range,
 using Mathlib's proved Popoviciu inequality. This is not empirical Bernstein. -/
def validationRadius (n : ℕ) (L s : ℝ) : ℝ :=
  bernsteinCutoff ((n:ℝ)*(L/2)^2) L s/(n:ℝ)

theorem bounded_validation_confidence {Ω : Type*} [MeasurableSpace Ω]
    {μ : Measure Ω} [IsProbabilityMeasure μ] {n : ℕ} (hn : 0 < n)
    {Y : Fin n → Ω → ℝ} {L m s : ℝ} (hL : 0 < L) (hs : 0 < s)
    (hY : ∀ i, Measurable (Y i)) (hindep : iIndepFun Y μ)
    (hbound : ∀ i, ∀ᵐ ω ∂μ, Y i ω ∈ Icc 0 L)
    (hmean : ∀ i, (∫ ω, Y i ω ∂μ)=m) :
    μ.real {ω | validationRadius n L s ≤ |empiricalMean Y ω-m|} ≤ 2*Real.exp (-s) := by
  let X : Fin n → Ω → ℝ := fun i ω ↦ Y i ω-m
  have hm : m ∈ Icc 0 L := by
    rw [← hmean ⟨0,hn⟩]
    exact bounded_mean_mem (hY _) hL.le (hbound _)
  have habsY (i : Fin n) : ∀ᵐ ω ∂μ, |Y i ω| ≤ L := by
    filter_upwards [hbound i] with ω hω
    simpa only [abs_of_nonneg hω.1] using hω.2
  have hX : ∀ i, Measurable (X i) := fun i ↦ (hY i).sub_const m
  have hXi : iIndepFun X μ := hindep.comp (fun _ y ↦ y-m) (fun _ ↦ measurable_id.sub_const m)
  have hXbound (i : Fin n) : ∀ᵐ ω ∂μ, |X i ω| ≤ L := by
    filter_upwards [hbound i] with ω hω
    apply abs_le.mpr
    dsimp [X]
    constructor <;> linarith [hω.1,hω.2,hm.1,hm.2]
  have hXcenter (i : Fin n) : (∫ ω, X i ω ∂μ)=0 := by
    rw [show X i = (fun ω ↦ Y i ω-m) by rfl,
      integral_sub (bounded_integrable (hY i) (habsY i)) (integrable_const m), hmean i]
    simp
  have hXsecond (i : Fin n) : (∫ ω, (X i ω)^2 ∂μ) ≤ (L/2)^2 := by
    have hv := variance_le_sq_of_bounded (hbound i) (hY i).aemeasurable
    rw [variance_eq_integral (hY i).aemeasurable, hmean i] at hv
    simpa only [sub_zero] using hv
  have hnR : 0 < (n:ℝ) := Nat.cast_pos.mpr hn
  have hV : 0 < ∑ _i : Fin n, (L/2)^2 := by
    simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
    positivity
  have htail := bernstein_confidence hX hXi hL.le hs hV hXbound hXcenter hXsecond
  simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul] at htail
  have hsum (ω : Ω) : (∑ i, X i ω) = (n:ℝ)*(empiricalMean Y ω-m) := by
    simp only [X, Finset.sum_sub_distrib, Finset.sum_const, Finset.card_univ,
      Fintype.card_fin, nsmul_eq_mul, empiricalMean]
    field_simp
  have hset : {ω | validationRadius n L s ≤ |empiricalMean Y ω-m|} =
      {ω | bernsteinCutoff ((n:ℝ)*(L/2)^2) L s ≤ |∑ i, X i ω|} := by
    ext ω
    simp only [mem_ofPred_eq, validationRadius, hsum, abs_mul, abs_of_pos hnR]
    rw [div_le_iff₀ hnR]
    constructor <;> intro h <;> simpa only [mul_comm] using h
  rw [hset]
  exact htail


def confidenceLog (K : ℕ) (δ : ℝ) : ℝ := Real.log (2*(K:ℝ)/δ)

theorem confidenceLog_pos {K : ℕ} (hK : 0 < K) {δ : ℝ} (hδ : δ ∈ Ioo (0:ℝ) 1) :
    0 < confidenceLog K δ := by
  apply Real.log_pos
  apply (lt_div_iff₀ hδ.1).mpr
  have hKR : (1:ℝ) ≤ K := by exact_mod_cast hK
  linarith [hδ.2]

theorem confidenceLog_cancel {K : ℕ} (hK : 0 < K) {δ : ℝ} (hδ : 0 < δ) :
    (K:ℝ)*(2*Real.exp (-confidenceLog K δ)) = δ := by
  have hKR : 0 < (K:ℝ) := Nat.cast_pos.mpr hK
  rw [confidenceLog, Real.exp_neg, Real.exp_log (by positivity)]
  field_simp

theorem simultaneous_validation_failure {Ω J : Type*} [MeasurableSpace Ω]
    [Fintype J] [Nonempty J] {μ : Measure Ω} [IsProbabilityMeasure μ]
    {n : ℕ} (hn : 0 < n) {Y : J → Fin n → Ω → ℝ} {risk : J → ℝ} {L δ : ℝ}
    (hL : 0 < L) (hδ : δ ∈ Ioo (0:ℝ) 1)
    (hY : ∀ j i, Measurable (Y j i)) (hindep : ∀ j, iIndepFun (Y j) μ)
    (hbound : ∀ j i, ∀ᵐ ω ∂μ, Y j i ω ∈ Icc 0 L)
    (hmean : ∀ j i, (∫ ω, Y j i ω ∂μ)=risk j) :
    μ.real {ω | ∃ j, validationRadius n L (confidenceLog (Fintype.card J) δ) ≤
      |empiricalMean (Y j) ω-risk j|} ≤ δ := by
  let s := confidenceLog (Fintype.card J) δ
  have hs : 0 < s := confidenceLog_pos Fintype.card_pos hδ
  have hc (j : J) := bounded_validation_confidence hn hL hs (hY j) (hindep j) (hbound j) (hmean j)
  have heq : {ω | ∃ j, validationRadius n L s ≤ |empiricalMean (Y j) ω-risk j|} =
      ⋃ j, {ω | validationRadius n L s ≤ |empiricalMean (Y j) ω-risk j|} := by ext ω; simp
  change μ.real {ω | ∃ j, validationRadius n L s ≤ |empiricalMean (Y j) ω-risk j|} ≤ δ
  rw [heq]
  calc
    _ ≤ ∑ j, μ.real {ω | validationRadius n L s ≤ |empiricalMean (Y j) ω-risk j|} :=
      measureReal_iUnion_fintype_le _
    _ ≤ ∑ _j : J, 2*Real.exp (-s) := Finset.sum_le_sum (fun j _ ↦ hc j)
    _ = δ := by
      simp only [Finset.sum_const, Finset.card_univ, nsmul_eq_mul]
      exact confidenceLog_cancel Fintype.card_pos hδ.1

theorem simultaneous_validation_good_probability {Ω J : Type*} [MeasurableSpace Ω]
    [Fintype J] [Nonempty J] {μ : Measure Ω} [IsProbabilityMeasure μ]
    {n : ℕ} (hn : 0 < n) {Y : J → Fin n → Ω → ℝ} {risk : J → ℝ} {L δ : ℝ}
    (hL : 0 < L) (hδ : δ ∈ Ioo (0:ℝ) 1)
    (hY : ∀ j i, Measurable (Y j i)) (hindep : ∀ j, iIndepFun (Y j) μ)
    (hbound : ∀ j i, ∀ᵐ ω ∂μ, Y j i ω ∈ Icc 0 L)
    (hmean : ∀ j i, (∫ ω, Y j i ω ∂μ)=risk j) :
    1-δ ≤ μ.real {ω | ∀ j, |empiricalMean (Y j) ω-risk j| <
      validationRadius n L (confidenceLog (Fintype.card J) δ)} := by
  let r := validationRadius n L (confidenceLog (Fintype.card J) δ)
  let bad : Set Ω := {ω | ∃ j, r ≤ |empiricalMean (Y j) ω-risk j|}
  have hbad : MeasurableSet bad := by
    have heq : bad = ⋃ j, {ω | r ≤ |empiricalMean (Y j) ω-risk j|} := by ext ω; simp [bad]
    rw [heq]
    apply MeasurableSet.iUnion
    intro j
    apply measurableSet_le measurable_const
    unfold empiricalMean
    fun_prop
  have hgood : {ω | ∀ j, |empiricalMean (Y j) ω-risk j| < r} = badᶜ := by
    ext ω
    simp only [bad, mem_ofPred_eq, mem_compl_iff, not_exists, not_le]
  change 1-δ ≤ μ.real {ω | ∀ j, |empiricalMean (Y j) ω-risk j| < r}
  rw [hgood, measureReal_compl hbad]
  have hb := simultaneous_validation_failure hn hL hδ hY hindep hbound hmean
  have hμ : μ.real univ = 1 := by simp
  rw [hμ]
  exact sub_le_sub_left hb 1

/-- The selector is a finite minimum over candidates passing the computed
 validation upper bound. It does not search outside this frozen family. -/
def chooseSmallest {J : Type*} [Fintype J] (width : J → ℕ) (score : J → ℝ) (ε : ℝ)
    (hpass : ∃ j, score j ≤ ε) : J := by
  classical
  let passing := Finset.univ.filter (fun j ↦ score j ≤ ε)
  have hp : passing.Nonempty := by
    obtain ⟨j,hj⟩ := hpass
    exact ⟨j,by simp [passing,hj]⟩
  exact Classical.choose (Finset.exists_min_image passing width hp)

theorem chooseSmallest_spec {J : Type*} [Fintype J] (width : J → ℕ) (score : J → ℝ) (ε : ℝ)
    (hpass : ∃ j, score j ≤ ε) :
    score (chooseSmallest width score ε hpass) ≤ ε ∧
      ∀ k, score k ≤ ε → width (chooseSmallest width score ε hpass) ≤ width k := by
  classical
  let passing := Finset.univ.filter (fun j ↦ score j ≤ ε)
  have hp : passing.Nonempty := by
    obtain ⟨j,hj⟩ := hpass
    exact ⟨j,by simp [passing,hj]⟩
  have h := Classical.choose_spec (Finset.exists_min_image passing width hp)
  refine ⟨(Finset.mem_filter.mp h.1).2, ?_⟩
  intro k hk
  exact h.2 k (by simp [passing,hk])

/-- On simultaneous validation success, the selected model meets the true
 risk threshold and is no wider than any tested candidate meeting the true
 threshold with the explicit 2r statistical margin. -/
theorem selected_risk_and_width {J : Type*} [Fintype J]
    (width : J → ℕ) (empirical risk : J → ℝ) (r ε : ℝ)
    (hgood : ∀ j, |empirical j-risk j| ≤ r)
    (hpass : ∃ j, empirical j+r ≤ ε) :
    let j := chooseSmallest width (fun k ↦ empirical k+r) ε hpass
    risk j ≤ ε ∧ ∀ k, risk k+2*r ≤ ε → width j ≤ width k := by
  dsimp only
  have hsel := chooseSmallest_spec width (fun k ↦ empirical k+r) ε hpass
  constructor
  · have h := (abs_le.mp (hgood (chooseSmallest width (fun k ↦ empirical k+r) ε hpass))).1
    linarith [hsel.1]
  · intro k hk
    apply hsel.2 k
    have h := (abs_le.mp (hgood k)).2
    linarith

theorem passing_exists_of_risk_margin {J : Type*} (empirical risk : J → ℝ) (r ε : ℝ)
    (hgood : ∀ j, |empirical j-risk j| ≤ r) (hmargin : ∃ k, risk k+2*r ≤ ε) :
    ∃ j, empirical j+r ≤ ε := by
  obtain ⟨k,hk⟩ := hmargin
  refine ⟨k,?_⟩
  have h := (abs_le.mp (hgood k)).2
  linarith


def squaredLoss {E : Type*} (f q : E → ℝ) (x : E) : ℝ := (f x-q x)^2

def squaredL2Risk {E : Type*} [MeasurableSpace E] (ν : Measure E) (f q : E → ℝ) : ℝ :=
  ∫ x, squaredLoss f q x ∂ν

theorem measurable_squaredLoss {E : Type*} [MeasurableSpace E] {f q : E → ℝ}
    (hf : Measurable f) (hq : Measurable q) : Measurable (squaredLoss f q) :=
  (hf.sub hq).pow_const 2

theorem squaredLoss_mem {E : Type*} {f q : E → ℝ} {B : ℝ} (hB : 0 ≤ B)
    (hf : ∀ x, |f x| ≤ B) (hq : ∀ x, |q x| ≤ B) (x : E) :
    squaredLoss f q x ∈ Icc 0 ((2*B)^2) := by
  refine ⟨sq_nonneg _, ?_⟩
  have habs : |f x-q x| ≤ 2*B := by
    exact (abs_sub (f x) (q x)).trans (by linarith [hf x,hq x])
  have h := (sq_le_sq₀ (abs_nonneg (f x-q x)) (by positivity : 0 ≤ 2*B)).mpr habs
  simp only [sq_abs] at h
  convert h using 1
  rfl

/-- Actual squared-L² validation for fixed candidate functions. The candidates
 are deterministic/frozen before sampling; only the validation input points
 are random and independent. Their law identifies the true risk integral. -/
theorem frozen_squaredLoss_confidence {E Ω J : Type*}
    [MeasurableSpace E] [MeasurableSpace Ω] [Fintype J] [Nonempty J]
    {ν : Measure E} [IsProbabilityMeasure ν] {μ : Measure Ω} [IsProbabilityMeasure μ]
    {n : ℕ} (hn : 0 < n) (samples : Fin n → Ω → E)
    (hmeas : ∀ i, Measurable (samples i)) (hindep : iIndepFun samples μ)
    (hlaw : ∀ i, HasLaw (samples i) ν μ)
    (f : E → ℝ) (Q : J → E → ℝ) (hf : Measurable f) (hQ : ∀ j, Measurable (Q j))
    {B δ : ℝ} (hB : 0 < B) (hδ : δ ∈ Ioo (0:ℝ) 1)
    (hfbound : ∀ x, |f x| ≤ B) (hQbound : ∀ j x, |Q j x| ≤ B) :
    1-δ ≤ μ.real {ω | ∀ j,
      |empiricalMean (fun i ω ↦ squaredLoss f (Q j) (samples i ω)) ω-squaredL2Risk ν f (Q j)| <
        validationRadius n ((2*B)^2) (confidenceLog (Fintype.card J) δ)} := by
  apply simultaneous_validation_good_probability hn (by positivity) hδ
  · intro j i
    exact (measurable_squaredLoss hf (hQ j)).comp (hmeas i)
  · intro j
    exact hindep.comp (fun _ ↦ squaredLoss f (Q j)) (fun _ ↦ measurable_squaredLoss hf (hQ j))
  · intro j i
    exact Filter.Eventually.of_forall (fun ω ↦ squaredLoss_mem hB.le hfbound (hQbound j) (samples i ω))
  · intro j i
    exact (hlaw i).integral_comp (measurable_squaredLoss hf (hQ j)).aestronglyMeasurable


/-- Readable form of the automatic bounded-loss Bernstein radius. -/
theorem validationRadius_eq {n : ℕ} (hn : 0 < n) {L s : ℝ} (hL : 0 ≤ L) (hs : 0 ≤ s) :
    validationRadius n L s = L*Real.sqrt (s/(2*(n:ℝ)))+2*L*s/(3*(n:ℝ)) := by
  have hnR : 0 < (n:ℝ) := Nat.cast_pos.mpr hn
  have hroot : Real.sqrt (2*((n:ℝ)*(L/2)^2)*s) =
      (n:ℝ)*L*Real.sqrt (s/(2*(n:ℝ))) := by
    apply (sq_eq_sq₀ (Real.sqrt_nonneg _) (by positivity)).mp
    rw [Real.sq_sqrt (by positivity), mul_pow, mul_pow, Real.sq_sqrt (by positivity)]
    field_simp
  unfold validationRadius bernsteinCutoff
  rw [hroot]
  field_simp


/-- Combined statistical-and-selection certificate. The good event is the
 explicit simultaneous deviation event already proved measurable above.
 Selection then uses only empirical upper bounds and the supplied widths. -/
theorem validation_selection_certificate {Ω J : Type*} [MeasurableSpace Ω]
    [Fintype J] [Nonempty J] {μ : Measure Ω} [IsProbabilityMeasure μ]
    {n : ℕ} (hn : 0 < n) {Y : J → Fin n → Ω → ℝ} {risk : J → ℝ} {L δ : ℝ}
    (hL : 0 < L) (hδ : δ ∈ Ioo (0:ℝ) 1)
    (hY : ∀ j i, Measurable (Y j i)) (hindep : ∀ j, iIndepFun (Y j) μ)
    (hbound : ∀ j i, ∀ᵐ ω ∂μ, Y j i ω ∈ Icc 0 L)
    (hmean : ∀ j i, (∫ ω, Y j i ω ∂μ)=risk j)
    (width : J → ℕ) (ε : ℝ) :
    let r := validationRadius n L (confidenceLog (Fintype.card J) δ)
    let good := {ω | ∀ j, |empiricalMean (Y j) ω-risk j| < r}
    1-δ ≤ μ.real good ∧
    ∀ ω ∈ good, (∃ k, risk k+2*r ≤ ε) →
      ∃ j, empiricalMean (Y j) ω+r ≤ ε ∧ risk j ≤ ε ∧
        (∀ k, empiricalMean (Y k) ω+r ≤ ε → width j ≤ width k) ∧
        (∀ k, risk k+2*r ≤ ε → width j ≤ width k) := by
  dsimp only
  constructor
  · exact simultaneous_validation_good_probability hn hL hδ hY hindep hbound hmean
  · intro ω hgood hmargin
    let r := validationRadius n L (confidenceLog (Fintype.card J) δ)
    let empirical : J → ℝ := fun j ↦ empiricalMean (Y j) ω
    have hgle : ∀ j, |empirical j-risk j| ≤ r := fun j ↦ (hgood j).le
    have hpass := passing_exists_of_risk_margin empirical risk r ε hgle hmargin
    let j := chooseSmallest width (fun k ↦ empirical k+r) ε hpass
    have hsel := chooseSmallest_spec width (fun k ↦ empirical k+r) ε hpass
    have hs := selected_risk_and_width width empirical risk r ε hgle hpass
    exact ⟨j,hsel.1,hs.1,hsel.2,hs.2⟩

end QIRadon.StatisticalSelection
