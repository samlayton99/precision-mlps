import QIRadon.Validation.Certification

/-! Finite-data, minimum-width selection with a deterministic uniform-error
certificate. The certificate is the actual maximum observed residual, plus
measurement uncertainty and a covering-radius regularity correction. -/

noncomputable section
namespace QIRadon.DeterministicSelection

variable {X ι J : Type*} [Fintype ι] [Nonempty ι]

/-- The largest residual on the supplied nonempty finite sample set. -/
def sampleResidual (sites : ι → X) (y : ι → ℝ) (Q : X → ℝ) : ℝ :=
  Finset.univ.sup' Finset.univ_nonempty (fun i ↦ |y i-Q (sites i)|)

theorem residual_le_sampleResidual (sites : ι → X) (y : ι → ℝ) (Q : X → ℝ) (i : ι) :
    |y i-Q (sites i)| ≤ sampleResidual sites y Q :=
  Finset.le_sup' (fun j ↦ |y j-Q (sites j)|) (Finset.mem_univ i)

theorem sampleResidual_le (sites : ι → X) (y : ι → ℝ) (Q : X → ℝ) {E : ℝ}
    (hE : ∀ i, |y i-Q (sites i)| ≤ E) : sampleResidual sites y Q ≤ E :=
  Finset.sup'_le _ _ (fun i _ ↦ hE i)

/-- All inputs are finite sample quantities or supplied quantitative regularity
bounds. No unknown true error is used in this certificate. -/
def sampleCertificate (sites : ι → X) (y : ι → ℝ) (Q : X → ℝ)
    (noise Lf LQ radius : ℝ) : ℝ :=
  sampleResidual sites y Q + noise + (Lf+LQ)*radius

variable [PseudoMetricSpace X]

theorem sampleCertificate_sound (D : Set X) (sites : ι → X) (y : ι → ℝ)
    (f Q : X → ℝ) (noise Lf LQ radius : ℝ)
    (hLf : 0 ≤ Lf) (hLQ : 0 ≤ LQ)
    (hsites : ∀ i, sites i ∈ D)
    (hcover : ∀ x ∈ D, ∃ i, dist x (sites i) ≤ radius)
    (hf : ∀ x ∈ D, ∀ z ∈ D, |f x-f z| ≤ Lf*dist x z)
    (hQ : ∀ x ∈ D, ∀ z ∈ D, |Q x-Q z| ≤ LQ*dist x z)
    (hnoise : ∀ i, |y i-f (sites i)| ≤ noise) :
    ∀ x ∈ D, |f x-Q x| ≤ sampleCertificate sites y Q noise Lf LQ radius :=
  QIRadon.sample_cover_error_bound D sites y f Q (sampleResidual sites y Q)
    noise Lf LQ radius hLf hLQ hsites hcover hf hQ hnoise
    (residual_le_sampleResidual sites y Q)

omit [PseudoMetricSpace X] in
/-- A candidate with true uniform error e passes whenever e plus the two
measurement-noise allowances and coverage correction fits the error budget. -/
theorem sampleCertificate_le_of_uniform_error (D : Set X) (sites : ι → X) (y : ι → ℝ)
    (f Q : X → ℝ) (noise Lf LQ radius e : ℝ)
    (hsites : ∀ i, sites i ∈ D)
    (hnoise : ∀ i, |y i-f (sites i)| ≤ noise)
    (herror : ∀ x ∈ D, |f x-Q x| ≤ e) :
    sampleCertificate sites y Q noise Lf LQ radius ≤ e+2*noise+(Lf+LQ)*radius := by
  have hr : sampleResidual sites y Q ≤ noise+e := by
    apply sampleResidual_le
    intro i
    calc
      |y i-Q (sites i)| = |(y i-f (sites i))+(f (sites i)-Q (sites i))| := by ring_nf
      _ ≤ |y i-f (sites i)|+|f (sites i)-Q (sites i)| := abs_add_le _ _
      _ ≤ noise+e := add_le_add (hnoise i) (herror _ (hsites i))
  unfold sampleCertificate
  linarith

/-- Select the smallest width among passing, actually computed sample
certificates. The selected candidate has true error at most epsilon and is no
wider than any tested candidate whose true error has the explicit margin.
Candidates may depend on these same samples; no independence is needed. -/
theorem exists_sample_selected_near_best [Fintype J]
    (D : Set X) (sites : ι → X) (y : ι → ℝ) (f : X → ℝ) (Q : J → X → ℝ)
    (width : J → ℕ) (LQ : J → ℝ) (noise Lf radius ε : ℝ)
    (hLf : 0 ≤ Lf) (hLQ : ∀ j, 0 ≤ LQ j)
    (hsites : ∀ i, sites i ∈ D)
    (hcover : ∀ x ∈ D, ∃ i, dist x (sites i) ≤ radius)
    (hf : ∀ x ∈ D, ∀ z ∈ D, |f x-f z| ≤ Lf*dist x z)
    (hQ : ∀ j, ∀ x ∈ D, ∀ z ∈ D, |Q j x-Q j z| ≤ LQ j*dist x z)
    (hnoise : ∀ i, |y i-f (sites i)| ≤ noise)
    (hpass : ∃ j, sampleCertificate sites y (Q j) noise Lf (LQ j) radius ≤ ε) :
    ∃ j,
      sampleCertificate sites y (Q j) noise Lf (LQ j) radius ≤ ε ∧
      (∀ x ∈ D, |f x-Q j x| ≤ ε) ∧
      (∀ k, sampleCertificate sites y (Q k) noise Lf (LQ k) radius ≤ ε →
        width j ≤ width k) ∧
      (∀ k e, (∀ x ∈ D, |f x-Q k x| ≤ e) →
        e+2*noise+(Lf+LQ k)*radius ≤ ε → width j ≤ width k) := by
  obtain ⟨j,hj,he,hmin⟩ := QIRadon.exists_smallest_certified_candidate D f Q width
    (fun j ↦ sampleCertificate sites y (Q j) noise Lf (LQ j) radius) ε
    (fun j ↦ sampleCertificate_sound D sites y f (Q j) noise Lf (LQ j) radius
      hLf (hLQ j) hsites hcover hf (hQ j) hnoise) hpass
  refine ⟨j,hj,he,hmin,?_⟩
  intro k e herror hmargin
  exact hmin k ((sampleCertificate_le_of_uniform_error D sites y f (Q k)
    noise Lf (LQ k) radius e hsites hnoise herror).trans hmargin)

omit [PseudoMetricSpace X] in
/-- A true-error margin supplies existence of a passing finite-data candidate;
there is no separate hypothesis that an observed certificate already passes. -/
theorem passing_exists_of_uniform_margin [Fintype J]
    (D : Set X) (sites : ι → X) (y : ι → ℝ) (f : X → ℝ) (Q : J → X → ℝ)
    (LQ : J → ℝ) (noise Lf radius ε : ℝ)
    (hsites : ∀ i, sites i ∈ D)
    (hnoise : ∀ i, |y i-f (sites i)| ≤ noise)
    (hfeasible : ∃ k e, (∀ x ∈ D, |f x-Q k x| ≤ e) ∧
      e+2*noise+(Lf+LQ k)*radius ≤ ε) :
    ∃ j, sampleCertificate sites y (Q j) noise Lf (LQ j) radius ≤ ε := by
  obtain ⟨k,e,he,hm⟩ := hfeasible
  exact ⟨k,(sampleCertificate_le_of_uniform_error D sites y f (Q k)
    noise Lf (LQ k) radius e hsites hnoise he).trans hm⟩

end QIRadon.DeterministicSelection
