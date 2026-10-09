import QIRadon.Radon.SourceProfiles
import QIRadon.Radon.RotationEvaluation
import QIRadon.Quadrature.OrbitOrder

/-! The actual bounded complex orbit extension of the canonical centered angular
integrand, constructed from original target data. -/
noncomputable section
open MeasureTheory Set Metric Filter
open scoped FourierTransform SchwartzMap ContDiff Topology RealInnerProductSpace
namespace QIRadon.Radon.LocalAnalyticSource
open QIRadon.Radon QIRadon.ComplexRotation QIRadon.Casimir
variable {n : ℕ} (src : LocalAnalyticSource n)

def centeredOrbit (x : EuclideanSpace ℝ (Fin (n+2))) {l : ℕ}
    (pairs : Fin l → Fin (n+2) × Fin (n+2)) (θ : sphere (0:EuclideanSpace ℝ (Fin (n+2))) 1)
    (angles : Fin l → ℂ) : ℂ :=
  src.jointProfile pairs θ (angles,rotationEvaluation pairs θ.val x angles) -
    src.jointProfile pairs θ (angles,0)

private theorem zero_mem_profileTube : (0:ℂ) ∈ scalarTube src.params.T src.params.h :=
  ofReal_mem_scalarTube src.params.h_pos ⟨by linarith [src.params.T_pos],src.params.T_pos.le⟩

/-- The actual centered orbit family is jointly complex smooth on a common ℓ¹
angle budget, for every word length and every real unit direction. -/
theorem centeredOrbit_smooth (x : EuclideanSpace ℝ (Fin (n+2))) (hx : ‖x‖ ≤ 1) {l : ℕ}
    (pairs : Fin l → Fin (n+2) × Fin (n+2)) (θ : sphere (0:EuclideanSpace ℝ (Fin (n+2))) 1) :
    ContDiffOn ℂ ∞ (src.centeredOrbit x pairs θ) (rotationBudget l src.params.b) := by
  have hEval : ContDiffOn ℂ ∞ (rotationEvaluation pairs θ.val x) (rotationBudget l src.params.b) :=
    ((rotationEvaluation_entire pairs θ.val x).contDiffOn isOpen_univ.uniqueDiffOn).mono (subset_univ _)
  have hleft := (src.jointProfile_smooth pairs θ).comp (contDiffOn_id.prodMk hEval)
    (show MapsTo (fun a : Fin l → ℂ => (a,rotationEvaluation pairs θ.val x a))
      (rotationBudget l src.params.b) (rotationSectionDomain l src.params.T (2*src.params.h) src.params.η) from by
        intro a ha
        exact ⟨lt_of_lt_of_le ha src.params.orbit_rotation_margin,
          scalarTube_mono le_rfl (by linarith [src.params.h_pos])
            (rotationEvaluation_mem_scalarTube pairs θ x hx src.params.one_lt_T.le ha src.params.orbit_scalar_margin)⟩)
  have hright := (src.jointProfile_smooth pairs θ).comp (contDiffOn_id.prodMk contDiffOn_const)
    (show MapsTo (fun a : Fin l → ℂ => (a,(0:ℂ)))
      (rotationBudget l src.params.b) (rotationSectionDomain l src.params.T (2*src.params.h) src.params.η) from by
        intro a ha
        exact ⟨lt_of_lt_of_le ha src.params.orbit_rotation_margin,
          scalarTube_mono le_rfl (by linarith [src.params.h_pos]) src.zero_mem_profileTube⟩)
  exact hleft.sub hright

/-- A single explicit bound controls all words and directions, without a
word-length-dependent target assumption. -/
theorem centeredOrbit_bound (x : EuclideanSpace ℝ (Fin (n+2))) (hx : ‖x‖ ≤ 1) {l : ℕ}
    (pairs : Fin l → Fin (n+2) × Fin (n+2)) (θ : sphere (0:EuclideanSpace ℝ (Fin (n+2))) 1)
    {angles : Fin l → ℂ} (ha : angles ∈ rotationBudget l src.params.b) :
    ‖src.centeredOrbit x pairs θ angles‖ ≤ 2*src.profileBound := by
  have haη : angles ∈ rotationBudget l src.params.η := lt_of_lt_of_le ha src.params.orbit_rotation_margin
  have heval := rotationEvaluation_mem_scalarTube pairs θ x hx src.params.one_lt_T.le ha src.params.orbit_scalar_margin
  exact (norm_sub_le _ _).trans ((add_le_add
    (src.jointProfile_bound pairs θ (p := (angles,rotationEvaluation pairs θ.val x angles)) ⟨haη,heval⟩)
    (src.jointProfile_bound pairs θ (p := (angles,0)) ⟨haη,src.zero_mem_profileTube⟩)).trans_eq (by ring))

/-- At real angles, this is exactly the canonical centered angular function at
the direction reached by the actual rotation word. -/
theorem centeredOrbit_real (x : EuclideanSpace ℝ (Fin (n+2))) (hx : ‖x‖ ≤ 1) {l : ℕ}
    (pairs : Fin l → Fin (n+2) × Fin (n+2)) (θ θ' : sphere (0:EuclideanSpace ℝ (Fin (n+2))) 1)
    (angles : Fin l → ℝ) (ha : (fun i => (angles i:ℂ)) ∈ rotationBudget l src.params.b)
    (hword : rotationWord pairs (fun i => (angles i:ℂ)) (realEmbedding θ.val)=realEmbedding θ'.val) :
    src.centeredOrbit x pairs θ (fun i => (angles i:ℂ)) = (canonicalAngular src.extendedReal x θ':ℂ) := by
  have haη : (fun i => (angles i:ℂ)) ∈ rotationBudget l src.params.η :=
    lt_of_lt_of_le ha src.params.orbit_rotation_margin
  have hθ : totalRealRotationWord pairs angles θ.val=θ'.val :=
    realEmbedding.injective ((rotationWord_ofReal_total pairs angles θ.val).symm.trans hword)
  have hdir : isometryDirection (totalRealRotationWord pairs angles) θ=θ' := Subtype.ext hθ
  have hs : |inner ℝ θ'.val x| ≤ src.params.T := by
    have hi := norm_inner_le_norm (𝕜 := ℝ) θ'.val x
    rw [Real.norm_eq_abs, mem_sphere_zero_iff_norm.mp θ'.property, one_mul] at hi
    exact (hi.trans hx).trans src.params.one_lt_T.le
  unfold centeredOrbit
  have hz := src.jointProfile_real_agreement pairs θ angles haη (show |(0:ℝ)| ≤ src.params.T by
    simpa using src.params.T_pos.le)
  simp only [Complex.ofReal_zero] at hz
  rw [rotationEvaluation_ofReal, hθ, src.jointProfile_real_agreement pairs θ angles haη hs, hz, hdir]
  rw [← src.complexify_extendedReal]
  exact (canonicalAngular_eq_filtered src.extendedReal x θ').symm

/-- Original target data constructs the actual orbit certificate used by the
Casimir/spherical-harmonic error theorem. No angular regularity is a premise. -/
def boundedOrbit (x : EuclideanSpace ℝ (Fin (n+2))) (hx : ‖x‖ ≤ 1) :
    BoundedOrbitExtension (canonicalAngular src.extendedReal x) src.params.b (2*src.profileBound) :=
  boundedOrbitExtension_of_outer (canonicalAngular src.extendedReal x) src.params.b (2*src.profileBound)
    (fun _l pairs θ => src.centeredOrbit x pairs θ)
    (fun _l pairs θ => src.centeredOrbit_smooth x hx pairs θ)
    (fun _l pairs θ _z hz => src.centeredOrbit_bound x hx pairs θ hz)
    (fun _l pairs θ t θ' ht hword => src.centeredOrbit_real x hx pairs θ θ' t ht hword)

end QIRadon.Radon.LocalAnalyticSource
