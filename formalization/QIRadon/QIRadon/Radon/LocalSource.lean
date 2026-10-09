import QIRadon.Radon.SourceParameters
import QIRadon.Radon.RotationFilter
import QIRadon.Radon.UniformVolume
import QIRadon.Radon.AngularContinuity

/-! Actual source data and its constructed compact extension. The only target
regularity input is a bounded local analytic extension of the supplied real
function. Profile analyticity and angular regularity are conclusions. -/
noncomputable section
open MeasureTheory Set Metric Filter
open scoped FourierTransform SchwartzMap ContDiff Topology
namespace QIRadon.Radon
open QIRadon.ComplexRotation

/-- Dimension is n+2, covering exactly the approved d≥2 source class. -/
structure LocalAnalyticSource (n : ℕ) where
  R : ℝ
  σ : ℝ
  B : ℝ
  params : SourceParameters R σ
  bound_nonneg : 0 ≤ B
  target : EuclideanSpace ℝ (Fin (n+2)) → ℝ
  complexTarget : EuclideanSpace ℂ (Fin (n+2)) → ℂ
  analytic : AnalyticOnNhd ℂ complexTarget (realBallTube realEmbedding R σ)
  agrees : ∀ x, ‖x‖ ≤ R → complexTarget (realEmbedding x)=(target x:ℂ)
  bound : ∀ z ∈ realBallTube realEmbedding R σ, ‖complexTarget z‖ ≤ B
  radiusDensity : ℝ → ℝ
  density_integrable : Integrable radiusDensity
  density_support : Function.support radiusDensity ⊆ Icc params.R₀ R
  density_smooth : ContDiff ℝ ∞ radiusDensity
  density_nonneg : ∀ a ∈ Icc params.R₀ R, 0 ≤ radiusDensity a
  density_normalized : (∫ a in Icc params.R₀ R, radiusDensity a)=1

namespace LocalAnalyticSource
variable {n : ℕ} (src : LocalAnalyticSource n)

/-- The compact extension is constructed from the source's fixed real radius
mixture; it is never assumed to be globally analytic. -/
def extendedComplex : 𝓢(EuclideanSpace ℝ (Fin (n+2)),ℂ) :=
  complexRadialExtensionSchwartz (realEmbedding (ι := Fin (n+2))) src.params.R₀_pos
    src.params.target_tube_pos src.analytic src.density_integrable src.density_support
    src.density_smooth src.density_normalized

def extendedReal : 𝓢(EuclideanSpace ℝ (Fin (n+2)),ℝ) :=
  SchwartzMap.postcompCLM (𝕜 := ℝ) Complex.reCLM src.extendedComplex

@[simp] theorem extendedComplex_apply (x : EuclideanSpace ℝ (Fin (n+2))) :
    src.extendedComplex x = src.complexTarget (realEmbedding x)*(radialCutoff src.radiusDensity src.params.R₀ src.R ‖x‖:ℂ) := rfl

@[simp] theorem extendedReal_apply (x : EuclideanSpace ℝ (Fin (n+2))) :
    src.extendedReal x=(src.extendedComplex x).re := rfl

theorem extendedComplex_eq_ofReal (x : EuclideanSpace ℝ (Fin (n+2))) :
    src.extendedComplex x=(src.extendedReal x:ℂ) := by
  rw [extendedReal_apply]
  by_cases hx : ‖x‖ ≤ src.R
  · rw [extendedComplex_apply, src.agrees x hx]
    simp
  · have heq : src.extendedComplex x=0 :=
      complexRadialExtension_eq_zero src.complexTarget realEmbedding src.radiusDensity (lt_of_not_ge hx)
    simp [heq]

theorem complexify_extendedReal : complexifySchwartz src.extendedReal=src.extendedComplex := by
  ext x
  exact (src.extendedComplex_eq_ofReal x).symm

/-- No approximation budget is spent on the finite boundary extension. -/
theorem extendedReal_eq_target {x : EuclideanSpace ℝ (Fin (n+2))} (hx : ‖x‖ ≤ src.params.R₀) :
    src.extendedReal x=src.target x := by
  rw [extendedReal_apply, extendedComplex_apply, radialCutoff_eq_one src.density_normalized hx,
    Complex.ofReal_one, mul_one, src.agrees x (hx.trans src.params.R₀_lt_R.le)]
  rfl

theorem extendedReal_eq_target_unitBall {x : EuclideanSpace ℝ (Fin (n+2))} (hx : ‖x‖ ≤ 1) :
    src.extendedReal x=src.target x :=
  src.extendedReal_eq_target (hx.trans (by linarith [src.params.one_lt_T, src.params.T_lt_S, src.params.S_lt_R₀]))

/-- Explicit common magnitude for all transverse projections. -/
def projectionBound : ℝ := unitBallVolume (n+1)*Real.sqrt (src.R^2+src.params.Q^2)^(n+1)*src.B

/-- Explicit finite Hilbert bound, including the outer real mass. -/
def hilbertBound : ℝ :=
  Real.pi⁻¹*(4*src.params.S*src.projectionBound/(4*src.params.h) +
    Real.log ((src.params.S+(src.params.T+2*src.params.h))/(src.params.S-(src.params.T+2*src.params.h)))*src.projectionBound +
    src.B*volume.real (closedBall (0:EuclideanSpace ℝ (Fin (n+2))) src.R)/(src.params.S-(src.params.T+2*src.params.h)))

/-- Magnitude of the exact cycles-Fourier normalization for derivative order n+1. -/
def filterFactor (_src : LocalAnalyticSource n) : ℝ :=
  (surfaceArea (EuclideanSpace ℝ (Fin (n+2)))/2)*((2*Real.pi)^(n+1))⁻¹

/-- Common magnitude of the canonical filtered profiles on the final scalar tube. -/
def profileBound : ℝ :=
  src.filterFactor*((n+1).factorial*(if Even (n+1) then src.projectionBound else src.hilbertBound)/src.params.h^(n+1))

theorem projectionBound_nonneg : 0 ≤ src.projectionBound := by
  unfold projectionBound
  exact mul_nonneg (mul_nonneg (unitBallVolume_nonneg _) (by positivity)) src.bound_nonneg

theorem hilbertBound_nonneg : 0 ≤ src.hilbertBound := by
  have hgap : 0 < src.params.S-(src.params.T+2*src.params.h) := by linarith [src.params.hilbert_margin]
  have hlog : 0 ≤ Real.log ((src.params.S+(src.params.T+2*src.params.h))/(src.params.S-(src.params.T+2*src.params.h))) := by
    apply Real.log_nonneg
    rw [le_div_iff₀ hgap]
    linarith [src.params.T_pos, src.params.h_pos]
  unfold hilbertBound
  have hS := src.params.S_pos
  have hh := src.params.h_pos
  have hB := src.bound_nonneg
  have hPB := src.projectionBound_nonneg
  positivity

theorem profileBound_nonneg : 0 ≤ src.profileBound := by
  have hc : 0 ≤ src.filterFactor := by
    unfold filterFactor
    have := surfaceArea_pos (V := EuclideanSpace ℝ (Fin (n+2)))
    positivity
  have hh := src.params.h_pos
  unfold profileBound
  apply mul_nonneg hc
  apply div_nonneg
  · apply mul_nonneg (by positivity)
    split_ifs
    · exact src.projectionBound_nonneg
    · exact src.hilbertBound_nonneg
  · positivity

/-- The common projection bound really equals each section's actual volume bound. -/
theorem projectionBound_eq (θ : sphere (0:EuclideanSpace ℝ (Fin (n+2))) 1) :
    src.projectionBound = volume.real (closedBall (0:Hyperplane θ) 1)*
      Real.sqrt (src.R^2+src.params.Q^2)^(Module.finrank ℝ (EuclideanSpace ℝ (Fin (n+2)))-1)*src.B := by
  rw [hyperplane_unitBall_volume θ (by simp)]
  simp [projectionBound]

end LocalAnalyticSource
end QIRadon.Radon
