import QIRadon.Radon.LocalSource
import QIRadon.Radon.SourceParameterChoice
import QIRadon.Radon.RadiusDensity

/-! Automatic construction of every boundary/margin ingredient from R>1 and
σ>0. The caller supplies only the known target and its bounded local analytic
extension; no radius density, profile, or angular certificate is an input. -/
set_option maxHeartbeats 800000
noncomputable section
open MeasureTheory Set Metric
open scoped ContDiff
namespace QIRadon.Radon.LocalAnalyticSource
open QIRadon.Radon QIRadon.ComplexRotation

/-- Fully specified source setup for the approved known-function construction. -/
def ofKnownTarget {n : ℕ} (R σ B : ℝ) (hR : 1 < R) (hσ : 0 < σ) (hB : 0 ≤ B)
    (f : EuclideanSpace ℝ (Fin (n+2)) → ℝ) (F : EuclideanSpace ℂ (Fin (n+2)) → ℂ)
    (hF : AnalyticOnNhd ℂ F (realBallTube realEmbedding R σ))
    (hagree : ∀ x, ‖x‖ ≤ R → F (realEmbedding x)=(f x:ℂ))
    (hbound : ∀ z ∈ realBallTube realEmbedding R σ, ‖F z‖ ≤ B) : LocalAnalyticSource n := by
  let p := standardSourceParameters R σ hR hσ
  exact {
    R := R
    σ := σ
    B := B
    params := p
    bound_nonneg := hB
    target := f
    complexTarget := F
    analytic := hF
    agrees := hagree
    bound := hbound
    radiusDensity := standardRadiusDensity p.R₀ R
    density_integrable := standardRadiusDensity_integrable p.R₀_lt_R
    density_support := standardRadiusDensity_support p.R₀_lt_R
    density_smooth := standardRadiusDensity_smooth p.R₀ R
    density_nonneg := fun a _ => standardRadiusDensity_nonneg p.R₀_lt_R a
    density_normalized := standardRadiusDensity_normalized p.R₀_lt_R }

end QIRadon.Radon.LocalAnalyticSource
