import QIRadon.Radon.CrossSection
import Mathlib.MeasureTheory.Measure.Lebesgue.VolumeOfBalls

/-! Direction-independent volume constants for the actual finite sections. -/
noncomputable section
open MeasureTheory Set Metric
namespace QIRadon.Radon
variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]
  [MeasurableSpace V] [BorelSpace V] [FiniteDimensional ℝ V]

/-- Exact unit-ball volume in dimension m. -/
def unitBallVolume (m : ℕ) : ℝ :=
  Real.sqrt Real.pi^m / Real.Gamma ((m:ℝ)/2+1)

theorem unitBallVolume_nonneg (m : ℕ) : 0 ≤ unitBallVolume m := by
  unfold unitBallVolume
  have hg := Real.Gamma_pos_of_pos (show 0 < (m:ℝ)/2+1 by positivity)
  positivity

/-- The transverse unit-ball volume is independent of direction. The approved
higher-dimensional theorem assumes d≥2, so the transverse space is nontrivial. -/
theorem hyperplane_unitBall_volume (θ : sphere (0:V) 1)
    (hd : 2 ≤ Module.finrank ℝ V) :
    volume.real (closedBall (0:Hyperplane θ) 1) = unitBallVolume (Module.finrank ℝ V-1) := by
  let : Nontrivial (Hyperplane θ) := Module.nontrivial_of_finrank_pos
    (show 0 < Module.finrank ℝ (Hyperplane θ) by rw [finrank_hyperplane θ]; omega)
  unfold Measure.real
  rw [InnerProductSpace.volume_closedBall, ENNReal.ofReal_one, one_pow, one_mul]
  change (ENNReal.ofReal (unitBallVolume (Module.finrank ℝ (Hyperplane θ)))).toReal = _
  rw [ENNReal.toReal_ofReal (unitBallVolume_nonneg (Module.finrank ℝ (Hyperplane θ)))]
  rw [finrank_hyperplane]

/-- The outer volume is equally explicit in terms of dimension and radius. -/
theorem outerBall_volume [Nontrivial V] {R : ℝ} (hR : 0 ≤ R) :
    volume.real (closedBall (0:V) R) = R^(Module.finrank ℝ V)*unitBallVolume (Module.finrank ℝ V) := by
  unfold Measure.real
  rw [InnerProductSpace.volume_closedBall, ENNReal.toReal_mul, ENNReal.toReal_pow,
    ENNReal.toReal_ofReal hR]
  change R^Module.finrank ℝ V*(ENNReal.ofReal (unitBallVolume (Module.finrank ℝ V))).toReal = _
  rw [ENNReal.toReal_ofReal (unitBallVolume_nonneg (Module.finrank ℝ V))]
end QIRadon.Radon
