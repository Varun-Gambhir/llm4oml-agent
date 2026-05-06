-- Minimal helper imports and small conveniences for ProofChecks
import Mathlib

namespace ProofChecks

/-! A tiny helper alias to make norm squared notation explicit for generated proofs. -/
abbrev norm_sq {E : Type _} [NormedAddCommGroup E] (x : E) : ℝ := ‖x‖ ^ 2

end ProofChecks
