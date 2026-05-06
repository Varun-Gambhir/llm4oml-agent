-- Minimal helper imports and small conveniences for ProofChecks
import Mathlib.Analysis.NormedSpace.Basic
import Mathlib.Analysis.InnerProductSpace.Basic
import Mathlib.Analysis.Convex.Basic
import Mathlib.Tactic.Basic

namespace ProofChecks

/-! A tiny helper alias to make norm squared notation explicit for generated proofs. -/
abbrev norm_sq {E : Type _} [NormedAddCommGroup E] (x : E) : ℝ := ‖x‖ ^ 2

end ProofChecks
