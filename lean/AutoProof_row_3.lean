import ProofChecks.Lemmas

theorem auto_abs_nonneg (x : ℝ) : 0 ≤ |x| := by
  exact abs_nonneg x