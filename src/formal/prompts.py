# src/formal/prompts.py

CLAIM_EXTRACTION_PROMPT = """
You are extracting key mathematical claims from a convergence proof for formal verification.

ALGORITHM: {algorithm}

PROOF (LaTeX):
{proof}

Extract the 3–6 most important mathematical claims that could be stated as formal theorems.
Focus on:
1. The main convergence rate theorem (e.g. E[f(w_T)] - f* ≤ C/√T)
2. Key intermediate inequalities (descent lemma, bounded variance, etc.)
3. Any assumption that the proof crucially depends on

Format as a numbered list. Be precise and mathematical.
Do NOT include proof steps — only theorem statements.
"""


LEAN_TRANSLATION_PROMPT = """
You are translating mathematical claims into Lean 4 theorem statements with mathlib.

ALGORITHM: {algorithm}

CLAIMS TO FORMALIZE:
{claims}

Write a complete Lean 4 file that:
1. Imports from Mathlib (use standard imports below)
2. Declares each claim as a `theorem` or `lemma`
3. Proves each theorem with real Lean code; do NOT use `sorry`
4. If a claim cannot be proved cleanly, omit it or rewrite it so it can be proved
5. Uses realistic types: ℝ for real numbers, ℕ for iteration counts,
   functions like `f : ℝ → ℝ` for objectives

Standard imports to include:
```
import Mathlib.Analysis.MeanInequalities
import Mathlib.Analysis.InnerProductSpace.Basic
import Mathlib.Topology.MetricSpace.Basic
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
```

CRITICAL RULES:
- Do NOT include any extra `import` statements. Use ONLY the 4 imports provided above.
- Every proof body must be a complete Lean proof with no `sorry`
- Use `∀`, `∃`, `∧`, `→` for quantifiers (not English)
- Use `Finset.sum` for finite sums ∑
- Use `|·|` for absolute value, `‖·‖` for norms
- If a claim cannot be typed cleanly or proved, write a comment `-- UNTYPEABLE: <reason>`
  and skip it

Return ONLY valid Lean 4 code, no explanations.
"""
