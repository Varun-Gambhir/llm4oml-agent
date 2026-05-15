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

You are generating STRICT Lean 4 code.

Rules:
1. Output ONLY valid Lean 4 code.
2. Use ONLY Mathlib imports.
3. Never invent identifiers.
4. Every theorem must compile under:
   import Mathlib
5. DO NOT translate full proofs. Produce theorem statements only, with minimal
  proof skeletons. Every proof body must be exactly `by sorry`.
6. Do not explain anything.
7. Do not use markdown fences.
8. All variables must have explicit types.
9. Prefer simple theorem skeletons over complex proofs.
10. Never use undefined notation.

Standard imports to include:
```
import Mathlib.Analysis.MeanInequalities
import Mathlib.Analysis.InnerProductSpace.Basic
import Mathlib.Topology.MetricSpace.Basic
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
```

CRITICAL RULES:
- Do NOT include any extra `import` statements. Use ONLY the 4 imports provided above.
- Every proof body must be exactly `by sorry`
- Do NOT translate full proofs; return only theorem statements and minimal `by sorry` bodies.
- Use `∀`, `∃`, `∧`, `→` for quantifiers (not English)
- Use `Finset.sum` for finite sums ∑
- Use `|·|` for absolute value, `‖·‖` for norms
- If a claim cannot be typed cleanly, write a comment `-- UNTYPEABLE: <reason>`
  and skip it

Return ONLY valid Lean 4 code, no explanations.
"""
