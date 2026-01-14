# ============================================================================
# File: src/utils/prompts.py
# ============================================================================
"""Prompt templates for the proof generation and evaluation system."""

from typing import Optional


class PromptManager:
    """Manages all prompt templates."""
    
    # Original prompts from your code
    GENERATE_PROMPT = r"""
## System Context
You are an expert in optimization algorithms and mathematical proofs. Please reason step-by-step and show your work carefully.

## Reference Examples
Here are examples of algorithm descriptions and their convergence theories and proofs to use as templates:

### Example 1: Stochastic Gradient Descent (SGD)
*Algorithm Description:*
•⁠  Update rule: $w_{{t+1}} = w_t - \eta_t g_t$
•⁠  $g_t = \nabla f(w_t; z_t)$ is a stochastic gradient computed on a random sample $z_t$ at time $t$
•⁠  Learning rate schedule: fixed ($\eta_t = \eta$) or decreasing (e.g., $\eta_t = \frac{{c}}{{\sqrt{{t}}}}$)

*Convergence Proof:*
Assumptions:
1.Objective function $f(w)$ is convex and differentiable
2.Gradients bounded: $\|\nabla f(w)\| \leq G$
3.Stochastic gradients unbiased: $\mathbb{{E}}[g_t | w_t] = \nabla f(w_t)$
4.Variance bound: $\mathbb{{E}}[\|g_t - \nabla f(w_t)\|^2] \leq \sigma^2$

Derivation:
From convexity:
$$f(w_t) - f(w^) \leq \langle \nabla f(w_t), w_t - w^ \rangle$$

Using update rule and expanding squared norm:
$$\|w_{{t+1}} - w^\|^2 = \|w_t - \eta_t g_t - w^\|^2 = \|w_t - w^\|^2 - 2\eta_t \langle g_t, w_t - w^ \rangle + \eta_t^2 \|g_t\|^2$$

Taking expectation:
$$\mathbb{{E}}[\|w_{{t+1}} - w^\|^2] \leq \mathbb{{E}}[\|w_t - w^\|^2] - 2\eta_t \mathbb{{E}}[f(w_t) - f(w^*)] + \eta_t^2 (\mathbb{{E}}[\|\nabla f(w_t)\|^2] + \sigma^2)$$

Final convergence rate: $\mathbb{{E}}[f(\bar{{w}}_T) - f(w^*)] = O\left(\frac{{1}}{{\sqrt{{T}}}}\right)$

### Example 2: Adam
*Algorithm Description:*
•⁠  First moment estimate: $m_t = \beta_1 m_{{t-1}} + (1 - \beta_1) g_t$
•⁠  Second moment estimate: $v_t = \beta_2 v_{{t-1}} + (1 - \beta_2) g_t^2$
•⁠  Bias correction: $\hat{{m}}_t = \frac{{m_t}}{{1 - \beta_1^t}}$, $\hat{{v}}_t = \frac{{v_t}}{{1 - \beta_2^t}}$
•⁠  Update rule: $w_{{t+1}} = w_t - \eta \cdot \frac{{\hat{{m}}_t}}{{\sqrt{{\hat{{v}}_t}} + \epsilon}}$

*Convergence Proof:*
Shows sublinear convergence: $\frac{{1}}{{T}} \sum_{{t=1}}^T \mathbb{{E}}[\|\nabla f(w_t)\|^2] \leq O\left(\frac{{\log T}}{{T}}\right)$

---

## Main Analysis Task

*ALGORITHM TO ANALYZE:*

{algorithm}

*ASSUMPTIONS:*

{assumptions}

## Required Analysis Components

### 1. Algorithm Details Expansion
Expand the algorithm description into a complete formal specification:

*Required Output:*
1.*Formal Algorithm Description*:
   - Precise mathematical formulation
   - Update rules and equations
   - Hyperparameters and their roles
   
2.*Pseudocode*:
   - Implementable pseudocode with clear steps
   - Include initialization, iteration, and termination conditions
   
3.*Dry Run Example*:
   - Apply the algorithm to a simple quadratic function: f(w) = (w - 3)²
   - Show 3 iterations with:
     - Initialization: w₀ = 0
     - Step size: η = 0.1
     - Minibatch size: 1
   - Show calculations for each step:
     - Gradient computation
     - Parameter update  
     - Objective value

*Structure:*
latex
\section*{{Algorithm Name}}
\section*{{Mathematical Formulation}}
\section*{{Pseudocode}}
\section*{{Dry Run Example}}


### 2. Convergence Theory Development
Develop a rigorous convergence theory:

*Required Output:*
1.*Formal Statement of Assumptions*:
   - Precise mathematical definitions
   - Parameter constraints
   - Function properties

2.*Convergence Theorems*:
   - Rate of convergence (e.g., O(1/T), linear, etc.)
   - Expected reduction properties
   - Special case behaviors

3.*Key Theoretical Properties*:
   - Stability analysis
   - Sensitivity to hyperparameters
   - Comparison to baseline methods

*Structure:*
latex
\section*{{Assumptions}}
\section*{{Main Convergence Theorem}}
\section*{{Theoretical Properties}}
\section*{{Discussion}}


### 3. Complete Convergence Proof
Provide a COMPLETE, STEP-BY-STEP convergence proof with:

*Required Elements:*
•⁠ Full mathematical expansions of all terms
•⁠ Explicit derivation of inequalities
•⁠ Detailed expectation calculations
•⁠ Complete algebraic manipulations
•⁠ Justification for every inequality/approximation

*Structure:*
latex
\section*{{Preliminaries (Definitions and Lemmas)}}
\section*{{Main Proof (Step-by-Step Derivation)}}
\section*{{Rate Derivation (With Constants)}}
\section*{{Special Cases}}


*IMPORTANT Requirements:*
•⁠ Derive everything from first principles
•⁠ Never skip algebraic steps
•⁠ Maintain strict adherence to the initial assumptions
•⁠ Use the reference examples as templates for mathematical rigor and presentation style

provide complete latex code and code only
"""
    
    EVALUATE_PROMPT = r"""
You are an expert at extracting convergence rates from optimization algorithm convergence proofs. 
        
        Below are examples of convergence rate extractions:
        
        Example 1: "Therefore, we have established that E[f(w_T) - f(w*)] ≤ O(1/√T)"
        Answer: O(1/√T)
        
        Example 2: "The regret bound gives us R(T) = O(log T), implying O(log T / T) convergence"
        Answer: O(log T / T)
        
        Example 3: "Under strong convexity, we achieve linear convergence: E[||w_t - w*||²] ≤ ρᵗ||w_0 - w*||²"
        Answer: ρᵗ (linear convergence)
        
        Example 4: "The analysis shows sublinear convergence at rate O(1/T)"
        Answer: O(1/T)
        
        Now, extract the convergence rate from the convergence proof in the pdf

Task: Evaluate whether a convergence proof properly utilizes all stated assumptions and whether 
        the reasoning includes non-trivial claims made without justification.
        
        Instructions:
        1. Carefully read the assumptions and the convergence proof.
        2. Identify whether the proof includes:
           - Any non-obvious mathematical claims stated without justification
           - Missing connections between assumptions and proof steps
           - Logical gaps in inequality derivations or expectation calculations
        3. All significant transformations—especially involving inequalities, bounds, or expectation operations—must be supported by:
           - Clear algebraic manipulation
           - Well-known inequalities or theorems (Jensen's, Hölder's, etc.)
           - Proper use of convexity, smoothness, or other stated assumptions
        4. Check if optimization-specific techniques are properly applied:
           - If Lipschitz continuity is assumed, verify it's used correctly for bounding gradients
           - If convexity is assumed, check that convex inequalities are properly applied
           - If smoothness is assumed, verify gradient Lipschitz properties are used appropriately
        
        Output Format:
        <Analysis>: Step-by-step explanation of whether the reasoning properly uses assumptions and is logically sound.
        <Flagged Reasoning Step (if applicable)>: Quote specific step(s) where unjustified claims or assumption misuse occurred.
        <Answer>: True or False. True if the reasoning is valid and properly uses assumptions; False if it contains unjustified steps.
        
        Key Focus Areas:
        1. Are smoothness assumptions (L-smoothness) properly used for gradient bounds?
        2. Are convexity assumptions correctly applied in inequality derivations?
        3. Are stochastic assumptions (unbiased gradients, bounded variance) properly utilized?
        4. Are all major algebraic steps in expectation calculations justified?

Task: Evaluate the mathematical rigor of a convergence proof for an optimization algorithm, 
        focusing on whether the reasoning relies on oversimplified cases or inappropriate approximations.
        
        Instructions:
        1. Carefully read the convergence proof.
        2. Identify whether the proof relies on:
           - Toy cases (e.g., setting learning rate to specific values, assuming gradients are constant)
           - Special parameter choices that don't generalize
           - Asymptotic behavior arguments without proper limiting analysis
           - Numerical approximations in place of exact mathematical expressions
        3. Flag these issues:
           - Using specific parameter values (like η = 0.1) to conclude general convergence behavior
           - Assuming "small enough" or "large enough" parameters without formal conditions
           - Replacing exact expressions with decimal approximations in derivations
           - Hand-waving through expectation calculations or probability arguments
        4. Acceptable practices (don't flag):
           - Using specific values for illustration after proving general result
           - Asymptotic notation (O, Ω, Θ) when properly derived
           - Standard optimization inequalities (AM-GM, Jensen's, Cauchy-Schwarz)
        Output Format: 
        <Analysis>: Explanation of whether the proof maintains mathematical rigor or relies on oversimplified reasoning.
        <Flagged Reasoning Step (if applicable)>: Quote specific step(s) where inappropriate simplifications occurred.
        <Answer>: True or False. True if the reasoning is mathematically rigorous; False if it contains oversimplifications.
        
        Key Considerations:
        1. General parameter analysis vs. specific parameter choices
        2. Proper limiting behavior vs. hand-waved asymptotics  
        3. Exact mathematical expressions vs. numerical approximations
        4. Rigorous probability/expectation arguments vs. informal reasoning

Task: Evaluate the correctness of computational steps in a convergence proof for an optimization algorithm.
        Focus on verifying mathematical computations and algebraic manipulations.
        
        Instructions:
        1. Identify all verifiable computational expressions in the proof, including:
           - Algebraic manipulations of inequalities
           - Expectation calculations with specific bounds
           - Series summations and their bounds
           - Norm computations and their properties
        2. Do not extract:
           - Purely symbolic manipulations without numerical content
           - General theoretical statements
           - Asymptotic expressions without explicit bounds
        3. Focus on expressions that can be mathematically verified, such as:
           - Specific inequality bounds with numerical constants
           - Expectation calculations with concrete expressions
           - Summation evaluations
           - Norm bound calculations
        Output Format:
        <Analysis>: Explain which computational steps were identified and why they need verification.
        <Expressions>: List all computational expressions that can be verified.
        <Verification_Notes>: For each expression, note what mathematical property or calculation it represents.
        
        Key Computational Areas:
        1. Gradient bound calculations
        2. Expectation and variance computations  
        3. Telescoping sum evaluations
        4. Convergence rate derivations with explicit constants
        5. Norm and inner product calculations

Task: Evaluate the completeness and structure of a convergence proof for an optimization algorithm.
        
        Instructions:
        1. Check if the proof includes all essential components:
           - Clear problem setup and algorithm description
           - Proper use of all stated assumptions
           - Step-by-step derivation from algorithm update to convergence bound
           - Final convergence rate or guarantee
        2. Evaluate proof structure:
           - Logical flow from assumptions to conclusion
           - Appropriate use of mathematical tools (inequalities, expectations, etc.)
           - Clear intermediate steps and their justifications
        3. Identify missing elements:
           - Undefined notation or variables
           - Skipped algebraic steps in key derivations
           - Missing connections between algorithm properties and convergence analysis
           - Incomplete treatment of stochastic elements (if applicable)

        Output Format:
        <Completeness_Analysis>: Assess what components are present/missing in the proof structure.
        <Structure_Quality>: Evaluate the logical flow and organization of the proof.
        <Missing_Elements>: List any critical missing components or unclear aspects.
        <Overall_Score>: Rate completeness from 1-5 (1=very incomplete, 5=comprehensive).
        
        Focus Areas:
        1. Does the proof clearly connect algorithm updates to convergence analysis?
        2. Are all assumptions properly introduced and utilized?
        3. Is the final convergence guarantee clearly stated and derived?
        4. Are intermediate bounds and inequalities properly justified?

now verify the proof in the pdf and provide all these metrics
1 Hallucination Error (HA) (yes/no)
2 Missing Step (MS) (yes/no)
3 Operator Error (OP) (yes/no)
4 Completeness Score (05)
5 Assumption Use Score (0-5)
6 Any fix required (None, Minor, Major, Rewrite)
"""
    
    CORRECT_PROMPT = r"""
You are a mathematical rigor specialist. Your task is to **correct and improve** a convergence proof based on detailed evaluation feedback. The proof is from a LaTeX document on deep learning/optimization convergence analysis.

---

## Input Materials

### 1. Original Proof Document
{previous_proof}

### 2. Evaluation Feedback
{feedback}

---

## Your Objectives

### Primary Goal
Generate a **corrected LaTeX document** that:
1. **Fixes all identified logical gaps** (especially Missing Steps - MS, Operator Errors - OP)
2. **Maintains all correct portions** of the original proof
3. **Preserves the document structure** (sections, theorem numbering, formatting)
4. **Adds missing lemmas/inequalities** where needed
5. **Ensures mathematical rigor** throughout

### Specific Requirements

#### A. Address Flagged Issues
For each issue marked with ❗ in the evaluation:
- **If MS (Missing Step):** Add the missing derivation, lemma, or intermediate inequality
- **If OP (Operator Error):** Correct the mathematical statement to match what is actually proved
- **If HA (Hallucination):** Remove or replace with correct content

#### B. Handle the Evaluation Sections

1. **Convergence Rate Issues**
   - If stated rate ≠ proved rate: Either fix the proof to support the stated rate OR adjust the theorem statement to match the proved rate
   - Add any missing bridging inequalities

2. **Assumption Usage**
   - Verify all assumptions (A1, A2, ...) are used correctly
   - If an assumption is cited but not properly applied, add the missing steps

3. **Logical Gaps**
   - For the specific flagged reasoning step: provide a complete derivation
   - Example: If evaluation says "Y= inf_t Σp_i alpha_{{t,i}}" is stated but "Y = min_i p_i · inf_t min_i alpha_{{t,i}}" is proved, you must either:
     - Add a lemma showing the sum-based bound, OR
     - Change the theorem statement to use the product-based bound

4. **Computational Verification**
   - Review all numbered expressions in <Expressions> section
   - Ensure each inequality chain is complete with no skipped steps

5. **Completeness**
   - Add any missing elements identified in <Missing_Elements>
   - Ensure all claims in the main theorems are supported by preceding lemmas

#### C. Maintain Document Quality
- Keep the same LaTeX structure (sections, algorithms, theorems, lemmas)
- Preserve all correct dry-run examples
- Maintain consistent notation throughout
- Keep helpful commentary (e.g., "Discussion" sections) unless incorrect

---

## Output Format

Provide the corrected proof as a **complete, compilable LaTeX document** with:

### 1. Document Header
```latex
% CORRECTED VERSION - [Brief summary of main fixes]
% Changes made:
% - [List 2-3 key corrections]
```

### 2. Full Document Content
- Include all sections from the original
- Mark substantial changes with comments like `% CORRECTED:` or `% ADDED:`
- For modified theorems/lemmas, include a comment explaining the change

### 3. Change Summary at End
```latex
% ===== CORRECTION SUMMARY =====
% 1. [Issue 1]: [How it was fixed]
% 2. [Issue 2]: [How it was fixed]
% ...
```

---

## Example Correction Pattern

**Original (with gap):**
```latex
\begin{{theorem}}
Under assumptions A1-A5 with $\gamma := \inf_t \sum_i p_i \alpha_{{t,i}}$, we have
$$\mathbb{{E}}[f(w_t) - f_\star] \leq (1-\mu\gamma)^t (f(w_0)-f_\star).$$
\end{{theorem}}

\begin{{proof}}
From Lemma 2, we have ... [derivation leads to]
$$\sum_i p_i \alpha_{{t,i}} (\nabla_i f)^2 \geq \underline{{\alpha}} \underline{{p}} \|\nabla f\|^2$$
where $\underline{{\alpha}} := \inf_t \min_i \alpha_{{t,i}}$ and $\underline{{p}} := \min_i p_i$.
[Then claims the theorem follows...]
\end{{proof}}
```

**Corrected Option A (fix proof):**
```latex
% CORRECTED: Added Lemma 3.5 to bridge gap
\begin{{lemma}}\label{{lem:sum_bound}}
If $\alpha_{{t,i}} = \eta/L_i$ for some $\eta > 0$, then
$$\sum_i p_i \alpha_{{t,i}} (\nabla_i f)^2 \geq \left(\sum_i p_i \alpha_{{t,i}}\right) \cdot \frac{{1}}{{L_{{\max}}}} \|\nabla f\|^2.$$
\end{{lemma}}
\begin{{proof}}[Proof of Lemma~\ref{{lem:sum_bound}}]
[Full derivation...]
\end{{proof}}

\begin{{theorem}}
[Same statement as before]
\end{{theorem}}

\begin{{proof}}
From Lemma 2 and Lemma~\ref{{lem:sum_bound}}, we have... [now derivation is complete]
\end{{proof}}
```

**Corrected Option B (fix statement):**
```latex
% CORRECTED: Theorem statement adjusted to match proved bound
\begin{{theorem}}
Under assumptions A1-A5 with $\underline{{p}} := \min_i p_i$ and $\underline{{\alpha}} := \inf_t \min_i \alpha_{{t,i}}$, we have
$$\mathbb{{E}}[f(w_t) - f_\star] \leq (1-\mu \underline{{p}}\underline{{\alpha}})^t (f(w_0)-f_\star).$$
\end{{theorem}}

\begin{{proof}}
[Same proof as before, now matches statement]
\end{{proof}}
```

---

## Quality Checklist

Before providing the corrected document, verify:
- [ ] All ❗ flagged issues from evaluation are addressed
- [ ] Every theorem statement is supported by its proof
- [ ] All lemma references are defined
- [ ] No new logical gaps introduced
- [ ] LaTeX syntax is correct (compiles without errors)
- [ ] Notation is consistent throughout
- [ ] Change summary clearly documents fixes

---

## Important Notes

1. **Prioritize mathematical correctness** over brevity
2. **When in doubt**, add the missing step rather than claim "it follows easily"
3. **Preserve original structure** - only modify what needs correction
4. **Be explicit** about which bound/rate is actually proved
5. If multiple correction approaches exist, choose the one requiring **minimal changes** to the original

---

## Begin Correction

Now, using the original proof and evaluation feedback provided above, generate the complete corrected LaTeX document.
"""
    
    VERIFY_PROMPT = r"""
Verify whether a corrected proof (V2) properly addresses the issues identified in the original evaluation (V1). Provide quantitative metrics and a clear assessment.

---

## Input Materials

### 1. Original Evaluation Report (V1)
{old_feedback}

### 2. Corrected Proof Document (V2)
{current_proof}

---

## Verification Requirements

### Part 1: Issue-by-Issue Check

For **each issue flagged with ❗ in V1 evaluation**, report:

```
### Issue #[N]: [Brief description]

**Original Problem:** [Quote from V1]
**Location:** [Theorem/Lemma]
**Type:** [MS/OP/HA]

**Status:** [Choose one]
- ✅ FIXED - Issue completely resolved
- ⚠️ PARTIALLY FIXED - Attempt made but gaps remain
- ❌ NOT FIXED - Issue unchanged
- 🆕 NEW ISSUE - Correction introduced different problem

**What Changed:** [Describe V2 modification]
**Assessment:** [Is the fix mathematically sound? Complete?]
**Remaining Concerns:** [Any lingering issues]
```

### Part 2: New Error Detection

Scan V2 for newly introduced problems:
- Logical gaps
- Incorrect inequalities
- Inconsistent notation
- Broken references
- Computational errors

Report each new issue with location and description.

---

## Scoring Metrics (Use Same Framework as V1)

### A. Error Resolution
```
Total V1 Issues: [N]
Fixed (✅): [N]
Partially Fixed (⚠️): [N]
Not Fixed (❌): [N]
New Issues (🆕): [N]

Resolution Rate: [% fixed]
Net Improvement: [(Fixed - New) / Total] x 100%
```

### B. Original Scores (0-5 scale)

| Metric | V1 Score | V2 Score | Change |
|--------|----------|----------|---------|
| **Hallucination Error (HA)** | [Yes/No] | [Yes/No] | [Better/Same/Worse] |
| **Missing Step (MS)** | [Yes/No] | [Yes/No] | [Better/Same/Worse] |
| **Operator Error (OP)** | [Yes/No] | [Yes/No] | [Better/Same/Worse] |
| **Completeness Score** | [0-5] | [0-5] | [Δ] |
| **Assumption Use Score** | [0-5] | [0-5] | [Δ] |
| **Overall Score** | [0-5] | [0-5] | [Δ] |

**Scoring Definitions:**
- **Completeness (0-5):** 5=no gaps, 4=minor omissions, 3=some incomplete, 2=major gaps, 1=mostly incomplete, 0=broken
- **Assumption Use (0-5):** 5=all correct, 4=minor issues, 3=some incorrect, 2=multiple errors, 1=fundamental misuse, 0=ignored
- **Overall (0-5):** 5=publication-ready, 4=solid, 3=acceptable with gaps, 2=significant issues, 1=major flaws, 0=unsound

### C. Critical Checks

**Convergence Rates:**
- Theorem 1: Stated [rate] ↔️ Proved [rate] → [✅ Match / ❌ Mismatch]
- Theorem 2: Stated [rate] ↔️ Proved [rate] → [✅ Match / ❌ Mismatch]
- Theorem 3: Stated [rate] ↔️ Proved [rate] → [✅ Match / ❌ Mismatch]

**Assumptions:**
- A1-A5 correctly used? [✅/❌]
- PL condition properly invoked? [✅/❌]
- Coordinate smoothness correctly applied? [✅/❌]

**Key Inequalities:**
- All steps in main derivations justified? [✅/❌]
- Missing intermediate bounds? [✅/❌]

---

## Final Assessment

### Overall Verdict (Choose ONE)
- 🟢 **PASS** - All critical issues fixed, ready for use
- 🟡 **PASS WITH MINOR REVISIONS** - Main issues fixed, minor polish needed
- 🟠 **CONDITIONAL** - Key fixes made but concerns remain
- 🔴 **FAIL** - Critical issues unresolved or new errors
- ⛔ **REJECT** - Fundamental problems, inadequate correction

### Justification
[2-3 sentences explaining verdict]

### Critical Remaining Issues
[List any blocking issues]

### Required Fixes (if not PASS)
1. [Specific action item]
2. [Another specific action item]

---

## Output Structure

1. **Executive Summary** (2-3 sentences)
2. **Issue-by-Issue Verification** (all V1 issues)
3. **New Errors** (problems introduced in V2)
4. **Scoring Metrics** (tables above)
5. **Final Assessment** (verdict + recommendations)

---

## Special Instructions
- Be rigorous: don't accept superficial fixes
- Verify new lemmas are mathematically correct
- Check that fixes don't just hide problems
- Confirm stated rates match proved rates

**Begin verification now.**
"""
    
    def __init__(self):
        pass
    
    def get_generation_prompt(self, algorithm: str, assumptions: str) -> str:
        """Get proof generation prompt."""
        return self.GENERATE_PROMPT.format(
            algorithm=algorithm,
            assumptions=assumptions
        )
    
    def get_initial_evaluation_prompt(self, proof: str) -> str:
        """Get initial evaluation prompt."""
        return self.EVALUATE_PROMPT.format(proof=proof)
    
    def get_correction_prompt(self, previous_proof: str, feedback: str) -> str:
        """Get correction prompt."""
        return self.CORRECT_PROMPT.format(
            previous_proof=previous_proof,
            feedback=feedback
        )
    
    def get_verification_prompt(self, old_feedback: str, current_proof: str) -> str:
        """Get verification prompt."""
        return self.VERIFY_PROMPT.format(
            old_feedback=old_feedback,
            current_proof=current_proof
        )
