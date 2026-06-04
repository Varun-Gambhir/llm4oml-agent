# DSAA Paper Structure: A* Grade Scientific Framing

## Title Ideas
1. Quantifying Correction Dynamics in Iterative LLM Reasoning Systems
2. Regression-Aware Evaluation of Self-Correcting LLM Proof Agents
3. A Framework for Measuring and Optimizing Iterative Correction Behavior in Mathematical Reasoning

## Abstract
- **Context:** Automated reasoning systems increasingly rely on iterative self-correction, but transitions between proof states are rarely monotonic.
- **Problem:** Iterative correction in LLMs is unstable because correcting one step often introduces regressions elsewhere. Existing metrics evaluate final correctness but ignore the quality and dynamics of the reasoning trajectory itself.
- **Method:** We formalize a regression-aware modeling framework for iterative proof correction. We introduce the Regression-Aware Correction Score (RACS), which explicitly balances error resolution, targeted compliance, and regression penalization. We study these dynamics using an asymmetric multi-agent architecture (frontier provers guided by lightweight evaluator ensembles).
- **Findings / Contribution:** 
  1. We formally define and evaluate metric properties for tracking non-monotonic reasoning trajectories.
  2. A systematic sensitivity analysis reveals that strong regression penalties suppress productive corrective exploration, leading to oscillatory failure.
  3. Regression-aware scoring closely aligns with human judgment compared to naive correction metrics.

---

## 1. Introduction
- **The Problem:** Iterative reasoning instability. Fixing a proof step often breaks prior correct steps, but standard static evaluation fails to capture this.
- **The Shift:** Moving from evaluating "is the final proof correct?" to "how stable and productive is the trajectory?"
- **Our Approach:** Regression-Aware Iterative Evaluation.
- **Scientific Hypotheses:**
  - **H1:** Iterative multi-agent correction systematically improves proof quality over single-pass generation.
  - **H2:** Regression-aware scoring correlates more strongly with human judgment than naive metrics.
  - **H3:** Strong regression penalties reduce oscillatory correction trajectories but may suppress overall search.
  - **H4:** Balanced correction incentives outperform aggressive error-resolution strategies.

## 2. Related Work
- **Automated Theorem Proving:** State of LLM integration.
- **Self-Refine & Multi-Agent Reasoning:** Critiques and the limits of symmetric self-correction.
- **Trajectory Evaluation:** Evaluating intermediate steps and reasoning graphs over static outcomes.
- **Iterative Alignment:** Reward models and stepwise reinforcement.

## 3. Problem Definition: Formalizing Correction Dynamics
- Formal formulation of an iterative proof correction sequence.
- State transitions mapping: `FIXED`, `REGRESSED`, `MIXED`, and `UNCHANGED`.
- The necessity of modeling regressions (why monotonic improvement is a flawed assumption).

## 4. Proposed Metric: Regression-Aware Correction Score (RACS)
- *Formulation:* $RACS = (w_{err} \cdot ERR + w_{tfp} \cdot TFP) \cdot (1 - w_{rp\_pen} \cdot RP_{clamped})$
- **Formal Metric Properties:**
  - *Property 1 (Boundedness):* $0 \le RACS \le 1$
  - *Property 2 (Perfect Correction Optimality):* A flawless targeted fix with zero regressions yields $RACS = 1$.
  - *Property 3 (Regression Monotonicity):* Increasing regression strictly reduces RACS.
  - *Property 4 (Mixed Transition Sensitivity):* Simultaneous fixes and regressions yield intermediate, penalized scores.

## 5. Agentic Framework (Brief Instrumentation Detail)
- *Note: Framing this as the instrument for studying the dynamics, not the core contribution.*
- Asymmetric architecture: SOTA mathematical provers (e.g., DeepSeek-R1) separated from lightweight, deterministic evaluator ensembles (e.g., Llama-3.3-70B, GPT-OSS-120B).
- State tracking and orchestration for gathering empirical trajectory data.

## 6. Experimental Setup
- **Dataset / Tasks:** Optimization algorithm convergence proofs.
- **Baselines for Comparison:**
  1. Single-pass prover (isolates iterative gain).
  2. Standard Self-Refine (standard iterative baseline).
  3. Majority-vote CoT (ensemble baseline).
  4. Naive correction score (for metric comparison).
- **Human Evaluation Protocol:** Ground-truth scoring methodology (e.g., 20 proofs, graduate reviewers, rubric scoring) to establish Pearson/Spearman alignment.

## 7. Correction Dynamics Analysis (Results Part I)
- **Overall Quality Improvement (Testing H1):** Empirical gains of the iterative ensemble over baselines.
- **Correction Transition Sankey Diagram:** Visualizing state flow across iterations (`FIXED` -> `MIXED` -> `REGRESSED`, etc.).
- **Oscillation and Convergence (Testing H3):** Quantifying how often new errors appear and defining the "productive regression" phenomenon vs. unbounded oscillation.

## 8. Correction Incentive Sensitivity Analysis (Results Part II)
- *Reframing the ablation study as a scientific study of correction incentive structures.*
- Evaluating tradeoffs: Aggressive fixing (High ERR) vs Judge compliance (High TFP) vs Conservative refinement (High RP).
- **Key Figure:** Heatmap of $w_{err}$ vs $w_{rp\_pen}$ colored by Spearman correlation to human evaluation.
- **Validating H2 & H4:** Demonstrating that balanced correction incentives (e.g., $w_{err}=0.6, w_{tfp}=0.4, w_{rp\_pen}=0.2$) achieve optimal alignment with human judgment and system stability.

## 9. Limitations
- Focus on informal/natural language proofs rather than formal verification languages (Lean/Coq).
- Judge dependence: Framework is upper-bounded by the reliability of the evaluator ensemble.

## 10. Conclusion
- Final remarks emphasizing that understanding the *dynamics* of reasoning transitions is just as critical as raw generation capabilities for building reliable AI mathematical reasoners.