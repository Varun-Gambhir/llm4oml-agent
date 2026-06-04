# Paper Outline: Heterogeneous Evaluator Ensembles for Iterative Proof Correction

## Title Ideas
1. Lightweight Evaluator Ensembles Guide Frontier Models in Iterative Proof Repair
2. Regression-Aware Iterative Correction of Convergence Proofs via Heterogeneous LLM Ensembles
3. Beyond Self-Correction: Guiding Mathematical Proof Generation with Evaluator Ensembles and Regression-Aware Metrics

## Abstract
- **Context:** Automated generation of optimization algorithm convergence proofs using LLMs.
- **Problem:** Frontier reasoning models often struggle with consistency, omissions, and hallucinations in long-form proofs. Naive self-correction often fails or leads to oscillation.
- **Method:** An iterative multi-agent framework separating "Proving" from "Judging". 
  - *Prover:* Frontier reasoning models (DeepSeek-R1) generate and refine the proof.
  - *Judges:* A heterogeneous ensemble of lightweight, deterministic evaluator models (GPT-OSS-120B, Llama-3.3-70b-instruct) verify specific proof steps.
  - *Metric:* Regression-Aware Correction Score (RACS).
- **Results / Contribution:** 
  1. Lightweight structured evaluators can reliably guide frontier provers to convergence (systems insight).
  2. RACS properly aligns with human judgment. 
  3. Moderate regression penalties encourage productive exploratory fixes while preventing unbounded oscillation.

---

## 1. Introduction
- Importance of mathematical optimization proofs (SGD, Adam, etc.)
- Limitations of current LLMs in producing zero-shot rigorous proofs.
- The misconception of symmetric self-correction ("big models checking big models").
- **Our Contribution:**
  - Asymmetric prover-judge architecture (frontier prover + diverse lightweight judges).
  - Novel quantitative tracking (RACS) to measure correction fidelity.
  - Behavioral sensitivity analysis demonstrating the necessity of "productive regressions" during iterative repair.

## 2. Related Work
- LLMs for formal mathematics and automated theorem proving.
- Multi-agent collaboration and iterative self-correction (debating LLMs, critic models).
- Evaluation of LLM generations and Judge Reliability.

## 3. System Architecture
### 3.1 Asymmetric Iterative Correction
- Distinguishing the Prover's role (creative, high temperature, deep reasoning) from the Judge's role (logical consistency, strict JSON formatting, deterministic).
- **Prover Models:** DeepSeek-R1, Qwen3.5 Reasoning.
- **Heterogeneous Judge Ensemble:** GPT-OSS-120B, Llama-3.3-70B-Instruct, Qwen3-32B-Instruct. Emphasizing diversity and evaluator discipline over raw mathematical ingenuity.

### 3.2 LangGraph Execution Workflow
- State machine transitions (Generation -> Evaluation -> Correction).
- Handling judge feedback and generating structured error sets ($HA, MS, OP$).
- Consensus formulation via Judge Reliability Score (JRS).

## 4. Methodology & Evaluation Metrics
### 4.1 Regression-Aware Correction Score (RACS)
- Formulation: $RACS = (w_{err} \cdot ERR + w_{tfp} \cdot TFP) \cdot (1 - w_{rp\_pen} \cdot RP_{clamped})$
- Deficiencies of naive correction tracking (ceiling artifacts in early versions).

### 4.2 Error Set Agreement & Judge Reliability
- Formulating Consensus Evaluation without diluting specific step-level fixes.

### 4.3 Behavioral Sensitivity Analysis of RACS
- *Objective:* Evaluate how RACS weighting influences convergence, correction stability, and human-alignment.
- *Experimental Setup:* Ablation study of 20 configurations over 63 algorithms (1260 runs).
- *Finding 1: Moderate regression penalties outperform strong penalties.* Overly aggressive regression penalization suppresses productive corrective exploration.
- *Finding 2: Balanced ERR/TFP weighting yields the most stable behavior.* (w_err=0.6, w_tfp=0.4).
- *Finding 3: Meaningful behavioral sensitivity.* RACS is highly sensitive to regression pressure.
- *Selection:* $w_{err}=0.6, w_{tfp}=0.4, w_{rp\_pen}=0.2$. Balances convergence stability and semantic correction fidelity.

## 5. Experiments
### 5.1 Baseline Comparison (Hypothesis H1)
- Iterative multi-agent correction systematically improves final proof quality over single-pass baseline.

### 5.2 Oscillation and Convergence Dynamics (Hypothesis H3)
- Impact of regression penalty on proof stability.
- Demonstrating that local instability / "exploratory fixes" often precede global convergence.

### 5.3 Judge Ensemble Performance
- Ablation on judge diversity vs. homogeneous frontier judges. 
- Demonstrative cost/efficiency wins without sacrificing mathematical rigor.

## 6. Results and Discussion
- Analysis of generated convergence proofs (SGD, Adam variations, etc.).
- Analysis of failures (when do lightweight judges fail to guide the prover?).

## 7. Conclusion
- Final remarks on system design for complex mathematical tasks relying on LLMs.
