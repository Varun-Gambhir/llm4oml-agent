# ============================================================================
# File: src/graph/nodes.py
# ============================================================================
"""Node implementations for LangGraph workflow."""

from typing import Dict, Any
from ..models.types import AgentState
from ..agents.prover_agent import ProverAgent
from ..evaluators.multi_judge import MultiJudgeEvaluator
from ..evaluators.single_judge import SingleJudge
from ..metrics.correction_metrics import CorrectionMetricsCalculator
from ..metrics.error_tracker import ErrorTracker
from ..utils.state import StateManager
from ..utils.parsers import ResponseParser


class WorkflowNodes:
    """Contains all node logic for the proof workflow."""
    
    def __init__(
        self,
        prover_model: str,
        evaluator_models: list,
        use_multi_judge: bool = True,
        temperature: float = 0.5
    ):
        self.prover = ProverAgent(prover_model, temperature=temperature)
        
        if use_multi_judge:
            self.evaluator = MultiJudgeEvaluator(evaluator_models, temperature=temperature)
        else:
            self.evaluator = SingleJudge(evaluator_models[0], "single_judge", temperature)
        
        self.use_multi_judge = use_multi_judge
        self.correction_calc = CorrectionMetricsCalculator()
        self.error_tracker = ErrorTracker()
    
    def prover_node(self, state: AgentState) -> Dict[str, Any]:
        """Generate or correct proof."""
        result = self.prover.execute(
            algorithm=state["algorithm_description"],
            assumptions=state["assumptions"],
            iteration=state["iteration"],
            previous_proof=state.get("current_proof", ""),
            feedback=state.get("feedback", "")
        )
        return result
    
    def evaluator_node(self, state: AgentState) -> Dict[str, Any]:
        """Evaluate proof with single or multiple judges."""
        print(f"[EvaluatorNode] Evaluating proof (Iteration {state['iteration']})...")
        
        current_proof = state["current_proof"]
        iteration = state["iteration"]
        feedback = state.get("feedback", "")
        
        if self.use_multi_judge:
            return self._multi_judge_evaluation(current_proof, feedback, iteration, state)
        else:
            return self._single_judge_evaluation(current_proof, feedback, iteration, state)
    
    def _single_judge_evaluation(
        self,
        proof: str,
        feedback: str,
        iteration: int,
        state: AgentState
    ) -> Dict[str, Any]:
        """Evaluate with single judge."""
        metrics = self.evaluator.evaluate(proof, feedback, iteration)
        
        # Extract error set
        error_set = {
            "hallucinations": metrics.hallucination_steps,
            "missing_steps": metrics.missing_step_indices,
            "operator_errors": metrics.operator_error_steps,
            "assumption_violations": metrics.assumption_violation_steps
        }
        
        # Compute CRS if not first iteration
        correction_metrics = None
        if iteration > 1 and "error_set_previous" in state:
            total_steps = len(proof.split('\n'))  # Simplified
            correction_metrics = self.correction_calc.compute_crs(
                state["error_set_previous"],
                error_set,
                metrics.flagged_steps,
                total_steps
            )
        
        return {
            "feedback": metrics.detailed_feedback,
            "metrics": {
                "HA": metrics.hallucination_error,
                "MS": metrics.missing_step,
                "OP": metrics.operator_error,
                "completeness_score": metrics.completeness_score,
                "assumption_use_score": metrics.assumption_use_score
            },
            "verdict": metrics.overall_verdict,
            "error_set_current": error_set,
            "correction_metrics": correction_metrics.dict() if correction_metrics else None,
            "iteration": iteration
        }
    
    def _multi_judge_evaluation(
        self,
        proof: str,
        feedback: str,
        iteration: int,
        state: AgentState
    ) -> Dict[str, Any]:
        """Evaluate with multiple judges."""
        consensus = self.evaluator.evaluate_parallel(proof, feedback, iteration)
        
        # Extract consensus error set
        consensus_errors = consensus.consensus_error_set.get("consensus_errors", set())
        error_set = {
            "hallucinations": set(),  # Can be refined based on error type
            "missing_steps": set(),
            "operator_errors": set(),
            "assumption_violations": consensus_errors  # Simplified
        }
        
        # Compute CRS if not first iteration
        correction_metrics = None
        if iteration > 1 and "error_set_previous" in state:
            # Use first judge's flagged steps as reference
            flagged = consensus.evaluations[0].metrics.flagged_steps
            total_steps = len(proof.split('\n'))
            
            correction_metrics = self.correction_calc.compute_crs(
                state["error_set_previous"],
                error_set,
                flagged,
                total_steps
            )
        
        # Aggregate feedback from all judges
        all_feedback = "\n\n---JUDGE CONSENSUS---\n\n"
        all_feedback += f"Consensus Verdict: {consensus.consensus_verdict}\n"
        all_feedback += f"Mean Completeness: {consensus.mean_completeness:.2f}\n"
        all_feedback += f"Outlier Judges: {consensus.outlier_judges}\n\n"
        
        for eval_result in consensus.evaluations:
            all_feedback += f"\n--- {eval_result.judge_id} ({eval_result.model_name}) ---\n"
            all_feedback += eval_result.metrics.detailed_feedback
        
        return {
            "feedback": all_feedback,
            "metrics": {
                "HA": any(e.metrics.hallucination_error for e in consensus.evaluations),
                "MS": any(e.metrics.missing_step for e in consensus.evaluations),
                "OP": any(e.metrics.operator_error for e in consensus.evaluations),
                "completeness_score": round(consensus.mean_completeness),
                "assumption_use_score": round(consensus.mean_assumption_score),
                "weighted_cfrs": consensus.weighted_cfrs
            },
            "verdict": consensus.consensus_verdict,
            "error_set_current": error_set,
            "correction_metrics": correction_metrics.dict() if correction_metrics else None,
            "judge_evaluations": [e.dict() for e in consensus.evaluations],
            "judge_reliability": consensus.judge_reliability_scores,
            "iteration": iteration
        }
