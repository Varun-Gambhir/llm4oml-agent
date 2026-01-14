# ============================================================================
# File: src/evaluators/multi_judge.py
# ============================================================================
"""Multi-judge ensemble evaluator."""

from typing import List, Optional
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

from .single_judge import SingleJudge
from ..models.schemas import EvaluationMetrics, JudgeEvaluation, ConsensusEvaluation
from ..models.types import ErrorSet
from ..metrics.judge_metrics import JudgeReliabilityCalculator
from ..metrics.error_tracker import ErrorTracker


class MultiJudgeEvaluator:
    """Ensemble of multiple LLM judges with consensus mechanism."""
    
    def __init__(
        self,
        judge_models: List[str],
        temperature: float = 0.5,
        max_tokens: int = 8192,
        outlier_threshold: float = 2.0
    ):
        """
        Initialize multi-judge evaluator.
        
        Args:
            judge_models: List of model names to use as judges
            temperature: Temperature for generation
            max_tokens: Max tokens per judge
            outlier_threshold: Z-score threshold for outlier detection
        """
        self.judges = [
            SingleJudge(
                model_name=model,
                judge_id=f"judge_{i}",
                temperature=temperature,
                max_tokens=max_tokens
            )
            for i, model in enumerate(judge_models)
        ]
        
        self.reliability_calc = JudgeReliabilityCalculator()
        self.outlier_threshold = outlier_threshold
    
    def evaluate_parallel(
        self,
        proof: str,
        feedback: Optional[str] = None,
        iteration: int = 1
    ) -> ConsensusEvaluation:
        """
        Evaluate proof using all judges in parallel.
        
        Returns consensus evaluation with judge reliability metrics.
        """
        judge_evaluations: List[JudgeEvaluation] = []
        
        # Evaluate in parallel
        with ThreadPoolExecutor(max_workers=len(self.judges)) as executor:
            futures = {
                executor.submit(
                    self._evaluate_single_judge,
                    judge,
                    proof,
                    feedback,
                    iteration
                ): judge
                for judge in self.judges
            }
            
            for future in as_completed(futures):
                try:
                    evaluation = future.result()
                    judge_evaluations.append(evaluation)
                except Exception as e:
                    judge = futures[future]
                    print(f"Judge {judge.judge_id} failed: {e}")
        
        # Compute consensus
        return self._compute_consensus(judge_evaluations)
    
    def _evaluate_single_judge(
        self,
        judge: SingleJudge,
        proof: str,
        feedback: Optional[str],
        iteration: int
    ) -> JudgeEvaluation:
        """Evaluate with single judge and track time."""
        import time
        start = time.time()
        
        metrics = judge.evaluate(proof, feedback, iteration)
        elapsed = time.time() - start
        
        return JudgeEvaluation(
            judge_id=judge.judge_id,
            model_name=judge.model_name,
            metrics=metrics,
            response_time=elapsed
        )
    
    def _compute_consensus(self, evaluations: List[JudgeEvaluation]) -> ConsensusEvaluation:
        """Compute consensus from multiple judge evaluations."""
        
        # Extract metrics and error sets
        error_sets = [self._extract_error_set(e.metrics) for e in evaluations]
        completeness_scores = [e.metrics.completeness_score for e in evaluations]
        assumption_scores = [e.metrics.assumption_use_score for e in evaluations]
        
        # Compute judge reliability
        judge_reliabilities = self.reliability_calc.compute_jrs(
            error_sets,
            completeness_scores
        )
        
        jrs_scores = [jm["jrs"] for jm in judge_reliabilities]
        z_scores = [jm["z_score"] for jm in judge_reliabilities]
        
        # Detect outliers
        outlier_indices = self.reliability_calc.detect_outliers(
            z_scores,
            self.outlier_threshold
        )
        outlier_judge_ids = [evaluations[i].judge_id for i in outlier_indices]
        
        # Compute weighted consensus (excluding outliers)
        non_outlier_indices = [i for i in range(len(evaluations)) if i not in outlier_indices]
        
        if not non_outlier_indices:
            # All judges are outliers - use all with equal weight
            non_outlier_indices = list(range(len(evaluations)))
        
        weights = np.array([jrs_scores[i] for i in non_outlier_indices])
        weights = weights / weights.sum()  # Normalize
        
        weighted_completeness = sum(
            w * completeness_scores[i]
            for i, w in zip(non_outlier_indices, weights)
        )
        
        weighted_assumption = sum(
            w * assumption_scores[i]
            for i, w in zip(non_outlier_indices, weights)
        )
        
        # Compute weighted CFRS (simplified - can be expanded)
        weighted_cfrs = (weighted_completeness + weighted_assumption) / 10.0
        
        # Consensus verdict via weighted voting
        verdict_scores = {"PASS": 5, "PASS_MINOR": 4, "CONDITIONAL": 3, "FAIL": 2, "REJECT": 1}
        verdicts = [e.metrics.overall_verdict for e in evaluations]
        
        weighted_verdict_score = sum(
            w * verdict_scores.get(verdicts[i], 0)
            for i, w in zip(non_outlier_indices, weights)
        )
        
        # Map back to verdict
        consensus_verdict = self._score_to_verdict(weighted_verdict_score)
        
        # Consensus error set (majority vote)
        consensus_error_set = self._compute_consensus_error_set(
            [error_sets[i] for i in non_outlier_indices]
        )
        
        return ConsensusEvaluation(
            num_judges=len(evaluations),
            evaluations=evaluations,
            mean_completeness=weighted_completeness,
            mean_assumption_score=weighted_assumption,
            consensus_verdict=consensus_verdict,
            judge_reliability_scores=jrs_scores,
            outlier_judges=outlier_judge_ids,
            consensus_error_set=consensus_error_set,
            weighted_cfrs=weighted_cfrs
        )
    
    def _extract_error_set(self, metrics: EvaluationMetrics) -> ErrorSet:
        """Extract error set from evaluation metrics."""
        return {
            "hallucinations": metrics.hallucination_steps,
            "missing_steps": metrics.missing_step_indices,
            "operator_errors": metrics.operator_error_steps,
            "assumption_violations": metrics.assumption_violation_steps
        }
    
    def _score_to_verdict(self, score: float) -> str:
        """Map weighted score back to verdict."""
        if score >= 4.5:
            return "PASS"
        elif score >= 3.5:
            return "PASS_MINOR"
        elif score >= 2.5:
            return "CONDITIONAL"
        elif score >= 1.5:
            return "FAIL"
        else:
            return "REJECT"
    
    def _compute_consensus_error_set(self, error_sets: List[ErrorSet]) -> dict:
        """Compute consensus error set via majority voting."""
        tracker = ErrorTracker()
        
        all_errors = set()
        for es in error_sets:
            all_errors.update(tracker.get_all_errors(es))
        
        # Majority vote for each error step
        threshold = len(error_sets) / 2
        consensus_errors = set()
        
        for error_step in all_errors:
            count = sum(
                1 for es in error_sets
                if error_step in tracker.get_all_errors(es)
            )
            if count > threshold:
                consensus_errors.add(error_step)
        
        return {"consensus_errors": consensus_errors}
