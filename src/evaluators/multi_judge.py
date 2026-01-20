# ============================================================================
# File: src/evaluators/multi_judge.py
# ============================================================================
"""
Multi-judge ensemble evaluator with reliability-aware consensus.

This implementation is fully aligned with:
- schemas.py (ConsensusEvaluation)
- nodes.py / workflow.py expectations
- judge_metrics.py (JRS, ESA, z-score)
- correction_metrics.py (CRS handled elsewhere)
"""

from typing import List, Optional
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

from .single_judge import SingleJudge
from ..models.schemas import EvaluationMetrics, JudgeEvaluation, ConsensusEvaluation
from ..models.types import ErrorSet
from ..metrics.judge_metrics import JudgeReliabilityCalculator
from ..metrics.error_tracker import ErrorTracker


class MultiJudgeEvaluator:
    """Ensemble of multiple LLM judges with reliability-aware consensus."""

    def __init__(
        self,
        provider_name: str,
        judge_models: List[str],
        api_key: str = None,
        temperature: float = 0.5,
        max_tokens: int = 8192,
        timeout: int = 600,
        max_retries: int = 3,
        outlier_threshold: float = 2.0,
    ):
        self.judges = [
            SingleJudge(
                provider_name=provider_name,
                model_name=model,
                judge_id=f"judge_{i}",
                api_key=api_key,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
                max_retries=max_retries,
            )
            for i, model in enumerate(judge_models)
        ]

        self.reliability_calc = JudgeReliabilityCalculator()
        self.outlier_threshold = outlier_threshold

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    def evaluate_parallel(
        self,
        proof: str,
        feedback: Optional[str] = None,
        iteration: int = 1,
    ) -> ConsensusEvaluation:

        evaluations: List[JudgeEvaluation] = []

        with ThreadPoolExecutor(max_workers=len(self.judges)) as executor:
            futures = {
                executor.submit(
                    self._evaluate_single_judge,
                    judge,
                    proof,
                    feedback,
                    iteration,
                ): judge
                for judge in self.judges
            }

            for future in as_completed(futures):
                try:
                    evaluations.append(future.result())
                except Exception as e:
                    judge = futures[future]
                    print(f"[MultiJudgeEvaluator] {judge.judge_id} failed: {e}")

        if not evaluations:
            raise RuntimeError("All judges failed. Cannot form consensus.")

        return self._compute_consensus(evaluations)

    # ------------------------------------------------------------------
    # Judge execution
    # ------------------------------------------------------------------
    def _evaluate_single_judge(
        self,
        judge: SingleJudge,
        proof: str,
        feedback: Optional[str],
        iteration: int,
    ) -> JudgeEvaluation:
        import time

        start = time.time()
        metrics = judge.evaluate(proof, feedback, iteration)
        elapsed = time.time() - start

        return JudgeEvaluation(
            judge_id=judge.judge_id,
            model_name=judge.model_name,
            metrics=metrics,
            response_time=elapsed,
        )

    # ------------------------------------------------------------------
    # Consensus logic
    # ------------------------------------------------------------------
    def _compute_consensus(
        self, evaluations: List[JudgeEvaluation]
    ) -> ConsensusEvaluation:

        tracker = ErrorTracker()

        error_sets = [self._extract_error_set(e.metrics) for e in evaluations]
        completeness = [e.metrics.completeness_score for e in evaluations]
        assumption_use = [e.metrics.assumption_use_score for e in evaluations]

        # ------------------------------------------------------------------
        # Judge reliability (JRS)
        # ------------------------------------------------------------------
        combined_scores = [
            0.5 * c + 0.5 * a for c, a in zip(completeness, assumption_use)
        ]

        judge_metrics = self.reliability_calc.compute_jrs(
            error_sets=error_sets,
            scores=combined_scores,
        )

        jrs = np.array([jm["jrs"] for jm in judge_metrics])
        z_scores = [jm["z_score"] for jm in judge_metrics]

        # ------------------------------------------------------------------
        # Outlier detection
        # ------------------------------------------------------------------
        outliers = set(
            self.reliability_calc.detect_outliers(
                z_scores, threshold=self.outlier_threshold
            )
        )

        valid_indices = [i for i in range(len(evaluations)) if i not in outliers]
        if not valid_indices:
            valid_indices = list(range(len(evaluations)))

        # ------------------------------------------------------------------
        # Stable weights
        # ------------------------------------------------------------------
        weights = np.array([max(jrs[i], 1e-6) for i in valid_indices])
        weights = weights / weights.sum()

        # ------------------------------------------------------------------
        # Ordinal-aware verdict (weighted median)
        # ------------------------------------------------------------------
        verdict_to_score = {
            "PASS": 5,
            "PASS_MINOR": 4,
            "CONDITIONAL": 3,
            "FAIL": 2,
            "REJECT": 1,
        }

        scored = sorted(
            (
                verdict_to_score[evaluations[i].metrics.overall_verdict],
                weights[j],
            )
            for j, i in enumerate(valid_indices)
        )

        cumulative = 0.0
        consensus_score = 1
        for score, w in scored:
            cumulative += w
            if cumulative >= 0.5:
                consensus_score = score
                break

        consensus_verdict = self._score_to_verdict(consensus_score)

        # ------------------------------------------------------------------
        # Reporting statistics (static quality proxy)
        # ------------------------------------------------------------------
        mean_completeness = sum(
            weights[j] * completeness[i] for j, i in enumerate(valid_indices)
        )
        mean_assumption = sum(
            weights[j] * assumption_use[i] for j, i in enumerate(valid_indices)
        )

        # IMPORTANT:
        # This is NOT CRS. It is a static quality proxy only.
        weighted_cfrs = (mean_completeness + mean_assumption) / 10.0

        # ------------------------------------------------------------------
        # Consensus error set (majority vote)
        # ------------------------------------------------------------------
        consensus_error_set = self._compute_consensus_error_set(
            [error_sets[i] for i in valid_indices]
        )

        return ConsensusEvaluation(
            num_judges=len(evaluations),
            evaluations=evaluations,
            mean_completeness=mean_completeness,
            mean_assumption_score=mean_assumption,
            consensus_verdict=consensus_verdict,
            judge_reliability_scores=jrs.tolist(),
            outlier_judges=[evaluations[i].judge_id for i in outliers],
            consensus_error_set=consensus_error_set,
            weighted_cfrs=weighted_cfrs,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _extract_error_set(self, metrics: EvaluationMetrics) -> ErrorSet:
        return {
            "hallucinations": metrics.hallucination_steps,
            "missing_steps": metrics.missing_step_indices,
            "operator_errors": metrics.operator_error_steps,
            "assumption_violations": metrics.assumption_violation_steps,
        }

    def _score_to_verdict(self, score: int) -> str:
        if score >= 5:
            return "PASS"
        if score == 4:
            return "PASS_MINOR"
        if score == 3:
            return "CONDITIONAL"
        if score == 2:
            return "FAIL"
        return "REJECT"

    def _compute_consensus_error_set(self, error_sets: List[ErrorSet]) -> dict:
        tracker = ErrorTracker()

        all_errors = set()
        for es in error_sets:
            all_errors |= tracker.get_all_errors(es)

        threshold = len(error_sets) / 2
        consensus = set()

        for step in all_errors:
            count = sum(
                step in tracker.get_all_errors(es) for es in error_sets
            )
            if count > threshold:
                consensus.add(step)

        return {"consensus_errors": consensus}

