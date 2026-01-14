# ============================================================================
# File: src/models/types.py
# ============================================================================
"""Type definitions for the convergence proof agent."""

from typing import TypedDict, List, Dict, Set, Optional, Literal
from typing_extensions import NotRequired

ErrorType = Literal["hallucination", "missing_step", "operator_error", "assumption_violation"]
VerdictType = Literal["PASS", "PASS_MINOR", "CONDITIONAL", "FAIL", "REJECT"]


class ErrorSet(TypedDict):
    """Set of error step indices by type."""
    hallucinations: Set[int]  # H
    missing_steps: Set[int]   # M
    operator_errors: Set[int]  # O
    assumption_violations: Set[int]  # A


class IterationMetrics(TypedDict):
    """Metrics for a single iteration."""
    hallucination_error: bool
    missing_step: bool
    operator_error: bool
    completeness_score: int  # 0-5
    assumption_use_score: int  # 0-5
    error_set: ErrorSet
    flagged_steps: Set[int]  # Steps explicitly flagged by evaluator


class CorrectionMetrics(TypedDict):
    """Correction Reasoning Score components."""
    err: float  # Error Resolution Rate
    rp: float   # Regression Penalty
    tfp: float  # Targeted Fix Precision
    crs: float  # Overall Correction Reasoning Score


class JudgeMetrics(TypedDict):
    """Judge reliability metrics."""
    esa: float  # Error Set Agreement
    sc: float   # Score Consistency (lower is better)
    z_score: float  # Standardized deviation
    jrs: float  # Judge Reliability Score


class AgentState(TypedDict):
    """State for the LangGraph agent workflow."""
    algorithm_description: str
    assumptions: str
    current_proof: str
    previous_proof: str
    feedback: str
    iteration: int
    metrics: Dict
    verdict: VerdictType
    error_set_current: NotRequired[ErrorSet]
    error_set_previous: NotRequired[ErrorSet]
    correction_metrics: NotRequired[CorrectionMetrics]
    
    # Multi-judge fields
    judge_feedbacks: NotRequired[List[str]]
    judge_metrics: NotRequired[List[IterationMetrics]]
    judge_reliability: NotRequired[List[JudgeMetrics]]
    consensus_verdict: NotRequired[VerdictType]