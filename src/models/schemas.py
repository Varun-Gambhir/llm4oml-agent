# ============================================================================
# File: src/models/schemas.py
# ============================================================================
"""Pydantic models for structured outputs and validation."""

from pydantic import BaseModel, Field, validator
from typing import Set, List, Optional


class EvaluationMetrics(BaseModel):
    """Structured output for evaluation step."""
    hallucination_error: bool = Field(description="Is there a Hallucination Error (HA)?")
    missing_step: bool = Field(description="Is there a Missing Step (MS)?")
    operator_error: bool = Field(description="Is there an Operator Error (OP)?")
    completeness_score: int = Field(ge=0, le=5, description="Score 0-5")
    assumption_use_score: int = Field(ge=0, le=5, description="Score 0-5")
    overall_verdict: str = Field(description="PASS, PASS_MINOR, CONDITIONAL, FAIL, REJECT")
    detailed_feedback: str = Field(description="Full text analysis and feedback")
    
    # v2.0: Step-level error tracking
    hallucination_steps: Set[int] = Field(default_factory=set)
    missing_step_indices: Set[int] = Field(default_factory=set)
    operator_error_steps: Set[int] = Field(default_factory=set)
    assumption_violation_steps: Set[int] = Field(default_factory=set)
    flagged_steps: Set[int] = Field(default_factory=set, 
                                    description="Steps explicitly flagged for correction")
    
    @validator('overall_verdict')
    def validate_verdict(cls, v):
        allowed = {"PASS", "PASS_MINOR", "CONDITIONAL", "FAIL", "REJECT"}
        if v not in allowed:
            raise ValueError(f"Verdict must be one of {allowed}")
        return v


class CorrectionMetricsOutput(BaseModel):
    """Output from correction metrics computation."""
    error_resolution_rate: float = Field(ge=0, le=1, description="ERR")
    regression_penalty: float = Field(ge=0, description="RP") 
    targeted_fix_precision: float = Field(ge=0, le=1, description="TFP")
    correction_reasoning_score: float = Field(ge=0, le=1, description="CRS (clipped)")
    crs_raw: float = Field(description="CRS before clipping")
    is_mixed_transition: bool = Field(default=False, description="Both fixes and new errors")

    
    errors_fixed: int
    errors_introduced: int
    flagged_fixed: int
    total_flagged: int


class JudgeEvaluation(BaseModel):
    """Single judge's evaluation."""
    judge_id: str
    model_name: str
    metrics: EvaluationMetrics
    response_time: float = Field(description="Time taken in seconds")


class ConsensusEvaluation(BaseModel):
    """Consensus from multiple judges."""
    num_judges: int
    evaluations: List[JudgeEvaluation]
    
    # Consensus metrics
    mean_completeness: float
    mean_assumption_score: float
    consensus_verdict: str
    
    # Judge reliability
    judge_reliability_scores: List[float]
    outlier_judges: List[str] = Field(default_factory=list)
    
    # Aggregated error sets
    consensus_error_set: dict  # Errors agreed upon by majority
    
    weighted_cfrs: float = Field(description="Weighted consensus CFRS")
