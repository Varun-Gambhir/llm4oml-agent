# ============================================================================
# File: src/evaluators/base_evaluator.py
# ============================================================================
"""Base evaluator interface."""

from abc import ABC, abstractmethod
from typing import Optional
from langchain_core.messages import HumanMessage
from ..models.schemas import EvaluationMetrics


class BaseEvaluator(ABC):
    """Abstract base class for proof evaluators."""
    
    def __init__(self, model_name: str, judge_id: str):
        self.model_name = model_name
        self.judge_id = judge_id
    
    @abstractmethod
    def evaluate(
        self,
        proof: str,
        feedback: Optional[str] = None,
        iteration: int = 1
    ) -> EvaluationMetrics:
        """
        Evaluate a convergence proof.
        
        Args:
            proof: The proof text to evaluate
            feedback: Previous feedback (for re-evaluation)
            iteration: Current iteration number
            
        Returns:
            EvaluationMetrics with scores and error sets
        """
        pass
