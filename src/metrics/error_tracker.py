# ============================================================================
# File: src/metrics/error_tracker.py
# ============================================================================
"""Step-level error set tracking and management."""

from typing import Set
from ..models.types import ErrorSet


class ErrorTracker:
    """Tracks error sets across iterations."""
    
    def __init__(self):
        self.history: list[ErrorSet] = []
    
    def add_iteration(self, error_set: ErrorSet) -> None:
        """Add error set for current iteration."""
        self.history.append(error_set)
    
    def get_all_errors(self, error_set: ErrorSet) -> Set[int]:
        """Get union of all error types."""
        return (
            error_set["hallucinations"] |
            error_set["missing_steps"] |
            error_set["operator_errors"] |
            error_set["assumption_violations"]
        )
    
    def compute_error_delta(self, prev: ErrorSet, curr: ErrorSet) -> tuple[Set[int], Set[int]]:
        """
        Compute errors fixed and errors introduced.
        
        Returns:
            (fixed_errors, new_errors)
        """
        prev_all = self.get_all_errors(prev)
        curr_all = self.get_all_errors(curr)
        
        fixed = prev_all - curr_all
        introduced = curr_all - prev_all
        
        return fixed, introduced