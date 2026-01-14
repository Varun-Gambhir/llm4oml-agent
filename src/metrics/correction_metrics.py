# ============================================================================
# File: src/metrics/correction_metrics.py
# ============================================================================
"""Correction Reasoning Score (CRS) computation."""

from typing import Set
import numpy as np
from ..models.types import ErrorSet, CorrectionMetrics
from ..models.schemas import CorrectionMetricsOutput
from .error_tracker import ErrorTracker


class CorrectionMetricsCalculator:
    """Computes ERR, RP, TFP, and overall CRS."""
    
    def __init__(
        self,
        w_err: float = 0.5,
        w_rp: float = 0.3,
        w_tfp: float = 0.2
    ):
        """
        Initialize with weights for CRS computation.
        
        Args:
            w_err: Weight for Error Resolution Rate
            w_rp: Weight for Regression Penalty (negative impact)
            w_tfp: Weight for Targeted Fix Precision
        """
        self.w_err = w_err
        self.w_rp = w_rp
        self.w_tfp = w_tfp
        self.tracker = ErrorTracker()
    
    def compute_err(self, prev_errors: Set[int], curr_errors: Set[int]) -> float:
        """
        Error Resolution Rate: fraction of previous errors that were fixed.
        
        ERR = |E^(k) \ E^(k+1)| / |E^(k)|
        """
        if len(prev_errors) == 0:
            return 1.0  # No errors to fix
        
        fixed_errors = prev_errors - curr_errors
        return len(fixed_errors) / len(prev_errors)
    
    def compute_rp(self, prev_errors: Set[int], curr_errors: Set[int], total_steps: int) -> float:
        """
        Regression Penalty: fraction of new errors introduced.
        
        RP = |E^(k+1) \ E^(k)| / T^(k+1)
        """
        if total_steps == 0:
            return 0.0
        
        new_errors = curr_errors - prev_errors
        return len(new_errors) / total_steps
    
    def compute_tfp(self, flagged_steps: Set[int], prev_errors: Set[int], 
                     curr_errors: Set[int]) -> float:
        """
        Targeted Fix Precision: fraction of flagged steps that were fixed.
        
        TFP = |F^(k) \ E^(k+1)| / |F^(k)|
        """
        if len(flagged_steps) == 0:
            return 1.0  # Nothing flagged
        
        fixed_flagged = flagged_steps - curr_errors
        return len(fixed_flagged) / len(flagged_steps)
    
    def compute_crs(
        self,
        prev_error_set: ErrorSet,
        curr_error_set: ErrorSet,
        flagged_steps: Set[int],
        total_steps_current: int
    ) -> CorrectionMetricsOutput:
        """
        Compute full Correction Reasoning Score.
        
        CRS = w_err * ERR - w_rp * RP + w_tfp * TFP
        """
        tracker = ErrorTracker()
        prev_all = tracker.get_all_errors(prev_error_set)
        curr_all = tracker.get_all_errors(curr_error_set)
        
        err = self.compute_err(prev_all, curr_all)
        rp = self.compute_rp(prev_all, curr_all, total_steps_current)
        tfp = self.compute_tfp(flagged_steps, prev_all, curr_all)
        
        crs = self.w_err * err - self.w_rp * rp + self.w_tfp * tfp
        crs = np.clip(crs, 0.0, 1.0)  # Bound to [0, 1]
        
        return CorrectionMetricsOutput(
            error_resolution_rate=err,
            regression_penalty=rp,
            targeted_fix_precision=tfp,
            correction_reasoning_score=crs,
            errors_fixed=len(prev_all - curr_all),
            errors_introduced=len(curr_all - prev_all),
            flagged_fixed=len(flagged_steps - curr_all),
            total_flagged=len(flagged_steps)
        )
