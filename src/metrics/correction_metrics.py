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
        r"""
        Error Resolution Rate: fraction of previous errors that were fixed.
        
        ERR = |E^(k) \ E^(k+1)| / |E^(k)|
        """
        if len(prev_errors) == 0:
            return 1.0  # No errors to fix
        
        fixed_errors = prev_errors - curr_errors
        return len(fixed_errors) / len(prev_errors)

    def compute_rp(self, prev_errors: Set[int], curr_errors: Set[int], total_steps: int) -> float:
        r"""
        Regression Penalty: fraction of new errors introduced relative to previous errors.
        
        RP = |E^(k+1) \ E^(k)| / max(|E^(k)|, 1)
        
        Note: Normalized against previous error count, not total steps,
        to make regression penalties meaningful even when total error count decreases.
        """
        new_errors = curr_errors - prev_errors
        
        if len(prev_errors) == 0:
            # No previous errors: any new error is severe
            return float(len(new_errors)) if new_errors else 0.0
        
        return len(new_errors) / len(prev_errors)
    
    def compute_tfp(self, flagged_steps: Set[int], prev_errors: Set[int], 
                     curr_errors: Set[int]) -> float:
        r"""
        Targeted Fix Precision: fraction of flagged steps that were fixed.
        
        TFP = |F^(k) \ E^(k+1)| / |F^(k)|
        """
        if len(flagged_steps) == 0:
            return 1.0  # Nothing flagged
        
        fixed_flagged = flagged_steps - curr_errors
        return len(fixed_flagged) / len(flagged_steps)
    
    def detect_mixed_transition(self, prev_errors: Set[int], curr_errors: Set[int]) -> bool:
        """
        Detect if both errors were fixed AND new errors were introduced.
        
        Returns:
            True if this is a mixed transition (progress + regression)
        """
        fixed = prev_errors - curr_errors
        introduced = curr_errors - prev_errors
        return bool(fixed) and bool(introduced)
    
    def compute_crs(
        self,
        prev_error_set: ErrorSet,
        curr_error_set: ErrorSet,
        flagged_steps: Set[int],
        total_steps_current: int
    ) -> CorrectionMetricsOutput:
        """Compute full Correction Reasoning Score."""
        tracker = ErrorTracker()
        prev_all = tracker.get_all_errors(prev_error_set)
        curr_all = tracker.get_all_errors(curr_error_set)
        
        err = self.compute_err(prev_all, curr_all)
        rp = self.compute_rp(prev_all, curr_all, total_steps_current)  # Uses new formula
        tfp = self.compute_tfp(flagged_steps, prev_all, curr_all)
        
        crs_raw = self.w_err * err - self.w_rp * rp + self.w_tfp * tfp
        crs = np.clip(crs_raw, 0.0, 1.0)
        
        # Detect mixed transitions
        is_mixed = self.detect_mixed_transition(prev_all, curr_all)
        
        if is_mixed:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(
                f"Mixed transition detected: "
                f"fixed {len(prev_all - curr_all)}, "
                f"introduced {len(curr_all - prev_all)}, "
                f"CRS={crs:.3f} (raw={crs_raw:.3f})"
            )
        
        return CorrectionMetricsOutput(
            error_resolution_rate=err,
            regression_penalty=rp,
            targeted_fix_precision=tfp,
            correction_reasoning_score=crs,
            crs_raw=crs_raw,  # NEW
            errors_fixed=len(prev_all - curr_all),
            errors_introduced=len(curr_all - prev_all),
            flagged_fixed=len(flagged_steps - curr_all),
            total_flagged=len(flagged_steps),
            is_mixed_transition=is_mixed  # NEW
        )