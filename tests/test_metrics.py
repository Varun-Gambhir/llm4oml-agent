# 
# # Additional File: tests/test_metrics.py (example)
# ============================================================================
import pytest
from src.metrics.correction_metrics import CorrectionMetricsCalculator
from src.metrics.judge_metrics import JudgeReliabilityCalculator
from src.models.types import ErrorSet


class TestCorrectionMetrics:
    def test_err_computation(self):
        calc = CorrectionMetricsCalculator()
        
        prev_errors = {1, 2, 3, 4, 5}
        curr_errors = {1, 2}  # Fixed 3, 4, 5
        
        err = calc.compute_err(prev_errors, curr_errors)
        assert err == 0.6  # 3/5 errors fixed
    
    def test_rp_computation(self):
        calc = CorrectionMetricsCalculator()
        
        prev_errors = {1, 2, 3}
        curr_errors = {1, 2, 6, 7}  # Introduced 6, 7
        total_steps = 10
        
        rp = calc.compute_rp(prev_errors, curr_errors, total_steps)
        assert rp == 0.2  # 2/10
    
    def test_tfp_computation(self):
        calc = CorrectionMetricsCalculator()
        
        flagged = {2, 3, 4}
        prev_errors = {1, 2, 3, 4, 5}
        curr_errors = {1, 2, 5}  # Fixed 3, 4 which were flagged
        
        tfp = calc.compute_tfp(flagged, prev_errors, curr_errors)
        assert tfp == 2/3  # 2 out of 3 flagged were fixed


class TestJudgeMetrics:
    def test_esa_computation(self):
        calc = JudgeReliabilityCalculator()
        
        error_set_1: ErrorSet = {
            "hallucinations": {1, 2, 3},
            "missing_steps": {4, 5},
            "operator_errors": set(),
            "assumption_violations": set()
        }
        
        error_set_2: ErrorSet = {
            "hallucinations": {1, 2},
            "missing_steps": {4, 5, 6},
            "operator_errors": set(),
            "assumption_violations": set()
        }
        
        esa = calc.compute_esa(error_set_1, error_set_2)
        # Intersection: {1,2,4,5} = 4
        # Union: {1,2,3,4,5,6} = 6
        assert esa == 4/6
    
    def test_outlier_detection(self):
        calc = JudgeReliabilityCalculator()
        
        scores = [3.0, 3.2, 3.1, 3.0, 5.0]  # 5.0 is outlier
        z_scores = calc.compute_z_scores(scores)
        outliers = calc.detect_outliers(z_scores, threshold=2.0)
        
        assert 4 in outliers  # Index 4 (score 5.0) should be outlier
