# tests/test_correction_metrics_fix.py

def test_rp_normalization_fix():
    """Test that RP is properly normalized against previous error count."""
    from src.metrics.correction_metrics import CorrectionMetricsCalculator
    from src.models.types import ErrorSet
    
    calc = CorrectionMetricsCalculator()
    
    # Scenario: Fixed 3 errors, introduced 1 new error
    prev_error_set: ErrorSet = {
        "hallucinations": {1, 3},
        "missing_steps": {4},
        "operator_errors": {2},
        "assumption_violations": set()
    }
    
    curr_error_set: ErrorSet = {
        "hallucinations": {2},  # e2 persists
        "missing_steps": {5},    # e5 is new
        "operator_errors": set(),
        "assumption_violations": set()
    }
    
    result = calc.compute_crs(
        prev_error_set=prev_error_set,
        curr_error_set=curr_error_set,
        flagged_steps=set(),
        total_steps_current=40
    )
    
    # Verify RP is meaningful
    assert result.regression_penalty == 0.25, \
        f"Expected RP=0.25 (1 new / 4 prev), got {result.regression_penalty}"
    
    # Verify ERR
    assert result.error_resolution_rate == 0.75, \
        f"Expected ERR=0.75 (3 fixed / 4 prev), got {result.error_resolution_rate}"
    
    # Verify mixed transition detected
    assert result.is_mixed_transition, "Should detect mixed transition"
    
    # Verify errors tracked correctly
    assert result.errors_fixed == 3
    assert result.errors_introduced == 1
    
    print("✅ RP normalization fix validated!")