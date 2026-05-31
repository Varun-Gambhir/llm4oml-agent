"""
Unit tests for the corrected CRS implementation (v3).

Run with: pytest tests/test_crs.py -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from src.metrics.correction_metrics import CorrectionMetricsCalculator
from src.models.schemas import EvaluationMetrics


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def calc():
    return CorrectionMetricsCalculator(w_err=0.50, w_tfp=0.50, w_rp_pen=0.40)


def make_error_set(hallucinations=(), missing_steps=(), operator_errors=(), assumption_violations=()):
    return {
        "hallucinations": set(hallucinations),
        "missing_steps": set(missing_steps),
        "operator_errors": set(operator_errors),
        "assumption_violations": set(assumption_violations),
    }


def make_metrics(**kwargs) -> EvaluationMetrics:
    defaults = dict(
        hallucination_error=False,
        missing_step=False,
        operator_error=False,
        completeness_score=3,
        assumption_use_score=3,
        correctness_verdict="PROBABLY_CORRECT",
        critical_errors=[],
        overall_verdict="CONDITIONAL",
        detailed_feedback="Test feedback",
        hallucination_steps=set(),
        missing_step_indices=set(),
        operator_error_steps=set(),
        assumption_violation_steps=set(),
        flagged_steps=set(),
    )
    defaults.update(kwargs)
    return EvaluationMetrics(**defaults)


# ============================================================================
# CRS formula correctness — the v2 ceiling bug must NOT appear
# ============================================================================

class TestCRSFormula:

    def test_perfect_correction_yields_1_0(self, calc):
        """
        Bug fix regression test:
        When all previous errors are fixed, no new errors introduced, all
        flagged steps resolved → CRS must equal 1.0 (not 0.7 as in v2).
        """
        prev = make_error_set(hallucinations={1, 2}, missing_steps={3})
        curr = make_error_set()                       # all errors gone
        flagged = {1, 2, 3}

        out = calc.compute_crs(prev, curr, flagged)

        assert out.error_resolution_rate == pytest.approx(1.0)
        assert out.regression_penalty == pytest.approx(0.0)
        assert out.targeted_fix_precision == pytest.approx(1.0)
        assert out.correction_reasoning_score == pytest.approx(1.0), (
            f"Expected CRS=1.0 for perfect correction, got {out.correction_reasoning_score:.4f}. "
            "The v2 ceiling bug (CRS always ~0.7) may have re-appeared."
        )

    def test_no_change_yields_intermediate_crs(self, calc):
        """If nothing changes, CRS should reflect partial quality."""
        prev = make_error_set(hallucinations={1}, missing_steps={2})
        curr = make_error_set(hallucinations={1}, missing_steps={2})  # same
        flagged = {1, 2}

        out = calc.compute_crs(prev, curr, flagged)

        assert out.error_resolution_rate == pytest.approx(0.0)
        assert out.regression_penalty == pytest.approx(0.0)   # no new errors
        assert out.targeted_fix_precision == pytest.approx(0.0)
        # quality = 0.5*0 + 0.5*0 = 0; penalty = 1 - 0.4*0 = 1; CRS = 0
        assert out.correction_reasoning_score == pytest.approx(0.0)

    def test_partial_fix_no_regression(self, calc):
        """Half the errors fixed, none introduced → CRS in (0, 1)."""
        prev = make_error_set(hallucinations={1, 2}, missing_steps={3, 4})
        curr = make_error_set(hallucinations={1}, missing_steps={3})  # {2,4} fixed
        flagged = {1, 2, 3, 4}

        out = calc.compute_crs(prev, curr, flagged)

        assert out.error_resolution_rate == pytest.approx(0.5)
        assert out.regression_penalty == pytest.approx(0.0)
        assert out.targeted_fix_precision == pytest.approx(0.5)
        # quality = 0.5*0.5 + 0.5*0.5 = 0.5; penalty factor = 1.0; CRS = 0.5
        assert out.correction_reasoning_score == pytest.approx(0.5)

    def test_regression_penalises_score(self, calc):
        """Introducing new errors should reduce CRS significantly."""
        prev = make_error_set(hallucinations={1, 2})
        curr = make_error_set(hallucinations={1, 2}, missing_steps={5, 6, 7})  # added 3 new
        flagged = set()

        out = calc.compute_crs(prev, curr, flagged)

        assert out.regression_penalty > 0
        assert out.correction_reasoning_score < 0.5, (
            "Score should be low when new errors are introduced without fixing any"
        )

    def test_mixed_transition_detected(self, calc):
        """Fixing some + introducing others = mixed transition flag."""
        prev = make_error_set(hallucinations={1, 2})
        curr = make_error_set(hallucinations={2}, missing_steps={5})  # fixed {1}, added {5}
        flagged = {1}

        out = calc.compute_crs(prev, curr, flagged)

        assert out.is_mixed_transition is True
        assert out.errors_fixed == 1
        assert out.errors_introduced == 1

    def test_empty_previous_errors_returns_full_err(self, calc):
        """With no previous errors, ERR = 1.0 (nothing to fix)."""
        prev = make_error_set()
        curr = make_error_set()
        flagged = set()

        out = calc.compute_crs(prev, curr, flagged)

        assert out.error_resolution_rate == pytest.approx(1.0)
        assert out.correction_reasoning_score == pytest.approx(1.0)

    def test_crs_always_in_unit_interval(self, calc):
        """CRS must always be clipped to [0, 1]."""
        cases = [
            (make_error_set(hallucinations={1}), make_error_set(hallucinations={1, 2, 3, 4, 5}), set()),
            (make_error_set(), make_error_set(), set()),
            (make_error_set(hallucinations={1, 2, 3}), make_error_set(), {1, 2, 3}),
        ]
        for prev, curr, flagged in cases:
            out = calc.compute_crs(prev, curr, flagged)
            assert 0.0 <= out.correction_reasoning_score <= 1.0, (
                f"CRS out of [0,1]: {out.correction_reasoning_score}"
            )


# ============================================================================
# First-iteration CRS (static quality proxy)
# ============================================================================

class TestFirstIterationCRS:

    def test_high_quality_proof_scores_high(self, calc):
        metrics = make_metrics(
            completeness_score=5,
            assumption_use_score=5,
            overall_verdict="PASS",
        )
        out = calc.compute_first_iteration_crs(metrics)
        assert out.correction_reasoning_score > 0.5

    def test_many_errors_reduces_first_crs(self, calc):
        """High error count should pull first-iteration CRS down."""
        metrics_clean = make_metrics(completeness_score=4, assumption_use_score=4)
        metrics_dirty = make_metrics(
            completeness_score=4, assumption_use_score=4,
            hallucination_steps={1, 2, 3, 4, 5},
            missing_step_indices={6, 7, 8, 9, 10},
            operator_error_steps={11, 12},
        )
        clean = calc.compute_first_iteration_crs(metrics_clean)
        dirty = calc.compute_first_iteration_crs(metrics_dirty)
        assert clean.correction_reasoning_score > dirty.correction_reasoning_score

    def test_first_iteration_crs_in_unit_interval(self, calc):
        for cs in range(6):
            for aus in range(6):
                m = make_metrics(completeness_score=cs, assumption_use_score=aus)
                out = calc.compute_first_iteration_crs(m)
                assert 0.0 <= out.correction_reasoning_score <= 1.0


# ============================================================================
# ERR / RP / TFP component tests
# ============================================================================

class TestComponents:

    def test_err_all_fixed(self, calc):
        assert calc.compute_err({1, 2, 3}, set()) == pytest.approx(1.0)

    def test_err_none_fixed(self, calc):
        assert calc.compute_err({1, 2}, {1, 2}) == pytest.approx(0.0)

    def test_err_half_fixed(self, calc):
        assert calc.compute_err({1, 2, 3, 4}, {1, 2}) == pytest.approx(0.5)

    def test_rp_none_introduced(self, calc):
        assert calc.compute_rp_normalised({1, 2}, {1}) == pytest.approx(0.0)

    def test_rp_introduced_clamped_to_1(self, calc):
        # Introducing way more than previously present → clamped to 1.0
        rp = calc.compute_rp_normalised({1}, {1, 2, 3, 4, 5, 6})
        assert rp == pytest.approx(1.0)

    def test_tfp_all_flagged_fixed(self, calc):
        assert calc.compute_tfp({1, 2, 3}, set()) == pytest.approx(1.0)

    def test_tfp_nothing_flagged(self, calc):
        assert calc.compute_tfp(set(), {1, 2}) == pytest.approx(1.0)

    def test_tfp_half_fixed(self, calc):
        assert calc.compute_tfp({1, 2, 3, 4}, {1, 2}) == pytest.approx(0.5)