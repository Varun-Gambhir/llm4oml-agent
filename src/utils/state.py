# ============================================================================
# File: src/utils/state.py
# ============================================================================
"""State management utilities."""

from typing import Dict, Any
from ..models.types import AgentState, ErrorSet


class StateManager:
    """Manages agent state transitions and updates."""
    
    @staticmethod
    def initialize_state(algorithm: str, assumptions: str) -> AgentState:
        """Create initial agent state."""
        return {
            "algorithm_description": algorithm,
            "assumptions": assumptions,
            "current_proof": "",
            "previous_proof": "",
            "feedback": "",
            "iteration": 0,
            "metrics": {},
            "verdict": "FAIL",
            "error_set_current": {
                "hallucinations": set(),
                "missing_steps": set(),
                "operator_errors": set(),
                "assumption_violations": set()
            },
            "error_set_previous": {
                "hallucinations": set(),
                "missing_steps": set(),
                "operator_errors": set(),
                "assumption_violations": set()
            }
        }
    
    @staticmethod
    def should_continue(state: AgentState, max_iterations: int = 3) -> str:
        """Determine next step in workflow."""
        verdict = state["verdict"]
        iteration = state["iteration"]
        
        if iteration > max_iterations:
            return "end"
        if verdict in ["PASS", "PASS_MINOR"]:
            return "end"
        if verdict == "FAIL" and "SYSTEM ERROR" in state["feedback"]:
            return "end"
        
        return "correct"
    
    @staticmethod
    def update_error_sets(state: AgentState, new_error_set: ErrorSet) -> Dict[str, Any]:
        """Update error sets in state."""
        return {
            "error_set_previous": state.get("error_set_current", new_error_set),
            "error_set_current": new_error_set
        }
