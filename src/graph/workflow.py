# ============================================================================
# File: src/graph/workflow.py
# ============================================================================
"""LangGraph workflow definition."""

from langgraph.graph import StateGraph, END
from ..models.types import AgentState
from ..utils.state import StateManager
from .nodes import WorkflowNodes


class ProofWorkflow:
    """Convergence proof generation and evaluation workflow."""
    
    def __init__(
        self,
        prover_model: str = "openai/gpt-oss-120b",
        evaluator_models: list = None,
        use_multi_judge: bool = True,
        max_iterations: int = 3,
        temperature: float = 0.5
    ):
        if evaluator_models is None:
            evaluator_models = ["openai/gpt-oss-120b"]
        
        self.nodes = WorkflowNodes(
            prover_model=prover_model,
            evaluator_models=evaluator_models,
            use_multi_judge=use_multi_judge,
            temperature=temperature
        )
        
        self.max_iterations = max_iterations
        self.state_manager = StateManager()
        self.app = self._build_graph()
    
    def _build_graph(self):
        """Build the LangGraph workflow."""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("prover", self.nodes.prover_node)
        workflow.add_node("evaluator", self.nodes.evaluator_node)
        
        # Set entry point
        workflow.set_entry_point("prover")
        
        # Add edges
        workflow.add_edge("prover", "evaluator")
        workflow.add_conditional_edges(
            "evaluator",
            lambda state: self.state_manager.should_continue(state, self.max_iterations),
            {
                "correct": "prover",
                "end": END
            }
        )
        
        return workflow.compile()
    
    def run(self, algorithm: str, assumptions: str) -> dict:
        """
        Run the complete workflow.
        
        Returns:
            Final state with all iterations logged
        """
        initial_state = self.state_manager.initialize_state(algorithm, assumptions)
        
        final_state = None
        for output in self.app.stream(initial_state):
            final_state = output
        
        return final_state
