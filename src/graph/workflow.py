# ============================================================================
# File: src/graph/workflow.py (UPDATED)
# ============================================================================
"""LangGraph workflow with provider abstraction."""

from langgraph.graph import StateGraph, END
from ..models.types import AgentState
from ..utils.state import StateManager
from ..utils.execution_tracker import ExecutionTracker
from .nodes import WorkflowNodes


class ProofWorkflow:
    """Convergence proof generation and evaluation workflow."""
    
    def __init__(
        self,
        provider_name: str = "nvidia",
        prover_model: str = "openai/gpt-oss-120b",
        evaluator_models: list = None,
        api_key: str = None,
        use_multi_judge: bool = True,
        max_iterations: int = 3,
        temperature: float = 0.5,
        timeout: int = 600,
        max_retries: int = 3,
        output_dir: str = "output"
    ):
        if evaluator_models is None:
            evaluator_models = [prover_model]
        
        self.nodes = WorkflowNodes(
            provider_name=provider_name,
            prover_model=prover_model,
            evaluator_models=evaluator_models,
            api_key=api_key,
            use_multi_judge=use_multi_judge,
            temperature=temperature,
            timeout=timeout,
            max_retries=max_retries
        )
        
        self.max_iterations = max_iterations
        self.state_manager = StateManager()
        
        # Initialize tracker
        self.tracker = ExecutionTracker(output_dir=output_dir)
        
        # Store configuration
        self.config = {
            "provider": provider_name,
            "prover_model": prover_model,
            "evaluator_models": evaluator_models,
            "use_multi_judge": use_multi_judge,
            "max_iterations": max_iterations,
            "temperature": temperature,
            "timeout": timeout,
            "max_retries": max_retries
        }
        
        self.app = self._build_graph()
    
    def _build_graph(self):
        """Build the LangGraph workflow."""
        workflow = StateGraph(AgentState)
        
        # Add nodes with tracking wrappers
        workflow.add_node("prover", self._tracked_prover_node)
        workflow.add_node("evaluator", self._tracked_evaluator_node)
        
        workflow.set_entry_point("prover")
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
    
    def _tracked_prover_node(self, state: AgentState) -> dict:
        """Prover node with state tracking."""
        print(f"\n{'='*60}")
        print(f"ITERATION {state['iteration'] + 1}: PROOF GENERATION")
        print(f"{'='*60}")
        
        result = self.nodes.prover_node(state)
        
        # Track this state
        self.tracker.track_iteration(
            iteration=result["iteration"],
            node_name="prover",
            state={**state, **result}
        )
        
        print(f"✓ Proof generated ({len(result['current_proof'])} chars)")
        return result
    
    def _tracked_evaluator_node(self, state: AgentState) -> dict:
        """Evaluator node with state tracking."""
        print(f"\n{'='*60}")
        print(f"ITERATION {state['iteration']}: EVALUATION")
        print(f"{'='*60}")
        
        result = self.nodes.evaluator_node(state)
        
        # Track this state
        self.tracker.track_iteration(
            iteration=result["iteration"],
            node_name="evaluator",
            state={**state, **result}
        )
        
        print(f"✓ Verdict: {result['verdict']}")
        if result.get("correction_metrics"):
            crs = result["correction_metrics"].get("correction_reasoning_score", 0)
            print(f"✓ CRS: {crs:.3f}")
        
        return result
    
    def run(self, algorithm: str, assumptions: str) -> tuple[dict, ExecutionTracker]:
        """
        Run the complete workflow with full state tracking.
        
        Returns:
            (final_state, tracker) tuple
        """
        # Set metadata
        self.tracker.set_metadata(algorithm, assumptions, self.config)
        
        # Initialize state
        initial_state = self.state_manager.initialize_state(algorithm, assumptions)
        
        print(f"\n{'#'*60}")
        print(f"# CONVERGENCE PROOF GENERATION STARTED")
        print(f"# Algorithm: {algorithm[:50]}...")
        print(f"# Max Iterations: {self.max_iterations}")
        print(f"# Multi-Judge: {self.config['use_multi_judge']}")
        print(f"{'#'*60}\n")
        
        # Run workflow
        final_state = None
        for output in self.app.stream(initial_state):
            for node_name, node_data in output.items():
                final_state = node_data
        
        # Finalize tracking
        if final_state:
            self.tracker.finalize(
                final_verdict=final_state.get("verdict", "UNKNOWN"),
                final_proof=final_state.get("current_proof", "")
            )
            
            # Save everything
            log_path = self.tracker.save()
            self.tracker.save_proofs_separately()
            
            print(f"\n{'#'*60}")
            print(f"# EXECUTION COMPLETE")
            print(f"# Final Verdict: {final_state.get('verdict')}")
            print(f"# Total Iterations: {final_state.get('iteration')}")
            print(f"# Log saved to: {log_path}")
            print(f"{'#'*60}\n")
        
        return final_state, self.tracker
