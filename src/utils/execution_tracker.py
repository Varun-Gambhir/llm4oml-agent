# ============================================================================
# File: src/utils/execution_tracker.py
# ============================================================================
"""Complete execution tracker that saves all iteration states."""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
from ..models.types import AgentState


class ExecutionTracker:
    """
    Tracks complete execution history including all iterations.
    Saves full proof content, feedback, metrics, and error sets for each iteration.
    """
    
    def __init__(self, output_dir: str = "output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        self.execution_log: Dict[str, Any] = {
            "metadata": {
                "start_time": datetime.now().isoformat(),
                "end_time": None,
                "status": "running"
            },
            "configuration": {},
            "algorithm_description": "",
            "assumptions": "",
            "iterations": []
        }
    
    def set_metadata(self, algorithm: str, assumptions: str, config: Dict[str, Any] = None):
        """Set initial metadata."""
        self.execution_log["algorithm_description"] = algorithm
        self.execution_log["assumptions"] = assumptions
        if config:
            self.execution_log["configuration"] = config
    
    def track_iteration(
        self,
        iteration: int,
        node_name: str,
        state: AgentState,
        additional_data: Optional[Dict[str, Any]] = None
    ):
        """
        Track a complete iteration with all state information.
        
        Args:
            iteration: Iteration number
            node_name: Name of node that produced this state
            state: Complete agent state
            additional_data: Any additional data to track
        """
        # Ensure we have enough iteration slots
        while len(self.execution_log["iterations"]) < iteration:
            self.execution_log["iterations"].append({
                "iteration_number": len(self.execution_log["iterations"]) + 1,
                "nodes": {}
            })
        
        # Get current iteration log
        iter_log = self.execution_log["iterations"][iteration - 1]
        
        # Create node entry
        node_entry = {
            "timestamp": datetime.now().isoformat(),
            "node_name": node_name
        }
        
        # Add state information based on node type
        if node_name == "prover":
            node_entry.update({
                "proof_content": state.get("current_proof", ""),
                "proof_length": len(state.get("current_proof", "")),
                "previous_proof": state.get("previous_proof", ""),
                "iteration": state.get("iteration", 0)
            })
        
        elif node_name == "evaluator":
            node_entry.update({
                "feedback": state.get("feedback", ""),
                "verdict": state.get("verdict", ""),
                "metrics": state.get("metrics", {}),
                "error_set": self._serialize_error_set(state.get("error_set_current", {})),
                "correction_metrics": state.get("correction_metrics"),
                "judge_evaluations": state.get("judge_evaluations", []),
                "judge_reliability": state.get("judge_reliability", [])
            })
        
        # Add additional data if provided
        if additional_data:
            node_entry.update(additional_data)
        
        # Store in iteration log
        iter_log["nodes"][node_name] = node_entry
        
        # Update iteration summary
        self._update_iteration_summary(iteration)
    
    def _serialize_error_set(self, error_set: Dict) -> Dict:
        """Convert error set to JSON-serializable format."""
        if not error_set:
            return {}
        
        return {
            "hallucinations": list(error_set.get("hallucinations", set())),
            "missing_steps": list(error_set.get("missing_steps", set())),
            "operator_errors": list(error_set.get("operator_errors", set())),
            "assumption_violations": list(error_set.get("assumption_violations", set()))
        }
    
    def _update_iteration_summary(self, iteration: int):
        """Update summary fields for an iteration."""
        iter_log = self.execution_log["iterations"][iteration - 1]
        
        # Extract key metrics from evaluator node
        if "evaluator" in iter_log["nodes"]:
            eval_data = iter_log["nodes"]["evaluator"]
            iter_log["summary"] = {
                "verdict": eval_data.get("verdict"),
                "completeness_score": eval_data.get("metrics", {}).get("completeness_score"),
                "has_errors": any([
                    eval_data.get("metrics", {}).get("HA", False),
                    eval_data.get("metrics", {}).get("MS", False),
                    eval_data.get("metrics", {}).get("OP", False)
                ])
            }
            
            # Add CRS if available
            if eval_data.get("correction_metrics"):
                iter_log["summary"]["crs"] = eval_data["correction_metrics"].get(
                    "correction_reasoning_score"
                )
    
    def finalize(self, final_verdict: str, final_proof: str):
        """Finalize the execution log."""
        self.execution_log["metadata"]["end_time"] = datetime.now().isoformat()
        self.execution_log["metadata"]["status"] = "completed"
        
        self.execution_log["final_results"] = {
            "verdict": final_verdict,
            "total_iterations": len(self.execution_log["iterations"]),
            "final_proof_length": len(final_proof),
            "convergence_achieved": final_verdict in ["PASS", "PASS_MINOR"]
        }
        
        # Calculate overall statistics
        self._calculate_statistics()
    
    def _calculate_statistics(self):
        """Calculate aggregate statistics across all iterations."""
        iterations = self.execution_log["iterations"]
        
        if not iterations:
            return
        
        stats = {
            "total_errors_identified": 0,
            "errors_by_type": {
                "hallucinations": 0,
                "missing_steps": 0,
                "operator_errors": 0,
                "assumption_violations": 0
            },
            "verdict_progression": [],
            "crs_progression": [],
            "proof_length_progression": []
        }
        
        for iter_log in iterations:
            # Verdict progression
            if "summary" in iter_log:
                stats["verdict_progression"].append(iter_log["summary"].get("verdict"))
                if iter_log["summary"].get("crs") is not None:
                    stats["crs_progression"].append(iter_log["summary"]["crs"])
            
            # Error counts
            if "evaluator" in iter_log["nodes"]:
                error_set = iter_log["nodes"]["evaluator"].get("error_set", {})
                for error_type, steps in error_set.items():
                    if steps:
                        stats["errors_by_type"][error_type] += len(steps)
                        stats["total_errors_identified"] += len(steps)
            
            # Proof length
            if "prover" in iter_log["nodes"]:
                length = iter_log["nodes"]["prover"].get("proof_length", 0)
                stats["proof_length_progression"].append(length)
        
        self.execution_log["statistics"] = stats
    
    def save(self, filename: str = None) -> Path:
        """
        Save execution log to JSON file.
        
        Args:
            filename: Optional custom filename
            
        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"execution_log_{timestamp}.json"
        
        filepath = self.output_dir / filename
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.execution_log, f, indent=2, default=str)
        
        return filepath
    
    def save_proofs_separately(self):
        """Save each iteration's proof as separate LaTeX file."""
        proofs_dir = self.output_dir / "proofs"
        proofs_dir.mkdir(exist_ok=True)
        
        for iter_log in self.execution_log["iterations"]:
            if "prover" in iter_log["nodes"]:
                iter_num = iter_log["iteration_number"]
                proof_content = iter_log["nodes"]["prover"]["proof_content"]
                
                proof_path = proofs_dir / f"proof_iteration_{iter_num}.tex"
                with open(proof_path, "w", encoding="utf-8") as f:
                    f.write(proof_content)
    
    def get_log(self) -> Dict[str, Any]:
        """Get current execution log."""
        return self.execution_log
