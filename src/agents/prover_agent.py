# ============================================================================
# File: src/agents/prover_agent.py
# ============================================================================
"""Proof generation agent."""

from langchain_core.messages import HumanMessage
from .base_agent import BaseAgent
from ..utils.parsers import ResponseParser


class ProverAgent(BaseAgent):
    """Agent for generating convergence proofs."""
    
    def execute(
        self,
        algorithm: str,
        assumptions: str,
        iteration: int = 0,
        previous_proof: str = "",
        feedback: str = ""
    ) -> dict:
        """
        Generate or regenerate a convergence proof.
        
        Returns:
            dict with 'current_proof', 'previous_proof', 'iteration'
        """
        print(f"[ProverAgent] Generating proof (Iteration {iteration + 1})...")
        
        try:
            if iteration == 0:
                # Initial proof generation
                prompt = self.prompt_manager.get_generation_prompt(algorithm, assumptions)
                response = self.llm.invoke([HumanMessage(content=prompt)])
                proof = ResponseParser.clean_latex(response.content)
                
                return {
                    "current_proof": proof,
                    "previous_proof": "",
                    "iteration": 1
                }
            else:
                # Correction iteration
                prompt = self.prompt_manager.get_correction_prompt(previous_proof, feedback)
                response = self.llm.invoke([HumanMessage(content=prompt)])
                proof = ResponseParser.clean_latex(response.content)
                
                return {
                    "previous_proof": previous_proof,
                    "current_proof": proof,
                    "iteration": iteration + 1
                }
                
        except Exception as e:
            print(f"[ProverAgent] Error: {e}")
            return {
                "current_proof": previous_proof + f"\n% ERROR: {str(e)}",
                "previous_proof": previous_proof,
                "iteration": iteration + 1
            }
