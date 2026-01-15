# ============================================================================
# File: src/agents/prover_agent.py (UPDATED)
# ============================================================================
"""Proof generation agent with provider abstraction."""

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
        """Generate or regenerate a convergence proof."""
        print(f"[ProverAgent] Generating proof (Iteration {iteration + 1})...")
        
        try:
            if iteration == 0:
                # Initial proof generation
                prompt = self.prompt_manager.get_generation_prompt(algorithm, assumptions)
                messages = [{"role": "user", "content": prompt}]
                
                response = self.llm.invoke(messages)
                proof = ResponseParser.clean_latex(response)
                
                return {
                    "current_proof": proof,
                    "previous_proof": "",
                    "iteration": 1
                }
            else:
                # Correction iteration
                prompt = self.prompt_manager.get_correction_prompt(previous_proof, feedback)
                messages = [{"role": "user", "content": prompt}]
                
                response = self.llm.invoke(messages)
                proof = ResponseParser.clean_latex(response)
                
                return {
                    "previous_proof": previous_proof,
                    "current_proof": proof,
                    "iteration": iteration + 1
                }
                
        except Exception as e:
            print(f"[ProverAgent] Error after all retries: {e}")
            error_msg = f"\n% ERROR: {str(e)}\n% Agent could not generate proof after retries."
            return {
                "current_proof": previous_proof + error_msg if previous_proof else error_msg,
                "previous_proof": previous_proof,
                "iteration": iteration + 1
            }
