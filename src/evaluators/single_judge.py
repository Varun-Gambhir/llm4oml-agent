# ============================================================================
# File: src/evaluators/single_judge.py (UPDATED)
# ============================================================================
"""Single LLM judge evaluator with provider abstraction."""

import time
from typing import Optional
from langchain_core.output_parsers import PydanticOutputParser

from .base_evaluator import BaseEvaluator
from ..models.schemas import EvaluationMetrics
from ..utils.prompts import PromptManager
from ..llm.provider_factory import LLMProviderFactory


class SingleJudge(BaseEvaluator):
    """Single LLM evaluator with retry logic."""
    
    def __init__(
        self,
        provider_name: str,
        model_name: str,
        judge_id: str,
        api_key: str = None,
        temperature: float = 0.5,
        max_tokens: int = 8192,
        timeout: int = 600,
        max_retries: int = 3
    ):
        super().__init__(model_name, judge_id)
        
        self.provider_name = provider_name
        self.llm = LLMProviderFactory.create(
            provider_name=provider_name,
            model_name=model_name,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            max_retries=max_retries
        )
        
        self.prompt_manager = PromptManager()
        self.parser = PydanticOutputParser(pydantic_object=EvaluationMetrics)
    
    def evaluate(
        self,
        proof: str,
        feedback: Optional[str] = None,
        iteration: int = 1
    ) -> EvaluationMetrics:
        """Evaluate proof using single LLM judge."""
        
        start_time = time.time()
        
        # Get appropriate prompt
        if iteration == 1:
            prompt = self.prompt_manager.get_initial_evaluation_prompt(proof)
        else:
            prompt = self.prompt_manager.get_verification_prompt(feedback, proof)
        
        # Try JSON parsing with format instructions
        result = self._try_json_parsing(prompt)
        
        # Emergency fallback
        if result is None:
            result = self._create_error_result(proof)
        
        return result
    
    def _try_json_parsing(self, prompt: str) -> Optional[EvaluationMetrics]:
        """Parse JSON response."""
        try:
            format_instructions = self.parser.get_format_instructions()
            full_prompt = f"{prompt}\n\n{format_instructions}\n\nIMPORTANT: Return ONLY valid JSON."
            
            messages = [{"role": "user", "content": full_prompt}]
            response = self.llm.invoke(messages)
            
            # Clean response
            clean_response = response.strip()
            if clean_response.startswith("```json"):
                clean_response = clean_response.replace("```json", "").replace("```", "").strip()
            
            result = self.parser.parse(clean_response)
            return result
            
        except Exception as e:
            print(f"[{self.judge_id}] JSON parsing failed: {e}")
            return None
    
    def _create_error_result(self, proof: str) -> EvaluationMetrics:
        """Emergency fallback result."""
        return EvaluationMetrics(
            hallucination_error=False,
            missing_step=False,
            operator_error=False,
            completeness_score=0,
            assumption_use_score=0,
            overall_verdict="FAIL",
            detailed_feedback=f"SYSTEM ERROR: Evaluation failed after retries. Judge: {self.judge_id}"
        )
