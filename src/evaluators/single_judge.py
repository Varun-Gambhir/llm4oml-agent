# ============================================================================
# File: src/evaluators/single_judge.py
# ============================================================================
"""Single LLM judge evaluator."""

import time
from typing import Optional
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import PydanticOutputParser

from .base_evaluator import BaseEvaluator
from ..models.schemas import EvaluationMetrics
from ..utils.prompts import PromptManager


class SingleJudge(BaseEvaluator):
    """Single LLM evaluator."""
    
    def __init__(
        self,
        model_name: str,
        judge_id: str,
        temperature: float = 0.5,
        max_tokens: int = 8192
    ):
        super().__init__(model_name, judge_id)
        
        self.llm = ChatNVIDIA(
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=1
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
        
        # Get appropriate prompt based on iteration
        if iteration == 1:
            prompt = self.prompt_manager.get_initial_evaluation_prompt(proof)
        else:
            prompt = self.prompt_manager.get_verification_prompt(feedback, proof)
        
        # Try structured output first
        result = self._try_structured_output(prompt)
        
        # Fallback to JSON parsing if needed
        if result is None:
            result = self._try_json_parsing(prompt)
        
        # Emergency fallback
        if result is None:
            result = self._create_error_result(proof)
        
        return result
    
    def _try_structured_output(self, prompt: str) -> Optional[EvaluationMetrics]:
        """Attempt to use structured output API."""
        try:
            structured_llm = self.llm.with_structured_output(EvaluationMetrics)
            result = structured_llm.invoke([HumanMessage(content=prompt)])
            return result
        except Exception as e:
            print(f"[{self.judge_id}] Structured API failed: {e}")
            return None
    
    def _try_json_parsing(self, prompt: str) -> Optional[EvaluationMetrics]:
        """Fallback: explicit JSON parsing."""
        try:
            format_instructions = self.parser.get_format_instructions()
            full_prompt = f"{prompt}\n\n{format_instructions}\n\nReturn ONLY valid JSON."
            
            response = self.llm.invoke([HumanMessage(content=full_prompt)])
            result = self.parser.parse(response.content)
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
            detailed_feedback=f"SYSTEM ERROR: Parsing failed. Judge: {self.judge_id}"
        )
