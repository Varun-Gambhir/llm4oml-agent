# ============================================================================
# File: src/agents/base_agent.py
# ============================================================================
"""Base agent class."""

from abc import ABC, abstractmethod
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from ..utils.prompts import PromptManager


class BaseAgent(ABC):
    """Abstract base class for agents."""
    
    def __init__(self, model_name: str, temperature: float = 0.5, max_tokens: int = 8192):
        self.model_name = model_name
        self.llm = ChatNVIDIA(
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=1
        )
        self.prompt_manager = PromptManager()
    
    @abstractmethod
    def execute(self, **kwargs) -> dict:
        """Execute agent logic."""
        pass
