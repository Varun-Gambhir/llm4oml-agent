# ============================================================================
# File: src/agents/base_agent.py (UPDATED)
# ============================================================================
"""Base agent class with provider abstraction."""

from abc import ABC, abstractmethod
from ..llm.provider_factory import LLMProviderFactory
from ..utils.prompts import PromptManager


class BaseAgent(ABC):
    """Abstract base class for agents."""
    
    def __init__(
        self,
        provider_name: str = "nvidia",
        model_name: str = "openai/gpt-oss-120b",
        api_key: str = None,
        temperature: float = 0.5,
        max_tokens: int = 8192,
        timeout: int = 600,
        max_retries: int = 3
    ):
        self.provider_name = provider_name
        self.model_name = model_name
        
        # Create LLM provider with retry logic
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
    
    @abstractmethod
    def execute(self, **kwargs) -> dict:
        """Execute agent logic."""
        pass
