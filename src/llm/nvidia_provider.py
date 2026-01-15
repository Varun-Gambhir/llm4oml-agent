# ============================================================================
# File: src/llm/nvidia_provider.py
# ============================================================================
"""NVIDIA AI Endpoints provider."""

from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.messages import HumanMessage, SystemMessage
from .base_provider import BaseLLMProvider


class NVIDIAProvider(BaseLLMProvider):
    """NVIDIA AI Endpoints LLM provider."""
    
    def __init__(self, model_name: str, api_key: str, **kwargs):
        super().__init__(model_name, api_key, **kwargs)
        
        self.client = ChatNVIDIA(
            model=model_name,
            api_key=api_key,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=1,
            timeout=self.timeout
        )
    
    def _make_request(self, messages: list, **kwargs) -> str:
        """Make request to NVIDIA API."""
        # Convert to LangChain message format
        lc_messages = []
        for msg in messages:
            if msg["role"] == "system":
                lc_messages.append(SystemMessage(content=msg["content"]))
            elif msg["role"] == "user":
                lc_messages.append(HumanMessage(content=msg["content"]))
        
        response = self.client.invoke(lc_messages)
        return response.content
