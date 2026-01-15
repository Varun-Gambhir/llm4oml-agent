# ============================================================================
# File: src/llm/openai_provider.py
# ============================================================================
"""OpenAI provider (for direct OpenAI API or compatible endpoints)."""

import requests
from .base_provider import BaseLLMProvider


class OpenAIProvider(BaseLLMProvider):
    """OpenAI-compatible provider."""
    
    def __init__(
        self,
        model_name: str,
        api_key: str,
        base_url: str = "https://api.openai.com/v1/chat/completions",
        **kwargs
    ):
        super().__init__(model_name, api_key, **kwargs)
        self.base_url = base_url
    
    def _make_request(self, messages: list, **kwargs) -> str:
        """Make request to OpenAI API."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }
        
        response = requests.post(
            self.base_url,
            headers=headers,
            json=payload,
            timeout=self.timeout
        )
        
        response.raise_for_status()
        data = response.json()
        
        return data["choices"][0]["message"]["content"]
