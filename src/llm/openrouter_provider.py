# ============================================================================
# File: src/llm/openrouter_provider.py
# ============================================================================
"""OpenRouter provider."""

import requests
from .base_provider import BaseLLMProvider


class OpenRouterProvider(BaseLLMProvider):
    """OpenRouter LLM provider."""
    
    def __init__(self, model_name: str, api_key: str, **kwargs):
        super().__init__(model_name, api_key, **kwargs)
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
    
    def _make_request(self, messages: list, **kwargs) -> str:
        """Make request to OpenRouter API."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": kwargs.get("http_referer", "http://localhost:3000"),
            "X-Title": kwargs.get("x_title", "Convergence Proof Agent")
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
