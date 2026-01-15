# ============================================================================
# File: src/llm/anthropic_provider.py
# ============================================================================
"""Anthropic Claude provider."""

import requests
from .base_provider import BaseLLMProvider


class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude provider."""
    
    def __init__(self, model_name: str, api_key: str, **kwargs):
        super().__init__(model_name, api_key, **kwargs)
        self.base_url = "https://api.anthropic.com/v1/messages"
        self.api_version = "2023-06-01"
    
    def _make_request(self, messages: list, **kwargs) -> str:
        """Make request to Anthropic API."""
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": self.api_version,
            "content-type": "application/json"
        }
        
        # Anthropic requires system message separately
        system_msg = None
        user_messages = []
        
        for msg in messages:
            if msg["role"] == "system":
                system_msg = msg["content"]
            else:
                user_messages.append(msg)
        
        payload = {
            "model": self.model_name,
            "messages": user_messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature
        }
        
        if system_msg:
            payload["system"] = system_msg
        
        response = requests.post(
            self.base_url,
            headers=headers,
            json=payload,
            timeout=self.timeout
        )
        
        response.raise_for_status()
        data = response.json()
        
        return data["content"][0]["text"]
