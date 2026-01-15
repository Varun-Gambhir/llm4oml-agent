# ============================================================================
# File: src/llm/provider_factory.py
# ============================================================================
"""Factory for creating LLM providers."""

import os
from typing import Optional
from .base_provider import BaseLLMProvider
from .nvidia_provider import NVIDIAProvider
from .openrouter_provider import OpenRouterProvider
from .openai_provider import OpenAIProvider
from .anthropic_provider import AnthropicProvider


class LLMProviderFactory:
    """Factory for creating LLM providers based on configuration."""
    
    PROVIDERS = {
        "nvidia": NVIDIAProvider,
        "openrouter": OpenRouterProvider,
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider
    }
    
    @classmethod
    def create(
        cls,
        provider_name: str,
        model_name: str,
        api_key: Optional[str] = None,
        **kwargs
    ) -> BaseLLMProvider:
        """
        Create an LLM provider instance.
        
        Args:
            provider_name: Name of provider (nvidia, openrouter, openai, anthropic)
            model_name: Model identifier
            api_key: API key (if None, reads from environment)
            **kwargs: Additional provider-specific parameters
            
        Returns:
            LLM provider instance
        """
        provider_name = provider_name.lower()
        
        if provider_name not in cls.PROVIDERS:
            raise ValueError(
                f"Unknown provider: {provider_name}. "
                f"Available: {list(cls.PROVIDERS.keys())}"
            )
        
        # Get API key from environment if not provided
        if api_key is None:
            env_var_map = {
                "nvidia": "NVIDIA_API_KEY",
                "openrouter": "OPENROUTER_API_KEY",
                "openai": "OPENAI_API_KEY",
                "anthropic": "ANTHROPIC_API_KEY"
            }
            
            env_var = env_var_map.get(provider_name)
            api_key = os.getenv(env_var)
            
            if not api_key:
                raise ValueError(
                    f"API key not found. Set {env_var} environment variable "
                    f"or pass api_key parameter."
                )
        
        provider_class = cls.PROVIDERS[provider_name]
        return provider_class(model_name, api_key, **kwargs)
    
    @classmethod
    def from_config(cls, config: dict) -> BaseLLMProvider:
        """
        Create provider from configuration dictionary.
        
        Args:
            config: Dict with keys: provider, model, api_key (optional), etc.
            
        Returns:
            LLM provider instance
        """
        provider_name = config.pop("provider")
        model_name = config.pop("model")
        api_key = config.pop("api_key", None)
        
        return cls.create(provider_name, model_name, api_key, **config)
